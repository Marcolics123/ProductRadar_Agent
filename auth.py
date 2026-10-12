import hashlib
import os
import secrets
from datetime import timedelta
from typing import Annotated
from uuid import uuid4

from argon2 import PasswordHasher
from argon2.exceptions import VerificationError, VerifyMismatchError
try:
    # 🟦【第四批兼容】argon2-cffi 新旧版本只是在异常类命名上不同。
    from argon2.exceptions import InvalidHashError
except ImportError:  # pragma: no cover - 由安装版本决定
    from argon2.exceptions import InvalidHash as InvalidHashError
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import delete, select, update
from sqlalchemy.exc import IntegrityError

from database import UserRow, UserSessionRow, connect, utc_now
from rate_limit import RateLimitExceeded, enforce_rate_limit


router = APIRouter(prefix="/auth", tags=["auth"])
password_hasher = PasswordHasher()
# 不存在的用户名也执行一次同等强度校验，减少根据响应时间枚举账号的风险。
DUMMY_PASSWORD_HASH = password_hasher.hash("productradar-dummy-password")
SESSION_COOKIE_NAME = "productradar_session"
CSRF_COOKIE_NAME = "productradar_csrf"
SESSION_TTL_DAYS = 7


# region 🟦【第四批修改 6】邀请制账号与登录数据结构
class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64, pattern=r"^[a-zA-Z0-9_.-]+$")
    password: str = Field(min_length=12, max_length=128)
    invite_code: str = Field(min_length=1, max_length=256)

    @field_validator("username", mode="after")
    @classmethod
    def normalize_username(cls, value: str) -> str:
        return value.strip().lower()


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)

    @field_validator("username", mode="after")
    @classmethod
    def normalize_username(cls, value: str) -> str:
        return value.strip().lower()


class AuthUser(BaseModel):
    user_id: str
    username: str


class AuthSession(BaseModel):
    user: AuthUser
    csrf_token: str
# endregion


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _cookie_secure() -> bool:
    value = os.environ.get("PRODUCTRADAR_COOKIE_SECURE", "true").strip().lower()
    if value not in {"true", "false"}:
        raise RuntimeError("PRODUCTRADAR_COOKIE_SECURE 必须是 true 或 false")
    return value == "true"


def _set_session_cookies(
    response: Response,
    session_token: str,
    csrf_token: str,
) -> None:
    """认证 Cookie 不允许 JavaScript 读取；CSRF Cookie 仅用于恢复同一会话的校验值。"""

    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_token,
        max_age=SESSION_TTL_DAYS * 24 * 60 * 60,
        httponly=True,
        secure=_cookie_secure(),
        samesite="strict",
        path="/",
    )
    response.set_cookie(
        key=CSRF_COOKIE_NAME,
        value=csrf_token,
        max_age=SESSION_TTL_DAYS * 24 * 60 * 60,
        httponly=False,
        secure=_cookie_secure(),
        samesite="strict",
        path="/",
    )


def _delete_session_cookie(response: Response) -> None:
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        secure=_cookie_secure(),
        httponly=True,
        samesite="strict",
        path="/",
    )
    response.delete_cookie(
        key=CSRF_COOKIE_NAME,
        secure=_cookie_secure(),
        httponly=False,
        samesite="strict",
        path="/",
    )


def _protect_auth_endpoint(request: Request, bucket: str, limit: int) -> None:
    client_host = request.client.host if request.client is not None else "unknown"
    try:
        enforce_rate_limit(
            bucket,
            client_host,
            limit=limit,
            window_seconds=300,
        )
    except RateLimitExceeded as error:
        raise HTTPException(
            status_code=429,
            detail=str(error),
            headers={"Retry-After": "300"},
        ) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


def _new_session(user_id: str) -> tuple[str, str]:
    raw_token = secrets.token_urlsafe(32)
    csrf_token = secrets.token_urlsafe(32)
    now = utc_now()
    with connect() as session:
        # 顺手清理当前用户的过期 Session，避免长期积累无用记录。
        session.execute(
            delete(UserSessionRow).where(
                UserSessionRow.user_id == user_id,
                UserSessionRow.expires_at <= now,
            )
        )
        session.add(
            UserSessionRow(
                token_hash=_token_hash(raw_token),
                csrf_token_hash=_token_hash(csrf_token),
                user_id=user_id,
                expires_at=now + timedelta(days=SESSION_TTL_DAYS),
                created_at=now,
            )
        )
    return raw_token, csrf_token


def _create_user(username: str, password: str) -> AuthUser:
    user_id = str(uuid4())
    try:
        with connect() as session:
            session.add(
                UserRow(
                    user_id=user_id,
                    username=username,
                    password_hash=password_hasher.hash(password),
                    is_active=True,
                    created_at=utc_now(),
                )
            )
    except IntegrityError as error:
        raise ValueError("用户名已存在") from error
    return AuthUser(user_id=user_id, username=username)


def _authenticate(username: str, password: str) -> AuthUser | None:
    with connect() as session:
        row = session.scalar(select(UserRow).where(UserRow.username == username))

    stored_hash = row.password_hash if row is not None else DUMMY_PASSWORD_HASH
    try:
        password_valid = password_hasher.verify(stored_hash, password)
    except (InvalidHashError, VerifyMismatchError, VerificationError):
        password_valid = False

    if row is None or not row.is_active or not password_valid:
        return None
    if password_hasher.check_needs_rehash(row.password_hash):
        with connect() as session:
            session.execute(
                update(UserRow)
                .where(UserRow.user_id == row.user_id)
                .values(password_hash=password_hasher.hash(password))
            )
    return AuthUser(user_id=row.user_id, username=row.username)


class _AuthenticatedSession(BaseModel):
    user: AuthUser
    session_token_hash: str
    csrf_token_hash: str


# region 🟦【第四批修改 7】Cookie Session、CSRF 校验和当前用户依赖
def _get_authenticated_session(request: Request) -> _AuthenticatedSession:
    raw_token = request.cookies.get(SESSION_COOKIE_NAME, "")
    if not raw_token:
        raise HTTPException(status_code=401, detail="请先登录")

    token_hash = _token_hash(raw_token)
    now = utc_now()
    with connect() as session:
        row = session.execute(
            select(UserSessionRow, UserRow)
            .join(UserRow, UserRow.user_id == UserSessionRow.user_id)
            .where(
                UserSessionRow.token_hash == token_hash,
                UserSessionRow.expires_at > now,
                UserRow.is_active.is_(True),
            )
        ).one_or_none()
        if row is None:
            raise HTTPException(status_code=401, detail="登录已失效，请重新登录")
        session_row, user_row = row
    return _AuthenticatedSession(
        user=AuthUser(user_id=user_row.user_id, username=user_row.username),
        session_token_hash=session_row.token_hash,
        csrf_token_hash=session_row.csrf_token_hash,
    )


def require_user(request: Request) -> AuthUser:
    authenticated = _get_authenticated_session(request)
    if request.method not in {"GET", "HEAD", "OPTIONS"}:
        supplied_csrf = request.headers.get("X-CSRF-Token", "")
        if not supplied_csrf or not secrets.compare_digest(
            _token_hash(supplied_csrf),
            authenticated.csrf_token_hash,
        ):
            raise HTTPException(status_code=403, detail="CSRF 校验失败，请刷新页面")
    return authenticated.user


CurrentUser = Annotated[AuthUser, Depends(require_user)]
# endregion


@router.post("/register", response_model=AuthSession)
def register(
    request: RegisterRequest,
    response: Response,
    http_request: Request,
) -> AuthSession:
    _protect_auth_endpoint(http_request, "register", 5)
    invite_code = os.environ.get("PRODUCTRADAR_INVITE_CODE", "")
    if not invite_code:
        raise HTTPException(status_code=403, detail="当前未开放新用户注册")
    if not secrets.compare_digest(request.invite_code, invite_code):
        raise HTTPException(status_code=403, detail="邀请码无效")

    try:
        user = _create_user(request.username, request.password)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    session_token, csrf_token = _new_session(user.user_id)
    _set_session_cookies(response, session_token, csrf_token)
    return AuthSession(user=user, csrf_token=csrf_token)


@router.post("/login", response_model=AuthSession)
def login(
    request: LoginRequest,
    response: Response,
    http_request: Request,
) -> AuthSession:
    _protect_auth_endpoint(http_request, "login", 10)
    user = _authenticate(request.username, request.password)
    if user is None:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    session_token, csrf_token = _new_session(user.user_id)
    _set_session_cookies(response, session_token, csrf_token)
    return AuthSession(user=user, csrf_token=csrf_token)


@router.get("/me", response_model=AuthSession)
def me(request: Request, response: Response) -> AuthSession:
    """页面刷新时恢复 CSRF Token，避免多个标签页互相使对方失效。"""

    authenticated = _get_authenticated_session(request)
    csrf_token = request.cookies.get(CSRF_COOKIE_NAME, "")
    if csrf_token and secrets.compare_digest(
        _token_hash(csrf_token),
        authenticated.csrf_token_hash,
    ):
        return AuthSession(user=authenticated.user, csrf_token=csrf_token)

    # 🟦【第四批修改：会话恢复】只在 CSRF Cookie 缺失或不匹配时轮换。
    csrf_token = secrets.token_urlsafe(32)
    with connect() as session:
        session.execute(
            update(UserSessionRow)
            .where(UserSessionRow.token_hash == authenticated.session_token_hash)
            .values(csrf_token_hash=_token_hash(csrf_token))
        )
    response.set_cookie(
        key=CSRF_COOKIE_NAME,
        value=csrf_token,
        max_age=SESSION_TTL_DAYS * 24 * 60 * 60,
        httponly=False,
        secure=_cookie_secure(),
        samesite="strict",
        path="/",
    )
    return AuthSession(user=authenticated.user, csrf_token=csrf_token)


@router.post("/logout", status_code=204)
def logout(request: Request, response: Response, _: CurrentUser) -> Response:
    raw_token = request.cookies.get(SESSION_COOKIE_NAME, "")
    if raw_token:
        with connect() as session:
            session.execute(
                delete(UserSessionRow).where(
                    UserSessionRow.token_hash == _token_hash(raw_token)
                )
            )
    _delete_session_cookie(response)
    response.status_code = 204
    return response
