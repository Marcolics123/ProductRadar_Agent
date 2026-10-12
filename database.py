import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    func,
    Integer,
    String,
    Text,
    create_engine,
    delete,
    inspect,
    select,
    text,
    update,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from agent import ResearchPlan, ResearchResult
from evaluation import EvalResult
from tracing import TraceEvent
from usage import UsageStats


# 🟦【第四批修复：本地与 Docker 统一密码来源】
# database.py 被 PyCharm、Alembic 或 Worker 单独导入时也会读取项目 .env。
load_dotenv(Path(__file__).parent / ".env")

_local_postgres_password = quote_plus(
    os.environ.get("POSTGRES_PASSWORD", "productradar")
)
DEFAULT_DATABASE_URL = (
    "postgresql+psycopg://productradar:"
    f"{_local_postgres_password}@127.0.0.1:15432/productradar"
)
DATABASE_URL = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ResearchStatus(str, Enum):
    WAITING_FOR_APPROVAL = "WAITING_FOR_APPROVAL"
    QUEUED = "QUEUED"
    RESEARCHING = "RESEARCHING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ResearchFeedback(BaseModel):
    """用户对一份已完成报告的最小反馈，用于后续在线 Eval。"""

    helpful: bool
    comment: str | None = None
    created_at: datetime
    updated_at: datetime


class ResearchTask(BaseModel):
    research_id: str
    # 第二批开始使用结构化产品字段；默认值用于兼容第一批已经保存的旧任务。
    own_product: str = ""
    competitors: list[str] = Field(default_factory=list)
    research_question: str
    research_plan: ResearchPlan
    status: ResearchStatus
    result: ResearchResult | None = None
    eval_result: EvalResult | None = None
    usage: UsageStats = Field(default_factory=UsageStats)
    feedback: ResearchFeedback | None = None
    current_step: str | None = None
    error_message: str | None = None
    attempt_count: int = Field(default=0, ge=0)
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class ResearchTaskSummary(BaseModel):
    """历史列表只返回摘要，避免每次都传输完整 Evidence 和 Analysis。"""

    research_id: str
    own_product: str
    competitors: list[str]
    research_question: str
    status: ResearchStatus
    current_step: str | None
    created_at: datetime
    updated_at: datetime


class Base(DeclarativeBase):
    pass


# region 🟦【第四批修改 5】多用户和持久会话数据表
class UserRow(Base):
    """只保存密码哈希，不保存明文密码。"""

    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    username: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, index=True
    )
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    is_active: Mapped[bool] = mapped_column(nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class UserSessionRow(Base):
    """数据库只保存 Session Token 的 SHA-256，不保存浏览器中的原始 Token。"""

    __tablename__ = "user_sessions"

    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    csrf_token_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
# endregion


class ResearchTaskRow(Base):
    __tablename__ = "research_tasks"
    __table_args__ = (
        CheckConstraint(
            "status IN ('WAITING_FOR_APPROVAL', 'QUEUED', 'RESEARCHING', "
            "'COMPLETED', 'FAILED')",
            name="ck_research_tasks_status",
        ),
    )

    research_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    # 🟦【第四批修改：数据隔离】旧任务允许为空；所有新任务必须由服务层传入 owner_id。
    owner_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    own_product: Mapped[str | None] = mapped_column(Text)
    competitors_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    research_question: Mapped[str] = mapped_column(Text, nullable=False)
    research_plan_json: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    result_json: Mapped[str | None] = mapped_column(Text)
    usage_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    current_step: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class EvalResultRow(Base):
    __tablename__ = "eval_results"

    research_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("research_tasks.research_id", ondelete="CASCADE"),
        primary_key=True,
    )
    eval_result_json: Mapped[str] = mapped_column(Text, nullable=False)


class ResearchFeedbackRow(Base):
    __tablename__ = "research_feedback"

    research_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("research_tasks.research_id", ondelete="CASCADE"),
        primary_key=True,
    )
    helpful: Mapped[bool] = mapped_column(nullable=False)
    comment: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class TraceEventRow(Base):
    __tablename__ = "trace_events"

    trace_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    research_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("research_tasks.research_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    stage: Mapped[str] = mapped_column(String(32), nullable=False)
    action: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    tool_call_id: Mapped[str | None] = mapped_column(String(128))
    input_json: Mapped[str] = mapped_column(Text, nullable=False)
    output_json: Mapped[str] = mapped_column(Text, nullable=False)
    error_type: Mapped[str | None] = mapped_column(String(128))
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


engine_options: dict[str, object] = {"pool_pre_ping": True}
if DATABASE_URL.startswith("sqlite"):
    engine_options["connect_args"] = {"check_same_thread": False}
else:
    # 🟦【质量报告稳定性修复】数据库停止时快速失败，避免 CLI 和启动过程长时间假死。
    engine_options["connect_args"] = {"connect_timeout": 5}

engine = create_engine(DATABASE_URL, **engine_options)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def connect() -> Iterator[Session]:
    session = SessionLocal()
    try:
        with session.begin():
            yield session
    finally:
        session.close()


def initialize_database() -> None:
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
    # 🟦【第四批修改：启动校验】公网版同时依赖用户、会话和研究任务表。
    required_tables = {"users", "user_sessions", "research_tasks"}
    existing_tables = set(inspect(engine).get_table_names())
    if not required_tables.issubset(existing_tables):
        raise RuntimeError("数据库尚未初始化，请先执行 alembic upgrade head")


def database_is_ready() -> bool:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        required_tables = {"users", "user_sessions", "research_tasks"}
        existing_tables = set(inspect(engine).get_table_names())
        return required_tables.issubset(existing_tables)
    except Exception:
        return False


def _parse_competitors(raw_value: str) -> list[str]:
    try:
        values = json.loads(raw_value)
    except (TypeError, json.JSONDecodeError):
        return []
    return [value for value in values if isinstance(value, str)] if isinstance(values, list) else []


def _parse_usage(raw_value: str) -> UsageStats:
    try:
        return UsageStats.model_validate_json(raw_value)
    except ValueError:
        return UsageStats()


def _to_feedback(row: ResearchFeedbackRow | None) -> ResearchFeedback | None:
    if row is None:
        return None
    return ResearchFeedback(
        helpful=row.helpful,
        comment=row.comment,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _to_research_task(
    row: ResearchTaskRow,
    eval_row: EvalResultRow | None,
    feedback_row: ResearchFeedbackRow | None,
) -> ResearchTask:
    return ResearchTask(
        research_id=row.research_id,
        own_product=row.own_product or "",
        competitors=_parse_competitors(row.competitors_json),
        research_question=row.research_question,
        research_plan=ResearchPlan.model_validate_json(row.research_plan_json),
        status=ResearchStatus(row.status),
        result=(
            ResearchResult.model_validate_json(row.result_json)
            if row.result_json is not None
            else None
        ),
        eval_result=(
            EvalResult.model_validate_json(eval_row.eval_result_json)
            if eval_row is not None
            else None
        ),
        usage=_parse_usage(row.usage_json),
        feedback=_to_feedback(feedback_row),
        current_step=row.current_step,
        error_message=row.error_message,
        attempt_count=row.attempt_count,
        created_at=row.created_at,
        updated_at=row.updated_at,
        started_at=row.started_at,
        completed_at=row.completed_at,
    )


class ActiveResearchLimitExceeded(RuntimeError):
    """单个用户的活跃研究数已达到保护上限。"""


ACTIVE_RESEARCH_STATUSES = {
    ResearchStatus.WAITING_FOR_APPROVAL.value,
    ResearchStatus.QUEUED.value,
    ResearchStatus.RESEARCHING.value,
}


def active_research_count(owner_id: str) -> int:
    """公开服务的容量保护，不属于商业套餐或计费逻辑。"""

    with connect() as session:
        return int(
            session.scalar(
                select(func.count())
                .select_from(ResearchTaskRow)
                .where(
                    ResearchTaskRow.owner_id == owner_id,
                    ResearchTaskRow.status.in_(ACTIVE_RESEARCH_STATUSES),
                )
            )
            or 0
        )


def create_research_task(
    research_task: ResearchTask,
    owner_id: str,
    max_active_tasks: int,
) -> None:
    """锁定用户行后检查活跃任务数，避免并发请求一起绕过上限。"""

    with connect() as session:
        # 🟦【第四批修改：并发保护】同一用户创建任务时串行检查容量。
        owner = session.scalar(
            select(UserRow)
            .where(UserRow.user_id == owner_id, UserRow.is_active.is_(True))
            .with_for_update()
        )
        if owner is None:
            raise LookupError("用户不存在或已停用")
        active_count = int(
            session.scalar(
                select(func.count())
                .select_from(ResearchTaskRow)
                .where(
                    ResearchTaskRow.owner_id == owner_id,
                    ResearchTaskRow.status.in_(ACTIVE_RESEARCH_STATUSES),
                )
            )
            or 0
        )
        if active_count >= max_active_tasks:
            raise ActiveResearchLimitExceeded(
                f"每个用户最多同时保留 {max_active_tasks} 个未完成研究任务"
            )

        session.add(
            ResearchTaskRow(
                research_id=research_task.research_id,
                owner_id=owner_id,
                own_product=research_task.own_product,
                competitors_json=json.dumps(
                    research_task.competitors,
                    ensure_ascii=False,
                ),
                research_question=research_task.research_question,
                research_plan_json=research_task.research_plan.model_dump_json(),
                status=research_task.status.value,
                result_json=None,
                usage_json=research_task.usage.model_dump_json(),
                current_step=research_task.current_step,
                error_message=None,
                attempt_count=0,
                created_at=research_task.created_at,
                updated_at=research_task.updated_at,
            )
        )


def get_research_task(research_id: str) -> ResearchTask | None:
    with connect() as session:
        row = session.get(ResearchTaskRow, research_id)
        if row is None:
            return None
        return _to_research_task(
            row,
            session.get(EvalResultRow, research_id),
            session.get(ResearchFeedbackRow, research_id),
        )


def get_owned_research_task(
    research_id: str,
    owner_id: str,
) -> ResearchTask | None:
    """用户入口必须同时匹配 owner_id，防止猜测 research_id 越权读取。"""

    with connect() as session:
        row = session.scalar(
            select(ResearchTaskRow).where(
                ResearchTaskRow.research_id == research_id,
                ResearchTaskRow.owner_id == owner_id,
            )
        )
        if row is None:
            return None
        return _to_research_task(
            row,
            session.get(EvalResultRow, research_id),
            session.get(ResearchFeedbackRow, research_id),
        )


def list_research_tasks(
    owner_id: str,
    limit: int = 20,
) -> list[ResearchTaskSummary]:
    """历史接口只查询轻量字段，与单任务详情接口形成明确区别。"""

    with connect() as session:
        rows = session.scalars(
            select(ResearchTaskRow)
            .where(ResearchTaskRow.owner_id == owner_id)
            .order_by(ResearchTaskRow.created_at.desc())
            .limit(limit)
        ).all()
        return [
            ResearchTaskSummary(
                research_id=row.research_id,
                own_product=row.own_product or "",
                competitors=_parse_competitors(row.competitors_json),
                research_question=row.research_question,
                status=ResearchStatus(row.status),
                current_step=row.current_step,
                created_at=row.created_at,
                updated_at=row.updated_at,
            )
            for row in rows
        ]


def list_research_tasks_for_metrics(limit: int = 500) -> list[ResearchTask]:
    """仅供本机质量报告使用；不经过公开 API，也不返回用户账号信息。"""

    with connect() as session:
        rows = session.scalars(
            select(ResearchTaskRow)
            .order_by(ResearchTaskRow.created_at.desc())
            .limit(limit)
        ).all()
        research_ids = [row.research_id for row in rows]
        eval_rows = (
            {
                row.research_id: row
                for row in session.scalars(
                    select(EvalResultRow).where(
                        EvalResultRow.research_id.in_(research_ids)
                    )
                ).all()
            }
            if research_ids
            else {}
        )
        return [
            _to_research_task(row, eval_rows.get(row.research_id), None)
            for row in rows
        ]


def get_tool_trace_counts(research_ids: list[str]) -> tuple[int, int]:
    """返回已结束的 Tool Trace 总数与失败数，started 事件不重复计数。"""

    if not research_ids:
        return 0, 0
    with connect() as session:
        terminal_filter = (
            TraceEventRow.research_id.in_(research_ids),
            TraceEventRow.stage == "tool",
            TraceEventRow.status.in_(["completed", "failed"]),
        )
        total = int(
            session.scalar(
                select(func.count())
                .select_from(TraceEventRow)
                .where(*terminal_filter)
            )
            or 0
        )
        failed = int(
            session.scalar(
                select(func.count())
                .select_from(TraceEventRow)
                .where(*terminal_filter, TraceEventRow.status == "failed")
            )
            or 0
        )
        return total, failed


def queue_research_task(
    research_id: str,
    owner_id: str,
    research_plan: ResearchPlan,
) -> bool:
    now = utc_now()
    with connect() as session:
        result = session.execute(
            update(ResearchTaskRow)
            .where(
                ResearchTaskRow.research_id == research_id,
                ResearchTaskRow.owner_id == owner_id,
                ResearchTaskRow.status == ResearchStatus.WAITING_FOR_APPROVAL.value,
            )
            .values(
                research_plan_json=research_plan.model_dump_json(),
                status=ResearchStatus.QUEUED.value,
                current_step="queued",
                error_message=None,
                updated_at=now,
            )
        )
        return result.rowcount == 1


# 🟩【第三批修改：失败重试】复用 checkpoint，并保留累计调用用量。
def retry_failed_research_task(research_id: str, owner_id: str) -> bool:
    """复用同一 checkpoint；累计用量不清零，防止重试绕过预算。"""

    now = utc_now()
    with connect() as session:
        result = session.execute(
            update(ResearchTaskRow)
            .where(
                ResearchTaskRow.research_id == research_id,
                ResearchTaskRow.owner_id == owner_id,
                ResearchTaskRow.status == ResearchStatus.FAILED.value,
            )
            .values(
                status=ResearchStatus.QUEUED.value,
                result_json=None,
                current_step="queued",
                error_message=None,
                started_at=None,
                completed_at=None,
                updated_at=now,
            )
        )
        if result.rowcount == 1:
            session.execute(
                delete(EvalResultRow).where(EvalResultRow.research_id == research_id)
            )
            return True
        return False


def start_queued_research_task(research_id: str) -> ResearchTask | None:
    now = utc_now()
    with connect() as session:
        result = session.execute(
            update(ResearchTaskRow)
            .where(
                ResearchTaskRow.research_id == research_id,
                ResearchTaskRow.status == ResearchStatus.QUEUED.value,
            )
            .values(
                status=ResearchStatus.RESEARCHING.value,
                current_step="research",
                error_message=None,
                attempt_count=ResearchTaskRow.attempt_count + 1,
                started_at=now,
                updated_at=now,
            )
        )
        if result.rowcount != 1:
            return None
    return get_research_task(research_id)


def resume_research_task(
    research_id: str,
    error_message: str | None = None,
) -> ResearchTask | None:
    now = utc_now()
    with connect() as session:
        result = session.execute(
            update(ResearchTaskRow)
            .where(
                ResearchTaskRow.research_id == research_id,
                ResearchTaskRow.status == ResearchStatus.RESEARCHING.value,
            )
            .values(
                current_step="research",
                error_message=error_message,
                attempt_count=ResearchTaskRow.attempt_count + 1,
                updated_at=now,
            )
        )
        if result.rowcount != 1:
            return None
    return get_research_task(research_id)


def restore_waiting_after_enqueue_failure(
    research_id: str,
    error_message: str,
) -> None:
    with connect() as session:
        session.execute(
            update(ResearchTaskRow)
            .where(
                ResearchTaskRow.research_id == research_id,
                ResearchTaskRow.status == ResearchStatus.QUEUED.value,
            )
            .values(
                status=ResearchStatus.WAITING_FOR_APPROVAL.value,
                current_step="approval",
                error_message=error_message[:2000],
                updated_at=utc_now(),
            )
        )


def restore_failed_after_retry_enqueue_failure(
    research_id: str,
    error_message: str,
) -> None:
    with connect() as session:
        session.execute(
            update(ResearchTaskRow)
            .where(
                ResearchTaskRow.research_id == research_id,
                ResearchTaskRow.status == ResearchStatus.QUEUED.value,
            )
            .values(
                status=ResearchStatus.FAILED.value,
                current_step="failed",
                error_message=error_message[:2000],
                completed_at=utc_now(),
                updated_at=utc_now(),
            )
        )


def save_usage_stats(research_id: str, usage: UsageStats) -> None:
    """研究执行过程中持续保存用量，页面刷新后统计也不会丢失。"""

    with connect() as session:
        session.execute(
            update(ResearchTaskRow)
            .where(ResearchTaskRow.research_id == research_id)
            .values(usage_json=usage.model_dump_json(), updated_at=utc_now())
        )


def complete_research_task(
    research_id: str,
    result: ResearchResult,
    eval_result: EvalResult | None,
    usage: UsageStats,
) -> None:
    now = utc_now()
    with connect() as session:
        update_result = session.execute(
            update(ResearchTaskRow)
            .where(
                ResearchTaskRow.research_id == research_id,
                ResearchTaskRow.status == ResearchStatus.RESEARCHING.value,
            )
            .values(
                result_json=result.model_dump_json(),
                usage_json=usage.model_dump_json(),
                status=ResearchStatus.COMPLETED.value,
                current_step="completed",
                error_message=None,
                completed_at=now,
                updated_at=now,
            )
        )
        if update_result.rowcount != 1:
            raise RuntimeError("Research Task 完成状态保存失败")

        if eval_result is not None:
            eval_row = session.get(EvalResultRow, research_id)
            if eval_row is None:
                session.add(
                    EvalResultRow(
                        research_id=research_id,
                        eval_result_json=eval_result.model_dump_json(),
                    )
                )
            else:
                eval_row.eval_result_json = eval_result.model_dump_json()


def save_research_feedback(
    research_id: str,
    owner_id: str,
    helpful: bool,
    comment: str | None,
) -> ResearchFeedback:
    """一个任务只保留一份最新反馈，重复提交是更新而不是新增重复记录。"""

    now = utc_now()
    with connect() as session:
        task_row = session.scalar(
            select(ResearchTaskRow).where(
                ResearchTaskRow.research_id == research_id,
                ResearchTaskRow.owner_id == owner_id,
            )
        )
        if task_row is None:
            raise LookupError("Research Task 不存在")
        if task_row.status != ResearchStatus.COMPLETED.value:
            raise RuntimeError("只有已完成的 Research Task 可以提交反馈")

        row = session.get(ResearchFeedbackRow, research_id)
        if row is None:
            row = ResearchFeedbackRow(
                research_id=research_id,
                helpful=helpful,
                comment=comment,
                created_at=now,
                updated_at=now,
            )
            session.add(row)
        else:
            row.helpful = helpful
            row.comment = comment
            row.updated_at = now
        session.flush()
        return _to_feedback(row)  # type: ignore[return-value]


def save_trace_event(research_id: str, event: TraceEvent) -> None:
    created_at = datetime.fromisoformat(event.created_at)
    with connect() as session:
        session.add(
            TraceEventRow(
                research_id=research_id,
                stage=event.stage,
                action=event.action,
                status=event.status,
                tool_call_id=event.tool_call_id,
                input_json=json.dumps(event.input_data, ensure_ascii=False),
                output_json=json.dumps(event.output_data, ensure_ascii=False),
                error_type=event.error_type,
                error_message=event.error_message,
                created_at=created_at,
            )
        )
        session.execute(
            update(ResearchTaskRow)
            .where(ResearchTaskRow.research_id == research_id)
            .values(current_step=event.stage, updated_at=utc_now())
        )


def fail_research_task(research_id: str, error_message: str) -> None:
    now = utc_now()
    with connect() as session:
        session.execute(
            update(ResearchTaskRow)
            .where(
                ResearchTaskRow.research_id == research_id,
                ResearchTaskRow.status.in_(
                    [
                        ResearchStatus.QUEUED.value,
                        ResearchStatus.RESEARCHING.value,
                    ]
                ),
            )
            .values(
                status=ResearchStatus.FAILED.value,
                current_step="failed",
                error_message=error_message[:2000],
                completed_at=now,
                updated_at=now,
            )
        )
