import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from redis import Redis
from starlette.middleware.trustedhost import TrustedHostMiddleware


PROJECT_DIR = Path(__file__).parent
load_dotenv(PROJECT_DIR / ".env")

from agent import ResearchPlan, create_research_plan
from auth import CurrentUser, router as auth_router
from database import (
    ActiveResearchLimitExceeded,
    ResearchFeedback,
    ResearchStatus,
    ResearchTask,
    ResearchTaskSummary,
    active_research_count,
    create_research_task,
    database_is_ready,
    get_owned_research_task,
    initialize_database,
    list_research_tasks,
    queue_research_task,
    restore_failed_after_retry_enqueue_failure,
    restore_waiting_after_enqueue_failure,
    retry_failed_research_task,
    save_research_feedback,
    save_trace_event,
    utc_now,
)
from logging_config import configure_logging
from research_context import build_research_context
from tracing import TraceEvent
from usage import use_usage_tracker
from worker import CELERY_BROKER_URL, run_research_job
# 🟩【第三批修改：入口依赖】接入可持久化的 LangGraph 工作流。
from workflow import (
    create_research_workflow_checkpoint,
    setup_workflow_checkpointer,
    workflow_checkpointer_is_ready,
)


configure_logging()
logger = logging.getLogger(__name__)
FRONTEND_DIR = PROJECT_DIR / "frontend"

CompetitorName = Annotated[str, Field(min_length=1, max_length=200)]


class ResearchRequest(BaseModel):
    """第二批使用结构化字段，区别于旧版把三项输入拼成一个字符串。"""

    own_product: str = Field(min_length=1, max_length=200)
    competitors: list[CompetitorName] = Field(min_length=1, max_length=10)
    research_question: str = Field(min_length=1, max_length=5000)

    @field_validator("own_product", "research_question", mode="after")
    @classmethod
    def validate_non_blank_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("字段不能为空")
        return value

    @field_validator("competitors", mode="after")
    @classmethod
    def normalize_competitors(cls, values: list[str]) -> list[str]:
        # 保留用户输入顺序并去重，让后续竞品覆盖率计算有稳定对象。
        normalized = list(dict.fromkeys(value.strip() for value in values))
        if any(not value for value in normalized):
            raise ValueError("竞争对手不能为空")
        return normalized


class ResearchApprovalRequest(BaseModel):
    research_plan: ResearchPlan | None = None


class ResearchFeedbackRequest(BaseModel):
    helpful: bool
    comment: str | None = Field(default=None, max_length=1000)

    @field_validator("comment", mode="after")
    @classmethod
    def normalize_comment(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


def broker_is_ready() -> bool:
    try:
        client = Redis.from_url(
            CELERY_BROKER_URL,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        return bool(client.ping())
    except Exception:
        return False


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_database()
    yield


app = FastAPI(lifespan=lifespan)
# 🟦【第四批修改：Host 防护】只接受部署域名和本地健康检查使用的 Host。
allowed_hosts = ["localhost", "127.0.0.1", "testserver"]
public_domain = os.environ.get("PRODUCTRADAR_DOMAIN", "").strip()
if public_domain:
    allowed_hosts.append(public_domain)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
# 🟦【第四批修改：认证解耦】公开页面与健康检查保持可访问，账号 API 独立成 Router。
app.include_router(auth_router)


def _max_active_tasks_per_user() -> int:
    """这是服务容量保护，不是套餐或商业配额。"""

    try:
        value = int(os.environ.get("MAX_ACTIVE_RESEARCH_TASKS_PER_USER", "2"))
    except ValueError as error:
        raise RuntimeError(
            "MAX_ACTIVE_RESEARCH_TASKS_PER_USER 必须是正整数"
        ) from error
    if value < 1:
        raise RuntimeError("MAX_ACTIVE_RESEARCH_TASKS_PER_USER 必须是正整数")
    return value


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health/live", include_in_schema=False)
def health_live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready", include_in_schema=False)
def health_ready() -> dict[str, str]:
    if (
        not database_is_ready()
        or not broker_is_ready()
        # 🟩【第三批修改：健康检查】就绪检查包含 checkpoint 存储。
        or not workflow_checkpointer_is_ready()
    ):
        raise HTTPException(status_code=503, detail="依赖服务尚未就绪")
    return {"status": "ready"}


@app.post("/research")
def research(request: ResearchRequest, current_user: CurrentUser) -> ResearchTask:
    max_active_tasks = _max_active_tasks_per_user()
    if active_research_count(current_user.user_id) >= max_active_tasks:
        raise HTTPException(
            status_code=429,
            detail=f"最多同时保留 {max_active_tasks} 个未完成研究任务",
        )

    research_context = build_research_context(
        request.own_product,
        request.competitors,
        request.research_question,
    )

    # Planning 单独建立 Usage 上下文，生成任务后将首次模型用量一起持久化。
    with use_usage_tracker() as usage_tracker:
        try:
            research_plan = create_research_plan(research_context)
        except ValueError as error:
            raise HTTPException(
                status_code=502,
                detail="Research Plan 生成失败",
            ) from error
        planning_usage = usage_tracker.snapshot()

    now = utc_now()
    research_task = ResearchTask(
        research_id=str(uuid4()),
        own_product=request.own_product,
        competitors=request.competitors,
        research_question=request.research_question,
        research_plan=research_plan,
        status=ResearchStatus.WAITING_FOR_APPROVAL,
        usage=planning_usage,
        current_step="planning",
        created_at=now,
        updated_at=now,
    )

    # 🟦【第四批修改：任务归属】创建时写入 owner_id，并在数据库事务内再次检查并发上限。
    try:
        create_research_task(
            research_task,
            current_user.user_id,
            max_active_tasks,
        )
    except ActiveResearchLimitExceeded as error:
        raise HTTPException(status_code=429, detail=str(error)) from error
    save_trace_event(
        research_task.research_id,
        TraceEvent(
            stage="planning",
            action="create_research_plan",
            status="completed",
            input_data={
                "own_product": request.own_product,
                "competitors": request.competitors,
                "research_question": request.research_question,
            },
            output_data={"research_plan": research_plan.model_dump()},
        ),
    )

    # 🟩【第三批修改：创建断点】提前运行到 interrupt，只保存审批点，不开始 Search。
    try:
        create_research_workflow_checkpoint(research_task)
    except Exception as error:
        logger.exception(
            "Research 工作流审批 checkpoint 创建失败",
            extra={
                "research_id": research_task.research_id,
                "stage": "approval",
            },
        )
        raise HTTPException(
            status_code=503,
            detail="Research 工作流暂时不可用，请稍后重试",
        ) from error

    saved_task = get_owned_research_task(
        research_task.research_id,
        current_user.user_id,
    )
    if saved_task is None:
        raise HTTPException(status_code=500, detail="Research Task 保存失败")
    return saved_task


@app.get("/research")
def research_history(
    current_user: CurrentUser,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[ResearchTaskSummary]:
    return list_research_tasks(current_user.user_id, limit)


@app.get("/research/{research_id}")
def get_research(research_id: str, current_user: CurrentUser) -> ResearchTask:
    research_task = get_owned_research_task(research_id, current_user.user_id)
    if research_task is None:
        raise HTTPException(status_code=404, detail="Research Task 不存在")
    return research_task


def _enqueue_research(research_id: str) -> None:
    """API 只投递 ID；完整任务由 Worker 从 PostgreSQL 读取。"""

    run_research_job.delay(research_id)


@app.post("/research/{research_id}/approve", status_code=202)
def approve_research(
    research_id: str,
    request: ResearchApprovalRequest,
    current_user: CurrentUser,
) -> ResearchTask:
    research_task = get_owned_research_task(research_id, current_user.user_id)
    if research_task is None:
        raise HTTPException(status_code=404, detail="Research Task 不存在")

    research_plan = request.research_plan or research_task.research_plan
    plan_is_unchanged = research_plan == research_task.research_plan
    if research_task.status != ResearchStatus.WAITING_FOR_APPROVAL:
        if plan_is_unchanged and research_task.status in {
            ResearchStatus.QUEUED,
            ResearchStatus.RESEARCHING,
            ResearchStatus.COMPLETED,
        }:
            return research_task
        raise HTTPException(
            status_code=409,
            detail="当前状态不允许批准或修改 Research Plan",
        )

    if not queue_research_task(
        research_id,
        current_user.user_id,
        research_plan,
    ):
        current_task = get_owned_research_task(research_id, current_user.user_id)
        if (
            current_task is not None
            and current_task.research_plan == research_plan
            and current_task.status
            in {
                ResearchStatus.QUEUED,
                ResearchStatus.RESEARCHING,
                ResearchStatus.COMPLETED,
            }
        ):
            return current_task
        raise HTTPException(
            status_code=409,
            detail="当前状态不允许批准或修改 Research Plan",
        )

    save_trace_event(
        research_id,
        TraceEvent(
            stage="approval",
            action="approve_research_plan",
            status="completed",
            input_data={
                "research_plan": research_plan.model_dump(),
                "edited": request.research_plan is not None,
            },
            output_data={"status": ResearchStatus.QUEUED.value},
        ),
    )

    try:
        _enqueue_research(research_id)
    except Exception as error:
        restore_waiting_after_enqueue_failure(research_id, str(error))
        logger.exception(
            "研究任务加入队列失败",
            extra={"research_id": research_id, "stage": "approval"},
        )
        raise HTTPException(
            status_code=503,
            detail="后台任务服务暂时不可用，请稍后重新批准",
        ) from error

    queued_task = get_owned_research_task(research_id, current_user.user_id)
    if queued_task is None:
        raise HTTPException(status_code=404, detail="Research Task 不存在")
    return queued_task


@app.post("/research/{research_id}/retry", status_code=202)
def retry_research(research_id: str, current_user: CurrentUser) -> ResearchTask:
    """失败重试与 Approve 不同：它复用已确认计划，不再要求重新审批。"""

    research_task = get_owned_research_task(research_id, current_user.user_id)
    if research_task is None:
        raise HTTPException(status_code=404, detail="Research Task 不存在")
    if research_task.status in {ResearchStatus.QUEUED, ResearchStatus.RESEARCHING}:
        return research_task
    if research_task.status != ResearchStatus.FAILED:
        raise HTTPException(status_code=409, detail="只有失败任务可以重新执行")
    if not retry_failed_research_task(research_id, current_user.user_id):
        raise HTTPException(status_code=409, detail="任务状态已经发生变化")

    save_trace_event(
        research_id,
        TraceEvent(
            stage="research",
            action="retry_failed_research",
            status="completed",
            input_data={"research_id": research_id},
            output_data={"status": ResearchStatus.QUEUED.value},
        ),
    )
    try:
        _enqueue_research(research_id)
    except Exception as error:
        restore_failed_after_retry_enqueue_failure(research_id, str(error))
        raise HTTPException(
            status_code=503,
            detail="后台任务服务暂时不可用，请稍后重试",
        ) from error

    queued_task = get_owned_research_task(research_id, current_user.user_id)
    if queued_task is None:
        raise HTTPException(status_code=404, detail="Research Task 不存在")
    return queued_task


@app.post("/research/{research_id}/feedback")
def submit_research_feedback(
    research_id: str,
    request: ResearchFeedbackRequest,
    current_user: CurrentUser,
) -> ResearchFeedback:
    """反馈只评价已有报告，不触发重新研究或修改结论。"""

    try:
        return save_research_feedback(
            research_id,
            current_user.user_id,
            request.helpful,
            request.comment,
        )
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


def run_database_migrations() -> None:
    from alembic import command
    from alembic.config import Config

    config = Config(str(PROJECT_DIR / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_DIR / "migrations"))
    command.upgrade(config, "head")
    # 🟩【第三批修改：初始化】Alembic 管理业务表，LangGraph setup 管理 checkpoint 表。
    setup_workflow_checkpointer()


if __name__ == "__main__":
    run_database_migrations()
    # 🟦【第四批修复：PyCharm 与 Docker 共存】
    # Docker 使用 8000；本地直接运行默认使用 8001，避免端口冲突。
    local_host = os.environ.get("APP_HOST", "127.0.0.1")
    local_port = int(os.environ.get("APP_PORT", "8001"))
    uvicorn.run(app, host=local_host, port=local_port)
