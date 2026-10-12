import logging
import os
from pathlib import Path

import httpx
from celery import Celery
from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).parent
load_dotenv(PROJECT_DIR / ".env")

from database import (
    ResearchStatus,
    complete_research_task,
    fail_research_task,
    resume_research_task,
    start_queued_research_task,
)
from logging_config import configure_logging
from workflow import run_research_workflow


configure_logging()
logger = logging.getLogger(__name__)

CELERY_BROKER_URL = os.environ.get(
    "CELERY_BROKER_URL",
    "redis://127.0.0.1:6379/0",
)
MAX_JOB_RETRIES = 2

celery_app = Celery("product_radar", broker=CELERY_BROKER_URL)
celery_app.conf.update(
    accept_content=["json"],
    task_serializer="json",
    task_ignore_result=True,
    task_track_started=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_publish_retry=False,
    worker_prefetch_multiplier=1,
    worker_hijack_root_logger=False,
    broker_connection_retry_on_startup=True,
    broker_transport_options={"visibility_timeout": 3600},
    task_soft_time_limit=1800,
    task_time_limit=1860,
)


def execute_research(research_id: str, is_retry: bool = False) -> None:
    research_task = (
        resume_research_task(research_id)
        if is_retry
        else start_queued_research_task(research_id)
    )
    if research_task is None:
        logger.info(
            "研究任务状态不允许 Worker 执行，已跳过",
            extra={"research_id": research_id, "stage": "research"},
        )
        return

    # 🟩【第三批修改：后台执行】Worker 恢复工作流，由 checkpoint 决定续跑位置。
    result, eval_result, usage = run_research_workflow(research_task)
    complete_research_task(
        research_id,
        result,
        eval_result,
        usage,
    )

    logger.info(
        "研究任务执行完成",
        extra={
            "research_id": research_id,
            "stage": "research",
            "status": ResearchStatus.COMPLETED.value,
        },
    )


@celery_app.task(bind=True, max_retries=MAX_JOB_RETRIES, name="research.run")
def run_research_job(self, research_id: str) -> None:
    delivery_info = getattr(self.request, "delivery_info", {}) or {}
    is_retry = self.request.retries > 0 or bool(delivery_info.get("redelivered"))

    try:
        execute_research(research_id, is_retry=is_retry)
    except httpx.TransportError as error:
        if self.request.retries < MAX_JOB_RETRIES:
            logger.warning(
                "外部服务暂时不可用，研究任务准备重试",
                extra={"research_id": research_id, "stage": "research"},
            )
            raise self.retry(
                exc=RuntimeError(str(error)),
                countdown=5 * (2**self.request.retries),
            ) from error

        fail_research_task(research_id, str(error))
        logger.exception(
            "研究任务重试次数已用尽",
            extra={
                "research_id": research_id,
                "stage": "research",
                "status": ResearchStatus.FAILED.value,
            },
        )
        raise
    except Exception as error:
        fail_research_task(research_id, str(error))
        logger.exception(
            "研究任务执行失败",
            extra={
                "research_id": research_id,
                "stage": "research",
                "status": ResearchStatus.FAILED.value,
            },
        )
        raise
