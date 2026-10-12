import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Literal

import httpx
from pydantic import BaseModel, Field


class ResearchBudgetExceeded(RuntimeError):
    """研究任务超过预设调用上限时终止，避免无限消耗外部服务额度。"""


class ResearchPhaseBudgetReached(RuntimeError):
    """初始研究额度已用完，但仍保留了用于覆盖缺口补搜的额度。"""


ResearchPhase = Literal["initial", "supplemental"]


class UsageStats(BaseModel):
    """面向用户展示的累计用量，不保存模型内部思维过程。"""

    model_calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    search_calls: int = Field(default=0, ge=0)
    search_results: int = Field(default=0, ge=0)
    pages_read: int = Field(default=0, ge=0)
    initial_search_calls: int = Field(default=0, ge=0)
    supplemental_search_calls: int = Field(default=0, ge=0)
    initial_pages_read: int = Field(default=0, ge=0)
    supplemental_pages_read: int = Field(default=0, ge=0)
    estimated_tavily_credits: int = Field(default=0, ge=0)
    estimated_qwen_cost_cny: float = Field(default=0.0, ge=0.0)
    qwen_cost_configured: bool = False


class UsageTracker:
    """在一次研究上下文中执行预算检查，并累计模型、搜索和网页用量。"""

    def __init__(self, initial: UsageStats | None = None) -> None:
        self.stats = (initial or UsageStats()).model_copy(deep=True)
        # 🟦【第四批修复：模型预算】为研究、Evidence、Analysis 和 Critic 共享的总上限。
        self.max_model_calls = int(os.environ.get("MAX_MODEL_CALLS", "50"))
        # 15 次是当前单任务的默认搜索硬上限；仍可通过环境变量按部署环境调整。
        self.max_search_calls = int(os.environ.get("MAX_SEARCH_CALLS", "15"))
        # 🟦【第四批修复：网页预算】与 15 次搜索上限对齐，给失效链接留出容错空间。
        self.max_pages_read = int(os.environ.get("MAX_PAGES_READ", "15"))
        reserved_search_calls = int(
            os.environ.get("RESERVED_SUPPLEMENTAL_SEARCH_CALLS", "7")
        )
        reserved_page_reads = int(
            os.environ.get("RESERVED_SUPPLEMENTAL_PAGE_READS", "7")
        )
        # 预留值不会突破总预算；默认形成 8 次初始 + 7 次补搜的分段预算。
        self.max_initial_search_calls = self.max_search_calls - min(
            max(reserved_search_calls, 0),
            self.max_search_calls,
        )
        self.max_initial_page_reads = self.max_pages_read - min(
            max(reserved_page_reads, 0),
            self.max_pages_read,
        )
        self.input_price = float(
            os.environ.get("QWEN_INPUT_PRICE_PER_MILLION_CNY", "0")
        )
        self.output_price = float(
            os.environ.get("QWEN_OUTPUT_PRICE_PER_MILLION_CNY", "0")
        )
        self.stats.qwen_cost_configured = (
            self.input_price > 0 and self.output_price > 0
        )

    def reserve_model_call(self) -> None:
        if self.stats.model_calls >= self.max_model_calls:
            raise ResearchBudgetExceeded("模型调用次数已达到当前研究预算上限")
        self.stats.model_calls += 1

    def record_model_response(self, response: httpx.Response) -> None:
        try:
            payload = response.json()
        except ValueError:
            return
        usage = payload.get("usage") if isinstance(payload, dict) else None
        if not isinstance(usage, dict):
            return

        input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
        output_tokens = usage.get(
            "completion_tokens",
            usage.get("output_tokens", 0),
        )
        if isinstance(input_tokens, int) and input_tokens >= 0:
            self.stats.input_tokens += input_tokens
        if isinstance(output_tokens, int) and output_tokens >= 0:
            self.stats.output_tokens += output_tokens

        # 单价由环境变量提供，避免把可能变化的供应商价格写死在代码中。
        if self.stats.qwen_cost_configured:
            self.stats.estimated_qwen_cost_cny = round(
                self.stats.input_tokens / 1_000_000 * self.input_price
                + self.stats.output_tokens / 1_000_000 * self.output_price,
                6,
            )

    def reserve_search_call(
        self,
        phase: ResearchPhase = "initial",
        estimated_credits: int = 1,
    ) -> None:
        if self.stats.search_calls >= self.max_search_calls:
            raise ResearchBudgetExceeded("搜索调用次数已达到当前研究预算上限")
        if (
            phase == "initial"
            and self.stats.initial_search_calls >= self.max_initial_search_calls
        ):
            raise ResearchPhaseBudgetReached(
                "初始搜索额度已用完，剩余额度保留给覆盖缺口补搜"
            )
        self.stats.search_calls += 1
        if phase == "initial":
            self.stats.initial_search_calls += 1
        else:
            self.stats.supplemental_search_calls += 1
        self.stats.estimated_tavily_credits += max(estimated_credits, 1)

    def record_search_results(self, result_count: int) -> None:
        self.stats.search_results += max(result_count, 0)

    def reserve_page_read(self, phase: ResearchPhase = "initial") -> None:
        if self.stats.pages_read >= self.max_pages_read:
            raise ResearchBudgetExceeded("网页读取次数已达到当前研究预算上限")
        if (
            phase == "initial"
            and self.stats.initial_pages_read >= self.max_initial_page_reads
        ):
            raise ResearchPhaseBudgetReached(
                "初始网页读取额度已用完，剩余额度保留给覆盖缺口补搜"
            )
        self.stats.pages_read += 1
        if phase == "initial":
            self.stats.initial_pages_read += 1
        else:
            self.stats.supplemental_pages_read += 1

    def snapshot(self) -> UsageStats:
        return self.stats.model_copy(deep=True)


_current_tracker: ContextVar[UsageTracker | None] = ContextVar(
    "product_radar_usage_tracker",
    default=None,
)


@contextmanager
def use_usage_tracker(initial: UsageStats | None = None) -> Iterator[UsageTracker]:
    """API 规划和 Worker 研究分别建立上下文，避免并发任务互相累计用量。"""

    tracker = UsageTracker(initial)
    token = _current_tracker.set(tracker)
    try:
        yield tracker
    finally:
        _current_tracker.reset(token)


def reserve_model_call() -> None:
    tracker = _current_tracker.get()
    if tracker is not None:
        tracker.reserve_model_call()


def record_model_response(response: httpx.Response) -> None:
    tracker = _current_tracker.get()
    if tracker is not None:
        tracker.record_model_response(response)


def reserve_search_call(
    phase: ResearchPhase = "initial",
    estimated_credits: int = 1,
) -> None:
    tracker = _current_tracker.get()
    if tracker is not None:
        tracker.reserve_search_call(phase, estimated_credits)


def record_search_results(result_count: int) -> None:
    tracker = _current_tracker.get()
    if tracker is not None:
        tracker.record_search_results(result_count)


def reserve_page_read(phase: ResearchPhase = "initial") -> None:
    tracker = _current_tracker.get()
    if tracker is not None:
        tracker.reserve_page_read(phase)
