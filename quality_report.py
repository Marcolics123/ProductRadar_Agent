import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from database import (
    ResearchStatus,
    ResearchTask,
    get_tool_trace_counts,
    list_research_tasks_for_metrics,
)


class QualityThresholds(BaseModel):
    """第一版质量门槛应保持明确，后续用固定评测结果再调整数值。"""

    minimum_sample_size: int = 10
    minimum_qualified_completion_rate: float = 0.80
    minimum_task_completion: float = 0.80
    minimum_evidence_accuracy: float = 0.85
    minimum_source_reliability: float = 0.70
    minimum_competitor_coverage: float = 0.90
    minimum_research_coverage: float = 0.80
    # 🟦【质量量化：Evidence 充足性】不让“只有一条证据链”的报告被误判为高质量。
    minimum_evidence_per_competitor: float = 1.0
    minimum_unique_sources_per_report: int = 2
    minimum_evidence_sufficiency_rate: float = 0.80
    maximum_unsupported_claims: int = 0
    minimum_execution_success_rate: float = 0.90
    minimum_first_pass_success_rate: float = 0.85
    maximum_budget_failure_rate: float = 0.05
    maximum_tool_failure_rate: float = 0.10


class ManualReviewItem(BaseModel):
    research_id: str
    reasons: list[str]


class QualityMetrics(BaseModel):
    terminal_task_count: int = Field(ge=0)
    completed_task_count: int = Field(ge=0)
    evaluated_report_count: int = Field(ge=0)
    qualified_report_count: int = Field(ge=0)
    qualified_completion_rate: float = Field(ge=0.0, le=1.0)
    average_task_completion: float = Field(ge=0.0, le=1.0)
    average_evidence_accuracy: float = Field(ge=0.0, le=1.0)
    average_source_reliability: float = Field(ge=0.0, le=1.0)
    average_competitor_coverage: float = Field(ge=0.0, le=1.0)
    average_research_coverage: float = Field(ge=0.0, le=1.0)
    coverage_matrix_report_count: int = Field(ge=0)
    average_coverage_matrix_rate: float = Field(ge=0.0, le=1.0)
    verified_coverage_cell_rate: float = Field(ge=0.0, le=1.0)
    average_coverage_quality_score: float = Field(ge=0.0, le=100.0)
    actionable_coverage_cell_count: int = Field(ge=0)
    search_funnel_report_count: int = Field(ge=0)
    search_results_per_query: float = Field(ge=0.0)
    diverse_results_per_query: float = Field(ge=0.0)
    funnel_result_to_read_rate: float = Field(ge=0.0, le=1.0)
    funnel_page_to_evidence_rate: float = Field(ge=0.0)
    partial_evidence_report_count: int = Field(ge=0)
    evidence_sufficiency_rate: float = Field(ge=0.0, le=1.0)
    sparse_evidence_report_count: int = Field(ge=0)
    zero_unsupported_claim_rate: float = Field(ge=0.0, le=1.0)
    average_evidence_count: float = Field(ge=0.0)
    average_evidence_per_competitor: float = Field(ge=0.0)
    average_unique_source_count: float = Field(ge=0.0)
    cross_source_supported_finding_rate: float = Field(ge=0.0, le=1.0)
    budget_constrained_report_count: int = Field(ge=0)


class StabilityMetrics(BaseModel):
    terminal_task_count: int = Field(ge=0)
    completed_task_count: int = Field(ge=0)
    failed_task_count: int = Field(ge=0)
    execution_success_rate: float = Field(ge=0.0, le=1.0)
    first_pass_success_rate: float = Field(ge=0.0, le=1.0)
    retry_rate: float = Field(ge=0.0, le=1.0)
    budget_failure_rate: float = Field(ge=0.0, le=1.0)
    runtime_p50_seconds: float | None = Field(default=None, ge=0.0)
    runtime_p95_seconds: float | None = Field(default=None, ge=0.0)
    tool_event_count: int = Field(ge=0)
    tool_failure_count: int = Field(ge=0)
    tool_failure_rate: float = Field(ge=0.0, le=1.0)
    failure_breakdown: dict[str, int]


class QualityStabilityReport(BaseModel):
    generated_at: datetime
    assessment: Literal["PASS", "FAIL", "INSUFFICIENT_DATA"]
    thresholds: QualityThresholds
    quality: QualityMetrics
    stability: StabilityMetrics
    manual_review_queue: list[ManualReviewItem]
    limitations: list[str]


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _average(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _percentile(values: list[float], percentile: float) -> float | None:
    """线性插值百分位；小样本时仍可计算，但报告会明确提示样本不足。"""

    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return round(ordered[lower], 1)
    value = ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)
    return round(value, 1)


def _competitor_coverage(task: ResearchTask) -> float:
    if not task.competitors:
        return 1.0
    evidence = task.result.evidence if task.result is not None else []
    searchable_text = " ".join(
        f"{item.claim} {item.title} {item.url} {item.source}"
        for item in evidence
    ).casefold()
    covered_count = sum(
        1 for competitor in task.competitors if competitor.casefold() in searchable_text
    )
    return _rate(covered_count, len(task.competitors))


def _research_coverage(task: ResearchTask) -> float:
    """新报告使用覆盖矩阵；历史报告没有矩阵时退回竞品覆盖率。"""

    if task.result is not None:
        matrix = task.result.coverage_matrix
        if matrix.total_cell_count:
            return matrix.coverage_rate
    return _competitor_coverage(task)


def _evidence_profile(
    task: ResearchTask,
    thresholds: QualityThresholds,
) -> tuple[float, int, bool]:
    """返回每竞品 Evidence、唯一来源数，以及是否达到最小证据链要求。"""

    evidence = task.result.evidence if task.result is not None else []
    competitor_count = max(len(task.competitors), 1)
    evidence_per_competitor = len(evidence) / competitor_count
    unique_source_count = len(
        {
            item.source.strip().casefold()
            for item in evidence
            if item.source.strip()
        }
    )
    sufficient = (
        evidence_per_competitor >= thresholds.minimum_evidence_per_competitor
        and unique_source_count >= thresholds.minimum_unique_sources_per_report
    )
    return evidence_per_competitor, unique_source_count, sufficient


def _finding_source_support(task: ResearchTask) -> tuple[int, int]:
    """统计有多少分析判断至少被两个不同来源共同支持；该指标只观察，不作为硬门槛。"""

    if task.result is None:
        return 0, 0
    source_by_evidence_id = {
        item.evidence_id: item.source.strip().casefold()
        for item in task.result.evidence
        if item.evidence_id is not None and item.source.strip()
    }
    analysis = task.result.analysis
    findings = [
        finding
        for category in (
            analysis.competitor_comparison,
            analysis.common_capabilities,
            analysis.differentiators,
            analysis.gaps,
            analysis.product_opportunities,
            analysis.mvp_recommendations,
        )
        for finding in category
    ]
    cross_source_count = sum(
        1
        for finding in findings
        if len(
            {
                source_by_evidence_id[evidence_id]
                for evidence_id in finding.evidence_ids
                if evidence_id in source_by_evidence_id
            }
        )
        >= 2
    )
    return len(findings), cross_source_count


def _failure_category(error_message: str | None) -> str:
    message = (error_message or "").casefold()
    if "预算" in message or "budget" in message:
        return "budget_exceeded"
    if "tavily" in message or "搜索" in message or "search" in message:
        return "search"
    if "网页" in message or "read_page" in message or "http status" in message:
        return "web_reader"
    if "dashscope" in message or "模型" in message or "llm" in message:
        return "model"
    if "psycopg" in message or "sqlalchemy" in message or "数据库" in message:
        return "database"
    if "redis" in message or "celery" in message or "worker" in message:
        return "worker"
    return "other"


def build_quality_stability_report(
    tasks: list[ResearchTask],
    *,
    tool_event_count: int = 0,
    tool_failure_count: int = 0,
    thresholds: QualityThresholds | None = None,
) -> QualityStabilityReport:
    """纯计算函数：不调用模型、搜索或网页，便于人工复核和自动测试。"""

    rules = thresholds or QualityThresholds()
    terminal_tasks = [
        task
        for task in tasks
        if task.status in {ResearchStatus.COMPLETED, ResearchStatus.FAILED}
    ]
    completed_tasks = [
        task for task in terminal_tasks if task.status == ResearchStatus.COMPLETED
    ]
    failed_tasks = [
        task for task in terminal_tasks if task.status == ResearchStatus.FAILED
    ]
    evaluated_tasks = [
        task for task in completed_tasks if task.eval_result is not None
    ]

    coverage_by_id = {
        task.research_id: _competitor_coverage(task) for task in completed_tasks
    }
    research_coverage_by_id = {
        task.research_id: _research_coverage(task) for task in completed_tasks
    }
    # 新版报告才有 Coverage Matrix，单独统计，避免历史报告的回退值混入该指标。
    matrix_tasks = [
        task
        for task in completed_tasks
        if task.result is not None
        and task.result.coverage_matrix.total_cell_count > 0
    ]
    matrix_cells = [
        cell
        for task in matrix_tasks
        for cell in task.result.coverage_matrix.cells
    ]
    funnel_summaries = [
        task.result.coverage_matrix.search_funnel_summary
        for task in matrix_tasks
        if task.result.coverage_matrix.search_funnel_summary.query_count > 0
    ]
    funnel_query_count = sum(item.query_count for item in funnel_summaries)
    funnel_result_count = sum(item.search_result_count for item in funnel_summaries)
    funnel_accepted_count = sum(
        item.accepted_result_count for item in funnel_summaries
    )
    funnel_pages_read = sum(item.pages_read for item in funnel_summaries)
    funnel_evidence_count = sum(item.evidence_count for item in funnel_summaries)
    evidence_profile_by_id = {
        task.research_id: _evidence_profile(task, rules)
        for task in completed_tasks
    }
    qualified_tasks: list[ResearchTask] = []
    manual_review: list[ManualReviewItem] = []

    for task in terminal_tasks:
        reasons: list[str] = []
        if task.status == ResearchStatus.FAILED:
            reasons.append(f"task_failed:{_failure_category(task.error_message)}")
        elif task.eval_result is None:
            reasons.append("missing_eval")
        else:
            evaluation = task.eval_result
            coverage = coverage_by_id[task.research_id]
            research_coverage = research_coverage_by_id[task.research_id]
            (
                evidence_per_competitor,
                unique_source_count,
                _,
            ) = evidence_profile_by_id[task.research_id]
            if evaluation.task_completion.rate < rules.minimum_task_completion:
                reasons.append("low_task_completion")
            if evaluation.evidence_accuracy.rate < rules.minimum_evidence_accuracy:
                reasons.append("low_evidence_accuracy")
            if evaluation.source_reliability.rate < rules.minimum_source_reliability:
                reasons.append("low_source_reliability")
            if coverage < rules.minimum_competitor_coverage:
                reasons.append("low_competitor_coverage")
            if research_coverage < rules.minimum_research_coverage:
                reasons.append("low_research_coverage")
            if evidence_per_competitor < rules.minimum_evidence_per_competitor:
                reasons.append("sparse_evidence")
            if unique_source_count < rules.minimum_unique_sources_per_report:
                reasons.append("low_source_diversity")
            if (
                evaluation.unsupported_claim_count
                > rules.maximum_unsupported_claims
            ):
                reasons.append("unsupported_claims")
            if task.result is not None and task.result.budget_exhausted:
                reasons.append("budget_constrained")

            quality_failures = [
                reason for reason in reasons if reason != "budget_constrained"
            ]
            if not quality_failures:
                qualified_tasks.append(task)

        if reasons:
            manual_review.append(
                ManualReviewItem(
                    research_id=task.research_id,
                    reasons=reasons,
                )
            )

    eval_results = [task.eval_result for task in evaluated_tasks]
    sufficient_evidence_count = sum(
        1
        for _, _, sufficient in evidence_profile_by_id.values()
        if sufficient
    )
    total_finding_count = 0
    cross_source_finding_count = 0
    for task in completed_tasks:
        finding_count, supported_count = _finding_source_support(task)
        total_finding_count += finding_count
        cross_source_finding_count += supported_count
    quality = QualityMetrics(
        terminal_task_count=len(terminal_tasks),
        completed_task_count=len(completed_tasks),
        evaluated_report_count=len(evaluated_tasks),
        qualified_report_count=len(qualified_tasks),
        qualified_completion_rate=_rate(
            len(qualified_tasks),
            len(terminal_tasks),
        ),
        average_task_completion=_average(
            [item.task_completion.rate for item in eval_results if item is not None]
        ),
        average_evidence_accuracy=_average(
            [item.evidence_accuracy.rate for item in eval_results if item is not None]
        ),
        average_source_reliability=_average(
            [item.source_reliability.rate for item in eval_results if item is not None]
        ),
        average_competitor_coverage=_average(list(coverage_by_id.values())),
        average_research_coverage=_average(
            list(research_coverage_by_id.values())
        ),
        coverage_matrix_report_count=len(matrix_tasks),
        average_coverage_matrix_rate=_average(
            [task.result.coverage_matrix.coverage_rate for task in matrix_tasks]
        ),
        verified_coverage_cell_rate=_rate(
            sum(1 for cell in matrix_cells if cell.status == "verified"),
            len(matrix_cells),
        ),
        average_coverage_quality_score=_average(
            [float(cell.quality_score) for cell in matrix_cells]
        ),
        actionable_coverage_cell_count=sum(
            1 for cell in matrix_cells if cell.next_action is not None
        ),
        search_funnel_report_count=len(funnel_summaries),
        search_results_per_query=(
            round(funnel_result_count / funnel_query_count, 4)
            if funnel_query_count
            else 0.0
        ),
        diverse_results_per_query=(
            round(funnel_accepted_count / funnel_query_count, 4)
            if funnel_query_count
            else 0.0
        ),
        funnel_result_to_read_rate=_rate(
            funnel_pages_read,
            funnel_accepted_count,
        ),
        funnel_page_to_evidence_rate=(
            round(funnel_evidence_count / funnel_pages_read, 4)
            if funnel_pages_read
            else 0.0
        ),
        partial_evidence_report_count=sum(
            1
            for task in completed_tasks
            if task.result is not None and task.result.evidence_status == "partial"
        ),
        evidence_sufficiency_rate=_rate(
            sufficient_evidence_count,
            len(completed_tasks),
        ),
        sparse_evidence_report_count=(
            len(completed_tasks) - sufficient_evidence_count
        ),
        zero_unsupported_claim_rate=_rate(
            sum(
                1
                for item in eval_results
                if item is not None and item.unsupported_claim_count == 0
            ),
            len(eval_results),
        ),
        average_evidence_count=_average(
            [float(len(task.result.evidence)) for task in completed_tasks if task.result]
        ),
        average_evidence_per_competitor=_average(
            [profile[0] for profile in evidence_profile_by_id.values()]
        ),
        average_unique_source_count=_average(
            [float(profile[1]) for profile in evidence_profile_by_id.values()]
        ),
        cross_source_supported_finding_rate=_rate(
            cross_source_finding_count,
            total_finding_count,
        ),
        budget_constrained_report_count=sum(
            1
            for task in completed_tasks
            if task.result is not None and task.result.budget_exhausted
        ),
    )

    durations = [
        (task.completed_at - task.started_at).total_seconds()
        for task in terminal_tasks
        if task.started_at is not None
        and task.completed_at is not None
        and task.completed_at >= task.started_at
    ]
    failure_breakdown = Counter(
        _failure_category(task.error_message) for task in failed_tasks
    )
    budget_failure_count = failure_breakdown.get("budget_exceeded", 0)
    stability = StabilityMetrics(
        terminal_task_count=len(terminal_tasks),
        completed_task_count=len(completed_tasks),
        failed_task_count=len(failed_tasks),
        execution_success_rate=_rate(len(completed_tasks), len(terminal_tasks)),
        first_pass_success_rate=_rate(
            sum(1 for task in completed_tasks if task.attempt_count <= 1),
            len(terminal_tasks),
        ),
        retry_rate=_rate(
            sum(1 for task in terminal_tasks if task.attempt_count > 1),
            len(terminal_tasks),
        ),
        budget_failure_rate=_rate(budget_failure_count, len(terminal_tasks)),
        runtime_p50_seconds=_percentile(durations, 0.50),
        runtime_p95_seconds=_percentile(durations, 0.95),
        tool_event_count=tool_event_count,
        tool_failure_count=tool_failure_count,
        tool_failure_rate=_rate(tool_failure_count, tool_event_count),
        failure_breakdown=dict(sorted(failure_breakdown.items())),
    )

    limitations: list[str] = []
    if len(terminal_tasks) < rules.minimum_sample_size:
        limitations.append(
            f"终态任务只有 {len(terminal_tasks)} 个，少于建议样本 {rules.minimum_sample_size} 个"
        )
    missing_eval_count = len(completed_tasks) - len(evaluated_tasks)
    if missing_eval_count:
        limitations.append(f"有 {missing_eval_count} 个已完成任务缺少 Eval")
    if not tool_event_count:
        limitations.append("没有可用于计算工具失败率的终态 Tool Trace")
    if quality.sparse_evidence_report_count:
        limitations.append(
            f"有 {quality.sparse_evidence_report_count} 份已完成报告的 Evidence 数量或来源多样性不足"
        )
    if not total_finding_count:
        limitations.append("没有可用于计算跨来源支持率的分析判断")
    if len(matrix_tasks) < len(completed_tasks):
        limitations.append(
            f"只有 {len(matrix_tasks)} 份已完成报告包含新版 Coverage Matrix"
        )
    if not funnel_summaries:
        limitations.append(
            "历史报告尚未保存搜索漏斗；需要完成一份新版研究后才能观察漏斗指标"
        )
    limitations.append("Evidence 正确率来自模型评估，重要报告仍需人工抽检")

    enough_data = (
        len(terminal_tasks) >= rules.minimum_sample_size
        and len(evaluated_tasks) >= rules.minimum_sample_size
    )
    quality_passed = (
        quality.qualified_completion_rate
        >= rules.minimum_qualified_completion_rate
        and quality.average_task_completion >= rules.minimum_task_completion
        and quality.average_evidence_accuracy >= rules.minimum_evidence_accuracy
        and quality.average_source_reliability >= rules.minimum_source_reliability
        and quality.average_competitor_coverage
        >= rules.minimum_competitor_coverage
        and quality.average_research_coverage >= rules.minimum_research_coverage
        and quality.evidence_sufficiency_rate
        >= rules.minimum_evidence_sufficiency_rate
        and quality.zero_unsupported_claim_rate == 1.0
    )
    stability_passed = (
        stability.execution_success_rate >= rules.minimum_execution_success_rate
        and stability.first_pass_success_rate
        >= rules.minimum_first_pass_success_rate
        and stability.budget_failure_rate <= rules.maximum_budget_failure_rate
        and stability.tool_failure_rate <= rules.maximum_tool_failure_rate
    )
    assessment: Literal["PASS", "FAIL", "INSUFFICIENT_DATA"]
    if not enough_data:
        assessment = "INSUFFICIENT_DATA"
    else:
        assessment = "PASS" if quality_passed and stability_passed else "FAIL"

    return QualityStabilityReport(
        generated_at=datetime.now(timezone.utc),
        assessment=assessment,
        thresholds=rules,
        quality=quality,
        stability=stability,
        manual_review_queue=manual_review[:20],
        limitations=limitations,
    )


def build_report_from_database(limit: int = 500) -> QualityStabilityReport:
    """管理员本机入口；只读取聚合所需数据，不调用外部服务。"""

    tasks = list_research_tasks_for_metrics(limit)
    research_ids = [task.research_id for task in tasks]
    tool_event_count, tool_failure_count = get_tool_trace_counts(research_ids)
    return build_quality_stability_report(
        tasks,
        tool_event_count=tool_event_count,
        tool_failure_count=tool_failure_count,
    )


def format_report_text(report: QualityStabilityReport) -> str:
    """把机器指标转成可直接阅读的中文报告，同时保留 JSON 输出选项。"""

    quality = report.quality
    stability = report.stability
    reason_labels = {
        "missing_eval": "缺少 Eval",
        "low_task_completion": "任务完成率低",
        "low_evidence_accuracy": "Evidence 正确率低",
        "low_source_reliability": "来源可靠率低",
        "low_competitor_coverage": "竞品覆盖率低",
        "low_research_coverage": "研究格子覆盖率低",
        "sparse_evidence": "Evidence 数量不足",
        "low_source_diversity": "来源多样性不足",
        "unsupported_claims": "存在无证据 Claim",
        "budget_constrained": "研究受到预算限制",
        "task_failed:budget_exceeded": "任务失败：超过研究预算",
        "task_failed:search": "任务失败：搜索异常",
        "task_failed:web_reader": "任务失败：网页读取异常",
        "task_failed:model": "任务失败：模型异常",
        "task_failed:database": "任务失败：数据库异常",
        "task_failed:worker": "任务失败：Worker 异常",
        "task_failed:other": "任务失败：其他异常",
    }

    def percent(value: float) -> str:
        return f"{value:.1%}"

    runtime_p50 = (
        f"{stability.runtime_p50_seconds:.1f} 秒"
        if stability.runtime_p50_seconds is not None
        else "暂无数据"
    )
    runtime_p95 = (
        f"{stability.runtime_p95_seconds:.1f} 秒"
        if stability.runtime_p95_seconds is not None
        else "暂无数据"
    )
    lines = [
        "ProductRadar 质量与稳定性报告",
        f"生成时间：{report.generated_at.astimezone().isoformat(timespec='seconds')}",
        f"综合判断：{report.assessment}",
        "",
        "质量指标",
        f"- 终态任务：{quality.terminal_task_count}",
        f"- 完成且通过质量门槛：{quality.qualified_report_count}，占 {percent(quality.qualified_completion_rate)}",
        f"- 任务完成率：{percent(quality.average_task_completion)}",
        f"- Evidence 正确率：{percent(quality.average_evidence_accuracy)}",
        f"- 来源可靠率：{percent(quality.average_source_reliability)}",
        f"- 竞品覆盖率：{percent(quality.average_competitor_coverage)}",
        f"- 研究覆盖率：{percent(quality.average_research_coverage)}",
        f"- 无证据 Claim 为零的报告占比：{percent(quality.zero_unsupported_claim_rate)}",
        f"- 平均 Evidence 数：{quality.average_evidence_count:.2f}",
        f"- 平均唯一来源数：{quality.average_unique_source_count:.2f}",
        "",
        "新版覆盖矩阵",
        f"- 含矩阵的报告：{quality.coverage_matrix_report_count}",
        f"- 平均格子覆盖率：{percent(quality.average_coverage_matrix_rate)}",
        f"- 交叉来源验证格子占比：{percent(quality.verified_coverage_cell_rate)}",
        f"- 平均 Evidence 质量分：{quality.average_coverage_quality_score:.1f}/100",
        f"- 仍有明确补救措施的格子：{quality.actionable_coverage_cell_count}",
        f"- Evidence 部分覆盖报告：{quality.partial_evidence_report_count}",
        "",
        "搜索漏斗指标",
        f"- 含漏斗的新报告：{quality.search_funnel_report_count}",
        f"- 每次查询平均候选结果：{quality.search_results_per_query:.2f}",
        f"- 多样性筛选后每次查询结果：{quality.diverse_results_per_query:.2f}",
        f"- 候选结果读取率：{percent(quality.funnel_result_to_read_rate)}",
        f"- 每个已读页面产生 Evidence：{quality.funnel_page_to_evidence_rate:.2f}",
        "",
        "稳定性指标",
        f"- 执行成功率：{percent(stability.execution_success_rate)}",
        f"- 首次执行成功率：{percent(stability.first_pass_success_rate)}",
        f"- 重试率：{percent(stability.retry_rate)}",
        f"- 预算导致失败率：{percent(stability.budget_failure_rate)}",
        f"- Tool 失败率：{percent(stability.tool_failure_rate)}",
        f"- 耗时 P50 / P95：{runtime_p50} / {runtime_p95}",
    ]

    if report.manual_review_queue:
        lines.extend(("", "需要人工复核"))
        for item in report.manual_review_queue:
            readable_reasons = [
                reason_labels.get(reason, reason) for reason in item.reasons
            ]
            lines.append(f"- {item.research_id}：{'、'.join(readable_reasons)}")

    if report.limitations:
        lines.extend(("", "限制说明"))
        lines.extend(f"- {limitation}" for limitation in report.limitations)
    return "\n".join(lines)


def main() -> None:
    # Windows/PyCharm 默认编码可能不是 UTF-8，显式设置后中文 JSON 不会乱码。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="生成 ProductRadar 质量与稳定性报告（不调用模型或搜索）"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="最多统计最近多少个研究任务，默认 500",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="输出格式：text 便于人工阅读，json 便于程序处理，默认 text",
    )
    arguments = parser.parse_args()
    if arguments.limit < 1:
        parser.error("--limit 必须大于 0")

    try:
        report = build_report_from_database(arguments.limit)
    except Exception as error:
        parser.exit(
            1,
            "无法生成报告：请先启动 PostgreSQL/Docker，"
            f"并检查 .env 数据库配置。\n原因：{error}\n",
        )
    if arguments.format == "json":
        print(
            json.dumps(
                report.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(format_report_text(report))


if __name__ == "__main__":
    main()
