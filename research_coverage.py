import json
import re
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, model_validator

from evidence import Evidence
from llm_client import request_json_object


# 预留的 7 次补搜预算中最多主动处理 5 个格子，仍给 Agent 留出 2 次机动搜索。
MAX_TARGETED_GAPS = 5
CoverageStatus = Literal["missing", "weak", "covered", "verified"]
CoverageQuality = Literal["none", "low", "medium", "high"]
TaskCoverageStatus = Literal["missing", "partial", "complete"]
SearchFunnelOutcome = Literal[
    "not_started",
    "search_failed",
    "searched",
    "pages_read",
    "evidence_found",
    "verified",
]
CandidateReadStatus = Literal["not_read", "read", "failed"]


class SearchCandidatePage(BaseModel):
    """保存搜索已发现的网页，让自动流程未采用或读取失败时仍可人工核查。"""

    title: str
    url: str
    domain: str
    # queries/cell_ids 解释这个 URL 为什么出现，以及它与哪些研究格子相关。
    queries: list[str] = Field(default_factory=list)
    cell_ids: list[str] = Field(default_factory=list)
    accepted: bool = False
    filter_reason: str | None = None
    read_status: CandidateReadStatus = "not_read"
    error: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)


PAGE_READ_ERROR_LABELS = {
    "unsafe_url": "URL 未通过公网安全检查",
    "too_many_redirects": "网页重定向次数过多",
    "invalid_redirect": "网页返回了无效重定向地址",
    "unsupported_content": "页面不是可读取的 HTML 正文",
    "content_too_large": "页面内容超过读取大小限制",
    "timeout": "网页连接或响应超时",
    "http_error": "网站返回了 4xx 或 5xx 错误",
    "request_failed": "网络连接失败或域名无法访问",
    "empty_content": "页面没有可提取正文，可能依赖 JavaScript 或存在反爬限制",
    "empty_page_content": "页面没有可提取正文",
    "initial_phase_budget_reached": "初始阶段页面读取预算已用完",
    "page_read_budget_exhausted": "本次研究的页面读取总预算已用完",
}


def _describe_page_read_error(error: str | None) -> str:
    """把内部错误码转换为用户能够采取行动的原因说明。"""

    if not error:
        return "未知读取原因"
    return PAGE_READ_ERROR_LABELS.get(error, error)

COVERAGE_PROMPT = """你是 ProductRadar 的 Evidence 覆盖检查组件。
输入包含覆盖格子和已有 Evidence。请判断每条 Evidence 是否直接支持对应竞品和研究任务。
只能引用输入中真实存在的 cell_id 和 evidence_id，不能补充事实，不能搜索。
同一条 Evidence 可以支持多个格子；没有直接证据的格子不要返回。
只返回 JSON：{"cells":[{"cell_id":"C1-T1","evidence_ids":["E1"]}]}。
"""


def build_baseline_query(competitor: str, research_tasks: list[str]) -> str:
    """每个竞品先获得一次稳定的基础检索机会，避免预算被少数竞品占满。"""

    task_context = " ".join(research_tasks)
    return (
        f"{competitor} {task_context} "
        "official product documentation release notes"
    )[:500].strip()


def likely_official_domains(competitor: str, urls: list[str]) -> list[str]:
    """仅在域名明显包含品牌词时把它作为官方域名偏好，避免误限制搜索。"""

    brand_tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", competitor.casefold())
        if len(token) >= 3
    ]
    compact_brand = "".join(brand_tokens)
    domains: list[str] = []
    for url in urls:
        hostname = (urlparse(url).hostname or "").removeprefix("www.").casefold()
        compact_domain = re.sub(r"[^a-z0-9]", "", hostname)
        if not hostname:
            continue
        if (
            compact_brand
            and compact_brand in compact_domain
            or any(token in compact_domain for token in brand_tokens)
        ):
            domains.append(hostname)
    return list(dict.fromkeys(domains))[:3]


class SearchFunnelMetrics(BaseModel):
    """记录单个研究格子从查询到 Evidence 的完整转化漏斗。"""

    queries: list[str] = Field(default_factory=list)
    query_count: int = Field(default=0, ge=0)
    search_result_count: int = Field(default=0, ge=0)
    accepted_result_count: int = Field(default=0, ge=0)
    unique_result_domains: list[str] = Field(default_factory=list)
    candidate_pages: list[SearchCandidatePage] = Field(default_factory=list)
    pages_read: int = Field(default=0, ge=0)
    page_read_failures: int = Field(default=0, ge=0)
    evidence_count: int = Field(default=0, ge=0)
    evidence_source_count: int = Field(default=0, ge=0)
    result_to_read_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    page_to_evidence_rate: float = Field(default=0.0, ge=0.0)
    outcome: SearchFunnelOutcome = "not_started"
    last_error: str | None = None


class SearchFunnelSummary(BaseModel):
    """面向最终报告的全部格子漏斗汇总；数量按格子归因，允许同页支持多个格子。"""

    query_count: int = Field(default=0, ge=0)
    search_result_count: int = Field(default=0, ge=0)
    accepted_result_count: int = Field(default=0, ge=0)
    pages_read: int = Field(default=0, ge=0)
    page_read_failures: int = Field(default=0, ge=0)
    evidence_count: int = Field(default=0, ge=0)
    result_to_read_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    page_to_evidence_rate: float = Field(default=0.0, ge=0.0)


class CoverageCell(BaseModel):
    """矩阵中的一个格子：一个竞品在一个研究任务上的 Evidence 覆盖。"""

    cell_id: str
    competitor: str
    research_task: str
    evidence_ids: list[str] = Field(default_factory=list)
    covered: bool = False
    suggested_query: str | None = None
    # 四层信息分别回答：当前状态、证据质量、为什么是这个状态、下一步怎么办。
    status: CoverageStatus = "missing"
    quality: CoverageQuality = "none"
    quality_score: int = Field(default=0, ge=0, le=100)
    evidence_count: int = Field(default=0, ge=0)
    source_count: int = Field(default=0, ge=0)
    reason: str = "尚未找到可直接支持该研究格子的公开 Evidence。"
    # 明确告诉用户“还缺哪类证据”，避免只给一条抽象的补搜建议。
    missing_evidence: list[str] = Field(default_factory=list)
    next_action: str | None = None
    query_variants: list[str] = Field(default_factory=list)
    search_funnel: SearchFunnelMetrics = Field(default_factory=SearchFunnelMetrics)

    @model_validator(mode="before")
    @classmethod
    def populate_legacy_fields(cls, value: object) -> object:
        """旧数据库只有 covered/evidence_ids，读取时补齐新字段而不要求迁移。"""

        if not isinstance(value, dict) or "status" in value:
            return value
        upgraded = dict(value)
        evidence_ids = upgraded.get("evidence_ids", [])
        evidence_count = len(evidence_ids) if isinstance(evidence_ids, list) else 0
        covered = bool(upgraded.get("covered") or evidence_count)
        upgraded.setdefault("status", "covered" if covered else "missing")
        upgraded.setdefault("quality", "medium" if covered else "none")
        upgraded.setdefault("quality_score", 60 if covered else 0)
        upgraded.setdefault("evidence_count", evidence_count)
        upgraded.setdefault("source_count", 1 if covered else 0)
        upgraded.setdefault(
            "reason",
            (
                "这是升级前的历史覆盖结果，已有 Evidence，但没有保存来源质量明细。"
                if covered
                else "升级前的历史结果中没有该格子的 Evidence。"
            ),
        )
        upgraded.setdefault(
            "missing_evidence",
            (
                ["至少一个可独立验证的公开来源"]
                if covered
                else ["能够直接支持该研究任务的公开原文与来源链接"]
            ),
        )
        upgraded.setdefault("next_action", upgraded.get("suggested_query"))
        upgraded.setdefault("query_variants", [])
        upgraded.setdefault("search_funnel", {})
        return upgraded


class TaskCoverageSummary(BaseModel):
    """按 Research Plan 任务汇总所有竞品，让用户看到每个任务完成了多少。"""

    research_task: str
    status: TaskCoverageStatus
    covered_competitor_count: int = Field(ge=0)
    total_competitor_count: int = Field(ge=0)
    completion_rate: float = Field(ge=0.0, le=1.0)
    average_quality_score: float = Field(ge=0.0, le=100.0)
    missing_competitors: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)


class CoverageMatrix(BaseModel):
    """保存覆盖结果，随 ResearchResult 一起进入数据库 JSON。"""

    cells: list[CoverageCell] = Field(default_factory=list)
    covered_cell_count: int = Field(default=0, ge=0)
    total_cell_count: int = Field(default=0, ge=0)
    coverage_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    assessment_method: Literal["model", "fallback", "no_evidence"] = "no_evidence"
    verified_cell_count: int = Field(default=0, ge=0)
    weak_cell_count: int = Field(default=0, ge=0)
    average_quality_score: float = Field(default=0.0, ge=0.0, le=100.0)
    task_summaries: list[TaskCoverageSummary] = Field(default_factory=list)
    search_funnel_summary: SearchFunnelSummary = Field(
        default_factory=SearchFunnelSummary
    )
    scope_note: str = (
        "覆盖率仅表示当前 Research Plan 范围内找到公开 Evidence 的程度，"
        "不代表已经覆盖整个互联网，也不代表未覆盖能力不存在。"
    )


def build_cell_query_variants(
    competitor: str,
    research_task: str,
) -> list[str]:
    """为一个覆盖格子准备官方、独立评测和变更记录三种查询意图。"""

    task_without_duplicate_name = re.sub(
        re.escape(competitor),
        " ",
        research_task,
        flags=re.IGNORECASE,
    )
    normalized_task = " ".join(task_without_duplicate_name.split())
    base_query = f'"{competitor}" {normalized_task}'.strip()
    variants = [
        f"{base_query} official product documentation help center",
        f"{base_query} independent review comparison analysis",
        f"{base_query} release notes changelog feature announcement",
    ]
    return list(dict.fromkeys(query[:500].strip() for query in variants))


def _suggested_query(competitor: str, research_task: str) -> str:
    """默认先查官方一手来源；后续补搜会切换到其他查询变体。"""

    return build_cell_query_variants(competitor, research_task)[0]


def _empty_cells(
    research_tasks: list[str],
    competitors: list[str],
) -> list[CoverageCell]:
    """按稳定顺序建立完整矩阵，方便后续比较每次研究的覆盖率。"""

    return [
        CoverageCell(
            cell_id=f"C{competitor_index}-T{task_index}",
            competitor=competitor,
            research_task=research_task,
            suggested_query=_suggested_query(competitor, research_task),
            query_variants=build_cell_query_variants(competitor, research_task),
        )
        for competitor_index, competitor in enumerate(competitors, start=1)
        for task_index, research_task in enumerate(research_tasks, start=1)
    ]


def build_coverage_cells(
    research_tasks: list[str],
    competitors: list[str],
) -> list[CoverageCell]:
    """建立稳定的竞品 × 任务格子，搜索与最终 Coverage Matrix 共用同一 ID。"""

    normalized_tasks = list(
        dict.fromkeys(task.strip() for task in research_tasks if task.strip())
    )
    normalized_competitors = list(
        dict.fromkeys(name.strip() for name in competitors if name.strip())
    )
    return _empty_cells(normalized_tasks, normalized_competitors)


def _build_matrix(
    cells: list[CoverageCell],
    evidence_by_cell: dict[str, list[str]],
    evidence: list[Evidence],
    method: Literal["model", "fallback", "no_evidence"],
    stop_reason: str | None = None,
    search_funnels: dict[str, SearchFunnelMetrics] | None = None,
) -> CoverageMatrix:
    evidence_by_id = {
        item.evidence_id: item for item in evidence if item.evidence_id is not None
    }
    completed_cells: list[CoverageCell] = []
    for cell in cells:
        evidence_ids = evidence_by_cell.get(cell.cell_id, [])
        matched_evidence = [
            evidence_by_id[evidence_id]
            for evidence_id in evidence_ids
            if evidence_id in evidence_by_id
        ]
        sources = {
            (urlparse(item.url).hostname or item.source).casefold()
            for item in matched_evidence
            if urlparse(item.url).hostname or item.source
        }
        evidence_count = len(matched_evidence)
        source_count = len(sources)
        funnel = (
            (search_funnels or {}).get(cell.cell_id)
            or cell.search_funnel
        ).model_copy(deep=True)
        query = cell.suggested_query or _suggested_query(
            cell.competitor,
            cell.research_task,
        )

        # 质量分只衡量 Evidence 数量与独立来源，不假装衡量“全网完整度”。
        if evidence_count == 0:
            status: CoverageStatus = "missing"
            quality: CoverageQuality = "none"
            quality_score = 0
            if funnel.query_count == 0:
                reason = "该格子尚未执行搜索，可能因为初始搜索预算已分配完。"
            elif funnel.last_error and funnel.search_result_count == 0:
                reason = f"搜索执行失败：{funnel.last_error}"
            elif funnel.search_result_count == 0:
                reason = "查询已经执行，但搜索服务没有返回候选网页。"
            elif funnel.accepted_result_count == 0:
                reason = "搜索返回了网页，但都因重复或来源不符合多样性要求而被过滤。"
            elif funnel.pages_read == 0:
                if funnel.page_read_failures > 0:
                    failure_reason = _describe_page_read_error(funnel.last_error)
                    reason = (
                        f"已经找到候选网页，但有 {funnel.page_read_failures} 次读取失败"
                        f"（{failure_reason}）。候选 URL 已保留，可人工打开核查。"
                    )
                elif funnel.last_error:
                    reason = (
                        "已经找到候选网页，但本次未继续读取正文："
                        f"{_describe_page_read_error(funnel.last_error)}。"
                        "候选 URL 已保留，可人工打开核查。"
                    )
                else:
                    reason = (
                        "已经找到候选网页，但 Agent 尚未在本次页面读取预算内选择这些网页。"
                        "这不表示网页无法访问；候选 URL 已保留，可人工打开核查。"
                    )
            else:
                reason = "已读取候选网页，但没有提取到能直接支持该格子的 Evidence。"
            missing_evidence = [
                f"{cell.competitor} 官方资料中直接回答“{cell.research_task}”的原文",
                "至少一条可访问、可追溯的网页来源（标题、URL 和原文引用）",
                "另一个独立来源，用于对关键事实进行交叉验证",
            ]
            next_action = (
                f"继续搜索：{query}；如仍无结果，请用户提供官方资料或缩小问题范围。"
            )
        elif evidence_count == 1:
            status = "weak"
            quality = "low"
            quality_score = 40
            reason = "仅有 1 条 Evidence，尚缺独立来源交叉验证。"
            missing_evidence = [
                "第二个独立域名中的公开来源",
                "能够直接支持现有 Claim 的原文引用，而不是仅有搜索摘要",
            ]
            next_action = f"寻找第二个独立来源：{query}"
        elif source_count < 2:
            status = "covered"
            quality = "medium"
            quality_score = 60
            reason = f"已有 {evidence_count} 条 Evidence，但都来自同一来源。"
            missing_evidence = [
                "至少一个不同域名的独立来源",
                "能够与现有 Evidence 互相印证的原文引用",
            ]
            next_action = f"寻找独立来源交叉验证：{query}"
        else:
            status = "verified"
            quality = "high"
            quality_score = min(95, 75 + min(source_count, 4) * 5)
            reason = (
                f"已有 {evidence_count} 条 Evidence，来自 {source_count} 个独立来源，"
                "可以交叉验证。"
            )
            missing_evidence = []
            next_action = None

        if stop_reason is not None and next_action is not None:
            reason = f"研究因“{stop_reason}”停止；{reason}"
            next_action = f"增加研究预算或重新发起任务后，{next_action}"

        funnel.evidence_count = evidence_count
        funnel.evidence_source_count = source_count
        funnel.result_to_read_rate = (
            min(1.0, funnel.pages_read / funnel.accepted_result_count)
            if funnel.accepted_result_count
            else 0.0
        )
        funnel.page_to_evidence_rate = (
            evidence_count / funnel.pages_read if funnel.pages_read else 0.0
        )
        if status == "verified":
            funnel.outcome = "verified"
        elif evidence_count:
            funnel.outcome = "evidence_found"
        elif funnel.pages_read:
            funnel.outcome = "pages_read"
        elif funnel.query_count:
            funnel.outcome = (
                "search_failed"
                if funnel.last_error and funnel.search_result_count == 0
                else "searched"
            )

        completed_cells.append(
            cell.model_copy(
                update={
                    "evidence_ids": evidence_ids,
                    "covered": evidence_count > 0,
                    "suggested_query": None if evidence_count > 0 else query,
                    "status": status,
                    "quality": quality,
                    "quality_score": quality_score,
                    "evidence_count": evidence_count,
                    "source_count": source_count,
                    "reason": reason,
                    "missing_evidence": missing_evidence,
                    "next_action": next_action,
                    "search_funnel": funnel,
                }
            )
        )

    covered_count = sum(1 for cell in completed_cells if cell.covered)
    total_count = len(completed_cells)
    research_tasks = list(dict.fromkeys(cell.research_task for cell in completed_cells))
    task_summaries: list[TaskCoverageSummary] = []
    for research_task in research_tasks:
        task_cells = [
            cell for cell in completed_cells if cell.research_task == research_task
        ]
        task_covered_count = sum(1 for cell in task_cells if cell.covered)
        task_total_count = len(task_cells)
        if task_covered_count == 0:
            task_status: TaskCoverageStatus = "missing"
        elif task_covered_count < task_total_count:
            task_status = "partial"
        else:
            task_status = "complete"
        task_summaries.append(
            TaskCoverageSummary(
                research_task=research_task,
                status=task_status,
                covered_competitor_count=task_covered_count,
                total_competitor_count=task_total_count,
                completion_rate=(
                    task_covered_count / task_total_count if task_total_count else 0.0
                ),
                average_quality_score=(
                    sum(cell.quality_score for cell in task_cells) / task_total_count
                    if task_total_count
                    else 0.0
                ),
                missing_competitors=[
                    cell.competitor for cell in task_cells if not cell.covered
                ],
                next_actions=[
                    f"{cell.competitor}：{cell.next_action}"
                    for cell in task_cells
                    if cell.next_action is not None
                ],
            )
        )
    funnel_query_count = sum(
        cell.search_funnel.query_count for cell in completed_cells
    )
    funnel_result_count = sum(
        cell.search_funnel.search_result_count for cell in completed_cells
    )
    funnel_accepted_count = sum(
        cell.search_funnel.accepted_result_count for cell in completed_cells
    )
    funnel_pages_read = sum(
        cell.search_funnel.pages_read for cell in completed_cells
    )
    funnel_evidence_count = sum(
        cell.search_funnel.evidence_count for cell in completed_cells
    )
    return CoverageMatrix(
        cells=completed_cells,
        covered_cell_count=covered_count,
        total_cell_count=total_count,
        coverage_rate=(covered_count / total_count if total_count else 0.0),
        assessment_method=method,
        verified_cell_count=sum(
            1 for cell in completed_cells if cell.status == "verified"
        ),
        weak_cell_count=sum(
            1 for cell in completed_cells if cell.status in {"weak", "covered"}
        ),
        average_quality_score=(
            sum(cell.quality_score for cell in completed_cells) / total_count
            if total_count
            else 0.0
        ),
        task_summaries=task_summaries,
        search_funnel_summary=SearchFunnelSummary(
            query_count=funnel_query_count,
            search_result_count=funnel_result_count,
            accepted_result_count=funnel_accepted_count,
            pages_read=funnel_pages_read,
            page_read_failures=sum(
                cell.search_funnel.page_read_failures
                for cell in completed_cells
            ),
            evidence_count=funnel_evidence_count,
            result_to_read_rate=(
                min(1.0, funnel_pages_read / funnel_accepted_count)
                if funnel_accepted_count
                else 0.0
            ),
            page_to_evidence_rate=(
                funnel_evidence_count / funnel_pages_read
                if funnel_pages_read
                else 0.0
            ),
        ),
    )


def _fallback_evidence_mapping(
    cells: list[CoverageCell],
    evidence: list[Evidence],
) -> dict[str, list[str]]:
    """模型不可用时，只按明确出现的竞品名称保守匹配，不伪造覆盖。"""

    mapping: dict[str, list[str]] = {}
    for cell in cells:
        competitor = cell.competitor.casefold()
        matched_ids: list[str] = []
        for item in evidence:
            searchable_text = " ".join(
                (item.claim, item.title, item.url, item.evidence_text)
            ).casefold()
            if competitor in searchable_text and item.evidence_id is not None:
                matched_ids.append(item.evidence_id)
        if matched_ids:
            mapping[cell.cell_id] = list(dict.fromkeys(matched_ids))
    return mapping


def enrich_coverage_matrix(
    matrix: CoverageMatrix,
    evidence: list[Evidence],
    stop_reason: str | None = None,
    search_funnels: dict[str, SearchFunnelMetrics] | None = None,
) -> CoverageMatrix:
    """用已保存的 Evidence 重建质量信息，使升级前的矩阵也能完整展示。"""

    if not matrix.cells:
        return matrix
    evidence_by_cell = {
        cell.cell_id: cell.evidence_ids
        for cell in matrix.cells
        if cell.evidence_ids
    }
    return _build_matrix(
        matrix.cells,
        evidence_by_cell,
        evidence,
        matrix.assessment_method,
        stop_reason,
        search_funnels,
    )


def build_coverage_matrix(
    research_tasks: list[str],
    competitors: list[str],
    evidence: list[Evidence],
    search_funnels: dict[str, SearchFunnelMetrics] | None = None,
) -> CoverageMatrix:
    """将 Evidence 映射到覆盖格子；所有模型返回 ID 都经过白名单校验。"""

    cells = build_coverage_cells(research_tasks, competitors)
    if not cells:
        return CoverageMatrix()
    if not evidence:
        return _build_matrix(
            cells,
            {},
            evidence,
            "no_evidence",
            search_funnels=search_funnels,
        )

    valid_cell_ids = {cell.cell_id for cell in cells}
    valid_evidence_ids = {
        item.evidence_id for item in evidence if item.evidence_id is not None
    }
    try:
        response = request_json_object(
            purpose="analysis",
            messages=[
                {"role": "system", "content": COVERAGE_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "cells": [
                                {
                                    "cell_id": cell.cell_id,
                                    "competitor": cell.competitor,
                                    "research_task": cell.research_task,
                                }
                                for cell in cells
                            ],
                            "evidence": [
                                {
                                    "evidence_id": item.evidence_id,
                                    "claim": item.claim,
                                    "source": item.source,
                                    "title": item.title,
                                }
                                for item in evidence
                                if item.evidence_id is not None
                            ],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            response_format={"type": "json_object"},
            timeout=60.0,
        )
    except Exception:
        fallback = _fallback_evidence_mapping(cells, evidence)
        return _build_matrix(
            cells,
            fallback,
            evidence,
            "fallback",
            search_funnels=search_funnels,
        )

    evidence_by_cell: dict[str, list[str]] = {}
    raw_cells = response.get("cells", [])
    if not isinstance(raw_cells, list):
        raw_cells = []
    for raw_cell in raw_cells:
        if not isinstance(raw_cell, dict):
            continue
        cell_id = raw_cell.get("cell_id")
        raw_evidence_ids = raw_cell.get("evidence_ids")
        if cell_id not in valid_cell_ids or not isinstance(raw_evidence_ids, list):
            continue
        checked_ids = [
            evidence_id
            for evidence_id in raw_evidence_ids
            if isinstance(evidence_id, str) and evidence_id in valid_evidence_ids
        ]
        if checked_ids:
            evidence_by_cell[cell_id] = list(dict.fromkeys(checked_ids))

    return _build_matrix(
        cells,
        evidence_by_cell,
        evidence,
        "model",
        search_funnels=search_funnels,
    )


def targeted_gap_cells(
    matrix: CoverageMatrix,
    limit: int = MAX_TARGETED_GAPS,
) -> list[CoverageCell]:
    """每轮只选择少量缺口，避免覆盖补搜挤占整个研究预算。"""

    if limit <= 0:
        return []
    actionable_cells = [
        cell for cell in matrix.cells if cell.next_action is not None
    ]
    priority = {"not_started": 0, "missing": 1, "weak": 2, "covered": 3}
    return sorted(
        actionable_cells,
        key=lambda cell: priority.get(
            (
                "not_started"
                if (
                    cell.search_funnel.query_count == 0
                    and cell.status == "missing"
                )
                else cell.status
            ),
            4,
        ),
    )[:limit]
