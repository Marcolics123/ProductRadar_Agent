import json
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from analysis import AnalysisFinding, CompetitiveAnalysis
from evidence import Evidence
# 🟦【第四批修改：模型网关】Critic 可与研究模型分开配置，避免单一模型故障阻断检查。
from llm_client import request_json_object

CRITIC_PROMPT = """你是 ProductRadar 的 Critic 检查组件。
你只能检查输入中的 CompetitiveAnalysis 和 Evidence，不能搜索、访问网页、补充外部事实或重写分析。
请检查：重要判断是否缺少证据、Evidence 是否可能过旧、Evidence 之间是否冲突，以及判断中是否包含 Evidence 未支持的具体事实。
只有 published_date 明确存在时才能判断 Evidence 是否过旧；日期缺失不等于证据过旧。
输出必须使用中文，并只返回以下格式的 JSON：
{
  "issues": [
    {
      "issue_type": "missing_evidence",
      "description": "...",
      "related_finding": "...",
      "evidence_ids": ["E1"]
    }
  ],
  "missing_research_tasks": ["需要补充了解什么"]
}
issue_type 只能是 missing_evidence、stale_evidence、conflicting_evidence 或 unsupported_fact。
related_finding 必须原样复制输入分析中的 finding；问题不对应单一 finding 时使用 null。
evidence_ids 只能引用输入中真实存在的编号。missing_evidence 可以没有 evidence_ids，stale_evidence 和 unsupported_fact 至少引用一个编号，conflicting_evidence 至少引用两个编号。
missing_research_tasks 最多三项，必须描述需要补充了解什么，不能写成具体搜索关键词。
如果没有问题，issues 和 missing_research_tasks 都返回空数组。
"""


class CriticIssue(BaseModel):
    issue_type: Literal[
        "missing_evidence",
        "stale_evidence",
        "conflicting_evidence",
        "unsupported_fact",
    ]
    description: str = Field(min_length=1)
    related_finding: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)


class CriticResult(BaseModel):
    passed: bool
    issues: list[CriticIssue]
    missing_research_tasks: list[str] = Field(max_length=3)

    @field_validator("missing_research_tasks", mode="after")
    @classmethod
    def validate_non_empty_and_blank(cls, tasks: list[str]) -> list[str]:
        if any(v.strip() == "" for v in tasks):
            raise ValueError("验证失败！missing_research_tasks内容为空！")
        return tasks


def apply_critic_evidence_gate(
    analysis: CompetitiveAnalysis,
    critic: CriticResult,
) -> tuple[CompetitiveAnalysis, CriticResult, int]:
    """删除 Critic 已确认缺少证据的判断，不让问题结论继续进入最终报告。"""

    blocked_findings = {
        issue.related_finding
        for issue in critic.issues
        if issue.issue_type in {"missing_evidence", "unsupported_fact"}
        and issue.related_finding is not None
    }
    if not blocked_findings:
        return analysis, critic, 0

    def allowed(findings: list[AnalysisFinding]) -> list[AnalysisFinding]:
        return [
            finding for finding in findings if finding.finding not in blocked_findings
        ]

    filtered_analysis = CompetitiveAnalysis(
        competitor_comparison=allowed(analysis.competitor_comparison),
        common_capabilities=allowed(analysis.common_capabilities),
        differentiators=allowed(analysis.differentiators),
        gaps=allowed(analysis.gaps),
        product_opportunities=allowed(analysis.product_opportunities),
        mvp_recommendations=allowed(analysis.mvp_recommendations),
    )
    # 被删除的 Finding 已不再属于最终结果，对应问题保留在 Trace 中即可。
    remaining_issues = [
        issue
        for issue in critic.issues
        if issue.related_finding not in blocked_findings
    ]
    filtered_critic = CriticResult(
        passed=not remaining_issues and not critic.missing_research_tasks,
        issues=remaining_issues,
        missing_research_tasks=critic.missing_research_tasks,
    )
    return filtered_analysis, filtered_critic, len(blocked_findings)


def review_analysis(
    research_question: str,
    analysis: CompetitiveAnalysis,
    evidence: list[Evidence],
) -> CriticResult:
    # 🟦【第四批修改：Critic 路由】切换模型只改变检查执行者，不允许它搜索或新增事实。
    response_data = request_json_object(
        purpose="critic",
        messages=[
            {"role": "system", "content": CRITIC_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "research_question": research_question,
                        "current_date": date.today().isoformat(),
                        "analysis": analysis.model_dump(),
                        "evidence": [item.model_dump() for item in evidence],
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        response_format={"type": "json_object"},
        timeout=60.0,
    )
    if "issues" not in response_data and "missing_research_tasks" not in response_data:
        raise ValueError("Critic 模型返回缺少必要字段")

    evidence_by_id = {
        item.evidence_id: item
        for item in evidence
        if item.evidence_id is not None
    }
    valid_evidence_ids = set(evidence_by_id)
    valid_findings = {
        finding.finding
        for findings in (
            analysis.competitor_comparison,
            analysis.common_capabilities,
            analysis.differentiators,
            analysis.gaps,
            analysis.product_opportunities,
            analysis.mvp_recommendations,
        )
        for finding in findings
    }

    def issue_references_are_valid(issue: CriticIssue) -> bool:
        evidence_ids = set(issue.evidence_ids)
        if not evidence_ids.issubset(valid_evidence_ids):
            return False
        if issue.related_finding is not None and issue.related_finding not in valid_findings:
            return False
        if issue.issue_type == "conflicting_evidence":
            return len(evidence_ids) >= 2
        if issue.issue_type == "stale_evidence":
            return len(evidence_ids) >= 1 and all(
                evidence_by_id[evidence_id].published_date is not None
                for evidence_id in evidence_ids
            )
        if issue.issue_type == "unsupported_fact":
            return len(evidence_ids) >= 1
        return True

    raw_issues = response_data.get("issues", [])
    raw_tasks = response_data.get("missing_research_tasks", [])
    response_structure_is_valid = (
        "issues" in response_data
        and "missing_research_tasks" in response_data
        and isinstance(raw_issues, list)
        and isinstance(raw_tasks, list)
    )

    issues: list[CriticIssue] = []
    if isinstance(raw_issues, list):
        for raw_issue in raw_issues:
            if not isinstance(raw_issue, dict):
                continue
            description = raw_issue.get("description")
            if not isinstance(description, str) or not description.strip():
                continue
            try:
                issue = CriticIssue.model_validate(raw_issue)
            except ValidationError:
                continue
            if issue_references_are_valid(issue):
                issues.append(issue)

    missing_research_tasks: list[str] = []
    if isinstance(raw_tasks, list):
        for raw_task in raw_tasks:
            if not isinstance(raw_task, str) or not raw_task.strip():
                continue
            task = raw_task.strip()
            if task not in missing_research_tasks:
                missing_research_tasks.append(task)
            if len(missing_research_tasks) == 3:
                break

    return CriticResult(
        passed=(
            response_structure_is_valid
            and not issues
            and not missing_research_tasks
        ),
        issues=issues,
        missing_research_tasks=missing_research_tasks,
    )
