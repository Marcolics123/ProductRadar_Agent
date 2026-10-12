import json

from pydantic import BaseModel, Field, model_validator

from agent import ResearchResult
# 🟦【第四批修改：模型网关】Eval 可以使用更便宜或独立的模型，并支持故障降级。
from llm_client import request_json_object

EVAL_PROMPT = """你是 ProductRadar 的结果评估组件。
你只能评估输入中的 Research Plan、Evidence、Analysis 和 Critic，不能搜索、访问网页或补充外部事实。
请返回以下格式的 JSON：
{
  "completed_tasks": ["原样复制已被 Evidence 和 Analysis 覆盖的计划任务"],
  "correct_evidence_ids": ["claim 被 evidence_text 直接支持的 Evidence ID"],
  "reliable_sources": ["可以根据现有 URL、标题和来源类型判断为官方、一手或可信来源的 source"]
}
只能返回输入中真实存在的任务、Evidence ID 和 source；无法判断时不要加入对应列表。
不要重新生成研究结论，也不要输出评分或解释。
"""


class RateMetric(BaseModel):
    matched_count: int = Field(ge=0)
    total_count: int = Field(ge=0)
    rate: float = Field(ge=0.0, le=1.0)

    @model_validator(mode='after')
    def validate_range(self):
        if self.matched_count > self.total_count:
            raise ValueError('matched_count 不能大于 total_count')
        
        return self


class EvalResult(BaseModel):
    task_completion: RateMetric
    evidence_accuracy: RateMetric
    source_reliability: RateMetric
    unsupported_claim_count: int = Field(ge=0)

#这是千问返回结果的中间数据结构，不是最终指标。
class EvalResponse(BaseModel):
    completed_tasks: list[str]
    correct_evidence_ids: list[str]
    reliable_sources: list[str]


def build_rate(matched_count: int, total_count: int) -> RateMetric:
    rate = matched_count / total_count if total_count else 0.0
    return RateMetric(
        matched_count=matched_count,
        total_count=total_count,
        rate=rate,
    )


def evaluate_research_result(
    research_question: str,
    result: ResearchResult,
) -> EvalResult:
    unsupported_findings = {
        issue.related_finding
        for issue in result.critic.issues
        if issue.issue_type in {"missing_evidence", "unsupported_fact"}
        and issue.related_finding is not None
    }

    if not result.evidence:
        return EvalResult(
            task_completion=build_rate(0, len(result.research_plan.tasks)),
            evidence_accuracy=build_rate(0, 0),
            source_reliability=build_rate(0, 0),
            unsupported_claim_count=len(unsupported_findings),
        )

    # 🟦【第四批修改：Eval 路由】Eval 仍只评价现有结果，不因换模型获得搜索权限。
    response_data = request_json_object(
        purpose="evaluation",
        messages=[
            {"role": "system", "content": EVAL_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "research_question": research_question,
                        "research_plan": result.research_plan.model_dump(),
                        "evidence": [
                            item.model_dump() for item in result.evidence
                        ],
                        "analysis": result.analysis.model_dump(),
                        "critic": result.critic.model_dump(),
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        response_format={"type": "json_object"},
        timeout=60.0,
    )
    expected_fields = {
        "completed_tasks",
        "correct_evidence_ids",
        "reliable_sources",
    }
    if not expected_fields.intersection(response_data):
        raise ValueError("Eval 模型返回缺少必要字段")

    def valid_strings(field_name: str) -> list[str]:
        values = response_data.get(field_name, [])
        if not isinstance(values, list):
            return []
        return [
            value.strip()
            for value in values
            if isinstance(value, str) and value.strip()
        ]

    eval_response = EvalResponse(
        completed_tasks=valid_strings("completed_tasks"),
        correct_evidence_ids=valid_strings("correct_evidence_ids"),
        reliable_sources=valid_strings("reliable_sources"),
    )

    completed_tasks = set(eval_response.completed_tasks).intersection(
        result.research_plan.tasks
    )
    evidence_ids = {
        item.evidence_id
        for item in result.evidence
        if item.evidence_id is not None
    }
    correct_evidence_ids = set(eval_response.correct_evidence_ids).intersection(
        evidence_ids
    )
    sources = {item.source for item in result.evidence}
    reliable_sources = set(eval_response.reliable_sources).intersection(sources)

    completed_task_count = sum(
        1 for task in result.research_plan.tasks if task in completed_tasks
    )

    return EvalResult(
        task_completion=build_rate(
            completed_task_count,
            len(result.research_plan.tasks),
        ),
        evidence_accuracy=build_rate(
            len(correct_evidence_ids),
            len(result.evidence),
        ),
        source_reliability=build_rate(
            len(reliable_sources),
            len(sources),
        ),
        unsupported_claim_count=len(unsupported_findings),
    )
