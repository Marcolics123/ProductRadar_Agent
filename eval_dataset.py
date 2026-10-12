import json
from pathlib import Path

from pydantic import BaseModel, Field, TypeAdapter

from database import ResearchStatus, ResearchTask


DATASET_PATH = Path(__file__).with_name("eval_cases.json")


class EvalCase(BaseModel):
    """固定问题和最低质量门槛，用于不同版本之间做可重复比较。"""

    case_id: str = Field(min_length=1)
    own_product: str = Field(min_length=1)
    competitors: list[str] = Field(min_length=1)
    research_question: str = Field(min_length=1)
    minimum_evidence: int = Field(ge=0)
    maximum_unsupported_claims: int = Field(ge=0)


class OfflineEvalScore(BaseModel):
    case_id: str
    task_completed: bool
    evidence_minimum_met: bool
    competitor_coverage_rate: float = Field(ge=0.0, le=1.0)
    unsupported_claim_limit_met: bool


def load_eval_cases() -> list[EvalCase]:
    """数据集是固定 JSON；区别于线上 Eval，它不会自行调用模型或搜索。"""

    raw_data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    return TypeAdapter(list[EvalCase]).validate_python(raw_data)


def score_research_task(case: EvalCase, task: ResearchTask) -> OfflineEvalScore:
    """对已完成任务做确定性检查，避免评测本身再次产生模型费用。"""

    result = task.result
    evidence = result.evidence if result is not None else []
    searchable_text = " ".join(
        f"{item.claim} {item.title} {item.url}" for item in evidence
    ).lower()
    covered_competitors = sum(
        1 for competitor in case.competitors if competitor.lower() in searchable_text
    )
    coverage_rate = (
        covered_competitors / len(case.competitors)
        if case.competitors
        else 0.0
    )
    unsupported_count = (
        task.eval_result.unsupported_claim_count
        if task.eval_result is not None
        else sum(
            1
            for issue in result.critic.issues
            if issue.issue_type in {"missing_evidence", "unsupported_fact"}
        )
        if result is not None
        else 0
    )
    return OfflineEvalScore(
        case_id=case.case_id,
        task_completed=task.status == ResearchStatus.COMPLETED,
        evidence_minimum_met=len(evidence) >= case.minimum_evidence,
        competitor_coverage_rate=coverage_rate,
        unsupported_claim_limit_met=(
            unsupported_count <= case.maximum_unsupported_claims
        ),
    )
