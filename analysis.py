import json

from pydantic import BaseModel, Field, ValidationError, field_validator

from evidence import Evidence
# 🟦【第四批修改：模型网关】Analyst 可独立选择模型，并共享统一降级策略。
from llm_client import request_json_object

ANALYSIS_PROMPT = """你是 ProductRadar 的竞品分析组件。
你只能使用输入中的 Evidence 作为事实依据，不能搜索、访问网页或补充外部事实。
输出必须使用中文，并只返回以下格式的 JSON：
{
  "competitor_comparison": [{"finding": "...", "evidence_ids": ["E1"]}],
  "common_capabilities": [{"finding": "...", "evidence_ids": ["E1", "E2"]}],
  "differentiators": [{"finding": "...", "evidence_ids": ["E1", "E2"]}],
  "gaps": [{"finding": "...", "evidence_ids": ["E1"]}],
  "product_opportunities": [{"finding": "...", "evidence_ids": ["E1", "E2"]}],
  "mvp_recommendations": [{"finding": "...", "evidence_ids": ["E1", "E2"]}]
}
每个判断至少引用一个输入中真实存在的 evidence_id。
不能把没有找到 Evidence 当作某项能力不存在。
product_opportunities 可以是基于 Evidence 的推论，但不能表述为已经证实的事实。
mvp_recommendations 必须是基于 Evidence 的最小产品建议，不能引入新的事实。
某一类没有足够 Evidence 时返回空数组。
"""


class AnalysisFinding(BaseModel):
    finding: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)

    @field_validator("finding", mode="after")
    @classmethod
    def normalize_finding(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("分析结论不能为空")
        return value

    @field_validator("evidence_ids", mode="after")
    @classmethod
    def normalize_evidence_ids(cls, values: list[str]) -> list[str]:
        # 保持引用顺序并去重，避免一个结论用重复 ID 伪装成多条证据。
        normalized = list(dict.fromkeys(value.strip() for value in values))
        if any(not value for value in normalized):
            raise ValueError("evidence_id 不能为空")
        return normalized


class CompetitiveAnalysis(BaseModel):
    competitor_comparison: list[AnalysisFinding]
    common_capabilities: list[AnalysisFinding]
    differentiators: list[AnalysisFinding]
    gaps: list[AnalysisFinding]
    product_opportunities: list[AnalysisFinding]
    mvp_recommendations: list[AnalysisFinding] = Field(default_factory=list)


def analyze_evidence(
    research_question: str,
    evidence: list[Evidence],
) -> CompetitiveAnalysis:
    if not evidence:
        return CompetitiveAnalysis(
            competitor_comparison=[],
            common_capabilities=[],
            differentiators=[],
            gaps=[],
            product_opportunities=[],
            mvp_recommendations=[],
        )

    # 🟦【第四批修改：Analysis 路由】仍然只传入已有 Evidence，模型切换不会扩大事实来源。
    analysis_data = request_json_object(
        purpose="analysis",
        messages=[
            {"role": "system", "content": ANALYSIS_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Research question:\n{research_question}\n\n"
                    "Evidence:\n"
                    f"{json.dumps([item.model_dump() for item in evidence], ensure_ascii=False)}"
                ),
            },
        ],
        response_format={"type": "json_object"},
        timeout=60.0,
    )

    valid_evidence_ids = {
        item.evidence_id for item in evidence if item.evidence_id is not None
    }

    def valid_findings(category: str) -> list[AnalysisFinding]:
        raw_findings = analysis_data.get(category, [])
        if not isinstance(raw_findings, list):
            return []

        findings: list[AnalysisFinding] = []
        for raw_finding in raw_findings:
            try:
                finding = AnalysisFinding.model_validate(raw_finding)
            except ValidationError:
                continue
            # 与旧版自由文本引用相比，只接受本次研究中真实存在的 Evidence ID。
            if set(finding.evidence_ids).issubset(valid_evidence_ids):
                findings.append(finding)
        return findings

    return CompetitiveAnalysis(
        competitor_comparison=valid_findings("competitor_comparison"),
        common_capabilities=valid_findings("common_capabilities"),
        differentiators=valid_findings("differentiators"),
        gaps=valid_findings("gaps"),
        product_opportunities=valid_findings("product_opportunities"),
        mvp_recommendations=valid_findings("mvp_recommendations"),
    )
