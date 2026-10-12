def build_research_context(
    own_product: str,
    competitors: list[str],
    research_question: str,
) -> str:
    """把结构化字段组装为模型上下文，但不改变数据库中的独立字段。"""

    if not own_product and not competitors:
        # 第一批旧数据已经是一段完整问题，保留兼容而不强行解析历史文本。
        return research_question
    return (
        f"自己的产品：{own_product}\n"
        f"竞争对手：{', '.join(competitors)}\n"
        f"研究问题：{research_question}"
    )
