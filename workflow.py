import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, TypedDict

# region 🟩【第三批新增】只允许安全的 checkpoint 类型反序列化
# 当前 State 仅使用 JSON 兼容数据。
os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")
# endregion

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agent import ResearchPlan, ResearchResult, run_agent
from evaluation import EvalResult, evaluate_research_result
from research_context import build_research_context
from tracing import TraceEvent
from usage import UsageStats, use_usage_tracker


# region 🟩【第三批修改 1】定义节点之间可持久化、可恢复的 State
class ResearchWorkflowState(TypedDict, total=False):
    """只保存节点之间必须共享的数据，不保存模型内部思维过程。"""

    research_id: str
    research_context: str
    competitors: list[str]
    research_plan: dict[str, Any]
    approved: bool
    result: dict[str, Any]
    eval_result: dict[str, Any] | None
    usage: dict[str, Any]


def _checkpoint_database_url() -> str:
    """把 SQLAlchemy URL 转成 psycopg 能直接识别的 PostgreSQL URL。"""

    from database import DATABASE_URL

    if not DATABASE_URL.startswith("postgresql"):
        raise RuntimeError("LangGraph checkpoint 当前只支持 PostgreSQL")
    return DATABASE_URL.replace("postgresql+psycopg://", "postgresql://", 1)


def workflow_config(research_id: str) -> dict[str, dict[str, str]]:
    # research_id 同时作为 LangGraph thread_id，是恢复同一工作流的稳定游标。
    return {"configurable": {"thread_id": research_id}}
# endregion


# region 🟩【第三批修改 2】研究计划确认前暂停工作流
def approval_node(state: ResearchWorkflowState) -> ResearchWorkflowState:
    """在任何搜索发生前暂停；恢复值包含批准后的最终 Research Plan。"""

    # interrupt 前不执行数据库写入或外部调用，因为恢复时本节点会从头运行。
    approval = interrupt(
        {
            "type": "research_plan_approval",
            "research_id": state["research_id"],
            "research_plan": state["research_plan"],
        }
    )
    if not isinstance(approval, dict) or approval.get("approved") is not True:
        raise ValueError("Research Plan 尚未获得批准")

    # Edit 和 Approve 使用同一恢复入口；Pydantic 继续负责最小结构校验。
    approved_plan = ResearchPlan.model_validate(
        approval.get("research_plan", state["research_plan"])
    )
    return {
        "approved": True,
        "research_plan": approved_plan.model_dump(mode="json"),
    }
# endregion


# region 🟩【第三批修改 3】把已有单 Agent 放入 Research 节点
def _current_usage(research_id: str, state: ResearchWorkflowState) -> UsageStats:
    """优先读取数据库累计用量，节点重试时不会把已经花掉的预算清零。"""

    from database import get_research_task

    research_task = get_research_task(research_id)
    if research_task is not None:
        return research_task.usage
    return UsageStats.model_validate(state.get("usage", {}))


def research_node(state: ResearchWorkflowState) -> ResearchWorkflowState:
    """复用现有单 Agent；节点成功后 LangGraph 才写入 Research checkpoint。"""

    from database import save_trace_event, save_usage_stats

    research_id = state["research_id"]
    research_plan = ResearchPlan.model_validate(state["research_plan"])
    initial_usage = _current_usage(research_id, state)
    competitors = state.get("competitors", [])
    if not competitors:
        # 兼容升级前已经写入、但尚未完成的 LangGraph checkpoint。
        from database import get_research_task

        current_task = get_research_task(research_id)
        competitors = current_task.competitors if current_task is not None else []

    with use_usage_tracker(initial_usage) as usage_tracker:

        def trace(event: TraceEvent) -> None:
            save_trace_event(research_id, event)
            save_usage_stats(research_id, usage_tracker.snapshot())

        try:
            trace(
                TraceEvent(
                    stage="research",
                    action="workflow_research_node",
                    status="started",
                    input_data={"research_plan": research_plan.model_dump()},
                )
            )
            result = run_agent(
                state["research_context"],
                research_plan,
                competitors=competitors,
                trace=trace,
            )
        finally:
            # 节点失败时也保存真实消耗；再次运行会从该累计值继续检查预算。
            save_usage_stats(research_id, usage_tracker.snapshot())

        trace(
            TraceEvent(
                stage="research",
                action="workflow_research_node",
                status="completed",
                output_data={"evidence_count": len(result.evidence)},
            )
        )
        return {
            "result": result.model_dump(mode="json"),
            "usage": usage_tracker.snapshot().model_dump(mode="json"),
        }
# endregion


# region 🟩【第三批修改 4】研究完成后进入独立 Eval 节点
def evaluation_node(state: ResearchWorkflowState) -> ResearchWorkflowState:
    """Eval 是独立 checkpoint；失败只记录，不推翻已经成功的研究结果。"""

    from database import save_trace_event, save_usage_stats

    research_id = state["research_id"]
    result = ResearchResult.model_validate(state["result"])
    initial_usage = _current_usage(research_id, state)

    with use_usage_tracker(initial_usage) as usage_tracker:

        def trace(event: TraceEvent) -> None:
            save_trace_event(research_id, event)
            save_usage_stats(research_id, usage_tracker.snapshot())

        eval_input = {
            "research_question": state["research_context"],
            "research_plan": result.research_plan.model_dump(),
            "evidence_count": len(result.evidence),
        }
        trace(
            TraceEvent(
                stage="eval",
                action="evaluate_research_result",
                status="started",
                input_data=eval_input,
            )
        )

        eval_result: EvalResult | None = None
        try:
            eval_result = evaluate_research_result(
                state["research_context"],
                result,
            )
        except Exception as error:
            trace(
                TraceEvent(
                    stage="eval",
                    action="evaluate_research_result",
                    status="failed",
                    input_data=eval_input,
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
            )
        else:
            trace(
                TraceEvent(
                    stage="eval",
                    action="evaluate_research_result",
                    status="completed",
                    input_data=eval_input,
                    output_data={"eval_result": eval_result.model_dump()},
                )
            )
        finally:
            save_usage_stats(research_id, usage_tracker.snapshot())

        return {
            "eval_result": (
                eval_result.model_dump(mode="json")
                if eval_result is not None
                else None
            ),
            "usage": usage_tracker.snapshot().model_dump(mode="json"),
        }
# endregion


# region 🟩【第三批修改 5】连接 Approval → Research → Evaluation
def _build_workflow(checkpointer: PostgresSaver):
    """节点边界就是 checkpoint 边界，避免一次失败重跑全部已完成阶段。"""

    builder = StateGraph(ResearchWorkflowState)
    builder.add_node("approval", approval_node)
    builder.add_node("research", research_node)
    builder.add_node("evaluation", evaluation_node)
    builder.add_edge(START, "approval")
    builder.add_edge("approval", "research")
    builder.add_edge("research", "evaluation")
    builder.add_edge("evaluation", END)
    return builder.compile(checkpointer=checkpointer)


@contextmanager
def open_research_workflow() -> Iterator[Any]:
    """每次运行使用独立 PostgreSQL 连接，适配 API 与 Celery 多进程。"""

    with PostgresSaver.from_conn_string(_checkpoint_database_url()) as checkpointer:
        yield _build_workflow(checkpointer)
# endregion


# region 🟩【第三批修改 6】初始化 PostgreSQL checkpoint 表
def setup_workflow_checkpointer() -> None:
    """部署阶段执行一次；setup 本身可重复执行，不在每个请求中建表。"""

    with PostgresSaver.from_conn_string(_checkpoint_database_url()) as checkpointer:
        checkpointer.setup()


def workflow_checkpointer_is_ready() -> bool:
    try:
        with PostgresSaver.from_conn_string(
            _checkpoint_database_url()
        ) as checkpointer:
            next(
                checkpointer.list(workflow_config("readiness-check"), limit=1),
                None,
            )
        return True
    except Exception:
        return False
# endregion


# region 🟩【第三批修改 7】新任务先保存审批断点，不执行搜索
def _initial_state(research_task: Any) -> ResearchWorkflowState:
    return {
        "research_id": research_task.research_id,
        "research_context": build_research_context(
            research_task.own_product,
            research_task.competitors,
            research_task.research_question,
        ),
        "competitors": research_task.competitors,
        "research_plan": research_task.research_plan.model_dump(mode="json"),
        "approved": False,
        "usage": research_task.usage.model_dump(mode="json"),
    }


def create_research_workflow_checkpoint(research_task: Any) -> None:
    """创建任务时运行到 approval interrupt，然后立即返回等待用户操作。"""

    config = workflow_config(research_task.research_id)
    with open_research_workflow() as graph:
        snapshot = graph.get_state(config)
        if snapshot.values:
            return
        output = graph.invoke(_initial_state(research_task), config=config)
        if "__interrupt__" not in output:
            raise RuntimeError("Research 工作流没有停在人工审批节点")
# endregion


# region 🟩【第三批修改 8】审批通过或失败重试时恢复工作流
def run_research_workflow(
    research_task: Any,
) -> tuple[ResearchResult, EvalResult | None, UsageStats]:
    """首次批准使用 Command.resume；失败重试使用 None 从未完成节点继续。"""

    config = workflow_config(research_task.research_id)
    with open_research_workflow() as graph:
        snapshot = graph.get_state(config)

        # 兼容第三批上线前已经创建、但尚未执行的旧 Research Task。
        if not snapshot.values:
            graph.invoke(_initial_state(research_task), config=config)
            snapshot = graph.get_state(config)

        if not snapshot.values.get("approved", False):
            graph.invoke(
                Command(
                    resume={
                        "approved": True,
                        "research_plan": research_task.research_plan.model_dump(
                            mode="json"
                        ),
                    }
                ),
                config=config,
            )
        elif snapshot.next:
            # 相同 thread_id + None 表示从最近 checkpoint 的失败节点继续。
            graph.invoke(None, config=config)

        final_state = graph.get_state(config)
        if final_state.next:
            raise RuntimeError(
                f"Research 工作流尚未完成，下一节点：{final_state.next}"
            )

        values = final_state.values
        if "result" not in values:
            raise RuntimeError("Research 工作流完成但缺少研究结果")

        result = ResearchResult.model_validate(values["result"])
        eval_value = values.get("eval_result")
        eval_result = (
            EvalResult.model_validate(eval_value)
            if isinstance(eval_value, dict)
            else None
        )
        usage = UsageStats.model_validate(values.get("usage", {}))
        return result, eval_result, usage
# endregion
