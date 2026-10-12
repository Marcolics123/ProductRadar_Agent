import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

import httpx

from usage import record_model_response, reserve_model_call


logger = logging.getLogger(__name__)

DEFAULT_QWEN_API_URL = (
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
)
DEFAULT_QWEN_MODEL = "qwen3.5-plus"

# region 🟦【第四批修改 1】按研究阶段选择模型
# 默认情况下所有阶段仍使用 qwen3.5-plus；只有配置对应环境变量时才会分流，
# 因此升级后不会改变现有部署行为。
PURPOSE_MODEL_ENV = {
    "planning": "PLANNING_MODEL",
    "agent": "RESEARCH_MODEL",
    "evidence": "EVIDENCE_MODEL",
    "analysis": "ANALYSIS_MODEL",
    "critic": "CRITIC_MODEL",
    "evaluation": "EVALUATION_MODEL",
}


@dataclass(frozen=True)
class ModelRoute:
    """一次模型请求需要的最小配置；API Key 永远不会写入日志或 Trace。"""

    name: str
    api_url: str
    api_key: str
    model: str


def _configured_routes(purpose: str) -> list[ModelRoute]:
    """返回主路由和可选降级路由，并对不完整配置立即报错。"""

    model_env = PURPOSE_MODEL_ENV.get(purpose)
    if model_env is None:
        raise ValueError(f"未知模型用途：{purpose}")

    primary_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not primary_key:
        raise RuntimeError("缺少 DASHSCOPE_API_KEY")

    default_model = os.environ.get("QWEN_MODEL", "").strip() or DEFAULT_QWEN_MODEL
    primary_model = os.environ.get(model_env, "").strip() or default_model
    routes = [
        ModelRoute(
            name="primary",
            api_url=(
                os.environ.get("QWEN_API_URL", "").strip()
                or DEFAULT_QWEN_API_URL
            ),
            api_key=primary_key,
            model=primary_model,
        )
    ]
    #降级模型
    fallback_values = {
        "api_url": os.environ.get("FALLBACK_LLM_API_URL", "").strip(),
        "api_key": os.environ.get("FALLBACK_LLM_API_KEY", "").strip(),
        "model": os.environ.get("FALLBACK_LLM_MODEL", "").strip(),
    }
    if any(fallback_values.values()):
        if not all(fallback_values.values()):
            raise RuntimeError(
                "FALLBACK_LLM_API_URL、FALLBACK_LLM_API_KEY 和 "
                "FALLBACK_LLM_MODEL 必须同时配置"
            )
        routes.append(
            ModelRoute(
                name="fallback",
                api_url=fallback_values["api_url"],
                api_key=fallback_values["api_key"],
                model=fallback_values["model"],
            )
        )
    return routes
# endregion


# region 🟦【第四批修改 2】统一验证兼容 OpenAI 协议的返回结构
def _response_message(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as error:
        raise ValueError("模型返回内容不是有效 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("模型返回格式无效")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("模型未返回 choices")
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ValueError("模型返回的 choice 格式无效")
    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("模型未返回有效 message")
    return message


def _json_object_from_response(response: httpx.Response) -> dict[str, Any]:
    """解析结构化输出，并兼容少数模型额外包裹的 Markdown JSON 围栏。"""

    message = _response_message(response)
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("模型未返回有效 content")

    normalized = content.strip()
    if normalized.startswith("```") and normalized.endswith("```"):
        lines = normalized.splitlines()
        if len(lines) >= 3:
            normalized = "\n".join(lines[1:-1]).strip()

    try:
        payload = json.loads(normalized)
    except json.JSONDecodeError as error:
        raise ValueError("模型 content 不是有效 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("模型 content 必须是 JSON 对象")
    return payload
# endregion


ResponseValue = TypeVar("ResponseValue")


# region 🟦【第四批修改 3】主模型故障时执行一次可选降级
def _request_with_routes(
    *,
    purpose: str,
    messages: list[dict[str, Any]],
    response_parser: Callable[[httpx.Response], ResponseValue],
    timeout: float,
    request_options: dict[str, Any],
) -> ResponseValue:
    """仅对网络错误、限流、服务端错误或无效响应使用下一条路由。"""

    routes = _configured_routes(purpose)
    last_error: Exception | None = None

    for route_index, route in enumerate(routes):
        try:
            # 每一次真实供应商请求都占用模型预算，降级调用也不会绕过成本限制。
            reserve_model_call()
            request_payload = {
                **request_options,
                "model": route.model,
                "messages": messages,
            }
            # DashScope 使用该字段关闭思考内容；通用降级供应商可能不认识它，
            # 所以不会把供应商专属参数发送给 fallback。
            if route.name == "primary":
                request_payload["enable_thinking"] = False

            response = httpx.post(
                route.api_url,
                headers={"Authorization": f"Bearer {route.api_key}"},
                json=request_payload,
                timeout=timeout,
            )
            response.raise_for_status()
            record_model_response(response)
            return response_parser(response)
        except httpx.HTTPStatusError as error:
            status_code = error.response.status_code
            # 认证、权限和请求格式错误不会换模型掩盖；只降级临时性故障。
            if status_code != 429 and status_code < 500:
            #如果不是限流（429）且不是服务端错误（5xx），说明是客户端错误（如 400 请求格式错误、401 认证失败、403 权限不足）。这类错误换模型也没用
                raise
            last_error = error
        except (httpx.TransportError, ValueError) as error:
            last_error = error

        if route_index + 1 < len(routes):
            logger.warning(
                "主模型调用失败，切换到降级模型",
                extra={"stage": purpose, "model_route": route.name},
            )

    if last_error is None:
        raise RuntimeError("没有可用的模型路由")
    raise last_error
# endregion


def request_chat_message(
    *,
    purpose: str,
    messages: list[dict[str, Any]],
    timeout: float = 60.0,
    **request_options: Any,
) -> dict[str, Any]:
    """用于需要 Tool Calling 的 Agent 请求，返回标准 message 对象。"""

    return _request_with_routes(
        purpose=purpose,
        messages=messages,
        response_parser=_response_message,
        timeout=timeout,
        request_options=request_options,
    )


def request_json_object(
    *,
    purpose: str,
    messages: list[dict[str, Any]],
    timeout: float = 60.0,
    **request_options: Any,
) -> dict[str, Any]:
    """用于 Plan、Evidence、Analysis、Critic 和 Eval 的结构化输出。"""

    return _request_with_routes(
        purpose=purpose,
        messages=messages,
        response_parser=_json_object_from_response,
        timeout=timeout,
        request_options=request_options,
    )
