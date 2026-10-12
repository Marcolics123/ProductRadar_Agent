from datetime import datetime, timezone
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field


class TraceEvent(BaseModel):
    stage: Literal[
        "planning",
        "approval",
        "agent",
        "tool",
        "evidence",
        "coverage",
        "analysis",
        "critic",
        "research",
        "eval",
    ]
    action: str = Field(min_length=1)
    status: Literal["started", "completed", "failed"]
    tool_call_id: str | None = None
    input_data: dict[str, Any] = Field(default_factory=dict)
    output_data: dict[str, Any] = Field(default_factory=dict)
    error_type: str | None = None
    error_message: str | None = None
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


TraceCallback = Callable[[TraceEvent], None]
