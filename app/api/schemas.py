"""Request and response models for the HTTP contract (TASK §9)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    session_id: StrictStr = Field(min_length=1, max_length=200)
    message: StrictStr = Field(min_length=1, max_length=4000)

    @field_validator("session_id", "message")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class Citation(BaseModel):
    doc_id: str
    locator: str


class ToolCall(BaseModel):
    name: str
    args: dict[str, Any]
    status: Literal["pending_confirmation", "success", "error"]


class ChatResponse(BaseModel):
    request_id: str
    answer: str
    route: Literal["rag", "tool", "hybrid", "clarify", "no_answer", "refuse"]
    citations: list[Citation]
    tool_calls: list[ToolCall]
    pending_confirmation: bool
