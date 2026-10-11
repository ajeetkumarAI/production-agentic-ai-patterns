"""Typed tool definition shared by the agent projects.

A ``Tool`` has a Pydantic argument model (validated before the call) and an
optional Pydantic output model (validated after the call), and can render
itself as an OpenAI function-tool spec.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, ValidationError


class ToolError(Exception):
    """Raised for invalid arguments, invalid outputs or failures inside a tool."""


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[..., Any]
    args_model: type[BaseModel] | None = None
    output_model: type[BaseModel] | None = None
    capabilities: set[str] = field(default_factory=set)
    required_scopes: set[str] = field(default_factory=set)
    side_effects: bool = False
    priority: int = 0          # higher wins when several tools offer a capability
    timeout_s: float = 10.0

    def to_openai(self) -> dict:
        params = self.args_model.model_json_schema() if self.args_model else {"type": "object", "properties": {}}
        params.pop("title", None)
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": params}}

    def validate_args(self, args: dict | str) -> dict:
        if isinstance(args, str):
            try:
                args = json.loads(args or "{}")
            except json.JSONDecodeError as e:
                raise ToolError(f"{self.name}: arguments are not valid JSON ({e})") from e
        if self.args_model is None:
            return dict(args)
        try:
            return self.args_model.model_validate(args).model_dump()
        except ValidationError as e:
            raise ToolError(f"{self.name}: invalid arguments: {e.errors()[:3]}") from e

    def validate_output(self, out: Any) -> Any:
        if self.output_model is None:
            return out
        try:
            if isinstance(out, BaseModel):
                out = out.model_dump()
            return self.output_model.model_validate(out).model_dump()
        except ValidationError as e:
            raise ToolError(f"{self.name}: tool returned invalid output: {e.errors()[:3]}") from e

    def __call__(self, args: dict | str) -> Any:
        clean = self.validate_args(args)
        try:
            out = self.func(**clean)
        except ToolError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ToolError(f"{self.name} failed: {type(e).__name__}: {e}") from e
        return self.validate_output(out)


def to_text(value: Any, limit: int = 4000) -> str:
    s = value if isinstance(value, str) else json.dumps(value, default=str)
    return s if len(s) <= limit else s[:limit] + "...[truncated]"
