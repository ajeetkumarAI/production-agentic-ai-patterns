"""Pydantic-validated structured generation with repair-retries.

This is the engine behind Project 1, shared by every other project that needs
reliable JSON from the model.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from .llm import BaseLLM, Message

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(Exception):
    def __init__(self, message: str, errors: list[str], last_raw: str | None):
        super().__init__(message)
        self.errors = errors
        self.last_raw = last_raw


@dataclass
class StructuredResult(Generic[T]):
    value: T
    attempts: int
    errors: list[str] = field(default_factory=list)
    raw: str = ""


def _strictify(schema: dict) -> dict:
    """Make a pydantic JSON schema acceptable to OpenAI strict mode:
    every object gets additionalProperties=false and all properties required."""
    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"].keys())
            for k in ("default", "title"):
                node.pop(k, None)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    schema = json.loads(json.dumps(schema))
    walk(schema)
    return schema


def schema_response_format(schema: type[BaseModel], strict: bool = False) -> dict:
    js = schema.model_json_schema()
    if strict:
        js = _strictify(js)
    return {"type": "json_schema", "json_schema": {"name": schema.__name__, "schema": js, "strict": strict}}


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str | None) -> str:
    """Pull the first JSON object/array out of a model reply (handles ``` fences and chatter)."""
    if not text:
        raise ValueError("empty model output")
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    text = text.strip()
    if text[:1] in "{[":
        return text
    start = min((i for i in (text.find("{"), text.find("[")) if i >= 0), default=-1)
    if start < 0:
        raise ValueError("no JSON object found in output")
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    raise ValueError("unbalanced JSON in output")


def generate_structured(
    llm: BaseLLM,
    messages: list[Message],
    schema: type[T],
    *,
    max_retries: int = 2,
    on_failure: Callable[[dict], None] | None = None,
    use_response_format: bool = True,
    strict: bool = False,
    **chat_kwargs: Any,
) -> StructuredResult[T]:
    """Call the LLM until its output validates against ``schema``.

    On a parse/validation error the exact error is fed back to the model as a
    repair instruction (up to ``max_retries`` extra attempts). Every failure is
    reported to ``on_failure`` so it can be logged.
    """
    msgs = list(messages)
    if not any(m["role"] == "system" and "JSON" in str(m.get("content", "")) for m in msgs):
        msgs.insert(0, {"role": "system", "content": (
            "Respond ONLY with a JSON object matching this JSON schema (no prose, no markdown):\n"
            + json.dumps(schema.model_json_schema()))})
    rf = schema_response_format(schema, strict=strict) if use_response_format else None
    errors: list[str] = []
    raw = None
    for attempt in range(1, max_retries + 2):
        resp = llm.chat(msgs, response_format=rf, **chat_kwargs)
        raw = resp.content
        try:
            value = schema.model_validate_json(extract_json(raw))
            return StructuredResult(value=value, attempts=attempt, errors=errors, raw=raw or "")
        except (ValidationError, ValueError) as e:
            err = _short_error(e)
            errors.append(err)
            if on_failure:
                on_failure({"schema": schema.__name__, "attempt": attempt, "error": err, "raw": raw,
                            "finish_reason": resp.finish_reason})
            msgs = msgs + [
                {"role": "assistant", "content": raw or ""},
                {"role": "user", "content": (
                    f"Your previous reply failed validation:\n{err}\n"
                    "Return a corrected JSON object only, fixing every listed problem.")},
            ]
    raise StructuredOutputError(f"{schema.__name__}: no valid output after {max_retries + 1} attempts",
                                errors, raw)


def _short_error(e: Exception) -> str:
    if isinstance(e, ValidationError):
        parts = []
        for err in e.errors()[:8]:
            loc = ".".join(str(x) for x in err.get("loc", ())) or "<root>"
            parts.append(f"- {loc}: {err.get('msg')}")
        return "\n".join(parts)
    return f"- {e}"
