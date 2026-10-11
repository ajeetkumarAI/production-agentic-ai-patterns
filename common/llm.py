"""LLM client abstraction used by every project.

* ``OpenAILLM``  - real calls to OpenAI Chat Completions (default model gpt-5-nano).
* ``FakeLLM``    - scripted responses for deterministic unit tests.
* ``MockLLM``    - offline mode: produces schema-valid JSON from the requested
                   ``response_format`` so every demo runs without an API key.

All clients record ``Usage`` (tokens, cost, latency) and notify global listeners,
which is how tracing (Project 11) and cost analytics (Project 7) hook in.
"""
from __future__ import annotations

import json
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from . import config

Message = dict[str, Any]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string as produced by the model

    def args(self) -> dict:
        try:
            return json.loads(self.arguments or "{}")
        except json.JSONDecodeError:
            return {}


@dataclass
class Usage:
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    tag: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=lambda: Usage(model="unknown"))
    finish_reason: str = "stop"

    def to_message(self) -> Message:
        """Assistant message to append back into the conversation."""
        msg: Message = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            msg["tool_calls"] = [
                {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
                for tc in self.tool_calls
            ]
        return msg


# --------------------------------------------------------------------------- listeners
_listeners: list[Callable[[Usage, list[Message], LLMResponse], None]] = []


def add_listener(fn: Callable[[Usage, list[Message], LLMResponse], None]) -> None:
    _listeners.append(fn)


def remove_listener(fn) -> None:
    if fn in _listeners:
        _listeners.remove(fn)


def estimate_tokens(text_or_messages: Any) -> int:
    """Cheap token estimate (~4 chars/token). Good enough for budgeting."""
    if isinstance(text_or_messages, str):
        text = text_or_messages
    else:
        text = json.dumps(text_or_messages, default=str)
    return max(1, len(text) // 4)


class BaseLLM:
    model: str = config.DEFAULT_MODEL

    def __init__(self) -> None:
        self.history: list[Usage] = []

    # subclasses implement _chat
    def _chat(self, messages: list[Message], **kw) -> LLMResponse:  # pragma: no cover
        raise NotImplementedError

    def chat(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        tools: list[dict] | None = None,
        tool_choice: Any = None,
        response_format: dict | None = None,
        reasoning_effort: str | None = None,
        max_completion_tokens: int | None = None,
        tag: str = "",
    ) -> LLMResponse:
        start = time.perf_counter()
        resp = self._chat(
            messages,
            model=model or self.model,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            reasoning_effort=reasoning_effort,
            max_completion_tokens=max_completion_tokens,
        )
        resp.usage.latency_ms = resp.usage.latency_ms or (time.perf_counter() - start) * 1000
        resp.usage.tag = tag
        if not resp.usage.cost_usd:
            resp.usage.cost_usd = config.cost_usd(
                resp.usage.model, resp.usage.prompt_tokens, resp.usage.completion_tokens, resp.usage.cached_tokens
            )
        self.history.append(resp.usage)
        for fn in list(_listeners):
            try:
                fn(resp.usage, messages, resp)
            except Exception:  # listeners must never break the agent
                pass
        return resp

    # convenience
    def text(self, prompt: str, system: str | None = None, **kw) -> str:
        msgs: list[Message] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return self.chat(msgs, **kw).content or ""

    @property
    def total_cost(self) -> float:
        return sum(u.cost_usd for u in self.history)

    @property
    def total_tokens(self) -> int:
        return sum(u.total_tokens for u in self.history)


# --------------------------------------------------------------------------- OpenAI
class OpenAILLM(BaseLLM):
    RETRYABLE = ("RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError")

    def __init__(self, model: str | None = None, reasoning_effort: str | None = None, max_retries: int = 4,
                 timeout: float = 120.0) -> None:
        super().__init__()
        from openai import OpenAI  # imported lazily so tests don't need it

        self.client = OpenAI(api_key=config.OPENAI_API_KEY, timeout=timeout, max_retries=0)
        self.model = model or config.DEFAULT_MODEL
        self.reasoning_effort = reasoning_effort or config.DEFAULT_REASONING_EFFORT
        self.max_retries = max_retries

    def _chat(self, messages, *, model, tools, tool_choice, response_format, reasoning_effort,
              max_completion_tokens) -> LLMResponse:
        kwargs: dict[str, Any] = {"model": model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice
        if response_format:
            kwargs["response_format"] = response_format
        effort = reasoning_effort or self.reasoning_effort
        if effort and model.startswith(("gpt-5", "o")):
            kwargs["reasoning_effort"] = effort
        if max_completion_tokens:
            kwargs["max_completion_tokens"] = max_completion_tokens

        delay = 1.0
        for attempt in range(self.max_retries + 1):
            try:
                start = time.perf_counter()
                r = self.client.chat.completions.create(**kwargs)
                latency = (time.perf_counter() - start) * 1000
                break
            except Exception as e:  # noqa: BLE001
                if type(e).__name__ in self.RETRYABLE and attempt < self.max_retries:
                    time.sleep(delay + random.random() * 0.5)
                    delay *= 2
                    continue
                raise
        choice = r.choices[0]
        msg = choice.message
        u = r.usage
        details = getattr(u, "completion_tokens_details", None)
        pdetails = getattr(u, "prompt_tokens_details", None)
        usage = Usage(
            model=model,
            prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(u, "completion_tokens", 0) or 0,
            reasoning_tokens=(getattr(details, "reasoning_tokens", 0) or 0) if details else 0,
            cached_tokens=(getattr(pdetails, "cached_tokens", 0) or 0) if pdetails else 0,
            latency_ms=latency,
        )
        calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments or "{}")
            for tc in (msg.tool_calls or [])
            if getattr(tc, "function", None)
        ]
        return LLMResponse(content=msg.content, tool_calls=calls, usage=usage,
                           finish_reason=choice.finish_reason or "stop")


# --------------------------------------------------------------------------- Fake (tests)
Script = str | LLMResponse | Callable[[list[Message], dict], "str | LLMResponse"]


class FakeLLM(BaseLLM):
    """Returns scripted responses in order. Once exhausted, repeats the last one
    (or uses ``default``). Records every call in ``calls`` for assertions."""

    def __init__(self, responses: Iterable[Script] = (), default: Script | None = None, model: str = "gpt-5-nano"):
        super().__init__()
        self.responses = list(responses)
        self.default = default
        self.model = model
        self.calls: list[dict] = []

    def _chat(self, messages, **kw) -> LLMResponse:
        self.calls.append({"messages": [dict(m) for m in messages], **kw})
        if self.responses:
            item = self.responses.pop(0)
        elif self.default is not None:
            item = self.default
        else:
            raise RuntimeError("FakeLLM ran out of scripted responses")
        if callable(item) and not isinstance(item, LLMResponse):
            item = item(messages, kw)
        if isinstance(item, LLMResponse):
            item.usage.model = kw.get("model") or self.model
            return item
        text = str(item)
        return LLMResponse(
            content=text,
            usage=Usage(model=kw.get("model") or self.model, prompt_tokens=estimate_tokens(messages),
                        completion_tokens=estimate_tokens(text)),
        )


def tool_call_response(name: str, arguments: dict, content: str = "") -> LLMResponse:
    """Helper for tests: an assistant turn that calls one tool."""
    return LLMResponse(content=content, tool_calls=[ToolCall(id=f"call_{uuid.uuid4().hex[:8]}", name=name,
                                                             arguments=json.dumps(arguments))])


# --------------------------------------------------------------------------- Mock (offline demos)
class MockLLM(BaseLLM):
    """Offline stand-in. When a JSON schema is requested it returns a minimal
    schema-valid object; otherwise an echo. Lets every demo run without a key
    so you can see the plumbing (retries, routing, logs) before spending tokens."""

    model = "mock"

    def _chat(self, messages, *, model, response_format=None, tools=None, **kw) -> LLMResponse:
        if response_format and response_format.get("type") == "json_schema":
            schema = response_format["json_schema"]["schema"]
            content = json.dumps(_example_from_schema(schema, schema.get("$defs", {})))
        elif response_format and response_format.get("type") == "json_object":
            content = "{}"
        else:
            last = next((m.get("content") for m in reversed(messages) if m.get("role") == "user"), "")
            content = f"[mock answer] {str(last)[:160]}"
        return LLMResponse(content=content, usage=Usage(model="mock", prompt_tokens=estimate_tokens(messages),
                                                        completion_tokens=estimate_tokens(content)))


def _example_from_schema(s: dict, defs: dict, depth: int = 0) -> Any:
    if depth > 8:
        return None
    if "$ref" in s:
        return _example_from_schema(defs.get(s["$ref"].split("/")[-1], {}), defs, depth + 1)
    for key in ("anyOf", "oneOf", "allOf"):
        if key in s:
            options = [o for o in s[key] if o.get("type") != "null"] or s[key]
            return _example_from_schema(options[0], defs, depth + 1)
    if "const" in s:
        return s["const"]
    if "enum" in s:
        return s["enum"][0]
    if "default" in s and s["default"] is not None:
        return s["default"]
    t = s.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), "string")
    if t == "object" or "properties" in s:
        props = s.get("properties", {})
        return {k: _example_from_schema(v, defs, depth + 1) for k, v in props.items()}
    if t == "array":
        n = s.get("minItems", 0)
        return [_example_from_schema(s.get("items", {"type": "string"}), defs, depth + 1) for _ in range(n)]
    if t in ("number", "integer"):
        lo = s.get("minimum", s.get("exclusiveMinimum", 0))
        hi = s.get("maximum", s.get("exclusiveMaximum", 1 if t == "number" else 10))
        val = lo + (hi - lo) * 0.2
        return int(val) if t == "integer" else round(val, 2)
    if t == "boolean":
        return True
    min_len = s.get("minLength", 0)
    text = "mock placeholder text"
    while len(text) < min_len:
        text += " mock"
    return text


# --------------------------------------------------------------------------- factory
def get_llm(model: str | None = None, reasoning_effort: str | None = None) -> BaseLLM:
    if config.use_real_llm():
        return OpenAILLM(model=model, reasoning_effort=reasoning_effort)
    print("[llm] No OPENAI_API_KEY found (or LLM_MODE=mock) -> using offline MockLLM. "
          "Add your key to .env for real gpt-5-nano calls.")
    return MockLLM()
