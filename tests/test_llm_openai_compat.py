"""The OpenAI-compatible backend (`llm/openai_compat.py`) and its dispatch from
`call_structured`, fully offline.

HTTP behaviour is tested through `httpx.MockTransport` — no socket is ever
opened. Dispatch tests call the real `call_structured` (the module-level import
below binds it before the autouse `no_llm_calls` tripwire patches the module
attribute, so the tripwire keeps guarding the pipeline tests untouched) with
`get_settings` monkeypatched to a copied `Settings`, and stub out either
`call_openai_compat` or `log_llm_call` where the test wants no side effects.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

import apcopilot.llm.client as llm_client
import apcopilot.llm.openai_compat as openai_compat
from apcopilot.config import get_settings
from apcopilot.llm.client import LLMUnavailableError, call_structured
from apcopilot.llm.openai_compat import call_openai_compat

BASE_URL = "https://example.test/v1"


class Fruit(BaseModel):
    name: str
    count: int


def _chat_response(
    *,
    arguments: str | None = None,
    content: str | None = None,
    finish_reason: str = "tool_calls",
    prompt_tokens: int = 11,
    completion_tokens: int = 7,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if arguments is not None:
        message["tool_calls"] = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "emit_result", "arguments": arguments},
            }
        ]
    return {
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


def _call(handler) -> tuple[dict[str, Any], int, int, str | None]:
    return asyncio.run(
        call_openai_compat(
            base_url=BASE_URL,
            api_key="test-key",
            model="gemini-2.5-flash-lite",
            system="sys",
            user="usr",
            response_model=Fruit,
            transport=httpx.MockTransport(handler),
        )
    )


def test_tool_call_happy_path() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_chat_response(arguments='{"name": "apple", "count": 3}'))

    tool_input, input_tokens, output_tokens, stop_reason = _call(handler)

    assert tool_input == {"name": "apple", "count": 3}
    assert (input_tokens, output_tokens) == (11, 7)
    assert stop_reason == "tool_calls"

    body = json.loads(requests[0].content)
    assert requests[0].url.path == "/v1/chat/completions"
    assert requests[0].headers["authorization"] == "Bearer test-key"
    assert body["tool_choice"] == {"type": "function", "function": {"name": "emit_result"}}
    assert body["tools"][0]["function"]["parameters"] == Fruit.model_json_schema()
    assert body["messages"][0] == {"role": "system", "content": "sys"}


def test_content_json_fallback_when_no_tool_call() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=_chat_response(content='{"name": "pear", "count": 1}', finish_reason="stop")
        )

    tool_input, _, _, stop_reason = _call(handler)
    assert tool_input == {"name": "pear", "count": 1}
    assert stop_reason == "stop"


def test_content_fallback_strips_markdown_fence() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        fenced = '```json\n{"name": "plum", "count": 2}\n```'
        return httpx.Response(200, json=_chat_response(content=fenced, finish_reason="stop"))

    tool_input, _, _, _ = _call(handler)
    assert tool_input == {"name": "plum", "count": 2}


def test_unparseable_response_degrades_to_empty_dict() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=_chat_response(arguments="not json", content="also not json")
        )

    tool_input, _, _, _ = _call(handler)
    assert tool_input == {}


def test_429_sleeps_per_retry_after_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(openai_compat.asyncio, "sleep", fake_sleep)

    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "2"})
        return httpx.Response(200, json=_chat_response(arguments='{"name": "apple", "count": 3}'))

    tool_input, _, _, _ = _call(handler)
    assert tool_input == {"name": "apple", "count": 3}
    assert calls == 2
    assert sleeps == [2.0]


def test_429_twice_gives_unavailable_and_caps_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(openai_compat.asyncio, "sleep", fake_sleep)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "120"})

    with pytest.raises(LLMUnavailableError, match="429"):
        _call(handler)
    assert sleeps == [30.0]  # Retry-After capped


def test_transport_error_raises_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(LLMUnavailableError, match="unreachable"):
        _call(handler)


# --- dispatch through call_structured -----------------------------------------


def _patched_settings(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> None:
    settings = get_settings().model_copy(update=overrides)
    monkeypatch.setattr(llm_client, "get_settings", lambda: settings)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


def _run_call_structured(**kwargs: Any) -> tuple[BaseModel, llm_client.LLMCallMeta]:
    return asyncio.run(
        call_structured(
            system="sys", user="usr", response_model=Fruit, model="test-model", node="test",
            **kwargs,
        )
    )


def test_off_mode_raises_before_any_http(monkeypatch: pytest.MonkeyPatch) -> None:
    _patched_settings(
        monkeypatch,
        llm_mode="off",
        llm_provider="openai_compat",
        openai_base_url=BASE_URL,
        openai_api_key="test-key",
    )

    async def explode(**kwargs: Any) -> Any:
        raise AssertionError("an HTTP call was attempted while llm_mode='off'")

    monkeypatch.setattr(openai_compat, "call_openai_compat", explode)
    with pytest.raises(LLMUnavailableError, match="off"):
        _run_call_structured()


def test_missing_base_url_raises_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    _patched_settings(
        monkeypatch,
        llm_mode="live",
        llm_provider="openai_compat",
        openai_base_url=None,
        openai_api_key="test-key",
    )
    with pytest.raises(LLMUnavailableError, match="base URL"):
        _run_call_structured()


def test_missing_key_raises_unavailable_for_remote_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patched_settings(
        monkeypatch,
        llm_mode="live",
        llm_provider="openai_compat",
        openai_base_url="https://api.x.ai/v1",
        openai_api_key=None,
    )
    with pytest.raises(LLMUnavailableError, match="key"):
        _run_call_structured()


def test_localhost_endpoint_needs_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _patched_settings(
        monkeypatch,
        llm_mode="live",
        llm_provider="openai_compat",
        openai_base_url="http://localhost:11434/v1",
        openai_api_key=None,
    )
    monkeypatch.setattr(llm_client, "log_llm_call", lambda **kwargs: None)

    captured: dict[str, Any] = {}

    async def fake_call(**kwargs: Any) -> tuple[dict[str, Any], int, int, str | None]:
        captured.update(kwargs)
        return {"name": "kiwi", "count": 4}, 5, 3, "tool_calls"

    monkeypatch.setattr(openai_compat, "call_openai_compat", fake_call)

    parsed, meta = _run_call_structured()
    assert parsed == Fruit(name="kiwi", count=4)
    assert captured["api_key"] is None
    assert captured["base_url"] == "http://localhost:11434/v1"
    assert (meta.input_tokens, meta.output_tokens) == (5, 3)


def test_validation_error_retry_appends_error_and_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patched_settings(
        monkeypatch,
        llm_mode="live",
        llm_provider="openai_compat",
        openai_base_url="http://localhost:11434/v1",
        openai_api_key=None,
    )
    logged: list[dict[str, Any]] = []
    monkeypatch.setattr(llm_client, "log_llm_call", lambda **kwargs: logged.append(kwargs))

    users: list[str] = []
    responses: Iterator[dict[str, Any]] = iter([{"name": "apple"}, {"name": "apple", "count": 3}])

    async def fake_call(**kwargs: Any) -> tuple[dict[str, Any], int, int, str | None]:
        users.append(kwargs["user"])
        return next(responses), 10, 5, "tool_calls"

    monkeypatch.setattr(openai_compat, "call_openai_compat", fake_call)

    parsed, meta = _run_call_structured(max_retries=1)
    assert parsed == Fruit(name="apple", count=3)
    assert meta.attempt == 2
    assert users[0] == "usr"
    assert "failed validation" in users[1]
    assert "count" in users[1]  # the pydantic error names the missing field
    assert [entry["attempt"] for entry in logged] == [1, 2]
