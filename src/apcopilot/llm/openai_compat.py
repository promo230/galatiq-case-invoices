"""OpenAI-compatible chat-completions backend for `call_structured`.

Speaks the `/chat/completions` dialect shared by Google Gemini's OpenAI
endpoint, xAI, and local Ollama, so switching to any of them is configuration
(`APCOPILOT_LLM_PROVIDER=openai_compat` plus a base URL), not new code. Uses
httpx directly — already a transitive dependency of the anthropic SDK — because
one endpoint does not justify depending on the `openai` package.

Structured output is forced the same way as the Anthropic path: a single
`emit_result` function tool carrying the response model's JSON schema, pinned
via `tool_choice`. The validation/self-correction retry loop lives in
`client.call_structured`; this module makes exactly one API call per attempt
(plus a single internal retry on HTTP 429, since free tiers rate-limit).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
from pydantic import BaseModel

from apcopilot.llm.client import LLMUnavailableError
from apcopilot.logging import get_logger

logger = get_logger(__name__)

_TOOL_NAME = "emit_result"
_TIMEOUT = httpx.Timeout(120.0, connect=10.0)  # generous read: free tiers can be slow
_MAX_RETRY_AFTER_S = 30.0
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})


def is_localhost(base_url: str) -> bool:
    """True for endpoints that plausibly need no API key (e.g. local Ollama)."""
    try:
        return httpx.URL(base_url).host in _LOCAL_HOSTS
    except httpx.InvalidURL:
        return False


def _retry_after_seconds(response: httpx.Response) -> float:
    """Seconds to sleep before the single 429 retry, capped at `_MAX_RETRY_AFTER_S`."""
    raw = response.headers.get("retry-after")
    try:
        seconds = float(raw) if raw is not None else 1.0
    except ValueError:
        seconds = 1.0
    return min(max(seconds, 0.0), _MAX_RETRY_AFTER_S)


def _parse_json_object(text: str) -> dict[str, Any] | None:
    """`text` parsed as a JSON object, tolerating a markdown code fence; else None."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        cleaned = cleaned.removeprefix("json").strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_tool_input(message: dict[str, Any]) -> dict[str, Any]:
    """The forced tool call's arguments, or `content` parsed as JSON as a fallback.

    Some OpenAI-compatible servers (Ollama especially; Gemini is usually fine)
    ignore a forced tool_choice and emit the JSON as plain assistant content, so
    both shapes are accepted. Anything unparseable degrades to {}, which fails
    schema validation upstream and triggers the shared self-correction retry —
    the same convention the Anthropic path uses when no tool_use block returns.
    """
    for call in message.get("tool_calls") or []:
        arguments = call.get("function", {}).get("arguments")
        if isinstance(arguments, str):
            parsed = _parse_json_object(arguments)
            if parsed is not None:
                return parsed
    content = message.get("content")
    if isinstance(content, str):
        parsed = _parse_json_object(content)
        if parsed is not None:
            return parsed
    return {}


async def call_openai_compat(
    *,
    base_url: str,
    api_key: str | None,
    model: str,
    system: str,
    user: str,
    response_model: type[BaseModel],
    transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[dict[str, Any], int, int, str | None]:
    """One structured call against `POST {base_url}/chat/completions`.

    Returns (tool_input, input_tokens, output_tokens, stop_reason) — the same
    contract as the Anthropic helper in client.py, with OpenAI-style
    prompt/completion token counts mapped onto input/output. Transport failures
    and HTTP errors (including a 429 that persists past one Retry-After sleep)
    raise LLMUnavailableError so callers degrade to the deterministic path.
    `transport` exists for tests (httpx.MockTransport); production leaves it None.
    """
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": _TOOL_NAME,
                    "description": (
                        f"Emit the final result as structured data matching the "
                        f"{response_model.__name__} schema. Always call this function "
                        f"exactly once."
                    ),
                    "parameters": response_model.model_json_schema(),
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": _TOOL_NAME}},
    }

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, transport=transport) as client:
            response = await client.post(url, json=payload, headers=headers)
            if response.status_code == 429:
                delay = _retry_after_seconds(response)
                logger.warning("llm_call.rate_limited", model=model, retry_after_s=delay)
                await asyncio.sleep(delay)
                response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "llm_call.http_error", model=model, url=url, status=exc.response.status_code
        )
        raise LLMUnavailableError(
            f"openai-compatible endpoint returned HTTP {exc.response.status_code}; "
            f"treating as unavailable"
        ) from exc
    except httpx.HTTPError as exc:
        logger.warning("llm_call.transport_error", model=model, url=url, error=str(exc))
        raise LLMUnavailableError(
            f"openai-compatible endpoint unreachable ({exc.__class__.__name__}); "
            f"treating as unavailable"
        ) from exc

    try:
        data = response.json()
    except json.JSONDecodeError as exc:
        raise LLMUnavailableError(
            "openai-compatible endpoint returned a non-JSON body; treating as unavailable"
        ) from exc

    usage = data.get("usage") or {}
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return (
        _extract_tool_input(message),
        int(usage.get("prompt_tokens") or 0),
        int(usage.get("completion_tokens") or 0),
        choice.get("finish_reason"),
    )


__all__ = ["call_openai_compat", "is_localhost"]
