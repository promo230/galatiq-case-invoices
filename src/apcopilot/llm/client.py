from __future__ import annotations

import hashlib
import json
import time
from decimal import Decimal
from functools import lru_cache, partial
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic
from pydantic import BaseModel, ValidationError

from apcopilot.config import get_settings
from apcopilot.db.runs import log_llm_call
from apcopilot.llm.pricing import compute_cost_usd
from apcopilot.logging import get_logger

logger = get_logger(__name__)

_TOOL_NAME = "emit_result"
_DEFAULT_MAX_TOKENS = 8192


class LLMCallMeta(BaseModel):
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: Decimal
    latency_ms: int
    stop_reason: str | None
    attempt: int
    prompt_name: str | None = None


class LLMUnavailableError(Exception):
    """Raised when the LLM cannot be called at all: mode is 'off', no API key
    is configured for a 'live'/'record' call, or (in 'replay' mode) no fixture
    matches the request. Callers are expected to catch this and fall back to a
    deterministic path rather than treat it as a hard failure."""


@lru_cache(maxsize=4)
def _get_client(api_key: str) -> AsyncAnthropic:
    return AsyncAnthropic(api_key=api_key)


def _sha256(*parts: str) -> str:
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _fixture_path(fixture_dir: Path, fixture_hash: str) -> Path:
    return fixture_dir / f"{fixture_hash}.json"


def _load_fixture(fixture_dir: Path, fixture_hash: str) -> dict[str, Any] | None:
    path = _fixture_path(fixture_dir, fixture_hash)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_fixture(
    fixture_dir: Path,
    fixture_hash: str,
    *,
    response: dict[str, Any],
    input_tokens: int,
    output_tokens: int,
) -> None:
    fixture_dir.mkdir(parents=True, exist_ok=True)
    path = _fixture_path(fixture_dir, fixture_hash)
    path.write_text(
        json.dumps(
            {"response": response, "input_tokens": input_tokens, "output_tokens": output_tokens},
            default=str,
            indent=2,
        ),
        encoding="utf-8",
    )


async def _call_anthropic(
    *,
    api_key: str,
    model: str,
    system: str,
    user: str,
    response_model: type[BaseModel],
) -> tuple[dict[str, Any], int, int, str | None]:
    """One structured call against the Anthropic Messages API.

    Returns (tool_input, input_tokens, output_tokens, stop_reason). `tool_input`
    is {} when the response carries no tool_use block; the caller's schema
    validation then fails and drives the self-correction retry.
    """
    client = _get_client(api_key)
    tool = {
        "name": _TOOL_NAME,
        "description": (
            f"Emit the final result as structured data matching the "
            f"{response_model.__name__} schema. Always call this tool exactly once."
        ),
        "input_schema": response_model.model_json_schema(),
    }
    response = await client.messages.create(
        model=model,
        max_tokens=_DEFAULT_MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user}],
        tools=[tool],
        tool_choice={"type": "tool", "name": _TOOL_NAME},
    )
    tool_use_block = next(
        (block for block in response.content if block.type == "tool_use"), None
    )
    tool_input: dict[str, Any] = dict(tool_use_block.input) if tool_use_block else {}
    return (
        tool_input,
        response.usage.input_tokens,
        response.usage.output_tokens,
        response.stop_reason,
    )


async def call_structured(
    *,
    system: str,
    user: str,
    response_model: type[BaseModel],
    model: str,
    node: str,
    run_id: str | None = None,
    prompt_name: str | None = None,
    max_retries: int = 1,
) -> tuple[BaseModel, LLMCallMeta]:
    """Call the configured LLM provider and force a structured response matching
    `response_model`.

    Structured output is forced via a single `emit_result` tool whose schema is
    `response_model.model_json_schema()`, with tool choice pinned to that tool:
    Anthropic tool use by default, or OpenAI-style function calling when
    `Settings.llm_provider` is "openai_compat" (Gemini / xAI / Ollama, selected
    by `Settings.openai_base_url`). If the tool's input fails pydantic
    validation, the validation error is appended to the user message and the
    call retries (up to `max_retries` additional attempts) so the model can
    self-correct.

    Respects `Settings.llm_mode`:
      - "off": raises LLMUnavailableError immediately, no API call.
      - "live"/"record": raises LLMUnavailableError if no API key (or, for
        openai_compat, no base URL) is resolved. A key is optional for
        localhost endpoints such as Ollama.
      - "record": on success, writes a fixture under `settings.fixture_dir`.
      - "replay": looks up a fixture by hash of (model, system, user); raises
        LLMUnavailableError if none exists. Never calls the network.

    Every attempt (success or a validation failure that triggers a retry) is
    logged via `apcopilot.db.runs.log_llm_call`.
    """
    settings = get_settings()
    mode = settings.llm_mode
    prompt_hash = _sha256(system, user)

    if mode == "off":
        raise LLMUnavailableError("llm_mode is 'off'; no LLM calls are permitted")

    if mode == "replay":
        return _replay(
            settings.fixture_dir,
            system=system,
            user=user,
            model=model,
            node=node,
            run_id=run_id,
            prompt_name=prompt_name,
            prompt_hash=prompt_hash,
            response_model=response_model,
        )

    # live/record: resolve provider credentials and bind the one-call helper.
    # Everything below the dispatch (retries, logging, cost, fixtures) is shared.
    if settings.llm_provider == "openai_compat":
        # Imported here rather than at module top: openai_compat imports
        # LLMUnavailableError from this module, so a top-level import would be
        # circular.
        from apcopilot.llm.openai_compat import call_openai_compat, is_localhost

        base_url = settings.openai_base_url
        if not base_url:
            raise LLMUnavailableError(
                "llm_provider is 'openai_compat' but no base URL is configured; "
                "treating as unavailable"
            )
        openai_key = settings.resolved_openai_key()
        if openai_key is None and not is_localhost(base_url):
            raise LLMUnavailableError(
                "no API key configured for the openai-compatible endpoint; "
                "treating as unavailable"
            )
        call_once = partial(
            call_openai_compat,
            base_url=base_url,
            api_key=openai_key,
            model=model,
            system=system,
            response_model=response_model,
        )
    else:
        api_key = settings.resolved_api_key()
        if api_key is None:
            raise LLMUnavailableError(
                "no Anthropic API key configured; treating as unavailable"
            )
        call_once = partial(
            _call_anthropic,
            api_key=api_key,
            model=model,
            system=system,
            response_model=response_model,
        )

    current_user = user
    total_attempts = max_retries + 1
    last_validation_error: ValidationError | None = None

    for attempt in range(1, total_attempts + 1):
        logger.debug("llm_call.start", model=model, node=node, attempt=attempt, mode=mode)
        start = time.monotonic()
        tool_input, input_tokens, output_tokens, stop_reason = await call_once(
            user=current_user
        )
        latency_ms = int((time.monotonic() - start) * 1000)
        cost_usd = compute_cost_usd(model, input_tokens, output_tokens)

        log_llm_call(
            run_id=run_id,
            node=node,
            model=model,
            prompt_name=prompt_name,
            prompt_hash=prompt_hash,
            attempt=attempt,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            stop_reason=stop_reason,
            request_preview=current_user[:500],
            response_preview=str(tool_input)[:500],
        )

        try:
            parsed = response_model(**tool_input)
        except ValidationError as exc:
            last_validation_error = exc
            logger.debug(
                "llm_call.validation_failed",
                model=model,
                node=node,
                attempt=attempt,
                mode=mode,
            )
            if attempt < total_attempts:
                current_user = (
                    f"{user}\n\n"
                    f"Your previous response (tool input: {tool_input}) failed validation "
                    f"against the required schema with this error:\n{exc}\n\n"
                    f"Please correct the issue and call the {_TOOL_NAME} tool again with "
                    f"input that satisfies the schema."
                )
                continue
            raise
        else:
            logger.info(
                "llm_call.success",
                model=model,
                node=node,
                attempt=attempt,
                mode=mode,
                stop_reason=stop_reason,
            )
            meta = LLMCallMeta(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
                stop_reason=stop_reason,
                attempt=attempt,
                prompt_name=prompt_name,
            )
            if mode == "record":
                fixture_hash = _sha256(model, system, user)
                _write_fixture(
                    settings.fixture_dir,
                    fixture_hash,
                    response=tool_input,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
            return parsed, meta

    # Unreachable: the loop above always either returns or raises on the last attempt.
    if last_validation_error is not None:
        raise last_validation_error
    raise LLMUnavailableError("LLM call failed with no response")  # pragma: no cover


def _replay(
    fixture_dir: Path,
    *,
    system: str,
    user: str,
    model: str,
    node: str,
    run_id: str | None,
    prompt_name: str | None,
    prompt_hash: str,
    response_model: type[BaseModel],
) -> tuple[BaseModel, LLMCallMeta]:
    fixture_hash = _sha256(model, system, user)
    data = _load_fixture(fixture_dir, fixture_hash)
    if data is None:
        raise LLMUnavailableError(
            f"replay mode: no fixture found for hash {fixture_hash} "
            f"(model={model!r}, node={node!r}); replay is offline-only and never "
            f"falls back to a live call"
        )

    input_tokens = int(data.get("input_tokens", 0))
    output_tokens = int(data.get("output_tokens", 0))
    cost_usd = compute_cost_usd(model, input_tokens, output_tokens)

    parsed = response_model(**data["response"])

    logger.debug("llm_call.replay_hit", model=model, node=node, attempt=1, mode="replay")
    log_llm_call(
        run_id=run_id,
        node=node,
        model=model,
        prompt_name=prompt_name,
        prompt_hash=prompt_hash,
        attempt=1,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        latency_ms=0,
        stop_reason="replay",
        request_preview=user[:500],
        response_preview=str(data["response"])[:500],
    )

    meta = LLMCallMeta(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        latency_ms=0,
        stop_reason="replay",
        attempt=1,
        prompt_name=prompt_name,
    )
    return parsed, meta


__all__ = ["LLMCallMeta", "LLMUnavailableError", "call_structured"]
