from __future__ import annotations

import hashlib
import json
import time
from decimal import Decimal
from functools import lru_cache
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
    """Call Claude and force a structured response matching `response_model`.

    Structured output is forced via tool use: a single `emit_result` tool whose
    `input_schema` is `response_model.model_json_schema()`, with `tool_choice`
    pinned to that tool. If the tool's input fails pydantic validation, the
    validation error is appended to the user message and the call retries (up
    to `max_retries` additional attempts) so the model can self-correct.

    Respects `Settings.llm_mode`:
      - "off": raises LLMUnavailableError immediately, no API call.
      - "live"/"record": raises LLMUnavailableError if no API key is resolved.
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

    api_key = settings.resolved_api_key()
    if mode in ("live", "record") and api_key is None:
        raise LLMUnavailableError(
            "no Anthropic API key configured; treating as unavailable"
        )

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

    assert api_key is not None  # narrowed above for live/record
    client = _get_client(api_key)
    tool = {
        "name": _TOOL_NAME,
        "description": (
            f"Emit the final result as structured data matching the "
            f"{response_model.__name__} schema. Always call this tool exactly once."
        ),
        "input_schema": response_model.model_json_schema(),
    }

    current_user = user
    total_attempts = max_retries + 1
    last_validation_error: ValidationError | None = None

    for attempt in range(1, total_attempts + 1):
        logger.debug("llm_call.start", model=model, node=node, attempt=attempt, mode=mode)
        start = time.monotonic()
        response = await client.messages.create(
            model=model,
            max_tokens=_DEFAULT_MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": current_user}],
            tools=[tool],
            tool_choice={"type": "tool", "name": _TOOL_NAME},
        )
        latency_ms = int((time.monotonic() - start) * 1000)

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost_usd = compute_cost_usd(model, input_tokens, output_tokens)
        stop_reason = response.stop_reason

        tool_use_block = next(
            (block for block in response.content if block.type == "tool_use"), None
        )
        tool_input: dict[str, Any] = dict(tool_use_block.input) if tool_use_block else {}

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
