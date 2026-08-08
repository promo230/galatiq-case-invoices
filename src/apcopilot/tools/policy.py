from __future__ import annotations

from decimal import Decimal
from functools import lru_cache
from typing import Any

import yaml

from apcopilot.config import get_settings


@lru_cache(maxsize=1)
def load_policies() -> dict[str, Any]:
    return yaml.safe_load(get_settings().policy_path.read_text(encoding="utf-8"))


def threshold(name: str) -> Decimal:
    return Decimal(str(load_policies()["thresholds"][name]))


def tolerance(name: str) -> Decimal:
    return Decimal(str(load_policies()["tolerances"][name]))


def fraud_config() -> dict[str, Any]:
    return load_policies()["fraud"]


def business_case() -> dict[str, Any]:
    return load_policies()["business_case"]


def get_policy(rule_id: str) -> dict[str, str] | None:
    """Resolve a policy citation. Returns None for an invented rule id, which is
    what lets verify_decision() catch a hallucinated citation."""
    rule = load_policies()["policy_rules"].get(rule_id)
    if rule is None:
        return None
    return {"rule_id": rule_id, "title": rule["title"], "text": rule["text"].strip()}


def all_policy_ids() -> list[str]:
    return sorted(load_policies()["policy_rules"])


def policy_digest() -> str:
    """Compact rendering of the full policy set for LLM system prompts."""
    lines = []
    for rule_id in all_policy_ids():
        rule = load_policies()["policy_rules"][rule_id]
        lines.append(f"[{rule_id}] {rule['title']}: {rule['text'].strip()}")
    return "\n".join(lines)
