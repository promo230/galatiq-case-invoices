"""VP-approval stage: deterministic guardrails around a proposer/critic LLM reflection loop.

Order of operations for every invoice:
  1. Hard guardrails (no LLM call): a CRITICAL flag is an automatic REJECT; an
     unpriced invoice (no total_usd) is an automatic NEEDS_HUMAN.
  2. If those pass, branch on `Settings.llm_mode`:
     - "off" (or the LLM path raising LLMUnavailableError): a deterministic
       rules-based fallback decision.
     - otherwise: the real proposer/critic reflection loop.
"""

from __future__ import annotations

from typing import cast

from pydantic import BaseModel, Field

from apcopilot.agents import prompts
from apcopilot.config import get_settings
from apcopilot.logging import get_logger
from apcopilot.models import (
    ApprovalDecision,
    ApprovalDecisionValue,
    ExtractedInvoice,
    Severity,
    ValidationResult,
)
from apcopilot.tools.policy import get_policy, threshold

logger = get_logger(__name__)


class CritiqueVerdict(BaseModel):
    """Local schema for the critic's structured verdict on a proposer draft."""

    approved: bool
    notes: list[str] = Field(default_factory=list)


async def run_approval(
    extraction: ExtractedInvoice,
    validation: ValidationResult,
    *,
    run_id: str,
) -> ApprovalDecision:
    guardrail_decision = _hard_guardrails(validation)
    if guardrail_decision is not None:
        logger.info(
            "approval.guardrail",
            run_id=run_id,
            decision=guardrail_decision.decision.value,
            rationale=guardrail_decision.rationale,
        )
        return guardrail_decision

    settings = get_settings()
    if settings.llm_mode == "off":
        decision = _deterministic_fallback(extraction, validation)
        logger.info(
            "approval.deterministic_fallback",
            run_id=run_id,
            decision=decision.decision.value,
            reason="llm_mode_off",
        )
        return decision

    # Imported lazily (rather than at module scope) so this module can be imported
    # and its "off"/guardrail paths exercised even before apcopilot.llm exists on
    # disk, and so a build in progress on that module never breaks this one.
    from apcopilot.llm import LLMUnavailableError

    try:
        return await _reflection_loop(extraction, validation, run_id=run_id)
    except LLMUnavailableError:
        decision = _deterministic_fallback(extraction, validation)
        logger.info(
            "approval.deterministic_fallback",
            run_id=run_id,
            decision=decision.decision.value,
            reason="llm_unavailable",
        )
        return decision


def _hard_guardrails(validation: ValidationResult) -> ApprovalDecision | None:
    """Deterministic checks that must happen before any LLM call.

    Returns None if neither guardrail fires, meaning the invoice proceeds to the
    LLM-mode branch.
    """
    critical_flags = [f for f in validation.flags if f.severity == Severity.CRITICAL]
    if critical_flags:
        rule_ids = sorted({f.rule_id for f in critical_flags})
        named = "; ".join(f"{f.rule_id} ({f.code}): {f.message}" for f in critical_flags)
        return ApprovalDecision(
            decision=ApprovalDecisionValue.REJECT,
            rationale=f"Rejected without review due to CRITICAL flag(s): {named}",
            cited_policy_ids=rule_ids,
            addressed_flag_rule_ids=rule_ids,
            confidence=1.0,
            rounds=0,
            decided_by="rules",
        )

    if validation.total_usd is None:
        return ApprovalDecision(
            decision=ApprovalDecisionValue.NEEDS_HUMAN,
            rationale=(
                "Could not determine a USD-equivalent total (missing price data or no FX "
                "rate on file); routed to human review."
            ),
            confidence=1.0,
            rounds=0,
            decided_by="rules",
        )

    return None


def _deterministic_fallback(
    extraction: ExtractedInvoice, validation: ValidationResult
) -> ApprovalDecision:
    """Rules-only decision used when the LLM path is off or unavailable.

    APPROVE iff: no HIGH/CRITICAL flag, and total_usd <= vendor_auto_approve_limit
    (when known), and total_usd <= threshold("high_value"). Otherwise NEEDS_HUMAN.
    Guardrails in _hard_guardrails() already guarantee total_usd is not None here.
    """
    del extraction  # not needed for the deterministic rules, kept for signature symmetry

    total = validation.total_usd
    assert total is not None, "guardrail should have handled total_usd is None"

    reasons: list[str] = []
    ok = True

    if validation.has_blocking:
        ok = False
        reasons.append(f"blocking flag present (max severity {validation.max_severity})")
    else:
        reasons.append("no HIGH/CRITICAL flags")

    if validation.vendor_auto_approve_limit is not None:
        if total > validation.vendor_auto_approve_limit:
            ok = False
            reasons.append(
                f"total {total} exceeds vendor auto-approve limit "
                f"{validation.vendor_auto_approve_limit}"
            )
        else:
            reasons.append(
                f"total {total} within vendor auto-approve limit "
                f"{validation.vendor_auto_approve_limit}"
            )
    else:
        reasons.append("vendor auto-approve limit unknown")

    high_value = threshold("high_value")
    if total > high_value:
        ok = False
        reasons.append(f"total {total} exceeds high-value threshold {high_value}")
    else:
        reasons.append(f"total {total} within high-value threshold {high_value}")

    decision = ApprovalDecisionValue.APPROVE if ok else ApprovalDecisionValue.NEEDS_HUMAN
    return ApprovalDecision(
        decision=decision,
        rationale="Deterministic rules fallback (LLM off/unavailable): " + "; ".join(reasons),
        confidence=1.0,
        rounds=0,
        decided_by="rules",
    )


def _verify_decision(draft: ApprovalDecision, validation: ValidationResult) -> list[str]:
    """Python-side verification of a proposer draft. Never trust the critic alone for these.

    Checks: (i) every cited policy id resolves via get_policy(); (ii) every
    HIGH/CRITICAL flag's rule_id is addressed; (iii) no silent APPROVE above the
    vendor limit or the high-value threshold.
    """
    problems: list[str] = []

    for policy_id in draft.cited_policy_ids:
        if get_policy(policy_id) is None:
            problems.append(
                f"Cited policy id '{policy_id}' does not resolve to a known policy rule "
                "(hallucinated citation)."
            )

    blocking_flags = validation.flags_at_or_above(Severity.HIGH)
    addressed = set(draft.addressed_flag_rule_ids)
    for flag in blocking_flags:
        if flag.rule_id not in addressed:
            problems.append(
                f"{flag.severity} flag '{flag.rule_id}' ({flag.code}) is not listed in "
                "addressed_flag_rule_ids."
            )

    total = validation.total_usd
    if draft.decision == ApprovalDecisionValue.APPROVE and total is not None:
        high_value = threshold("high_value")
        if total > high_value:
            problems.append(
                f"Decision is APPROVE but total {total} exceeds the high-value threshold "
                f"{high_value} (POL-THRESHOLD-10K); must be NEEDS_HUMAN or REJECT."
            )
        if (
            validation.vendor_auto_approve_limit is not None
            and total > validation.vendor_auto_approve_limit
        ):
            problems.append(
                f"Decision is APPROVE but total {total} exceeds the vendor auto-approve "
                f"limit {validation.vendor_auto_approve_limit}; must be NEEDS_HUMAN or REJECT."
            )

    return problems


async def _reflection_loop(
    extraction: ExtractedInvoice,
    validation: ValidationResult,
    *,
    run_id: str,
) -> ApprovalDecision:
    from apcopilot.llm import call_structured

    settings = get_settings()
    max_rounds = int(threshold("max_approval_rounds"))
    context = prompts.build_context_block(extraction, validation)

    critique_notes: list[str] = []
    draft: ApprovalDecision | None = None

    for round_num in range(1, max_rounds + 1):
        proposer_user_msg = prompts.build_proposer_user_message(context, critique_notes)
        raw_draft, _meta = await call_structured(
            system=prompts.proposer_system_prompt(),
            user=proposer_user_msg,
            response_model=ApprovalDecision,
            model=settings.approval_model,
            node="approve",
            run_id=run_id,
            prompt_name="approval_propose",
        )
        draft = cast(ApprovalDecision, raw_draft)

        python_problems = _verify_decision(draft, validation)

        critic_user_msg = prompts.build_critic_user_message(context, draft)
        raw_verdict, _meta2 = await call_structured(
            system=prompts.critic_system_prompt(),
            user=critic_user_msg,
            response_model=CritiqueVerdict,
            model=settings.critic_model,
            node="approve",
            run_id=run_id,
            prompt_name="approval_critique",
        )
        verdict = cast(CritiqueVerdict, raw_verdict)

        round_problems = list(python_problems)
        if not verdict.approved:
            round_problems.extend(
                verdict.notes or ["Critic rejected the decision without specific notes."]
            )

        passed = not round_problems

        logger.info(
            "approval.round",
            run_id=run_id,
            round=round_num,
            decision=draft.decision.value,
            confidence=draft.confidence,
            critic_approved=verdict.approved,
            python_problems=python_problems,
            critic_passed=passed,
        )

        if passed:
            draft.rounds = round_num
            draft.critique_notes = critique_notes
            # Stamped here, not trusted from the model: ApprovalDecision.decided_by
            # defaults to "rules", and the proposer fills the schema without knowing
            # which actor it is. Provenance is the harness's call.
            draft.decided_by = "vp_agent"
            return draft

        critique_notes.extend(round_problems)

    # Cap reached without a policy-compliant decision: force NEEDS_HUMAN.
    last_rationale = draft.rationale if draft is not None else "no draft produced"
    outstanding = "; ".join(critique_notes) if critique_notes else "none recorded"
    forced = ApprovalDecision(
        decision=ApprovalDecisionValue.NEEDS_HUMAN,
        rationale=(
            f"Reflection loop could not reach a policy-compliant decision within "
            f"{max_rounds} round(s); routed to human review. Last draft rationale: "
            f"{last_rationale}. Outstanding issues: {outstanding}."
        ),
        cited_policy_ids=draft.cited_policy_ids if draft is not None else [],
        addressed_flag_rule_ids=draft.addressed_flag_rule_ids if draft is not None else [],
        confidence=draft.confidence if draft is not None else 0.0,
        rounds=max_rounds,
        decided_by="vp_agent",
        critique_notes=critique_notes,
    )
    logger.info(
        "approval.exhausted",
        run_id=run_id,
        max_rounds=max_rounds,
        outstanding=critique_notes,
    )
    return forced


__all__ = ["CritiqueVerdict", "run_approval"]
