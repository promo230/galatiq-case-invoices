from __future__ import annotations

from decimal import Decimal

from apcopilot.models import ExtractedInvoice, Severity, ValidationFlag
from apcopilot.tools.policy import tolerance

RULE_ID = "POL-MATH-01"


def check_math(extraction: ExtractedInvoice) -> list[ValidationFlag]:
    """POL-MATH-01: subtotal + tax must equal total within total_residual_abs,
    and (when computable) the line items must sum to the stated subtotal
    within money_abs/money_rel. Only runs when the needed fields are present so
    it never double-flags on top of a POL-DATA-01 missing-total flag."""
    flags: list[ValidationFlag] = []

    if (
        extraction.subtotal is not None
        and extraction.tax is not None
        and extraction.total is not None
    ):
        expected = extraction.subtotal + extraction.tax
        residual = extraction.total - expected
        if abs(residual) > tolerance("total_residual_abs"):
            flags.append(
                ValidationFlag(
                    rule_id=RULE_ID,
                    code="MATH_MISMATCH",
                    severity=Severity.HIGH,
                    message=(
                        f"Total ({extraction.total}) does not equal subtotal + tax "
                        f"({extraction.subtotal} + {extraction.tax} = {expected}); "
                        f"residual {residual}."
                    ),
                    evidence={
                        "subtotal": str(extraction.subtotal),
                        "tax": str(extraction.tax),
                        "total": str(extraction.total),
                        "residual": str(residual),
                    },
                )
            )

    if extraction.subtotal is not None and extraction.line_items:
        computed = Decimal(0)
        computable = True
        for item in extraction.line_items:
            if item.line_total is not None:
                computed += item.line_total
            elif item.unit_price is not None:
                computed += item.unit_price * item.quantity
            else:
                computable = False
                break
        if computable:
            residual = computed - extraction.subtotal
            allowed = max(tolerance("money_abs"), abs(extraction.subtotal) * tolerance("money_rel"))
            if abs(residual) > allowed:
                flags.append(
                    ValidationFlag(
                        rule_id=RULE_ID,
                        code="MATH_MISMATCH",
                        severity=Severity.HIGH,
                        message=(
                            f"Sum of line items ({computed}) does not match the stated "
                            f"subtotal ({extraction.subtotal}); residual {residual}."
                        ),
                        evidence={
                            "line_items_sum": str(computed),
                            "subtotal": str(extraction.subtotal),
                            "residual": str(residual),
                        },
                    )
                )

    return flags


__all__ = ["check_math"]
