from __future__ import annotations

from decimal import Decimal

# NOTE: approximate/placeholder pricing (USD per million tokens), based on
# published Anthropic list pricing as of early 2026. Not guaranteed current —
# verify against https://platform.claude.com/docs/en/pricing before relying on
# this for real billing. Unknown/unpriced models fall back to Decimal(0) in
# compute_cost_usd rather than raising.
_PRICING_PER_MILLION_USD: dict[str, tuple[Decimal, Decimal]] = {
    # model: (input price per 1M tokens, output price per 1M tokens)
    "claude-haiku-4-5": (Decimal("1.00"), Decimal("5.00")),
    "claude-sonnet-5": (Decimal("3.00"), Decimal("15.00")),
}

_ONE_MILLION = Decimal(1_000_000)


def compute_cost_usd(model: str, input_tokens: int, output_tokens: int) -> Decimal:
    """Compute call cost in USD from token counts using the static pricing table.

    Returns Decimal(0) for models not in the table rather than raising, so an
    unrecognized/new model id never crashes a call — it just costs $0 in the log.
    """
    prices = _PRICING_PER_MILLION_USD.get(model)
    if prices is None:
        return Decimal(0)
    input_price_per_million, output_price_per_million = prices
    input_cost = (Decimal(input_tokens) / _ONE_MILLION) * input_price_per_million
    output_cost = (Decimal(output_tokens) / _ONE_MILLION) * output_price_per_million
    return input_cost + output_cost


__all__ = ["compute_cost_usd"]
