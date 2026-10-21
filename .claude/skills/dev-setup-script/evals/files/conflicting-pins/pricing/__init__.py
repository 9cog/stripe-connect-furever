def quote(unit_cents: int, qty: int, discount_pct: float = 0.0) -> int:
    """Total in cents, discount applied, rounded half-up."""
    gross = unit_cents * qty
    return int(gross * (1 - discount_pct / 100) + 0.5)
