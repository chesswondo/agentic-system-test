"""Batch summary: success, golden and convergence rates."""

from __future__ import annotations

from typing import Any

from picos.runtime.batch import format_table, outcome_signature, summarize

GOLDEN = {
    "final_plan_delivered": True,
    "date": "2026-11-22",
    "trail_id": "vega-de-ario",
    "bookings": 1,
    "rentals": 1,
    "rented_items": ["gaiters"],
    "must_bring": {},
}


def _row(
    seed: int, outcome: dict[str, Any], success: bool = True, golden: bool = True, cost=0.1
) -> dict[str, Any]:
    return {
        "seed": seed,
        "termination": "success" if success else "max_ticks",
        "ticks": 9,
        "success": success,
        "matches_expected": golden,
        "outcome_signature": outcome_signature(outcome),
        "llm_calls": 20,
        "input_tokens": 1000,
        "cost_usd": cost,
        "unpriced_models": [],
        "duration_s": 60.0,
        "llm_latency_mean_ms": 900.0,
        "retries": 1,
        "tool_error_rate": 0.05,
    }


def test_signature_ignores_ids_and_needs_a_plan():
    assert outcome_signature(GOLDEN) == outcome_signature(GOLDEN | {"booking_id": "BKG-9"})
    assert outcome_signature(GOLDEN | {"final_plan_delivered": False}) is None


def test_convergence_counts_modal_golden_outcome():
    other = GOLDEN | {"date": "2026-11-21"}
    rows = [
        _row(1, GOLDEN),
        _row(2, GOLDEN),
        _row(3, GOLDEN, cost=0.3),
        _row(4, other, golden=False),
        _row(5, {}, success=False, golden=False),
    ]
    s = summarize(rows)
    assert (s["success_rate"], s["golden_rate"], s["convergence_rate"]) == (0.8, 0.6, 0.6)
    assert s["distinct_outcomes"] == 2 and s["modal_outcome_is_golden"] is True
    assert s["terminations"] == {"max_ticks": 1, "success": 4}
    assert s["cost_usd"]["max"] == 0.3
    table = format_table(rows, s)
    assert "convergence 60%" in table
    assert len(table.splitlines()) == 2 + 5 + 3  # header, rule, rows, blank, 2 summary lines


def test_modal_outcome_that_is_not_golden_gives_zero_convergence():
    wrong = GOLDEN | {"trail_id": "ruta-del-cares"}
    rows = [_row(1, wrong, golden=False), _row(2, wrong, golden=False), _row(3, GOLDEN)]
    s = summarize(rows)
    assert s["convergence_rate"] == 0.0 and s["modal_outcome_is_golden"] is False
