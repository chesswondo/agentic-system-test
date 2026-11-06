"""Batch of runs with different seeds and a convergence report.

Runs are sequential: each one starts its own MCP server processes, and live runs share one
provider rate limit. Every run is a normal run directory nested in `runs/batch_<id>/`.

Convergence rate = share of runs whose outcome signature equals the modal signature, counted
only if the modal outcome is also the golden one (otherwise 0).
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from collections.abc import Sequence
from typing import Any

OUTCOME_KEYS = ("date", "trail_id", "bookings", "rentals", "rented_items", "must_bring")


def outcome_signature(outcome: dict[str, Any]) -> str | None:
    """Canonical string of the outcome fields that define the result; None if no plan."""
    if not outcome.get("final_plan_delivered"):
        return None
    return json.dumps({k: outcome.get(k) for k in OUTCOME_KEYS}, sort_keys=True)


def run_row(trajectory: dict[str, Any]) -> dict[str, Any]:
    m = trajectory["metrics"]
    return {
        "run_id": trajectory["run_id"],
        "seed": trajectory["seed"],
        "termination": m["termination_reason"],
        "ticks": m["ticks"],
        "success": m["success"],
        "matches_expected": m["outcome_matches_expected"],
        "outcome_signature": outcome_signature(trajectory["outcome"]),
        "llm_calls": m["llm"]["calls"],
        "input_tokens": m["llm"]["input_tokens"],
        "output_tokens": m["llm"]["output_tokens"],
        "cost_usd": m["llm"]["cost_usd"],
        "unpriced_models": m["llm"]["unpriced_models"],
        "duration_s": m["duration_s"],
        "llm_latency_mean_ms": m["llm"]["latency_ms"]["mean"],
        "tool_error_rate": m["tools"]["error_rate"],
        "retries": m["retries"]["tool"] + m["retries"]["llm"],
        "violations": m["violations"]["total"],
        "budget_exhaustions": m["budget_exhaustions"],
    }


def _mean(values: Sequence[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return round(statistics.fmean(present), 4) if present else None


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    signatures = Counter(r["outcome_signature"] for r in rows if r["outcome_signature"])
    modal, modal_count = signatures.most_common(1)[0] if signatures else (None, 0)
    modal_is_golden = any(
        r["matches_expected"] for r in rows if r["outcome_signature"] == modal and modal
    )
    costs = [r["cost_usd"] for r in rows]
    return {
        "runs": n,
        "success_rate": round(sum(r["success"] for r in rows) / n, 4) if n else 0.0,
        "golden_rate": round(sum(r["matches_expected"] for r in rows) / n, 4) if n else 0.0,
        "convergence_rate": round(modal_count / n, 4) if n and modal_is_golden else 0.0,
        "modal_outcome": json.loads(modal) if modal else None,
        "modal_outcome_is_golden": modal_is_golden,
        "distinct_outcomes": len(signatures),
        "terminations": dict(sorted(Counter(r["termination"] for r in rows).items())),
        "cost_usd": {
            "mean": _mean(costs),
            "max": max(costs, default=None),
            "total": round(sum(costs), 6),
        },
        "unpriced_models": sorted({m for r in rows for m in r["unpriced_models"]}),
        "ticks_mean": _mean([r["ticks"] for r in rows]),
        "duration_s_mean": _mean([r["duration_s"] for r in rows]),
        "llm_latency_mean_ms": _mean([r["llm_latency_mean_ms"] for r in rows]),
        "llm_calls_mean": _mean([r["llm_calls"] for r in rows]),
        "input_tokens_mean": _mean([r["input_tokens"] for r in rows]),
    }


TABLE_COLUMNS = (
    ("seed", "seed", "{}"),
    ("termination", "termination", "{}"),
    ("ticks", "ticks", "{}"),
    ("golden", "matches_expected", "{}"),
    ("llm", "llm_calls", "{}"),
    ("in_tok", "input_tokens", "{}"),
    ("cost$", "cost_usd", "{:.4f}"),
    ("dur_s", "duration_s", "{:.1f}"),
    ("retries", "retries", "{}"),
    ("tool_err", "tool_error_rate", "{:.2f}"),
)


def format_table(rows: Sequence[dict[str, Any]], summary: dict[str, Any]) -> str:
    cells = [[h for h, _, _ in TABLE_COLUMNS]]
    cells += [[fmt.format(r[key]) for _, key, fmt in TABLE_COLUMNS] for r in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(TABLE_COLUMNS))]
    lines = [
        "  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)).rstrip() for row in cells
    ]
    lines.insert(1, "  ".join("-" * w for w in widths))
    cost = summary["cost_usd"]
    lines += [
        "",
        f"success rate {summary['success_rate']:.0%}, golden {summary['golden_rate']:.0%}, "
        f"convergence {summary['convergence_rate']:.0%} "
        f"({summary['distinct_outcomes']} distinct outcome(s))",
        f"mean ticks {summary['ticks_mean']:.1f}, "
        f"mean duration {summary['duration_s_mean']:.1f} s, "
        f"est. cost mean ${cost['mean']:.4f} / max ${cost['max']:.4f} / "
        f"total ${cost['total']:.4f}",
    ]
    if summary["unpriced_models"]:
        lines.append(f"no price configured (not in cost): {', '.join(summary['unpriced_models'])}")
    return "\n".join(lines)
