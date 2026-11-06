"""Per-run metrics, computed from the trajectory alone.

A pure function of the recorded events plus the price table, so metrics can be recomputed for
any saved trajectory. Cost is estimated from `usage_metadata` token counts; calls to a model
without a price are listed in `unpriced_models` instead of being counted as free.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping
from typing import Any

from picos.config import Price


def _latency_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50_ms": None, "max_ms": None}
    return {"p50_ms": round(statistics.median(values), 1), "max_ms": round(max(values), 1)}


def compute_metrics(trajectory: Mapping[str, Any], prices: Mapping[str, Price]) -> dict[str, Any]:
    events: list[dict[str, Any]] = trajectory["events"]
    termination = trajectory["termination"]
    outcome = trajectory.get("outcome") or {}

    llm: dict[str, dict[str, Any]] = {}
    llm_latency: list[float] = []
    unpriced: set[str] = set()
    tool_calls: Counter[str] = Counter()
    attempts: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    injected: Counter[str] = Counter()  # failures caused by the fault-injection layer
    latencies: defaultdict[str, list[float]] = defaultdict(list)
    retries = {"tool": 0, "llm": 0}
    edges: Counter[str] = Counter()
    violations: Counter[str] = Counter()
    budget_exhausted = 0

    for e in events:
        kind = e["kind"]
        if kind == "llm_call":
            row = llm.setdefault(
                e["actor"],
                {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
            )
            row["calls"] += 1
            row["input_tokens"] += e.get("input_tokens", 0)
            row["output_tokens"] += e.get("output_tokens", 0)
            price = prices.get(e.get("model", ""))
            if price is None:
                unpriced.add(e.get("model", "?"))
            else:
                row["cost_usd"] += (
                    e.get("input_tokens", 0) * price.input
                    + e.get("output_tokens", 0) * price.output
                ) / 1_000_000
            if "latency_ms" in e:
                llm_latency.append(e["latency_ms"])
        elif kind == "tool_call":
            tool_calls[e["tool"]] += 1
        elif kind == "tool_result":
            attempts[e["tool"]] += 1
            if not e.get("ok"):
                failures[e["tool"]] += 1
            if e.get("fault") in ("error", "timeout"):
                injected[e["tool"]] += 1
            if "latency_ms" in e:
                latencies[e["tool"]].append(e["latency_ms"])
        elif kind == "retry":
            retries["llm" if e.get("target") == "llm" else "tool"] += 1
        elif kind == "message":
            edges[f"{e['from']}->{e['to']}"] += 1
        elif kind == "violation":
            violations[e.get("rule", "?")] += 1
        elif kind == "budget_exhausted":
            budget_exhausted += 1

    for row in llm.values():
        row["cost_usd"] = round(row["cost_usd"], 6)
    total_attempts = sum(attempts.values())
    return {
        "termination_reason": termination["reason"],
        "ticks": termination["tick"],
        "success": termination["reason"] == "success",
        "outcome_matches_expected": bool(outcome.get("matches_expected")),
        "duration_s": max((e.get("ts", 0.0) for e in events), default=0.0),
        "llm": {
            "per_actor": dict(sorted(llm.items())),
            "calls": sum(r["calls"] for r in llm.values()),
            "input_tokens": sum(r["input_tokens"] for r in llm.values()),
            "output_tokens": sum(r["output_tokens"] for r in llm.values()),
            "cost_usd": round(sum(r["cost_usd"] for r in llm.values()), 6),
            "unpriced_models": sorted(unpriced),
            "latency_ms": {
                "mean": round(statistics.fmean(llm_latency), 1) if llm_latency else None,
                **_latency_stats(llm_latency),
            },
        },
        "tools": {
            "per_tool": {
                tool: {
                    "calls": tool_calls[tool],
                    "attempts": attempts[tool],
                    "errors": failures[tool],
                    "injected_faults": injected[tool],
                    **_latency_stats(latencies[tool]),
                }
                for tool in sorted(set(tool_calls) | set(attempts))
            },
            "attempts": total_attempts,
            "errors": sum(failures.values()),
            "injected_faults": sum(injected.values()),
            "error_rate": round(sum(failures.values()) / total_attempts, 4)
            if total_attempts
            else 0.0,
        },
        "retries": retries,
        "messages_per_edge": dict(sorted(edges.items())),
        "violations": {
            "total": sum(violations.values()),
            "by_rule": dict(sorted(violations.items())),
        },
        "budget_exhaustions": budget_exhausted,
    }
