"""End-to-end simulation offline: scripted LLMs (scenarios/default.fake.yaml), scripted humans,
real MCP servers with seeded fault injection. Covers the full message chain, faults, topology,
termination and determinism."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from langchain_core.messages import ToolMessage

from picos.agents.fake import ScriptedChatModel
from picos.agents.fake_script import FakeScripts, load_fake_scripts
from picos.config import AppConfig, load_config
from picos.runtime.simulation import Simulation, scripted_humans
from picos.runtime.trajectory import check_trajectory, comparable
from tests.helpers import ROOT, SCENARIO_PATH, text

E2E_SEED = 1  # injects a get_forecast 503 and a book_trip fault, both recovered by retries
FAKE_SCRIPT = ROOT / "scenarios" / "default.fake.yaml"


def make_cfg(runs_dir: Path, **runtime: Any) -> AppConfig:
    cfg = load_config(ROOT / "config" / "app.yaml")
    no_wait = cfg.faults.retry.model_copy(update={"base_delay_s": 0.0, "max_delay_s": 0.0})
    return cfg.model_copy(
        update={
            "llm": cfg.llm.model_copy(update={"provider": "fake"}),
            "runtime": cfg.runtime.model_copy(update={"runs_dir": runs_dir, **runtime}),
            "faults": cfg.faults.model_copy(update={"retry": no_wait}),
        }
    )


async def simulate(
    runs_dir: Path, seed: int = E2E_SEED, scripts: FakeScripts | None = None, **runtime: Any
) -> tuple[dict[str, Any], dict[str, ScriptedChatModel], Simulation]:
    scripts = scripts or load_fake_scripts(FAKE_SCRIPT)
    llms = {a: ScriptedChatModel(script=scripts.messages(a)) for a in ("agent_a", "agent_b")}
    sim = Simulation(
        make_cfg(runs_dir, **runtime),
        SCENARIO_PATH,
        seed,
        run_id=f"test-seed{seed}",
        agent_llms=llms,
        human_provider=scripted_humans,
        root=ROOT,
    )
    trajectory = await sim.run()
    on_disk = json.loads((sim.run_dir / "trajectory.json").read_text(encoding="utf-8"))
    assert on_disk == json.loads(json.dumps(trajectory))
    check_trajectory(on_disk)
    return trajectory, llms, sim


def of_kind(trajectory: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    return [e for e in trajectory["events"] if e["kind"] == kind]


def chain(trajectory: dict[str, Any]) -> list[tuple[str, str, str]]:
    return [
        (e["from"], e["to"], e.get("performative") or e["message_kind"])
        for e in of_kind(trajectory, "message")
    ]


def tool_result(llm: ScriptedChatModel, name: str) -> str:
    """Content of the last result of tool `name` that the model actually saw."""
    msgs = [m for m in llm.seen[-1] if isinstance(m, ToolMessage) and m.name == name]
    assert msgs, f"model never saw a {name} result"
    return text(msgs[-1])


def steps(*calls: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    return {"tool_calls": [{"name": n, "args": a} for n, a in calls]}


def send(performative: str, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    return "send_to_peer", {
        "performative": performative,
        "payload_json": json.dumps(payload),
        "text": performative.lower(),
    }


END = ("end_turn", {"reason": "done"})
REQUEST = send(
    "REQUEST_CONSTRAINTS",
    {"window_start": "2026-11-17", "window_end": "2026-11-30", "trip_type": "day_hike"},
)


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def e2e(tmp_path_factory) -> tuple[dict[str, Any], dict[str, ScriptedChatModel]]:
    trajectory, llms, _ = await simulate(tmp_path_factory.mktemp("e2e"))
    return trajectory, llms


async def test_full_chain_ends_in_success_with_golden_outcome(e2e):
    trajectory, _ = e2e
    assert trajectory["termination"] == {"reason": "success", "tick": 9}
    outcome = trajectory["outcome"]
    assert outcome["matches_expected"] is True
    assert (outcome["date"], outcome["trail_id"]) == ("2026-11-22", "vega-de-ario")
    assert (outcome["bookings"], outcome["rentals"]) == (1, 1)
    assert outcome["booking_confirmed"] is True
    assert outcome["rented_items"] == ["gaiters", "trekking_poles"]
    assert outcome["must_bring"]["human_a"] == ["headlamp"]
    # The intended causal chain, opened by one message from each human.
    assert chain(trajectory) == [
        ("human_a", "agent_a", "info"),
        ("human_b", "agent_b", "info"),
        ("agent_a", "agent_b", "REQUEST_CONSTRAINTS"),
        ("agent_b", "human_b", "question"),
        ("human_b", "agent_b", "info"),
        ("agent_b", "agent_a", "CONSTRAINTS"),
        ("agent_a", "agent_b", "PROPOSAL"),
        ("agent_b", "human_b", "question"),
        ("human_b", "agent_b", "info"),
        ("agent_b", "agent_a", "ACCEPT"),
        ("agent_a", "human_a", "final_plan"),
        ("human_a", "agent_a", "info"),
    ]
    assert "afraid of heights" in of_kind(trajectory, "message")[4]["text"]
    assert of_kind(trajectory, "violation") == [] and of_kind(trajectory, "error") == []


async def test_injected_faults_are_retried_and_recovered(e2e):
    trajectory, _ = e2e
    faulted = [
        r for r in of_kind(trajectory, "tool_result") if r.get("fault") in ("error", "timeout")
    ]
    assert faulted, "this seed must inject at least one fault"
    retries = of_kind(trajectory, "retry")
    assert {r["call_id"] for r in retries} == {r["call_id"] for r in faulted}
    for fault in faulted:
        later = [
            r
            for r in of_kind(trajectory, "tool_result")
            if r["call_id"] == fault["call_id"] and r["attempt"] > fault["attempt"]
        ]
        assert later and later[-1]["ok"] is True, f"{fault['call_id']} never recovered"
    assert any(f["tool"] == "book_trip" for f in faulted)  # side effect retried, still 1 booking


async def test_decision_tools_agree_with_the_scripted_choices(e2e):
    """The script is fixed, so check the real tool results the model saw support it."""
    _, llms = e2e
    a, b = llms["agent_a"], llms["agent_b"]
    dates = json.loads(tool_result(a, "rank_dates"))
    assert dates[0]["date"] == "2026-11-22"
    trails = json.loads(tool_result(a, "rank_trails"))
    assert trails["ranked"][0]["id"] == "vega-de-ario"
    assert "exposure" in trails["excluded"]["ruta-del-cares"]
    proposal_gear = json.loads(
        next(
            c["args"]["payload_json"]
            for m in a.script
            for c in m.tool_calls
            if c["args"].get("performative") == "PROPOSAL"
        )
    )["required_gear"]
    assert json.loads(tool_result(a, "required_gear"))["items"] == proposal_gear
    assert json.loads(tool_result(a, "check_equipment"))["missing"] == ["headlamp"]
    assert json.loads(tool_result(b, "check_equipment"))["missing"] == [
        "gaiters",
        "trekking_poles",
    ]
    assert json.loads(tool_result(b, "rent_equipment"))["rental_id"] == "RNT-0001"


async def test_agents_activate_concurrently_within_a_tick(e2e):
    trajectory, _ = e2e

    def interval(actor: str) -> tuple[float, float]:
        start, end = (
            e["ts"]
            for e in trajectory["events"]
            if e["tick"] == 0 and e["actor"] == actor and e["kind"] in ("turn_start", "turn_end")
        )
        return start, end

    (a0, a1), (b0, b1) = interval("agent_a"), interval("agent_b")
    assert a0 < b1 and b0 < a1, "tick-0 activations of agent_a and agent_b do not overlap"


async def test_same_seed_gives_identical_events_and_other_seed_differs(tmp_path, e2e):
    trajectory, _ = e2e
    again, _, _ = await simulate(tmp_path / "again")
    assert comparable(again["events"]) == comparable(trajectory["events"])
    other, _, _ = await simulate(tmp_path / "other", seed=E2E_SEED + 1)
    assert other["termination"]["reason"] == "success"
    assert comparable(other["events"]) != comparable(trajectory["events"])


async def test_nobody_replies_ends_in_quiescence(tmp_path):
    scripts = FakeScripts.model_validate(
        {
            "agent_a": [steps(REQUEST, END)],
            "agent_b": [steps(END), steps(END)],  # ignores the request
        }
    )
    trajectory, _, _ = await simulate(tmp_path, scripts=scripts)
    assert trajectory["termination"] == {"reason": "quiescence", "tick": 1}
    assert trajectory["outcome"]["matches_expected"] is False


async def test_ping_pong_loop_ends_at_max_ticks(tmp_path):
    ping = steps(send("INFORM", {"details": "still thinking"}), END)
    scripts = FakeScripts.model_validate({"agent_a": [ping] * 5, "agent_b": [ping] * 5})
    trajectory, _, _ = await simulate(tmp_path, scripts=scripts, max_ticks=5)
    assert trajectory["termination"] == {"reason": "max_ticks", "tick": 4}
    assert sum(1 for c in chain(trajectory) if c[2] == "INFORM") == 10


async def test_unrecoverable_exception_ends_in_error_with_trajectory(tmp_path):
    scripts = FakeScripts.model_validate({"agent_a": [], "agent_b": [steps(END)]})
    trajectory, _, _ = await simulate(tmp_path, scripts=scripts)
    assert trajectory["termination"]["reason"] == "error"
    assert "ScriptExhaustedError" in trajectory["termination"]["detail"]
    [error] = of_kind(trajectory, "error")
    assert error["actor"] == "agent_a" and error["traceback"]
    # The concurrent activation of agent_b still completed and was flushed.
    assert any(e["actor"] == "agent_b" and e["kind"] == "turn_end" for e in trajectory["events"])


def test_existing_run_directory_is_refused(tmp_path):
    (tmp_path / "test-seed1").mkdir()
    with pytest.raises(FileExistsError):
        Simulation(
            make_cfg(tmp_path),
            SCENARIO_PATH,
            1,
            run_id="test-seed1",
            agent_llms={},
            human_provider=scripted_humans,
            root=ROOT,
        )
