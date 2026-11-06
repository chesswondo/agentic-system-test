"""Live end-to-end run with the real LLM provider. Deselected by default; run with
`uv run pytest -q -m live`. Needs GOOGLE_API_KEY in the environment or `.env`."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from picos.config import load_config, load_env
from picos.runtime.simulation import Simulation, build_participants
from tests.helpers import ROOT, SCENARIO_PATH

LIVE_SEED = 7


@pytest.mark.live
async def test_live_run_reaches_golden_outcome(tmp_path: Path) -> None:
    load_env()
    if not os.environ.get("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY is not set")
    cfg = load_config(ROOT / "config" / "app.yaml")
    cfg = cfg.model_copy(
        update={
            "llm": cfg.llm.model_copy(update={"provider": "google_genai"}),
            "humans": cfg.humans.model_copy(update={"provider": "llm"}),
            "runtime": cfg.runtime.model_copy(update={"runs_dir": tmp_path}),
        }
    )
    agent_llms, humans = build_participants(cfg, ROOT, fake_script_path=None)
    sim = Simulation(
        cfg,
        SCENARIO_PATH,
        LIVE_SEED,
        run_id=f"live-seed{LIVE_SEED}",
        agent_llms=agent_llms,
        human_provider=humans,
        root=ROOT,
    )
    trajectory = await sim.run()
    assert trajectory["termination"]["reason"] == "success", trajectory["termination"]
    assert trajectory["outcome"]["matches_expected"], trajectory["outcome"]
