"""Shared fixtures: the real scenario, the real agent specs, run-scoped env dirs."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from picos.agents.spec import AgentSpec, load_agent_spec
from picos.domain import Scenario, load_scenario
from picos.tools.mcp_client import EnvContext
from tests.helpers import ROOT, SCENARIO_PATH


@pytest.fixture(scope="session")
def scenario() -> Scenario:
    return load_scenario(SCENARIO_PATH)


@pytest.fixture(scope="session")
def specs() -> dict[str, AgentSpec]:
    return {
        n: load_agent_spec(ROOT / "config" / "agents" / f"{n}.yaml", base_dir=ROOT)
        for n in ("agent_a", "agent_b")
    }


@pytest.fixture
def env(tmp_path: Path) -> Callable[[int], EnvContext]:
    def _env(seed: int = 42) -> EnvContext:
        return EnvContext(
            run_seed=seed,
            scenario_path=SCENARIO_PATH,
            data_dir=ROOT / "data",
            state_dir=tmp_path / "env",
        )

    return _env
