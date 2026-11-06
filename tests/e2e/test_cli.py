import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

import picos.cli
from picos.cli import app
from picos.runtime.trajectory import check_trajectory
from tests.helpers import ROOT, assert_valid_mermaid

runner = CliRunner()


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.output and "batch" in result.output


def test_bad_config_exits_2(tmp_path: Path) -> None:
    result = runner.invoke(app, ["batch", "--config", str(tmp_path / "missing.yaml")])
    assert result.exit_code == 2


def test_real_run_without_key_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(picos.cli, "load_env", lambda: None)  # never read a real .env here
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.chdir(ROOT)
    result = runner.invoke(app, ["run", "--llm", "google_genai", "--run-id", "never-created"])
    assert result.exit_code == 2
    assert "GOOGLE_API_KEY" in result.output
    assert not (ROOT / "runs" / "never-created").exists()


def _config_in(tmp_path: Path) -> Path:
    raw = yaml.safe_load((ROOT / "config" / "app.yaml").read_text(encoding="utf-8"))
    raw["runtime"]["runs_dir"] = str(tmp_path)
    config = tmp_path / "app.yaml"
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return config


def test_fake_run_writes_valid_trajectory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config_in(tmp_path)
    monkeypatch.chdir(ROOT)
    args = ["run", "--scenario", "scenarios/default.yaml", "--seed", "1", "--llm", "fake"]
    result = runner.invoke(app, [*args, "--config", str(config), "--run-id", "cli"])
    assert result.exit_code == 0, result.output
    trajectory = json.loads((tmp_path / "cli" / "trajectory.json").read_text(encoding="utf-8"))
    check_trajectory(trajectory)
    assert trajectory["termination"]["reason"] == "success"
    assert trajectory["config_snapshot"]["app"]["llm"]["provider"] == "fake"
    run_dir = tmp_path / "cli"
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics == trajectory["metrics"]
    assert metrics["success"] and metrics["outcome_matches_expected"]
    assert metrics["llm"]["per_actor"]["agent_a"]["calls"] > 0
    assert metrics["messages_per_edge"]["agent_a->agent_b"] > 0
    for tool, row in metrics["tools"]["per_tool"].items():
        assert row["injected_faults"] <= row["errors"] <= row["attempts"], tool
    mmd = (run_dir / "sequence.mmd").read_text(encoding="utf-8")
    assert_valid_mermaid(mmd)
    n_messages = sum(e["kind"] == "message" for e in trajectory["events"])
    assert mmd.count("->>") == n_messages


def test_fake_batch_writes_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config_in(tmp_path)
    monkeypatch.chdir(ROOT)
    args = ["batch", "--n", "2", "--seed-start", "3", "--llm", "fake", "--batch-id", "b"]
    result = runner.invoke(app, [*args, "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert "llm=fake humans=scripted seeds=[3, 4]" in result.output
    assert "convergence 100%" in result.output
    report = json.loads((tmp_path / "b" / "report.json").read_text(encoding="utf-8"))
    assert report["seeds"] == [3, 4] and report["provider"] == "fake"
    assert [r["run_id"] for r in report["runs"]] == ["b_seed3", "b_seed4"]
    assert report["summary"]["success_rate"] == 1.0
    assert report["summary"]["convergence_rate"] == 1.0
    for seed in (3, 4):
        assert (tmp_path / "b" / f"seed{seed}" / "trajectory.json").exists()
    again = runner.invoke(app, [*args, "--config", str(config)])
    assert again.exit_code == 2  # an existing batch dir is never reused
