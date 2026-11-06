"""Typer CLI: `run` executes one simulation; `batch` runs N seeds and writes a report."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

import typer
from pydantic import ValidationError

from picos.agents.fake_script import default_fake_script_path
from picos.config import DEFAULT_CONFIG_PATH, AppConfig, ConfigError, load_config, load_env
from picos.logging_setup import setup_logging
from picos.observability import configure_langsmith
from picos.runtime.batch import format_table, run_row, summarize
from picos.runtime.simulation import Simulation, build_participants, make_run_id
from picos.runtime.trajectory import write_json

app = typer.Typer(help="Picos de Europa multi-agent trip planner.", no_args_is_help=True)
log = logging.getLogger(__name__)

ConfigOpt = Annotated[Path, typer.Option("--config", help="Path to app config YAML.")]
ScenarioOpt = Annotated[Path, typer.Option("--scenario", help="Path to scenario YAML.")]
LLMOpt = Annotated[
    Literal["google_genai", "fake"] | None,
    typer.Option("--llm", help="Override llm.provider (fake = scripted, offline)."),
]
HumansOpt = Annotated[
    Literal["llm", "scripted"] | None,
    typer.Option("--humans", help="Override humans.provider (fake LLM always uses scripted)."),
]
FakeScriptOpt = Annotated[
    Path | None,
    typer.Option(help="Scripted LLM responses for --llm fake (default: <scenario>.fake.yaml)."),
]


def _bootstrap(config_path: Path) -> AppConfig:
    load_env()
    try:
        cfg = load_config(config_path)
    except (OSError, ValidationError) as exc:
        typer.echo(f"Invalid config {config_path}: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    setup_logging(cfg.observability.log_level)
    configure_langsmith(cfg.observability.langsmith)
    return cfg


def _with_overrides(cfg: AppConfig, llm: str | None, humans: str | None) -> AppConfig:
    update: dict[str, object] = {}
    if llm:
        update["llm"] = cfg.llm.model_copy(update={"provider": llm})
    if humans:
        update["humans"] = cfg.humans.model_copy(update={"provider": humans})
    return cfg.model_copy(update=update)


def _simulation(
    cfg: AppConfig,
    scenario: Path,
    seed: int,
    run_id: str,
    fake_script: Path | None,
    run_dir: Path | None = None,
) -> Simulation:
    """Build one run; a problem the user must fix exits with code 2."""
    try:
        agent_llms, human_provider = build_participants(
            cfg, Path("."), fake_script or default_fake_script_path(scenario)
        )
        return Simulation(
            cfg,
            scenario,
            seed,
            run_id=run_id,
            agent_llms=agent_llms,
            human_provider=human_provider,
            run_dir=run_dir,
        )
    except (ConfigError, OSError, ValidationError, ValueError) as exc:
        typer.echo(f"Cannot start run: {exc}", err=True)
        raise typer.Exit(code=2) from exc


@app.command()
def run(
    scenario: ScenarioOpt = Path("scenarios/default.yaml"),
    seed: Annotated[int, typer.Option(help="Run seed.")] = 42,
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
    llm: LLMOpt = None,
    humans: HumansOpt = None,
    fake_script: FakeScriptOpt = None,
    run_id: Annotated[
        str | None, typer.Option(help="Run id (default: <UTC time>_seed<n>).")
    ] = None,
) -> None:
    """Run one simulation and write trajectory.json, metrics.json and sequence.mmd to
    runs/<run_id>/. Exit 0 only on success."""
    cfg = _with_overrides(_bootstrap(config), llm, humans)
    sim = _simulation(cfg, scenario, seed, run_id or make_run_id(seed), fake_script)
    log.info("run: scenario=%s seed=%d provider=%s", scenario, seed, cfg.llm.provider)
    trajectory = asyncio.run(sim.run())
    term, metrics = trajectory["termination"], trajectory["metrics"]
    typer.echo(
        f"{trajectory['run_id']}: {term['reason']} at tick {term['tick']}, "
        f"outcome matches expected: {trajectory['outcome']['matches_expected']}, "
        f"LLM calls {metrics['llm']['calls']}, est. cost ${metrics['llm']['cost_usd']:.4f}"
    )
    typer.echo(f"run dir: {sim.run_dir}")
    raise typer.Exit(code=0 if term["reason"] == "success" else 1)


@app.command()
def batch(
    scenario: ScenarioOpt = Path("scenarios/default.yaml"),
    n: Annotated[int, typer.Option(help="Number of runs.", min=1)] = 5,
    seed_start: Annotated[int, typer.Option(help="Seed of the first run; then +1 each.")] = 1,
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
    llm: LLMOpt = None,
    humans: HumansOpt = None,
    fake_script: FakeScriptOpt = None,
    batch_id: Annotated[
        str | None, typer.Option(help="Batch id (default: batch_<UTC time>).")
    ] = None,
) -> None:
    """Run N simulations with consecutive seeds, sequentially. Writes
    runs/<batch_id>/report.json and prints a table. Exit 0 when the report is written."""
    cfg = _with_overrides(_bootstrap(config), llm, humans)
    batch_id = batch_id or "batch_" + datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
    batch_dir = cfg.runtime.runs_dir / batch_id
    if batch_dir.exists():
        typer.echo(f"Cannot start batch: {batch_dir} already exists", err=True)
        raise typer.Exit(code=2)
    seeds = list(range(seed_start, seed_start + n))
    humans_provider = "scripted" if cfg.llm.provider == "fake" else cfg.humans.provider
    typer.echo(f"batch {batch_id}: llm={cfg.llm.provider} humans={humans_provider} seeds={seeds}")
    log.info("batch %s: seeds=%s provider=%s", batch_id, seeds, cfg.llm.provider)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        run_id = f"{batch_id}_seed{seed}"
        sim = _simulation(cfg, scenario, seed, run_id, fake_script, batch_dir / f"seed{seed}")
        rows.append(run_row(asyncio.run(sim.run())))
        typer.echo(f"seed {seed}: {rows[-1]['termination']} at tick {rows[-1]['ticks']}")
    summary = summarize(rows)
    report = {
        "batch_id": batch_id,
        "scenario": scenario.as_posix(),
        "provider": cfg.llm.provider,
        "humans": humans_provider,
        "seeds": seeds,
        "summary": summary,
        "runs": rows,
    }
    write_json(batch_dir / "report.json", report)
    typer.echo(format_table(rows, summary))
    typer.echo(f"report: {batch_dir / 'report.json'}")
