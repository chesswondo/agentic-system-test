"""Discrete-time simulation loop.

Tick t: deliver the messages sent during t-1 (tick 0: the scenario's initial messages),
activate every actor with a non-empty inbox concurrently (`asyncio.gather`), then flush the
buffered events in canonical order and check termination:
success > error > quiescence (nothing in flight) > max_ticks.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from collections.abc import Callable, Mapping, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from picos.agents.fake_script import load_fake_scripts
from picos.agents.graph import AgentOptions, TurnResult, build_agent, run_turn
from picos.agents.llm import make_llm
from picos.agents.spec import AgentSpec, load_agent_spec
from picos.config import AppConfig
from picos.domain import HumanId, HumanSpec, Scenario, load_scenario
from picos.humans.base import HumanActivation, HumanProvider, SimulatedHuman
from picos.humans.llm import LLMHumanProvider
from picos.humans.scripted import ScriptedHumanProvider
from picos.protocol import AgentId, HumanMessage
from picos.runtime.bus import ACTORS, Message, MessageBus
from picos.runtime.mermaid import sequence_diagram
from picos.runtime.metrics import compute_metrics
from picos.runtime.outcome import extract_outcome
from picos.runtime.recorder import Recorder
from picos.runtime.trajectory import TrajectoryRecorder, write_json
from picos.tools.comms import Outbox
from picos.tools.mcp_client import EnvContext, open_toolset

log = logging.getLogger(__name__)

AGENT_IDS: tuple[AgentId, ...] = ("agent_a", "agent_b")
TerminationReason = Literal["success", "quiescence", "max_ticks", "error"]
HumanProviderFactory = Callable[[HumanId, HumanSpec, Recorder], HumanProvider]


@dataclass(frozen=True)
class Termination:
    reason: TerminationReason
    tick: int
    detail: str | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"reason": self.reason, "tick": self.tick}
        if self.detail:
            out["detail"] = self.detail
        return out


@dataclass
class _Agent:
    spec: AgentSpec
    graph: CompiledStateGraph
    outbox: Outbox


def make_run_id(seed: int, now: datetime | None = None) -> str:
    stamp = (now or datetime.now(UTC)).strftime("%Y-%m-%dT%H-%M-%S")
    return f"{stamp}_seed{seed}"


def error_fields(exc: BaseException, frames: int = 5) -> dict[str, Any]:
    tb = traceback.extract_tb(exc.__traceback__)[-frames:]
    return {
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": [f"{Path(f.filename).name}:{f.lineno} in {f.name}" for f in tb],
    }


class Simulation:
    def __init__(
        self,
        cfg: AppConfig,
        scenario_path: Path,
        seed: int,
        *,
        run_id: str,
        agent_llms: Mapping[AgentId, BaseChatModel],
        human_provider: HumanProviderFactory,
        root: Path = Path("."),
        run_dir: Path | None = None,
    ) -> None:
        """`run_dir` defaults to `<root>/<runtime.runs_dir>/<run_id>` (batch nests its runs)."""
        self.cfg = cfg
        self.scenario_path = scenario_path
        self.scenario: Scenario = load_scenario(scenario_path)
        self.seed = seed
        self.run_id = run_id
        self.root = root
        self.run_dir = run_dir or root / cfg.runtime.runs_dir / run_id
        if self.run_dir.exists():  # stale ledgers would leak idempotency state into this run
            raise FileExistsError(f"run directory {self.run_dir} already exists")
        self.agent_llms = agent_llms
        self.recorder = TrajectoryRecorder()
        self.bus = MessageBus(self.recorder, cfg.runtime.max_negotiation_rounds)
        self.conversation_id = f"{self.scenario.name}-seed{seed}"
        self.humans: dict[str, SimulatedHuman] = {
            hid: SimulatedHuman(
                hid, human_provider(hid, spec, self.recorder), self.recorder, self.bus.accept
            )
            for hid, spec in self.scenario.humans.items()
        }
        self.agents: dict[str, _Agent] = {}
        self.tick = 0

    async def run(self) -> dict[str, Any]:
        """Run to termination and write trajectory.json, metrics.json and sequence.mmd to the
        run directory; return the trajectory."""
        log.info("run %s: start (seed=%d)", self.run_id, self.seed)
        try:
            async with AsyncExitStack() as stack:
                await self._open_agents(stack)
                termination = await self._loop()
        except Exception as exc:  # unrecoverable: recorded, never swallowed silently
            log.exception("run %s: aborted", self.run_id)
            self.recorder.record("runtime", "error", **error_fields(exc))
            termination = Termination("error", self.tick, f"{type(exc).__name__}: {exc}")
        self.recorder.flush(self.tick)
        trajectory = self._trajectory(termination)
        write_json(self.run_dir / "trajectory.json", trajectory)
        write_json(self.run_dir / "metrics.json", trajectory["metrics"])
        names = {hid: h.name for hid, h in self.scenario.humans.items()}
        (self.run_dir / "sequence.mmd").write_text(
            sequence_diagram(trajectory, names), encoding="utf-8"
        )
        log.info("run %s: %s at tick %d", self.run_id, termination.reason, termination.tick)
        return trajectory

    async def _open_agents(self, stack: AsyncExitStack) -> None:
        env = EnvContext(
            run_seed=self.seed,
            scenario_path=self.scenario_path,
            data_dir=self.root / "data",
            state_dir=self.run_dir / "env",
        )
        for name in AGENT_IDS:
            spec = load_agent_spec(
                self.root / "config" / "agents" / f"{name}.yaml", base_dir=self.root
            )
            outbox = Outbox(name, self.conversation_id, gate=self.bus.accept)
            tools = await stack.enter_async_context(
                open_toolset(spec, self.scenario, self.cfg.faults, env, outbox, self.recorder)
            )
            graph = build_agent(
                spec,
                tools,
                self.agent_llms[name],
                InMemorySaver(),  # one checkpointer per agent: agent states stay independent
                recorder=self.recorder,
                options=AgentOptions.from_config(self.cfg, spec),
                context=agent_context(self.scenario, spec),
            )
            self.agents[name] = _Agent(spec, graph, outbox)

    def _send_initial_messages(self) -> None:
        for m in self.scenario.initial_messages:
            human = self.humans[m.sender]
            if m.recipient != human.agent:
                raise ValueError(f"initial message {m.sender} -> {m.recipient} breaks topology")
            human.initial(m.text)

    async def _loop(self) -> Termination:
        max_ticks = self.cfg.runtime.max_ticks
        for tick in range(max_ticks):
            self.tick = tick
            if tick == 0:
                self._send_initial_messages()
                self.recorder.flush(tick)  # log the trigger before the turns it causes
            inboxes = self.bus.deliver()
            actors = [a for a in ACTORS if a in inboxes]
            results = await asyncio.gather(
                *(self._activate(a, inboxes[a]) for a in actors), return_exceptions=True
            )
            for agent in self.agents.values():
                agent.outbox.drain()  # already accepted by the bus at send time
            errors: list[tuple[str, BaseException]] = []
            success = False
            for actor, result in zip(actors, results, strict=True):
                if isinstance(result, BaseException):
                    if not isinstance(result, Exception):
                        raise result  # cancellation / interrupt: not ours to handle
                    self.recorder.record(actor, "error", **error_fields(result))
                    errors.append((actor, result))
                elif isinstance(result, HumanActivation) and actor == "human_a":
                    success = result.accepted_final_plan
            self.recorder.flush(tick)

            if success:
                return Termination("success", tick)
            if errors:
                actor, exc = errors[0]
                return Termination("error", tick, f"{actor}: {type(exc).__name__}: {exc}")
            if not self.bus.has_pending():
                return Termination("quiescence", tick)
        return Termination("max_ticks", max_ticks - 1)

    async def _activate(self, actor: str, inbox: Sequence[Message]) -> TurnResult | HumanActivation:
        if actor in self.agents:
            trace = {"run_id": self.run_id, "seed": self.seed, "tick": self.tick}
            return await run_turn(
                self.agents[actor].graph, actor, inbox, self.recorder, trace=trace
            )
        human_inbox = [m for m in inbox if isinstance(m, HumanMessage)]
        if len(human_inbox) != len(inbox):
            raise TypeError(f"{actor} received a non-human message")
        return await self.humans[actor].activate(human_inbox)

    def _trajectory(self, termination: Termination) -> dict[str, Any]:
        trajectory: dict[str, Any] = {
            "run_id": self.run_id,
            "scenario": self.scenario_path.as_posix(),
            "seed": self.seed,
            "config_snapshot": {
                "app": self.cfg.model_dump(mode="json"),
                "agents": {n: a.spec.model_dump(mode="json") for n, a in self.agents.items()},
            },
            "events": self.recorder.events,
            "termination": termination.as_dict(),
            "outcome": extract_outcome(self.scenario, self.bus.log, self.run_dir / "env"),
            "metrics": {},
        }
        trajectory["metrics"] = compute_metrics(trajectory, self.cfg.prices)
        return trajectory


def build_participants(
    cfg: AppConfig, root: Path, fake_script_path: Path | None
) -> tuple[dict[AgentId, BaseChatModel], HumanProviderFactory]:
    """Models for both agents and the human provider factory, per config.

    `fake` replays `fake_script_path` and always uses scripted humans. `google_genai` fails
    fast without GOOGLE_API_KEY (ConfigError from make_llm)."""
    if cfg.llm.provider == "fake":
        if fake_script_path is None:
            raise ValueError("llm.provider 'fake' needs a fake script file")
        scripts = load_fake_scripts(fake_script_path)
        llms: dict[AgentId, BaseChatModel] = {
            a: make_llm(cfg.llm, "agents", fake_script=scripts.messages(a)) for a in AGENT_IDS
        }
        return llms, scripted_humans

    llms = {a: make_llm(cfg.llm, "agents") for a in AGENT_IDS}
    if cfg.humans.provider == "scripted":
        return llms, scripted_humans
    make_llm(cfg.llm, "humans")  # fail fast before the run starts
    templates = {
        hid: (root / path).read_text(encoding="utf-8") for hid, path in cfg.humans.prompts.items()
    }

    def llm_humans(hid: HumanId, spec: HumanSpec, recorder: Recorder) -> HumanProvider:
        return LLMHumanProvider(
            hid,
            spec,
            make_llm(cfg.llm, "humans"),
            templates[hid],
            retry=cfg.llm.retry,
            recorder=recorder,
        )

    return llms, llm_humans


def agent_context(scenario: Scenario, spec: AgentSpec) -> str:
    """Run facts every agent needs and cannot look up with a tool."""
    sc = scenario
    other = next(hid for hid in sc.humans if hid != spec.human)
    peer = next(a for a in AGENT_IDS if a != spec.name)
    lines = [
        f"- Today: {sc.today.isoformat()} ({sc.today.strftime('%A')})",
        f"- Planning window: {sc.window_start.isoformat()} to {sc.window_end.isoformat()}",
        f"- Trip type: {sc.trip_type}",
        f"- Your human: {sc.humans[spec.human].name} ({spec.human})",
        f"- Other participant: {sc.humans[other].name} ({other}), represented by the other "
        f"assistant ({peer})",
    ]
    return "\n".join(lines)


def scripted_humans(hid: HumanId, spec: HumanSpec, recorder: Recorder) -> HumanProvider:
    return ScriptedHumanProvider(spec)
