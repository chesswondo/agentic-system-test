"""Runtime building blocks: message bus guard, trajectory ordering, simulated humans."""

from __future__ import annotations

from datetime import date

import pytest
from langchain_core.messages import AIMessage

from picos.agents.fake import ScriptedChatModel
from picos.config import RetryConfig
from picos.domain import FinalPlan, ParticipantGear
from picos.humans.llm import HumanReplyError, LLMHumanProvider
from picos.humans.scripted import ScriptedHumanProvider
from picos.protocol import AgentMessage, HumanMessage
from picos.runtime.bus import BusViolation, MessageBus
from picos.runtime.recorder import ListRecorder, tool_slot
from picos.runtime.trajectory import TrajectoryRecorder, check_trajectory
from picos.tools.comms import Outbox, comms_functions
from picos.tools.local import LocalToolFactory

NO_WAIT = RetryConfig(max_attempts=2, base_delay_s=0.0, max_delay_s=0.0)
PLAN = FinalPlan(
    date=date(2026, 11, 22),
    trail_id="vega-de-ario",
    trail_name="Vega de Ario",
    booking_id="BKG-0001",
    weather_summary="dry",
    participants=[ParticipantGear(participant="human_a", must_bring=["headlamp"])],
)


def human_msg(msg_id: str, sender: str, recipient: str, **kw) -> HumanMessage:
    return HumanMessage(msg_id=msg_id, sender=sender, recipient=recipient, text="hi", **kw)


def inform(msg_id: str, sender: str = "agent_a", recipient: str = "agent_b") -> AgentMessage:
    return AgentMessage(
        msg_id=msg_id,
        conversation_id="c",
        sender=sender,
        recipient=recipient,
        performative="INFORM",
        payload={},
        text="fyi",
    )


def proposal(msg_id: str) -> AgentMessage:
    payload = {
        "date": "2026-11-22",
        "trail_id": "vega-de-ario",
        "trail_summary": "x",
        "required_gear": [],
        "proposal_round": 1,
    }
    return AgentMessage(
        msg_id=msg_id,
        conversation_id="c",
        sender="agent_a",
        recipient="agent_b",
        performative="PROPOSAL",
        payload=payload,
        text="p",
    )


# Bus

@pytest.mark.parametrize(
    ("sender", "recipient"),
    [
        ("agent_a", "human_b"),
        ("agent_b", "human_a"),
        ("human_a", "human_b"),
        ("human_a", "agent_b"),
    ],
)
def test_bus_rejects_edges_outside_topology(sender, recipient):
    rec = ListRecorder()
    bus = MessageBus(rec, max_negotiation_rounds=4)
    with pytest.raises(BusViolation, match="topology"):
        bus.accept(human_msg("m1", sender, recipient))
    [violation] = rec.of_kind("violation")
    assert violation["rule"] == "topology"
    assert (violation["from"], violation["to"]) == (sender, recipient)
    assert rec.of_kind("message") == []
    assert bus.deliver() == {} and bus.log == []


async def test_comms_tool_rejected_by_bus_gets_error_result_and_violation():
    """Agent A mis-wired to Human B: the send is refused at the bus, the model sees an error."""
    rec = ListRecorder()
    bus = MessageBus(rec, max_negotiation_rounds=4)
    outbox = Outbox("agent_a", "c", gate=bus.accept)
    tools = {
        f.__name__: LocalToolFactory("agent_a", rec).make(f)
        for f in comms_functions("agent_a", "human_b", outbox)
    }
    result = await tools["send_to_human"].ainvoke(
        {"name": "send_to_human", "args": {"text": "hi Bob"}, "id": "t1", "type": "tool_call"}
    )
    assert result.status == "error" and "topology" in result.content
    [violation] = rec.of_kind("violation")
    assert (violation["actor"], violation["from"], violation["to"]) == (
        "agent_a",
        "agent_a",
        "human_b",
    )
    assert outbox.peek() == [] and outbox.next_msg_id() == "agent_a-1"  # no id consumed
    assert bus.deliver() == {}


def test_bus_revalidates_final_plan_payload():
    bus = MessageBus(ListRecorder(), max_negotiation_rounds=4)
    forged = HumanMessage.model_construct(
        msg_id="m1",
        sender="agent_a",
        recipient="human_a",
        kind="final_plan",
        text="x",
        final_plan=None,
    )
    with pytest.raises(BusViolation, match="invalid_payload"):
        bus.accept(forged)
    bus.accept(human_msg("m2", "agent_a", "human_a", kind="final_plan", final_plan=PLAN))
    with pytest.raises(BusViolation, match="only an agent"):
        bus.accept(human_msg("m3", "human_a", "agent_a", kind="final_plan", final_plan=PLAN))


def test_bus_rejects_duplicate_ids_and_proposals_beyond_limit():
    rec = ListRecorder()
    bus = MessageBus(rec, max_negotiation_rounds=2)
    bus.accept(proposal("p1"))
    with pytest.raises(BusViolation, match="duplicate_msg_id"):
        bus.accept(proposal("p1"))
    bus.accept(proposal("p2"))
    with pytest.raises(BusViolation, match="negotiation_limit"):
        bus.accept(proposal("p3"))
    bus.accept(inform("i1"))  # other performatives are not limited
    assert [v["rule"] for v in rec.of_kind("violation")] == [
        "duplicate_msg_id",
        "negotiation_limit",
    ]


def test_bus_delivers_in_sender_canonical_then_send_order():
    bus = MessageBus(ListRecorder(), max_negotiation_rounds=4)
    bus.accept(inform("b1", "agent_b", "agent_a"))
    bus.accept(human_msg("h1", "human_a", "agent_a"))
    bus.accept(inform("b2", "agent_b", "agent_a"))
    bus.accept(human_msg("hb", "human_b", "agent_b"))
    inboxes = bus.deliver()
    assert [m.msg_id for m in inboxes["agent_a"]] == ["h1", "b1", "b2"]
    assert [m.msg_id for m in inboxes["agent_b"]] == ["hb"]
    assert not bus.has_pending()


# Trajectory recorder

def test_flush_orders_actors_canonically_and_parallel_tool_events_by_slot():
    rec = TrajectoryRecorder(clock=iter(range(100)).__next__)
    rec.record("agent_b", "turn_start")
    rec.record("runtime", "error", error="x")
    rec.record("agent_a", "llm_call")

    def in_slot(slot: int, kind: str, call_id: str) -> None:
        token = tool_slot.set(slot)
        rec.record("agent_a", kind, call_id=call_id)
        tool_slot.reset(token)

    # Completion order of two parallel calls: c2 finishes first, then c1 retries.
    in_slot(1, "tool_call", "c2")
    in_slot(0, "tool_call", "c1")
    in_slot(1, "tool_result", "c2")
    in_slot(0, "retry", "c1")
    in_slot(0, "tool_result", "c1")
    rec.record("agent_a", "llm_call")
    rec.record("human_a", "message")

    events = rec.flush(3)
    assert [(e["actor"], e["kind"], e.get("call_id")) for e in events] == [
        ("human_a", "message", None),
        ("agent_a", "llm_call", None),
        ("agent_a", "tool_call", "c1"),
        ("agent_a", "retry", "c1"),
        ("agent_a", "tool_result", "c1"),
        ("agent_a", "tool_call", "c2"),
        ("agent_a", "tool_result", "c2"),
        ("agent_a", "llm_call", None),
        ("agent_b", "turn_start", None),
        ("runtime", "error", None),
    ]
    assert [e["seq"] for e in events] == list(range(1, 11))
    assert {e["tick"] for e in events} == {3}
    assert rec.flush(4) == []


def test_check_trajectory_rejects_gaps_and_unknown_kinds():
    good = {
        "run_id": "r",
        "scenario": "s",
        "seed": 1,
        "config_snapshot": {},
        "events": [{"tick": 0, "seq": 1, "actor": "human_a", "kind": "message"}],
        "termination": {"reason": "success", "tick": 0},
        "outcome": {},
        "metrics": {},
    }
    check_trajectory(good)
    with pytest.raises(ValueError, match="gap-free"):
        check_trajectory(good | {"events": [{**good["events"][0], "seq": 2}]})
    with pytest.raises(ValueError, match="bad kind"):
        check_trajectory(good | {"events": [{**good["events"][0], "kind": "chat"}]})
    with pytest.raises(ValueError, match="missing"):
        check_trajectory({k: v for k, v in good.items() if k != "termination"})


# Simulated humans


async def test_scripted_bob_consents_reveals_hidden_fact_rents_and_accepts(scenario):
    bob = ScriptedHumanProvider(scenario.humans["human_b"])
    ask = human_msg("q1", "agent_b", "human_b", kind="question").model_copy(
        update={"text": "May I share your limits? Any restrictions?"}
    )
    reply = await bob.reply([ask])
    assert "share" in reply.text and "afraid of heights" in reply.text
    assert reply.accepts_final_plan is None

    rent = ask.model_copy(update={"text": "You miss poles and gaiters. Shall I rent them?"})
    reply = await bob.reply([rent])
    assert reply.text == "Yes, please rent the missing gear for me."

    info = human_msg("i1", "agent_b", "human_b")
    assert (await bob.reply([info])).text == "Okay, thanks."

    final = human_msg("f1", "agent_a", "human_a", kind="final_plan", final_plan=PLAN)
    reply = await ScriptedHumanProvider(scenario.humans["human_a"]).reply([final])
    assert reply.accepts_final_plan is True


def _llm_human(scenario, script: list) -> tuple[LLMHumanProvider, ScriptedChatModel, ListRecorder]:
    rec = ListRecorder()
    llm = ScriptedChatModel(script=script)
    provider = LLMHumanProvider(
        "human_a",
        scenario.humans["human_a"],
        llm,
        "You are {name}. {style}\n{known_facts}\n{hidden_facts}\n{decisions}",
        retry=NO_WAIT,
        recorder=rec,
    )
    return provider, llm, rec


async def test_llm_human_returns_structured_reply_and_keeps_history(scenario):
    answer = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "HumanReply",
                "args": {"text": "Perfect, thanks!", "accepts_final_plan": True},
                "id": "h1",
                "type": "tool_call",
            }
        ],
        usage_metadata={"input_tokens": 50, "output_tokens": 7, "total_tokens": 57},
    )
    provider, llm, rec = _llm_human(scenario, [answer])
    final = human_msg("f1", "agent_a", "human_a", kind="final_plan", final_plan=PLAN)
    reply = await provider.reply([final])

    assert (reply.text, reply.accepts_final_plan) == ("Perfect, thanks!", True)
    system, user = llm.seen[0]
    assert system.content.startswith("You are Alice.") and "vega-de-ario" in user.content
    [call] = rec.of_kind("llm_call")
    assert (call["actor"], call["input_tokens"], call["output_tokens"]) == ("human_a", 50, 7)
    assert [m.content for m in provider.history][1] == "Perfect, thanks!"


async def test_llm_human_without_structured_reply_raises(scenario):
    provider, _, _ = _llm_human(scenario, [AIMessage(content="sure")])
    with pytest.raises(HumanReplyError):
        await provider.reply([human_msg("i1", "agent_a", "human_a")])
