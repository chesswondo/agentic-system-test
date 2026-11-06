"""Scripted human: a small rule table driven by the scenario persona (offline tests and runs).

Rules, applied to the messages of one activation:
1. a `final_plan` -> accept or decline per `decisions.accept_final_plan`;
2. a question about renting -> yes/no per `decisions.rent_missing_gear`;
3. a question about sharing, consent or restrictions -> consent per
   `decisions.consent_to_share`, plus every hidden fact (that is when the persona reveals it);
4. anything else -> a short acknowledgement.
The keyword lists are the rule table itself, not tunables: they define what this test double
recognises, and the LLM provider replaces it for real runs.
"""

from __future__ import annotations

from collections.abc import Sequence

from picos.domain import HumanSpec
from picos.protocol import HumanMessage, HumanReply

RENT_WORDS = ("rent",)
CONSENT_WORDS = ("share", "consent", "restriction", "limitation", "health", "concern")


def _mentions(text: str, words: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(w in lowered for w in words)


class ScriptedHumanProvider:
    def __init__(self, spec: HumanSpec) -> None:
        self.spec = spec

    async def reply(self, inbox: Sequence[HumanMessage]) -> HumanReply:
        decisions = self.spec.persona.decisions
        if any(m.kind == "final_plan" for m in inbox):
            if decisions.accept_final_plan:
                return HumanReply(text="Looks great, count me in!", accepts_final_plan=True)
            return HumanReply(
                text="Sorry, this plan does not work for me.", accepts_final_plan=False
            )

        parts: list[str] = []
        questions = [m.text for m in inbox if m.kind == "question"]
        if any(_mentions(q, CONSENT_WORDS) for q in questions):
            if decisions.consent_to_share:
                parts.append("Yes, you can share what is needed with the other assistant.")
                parts.extend(
                    f"One thing you should know: {f.text}" for f in self.spec.persona.hidden_facts
                )
            elif decisions.consent_to_share is False:
                parts.append("No, please do not share my details.")
        if any(_mentions(q, RENT_WORDS) for q in questions):
            if decisions.rent_missing_gear:
                parts.append("Yes, please rent the missing gear for me.")
            elif decisions.rent_missing_gear is False:
                parts.append("No, please do not rent anything.")
        return HumanReply(text=" ".join(parts) or "Okay, thanks.")
