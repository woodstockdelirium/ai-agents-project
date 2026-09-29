"""The classifier and the policy layer. TODO 2 and 3.

Two calls per request in the routed system: one to decide, one to answer.
The first is cheap and its output is inspectable. The second is the work.

The policy layer between them is about three lines of code and three design
decisions, and the decisions are where the marks are.
"""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from routes import ROUTES, SPECIALISTS, SYSTEM_ROUTER

from project.models import BASE_URL, API_KEY, SMALL


class Decision(BaseModel):
    route: Literal["request", "info", "status", "complaint", "other"]
    confidence: float = Field(ge=0, le=1)
    evidence: str = Field(max_length=200)


class Routed(BaseModel):
    """What the policy layer decided, and why."""

    decision: Decision
    applied_route: str
    policy_fired: str | None = None      # None means the decision stood
    evidence_ok: bool = True


# --------------------------------------------------------------------------
# TODO 2. The classifying call.
# --------------------------------------------------------------------------

def classify(client, text: str, model: str = SMALL.name,
             temperature: float = 0.0) -> tuple[Decision | None, dict]:
    """One cheap call whose only job is to pick a route."""
    t0 = time.perf_counter()
    try:
        reply = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=200,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "decision",
                    "schema": Decision.model_json_schema(),
                },
            },
            messages=[
                {"role": "system", "content": SYSTEM_ROUTER},
                {"role": "user", "content": text},
            ],
        )
        raw = reply.choices[0].message.content
        meta = {
            "seconds": time.perf_counter() - t0,
            "prompt_tokens": reply.usage.prompt_tokens,
            "completion_tokens": reply.usage.completion_tokens,
            "raw": raw,
            "error": None,
        }
        try:
            return Decision.model_validate_json(raw), meta
        except ValidationError as exc:
            meta["error"] = str(exc).splitlines()[0]
            return None, meta
    except Exception as exc:
        return None, {
            "seconds": time.perf_counter() - t0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "raw": "",
            "error": str(exc),
        }


# --------------------------------------------------------------------------
# TODO 3. The policy layer.
# --------------------------------------------------------------------------

CONFIDENCE_FLOOR = 0.01
SAFE_DEFAULT = "info"


def apply_policy(decision: Decision | None, text: str) -> Routed:
    """Take a Decision and return what will actually happen."""
    if decision is None:
        return Routed(
            decision=Decision(route=SAFE_DEFAULT, confidence=0.0, evidence=""),
            applied_route=SAFE_DEFAULT,
            policy_fired="invalid_decision",
            evidence_ok=False,
        )

    evidence_ok = bool(decision.evidence and decision.evidence in text)
    if not evidence_ok:
        return Routed(
            decision=decision,
            applied_route=SAFE_DEFAULT,
            policy_fired="evidence_failed",
            evidence_ok=False,
        )

    if decision.confidence < CONFIDENCE_FLOOR:
        return Routed(
            decision=decision,
            applied_route=SAFE_DEFAULT,
            policy_fired="low_confidence",
            evidence_ok=evidence_ok,
        )

    return Routed(
        decision=decision,
        applied_route=decision.route,
        policy_fired=None,
        evidence_ok=evidence_ok,
    )


# --------------------------------------------------------------------------
# Given. The answering call.
# --------------------------------------------------------------------------

def respond(client, system: str, text: str,
            model: str = SMALL.name) -> tuple[str, dict]:
    t0 = time.perf_counter()
    reply = client.chat.completions.create(
        model=model, temperature=0.0, max_tokens=250,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": text}])
    return reply.choices[0].message.content, {
        "seconds": time.perf_counter() - t0,
        "prompt_tokens": reply.usage.prompt_tokens,
        "completion_tokens": reply.usage.completion_tokens,
    }