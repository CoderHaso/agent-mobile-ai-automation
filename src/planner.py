"""Planner Agent (Strategist).

Takes a free-form, high-level goal from the user (e.g. "Create a new Google
account") and asks the LLM to decompose it into a small list of MILESTONES
— ordered objectives, NOT atomic UI actions.

Why milestones, not actions?
    Real Android flows are non-deterministic and OEM-specific:
      - Recovery email / phone steps appear only sometimes.
      - The same flow looks different on Samsung, Xiaomi, Huawei, stock.
      - Apps update their UI frequently.
    A rigid "tap → type → tap" list breaks the moment the device
    diverges from the planner's mental model. The Executor (a separate
    ReAct agent) decides the actual taps at runtime by *looking at the
    screen* and reasoning about which milestone it's currently working on.

Each milestone:
    {
        "step_id": int,                    # ordering, 1..N
        "action_description": str,         # the OBJECTIVE, e.g.
                                           # "Reach the Create Account screen"
        "expected_outcome": str,           # how to recognize success
        "is_optional": bool,               # may be skipped if flow doesn't
                                           # ask for it (e.g. recovery email)
        "status": "pending"
    }
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator

from .llm_client import LLMClient, LLMResponseError

log = logging.getLogger(__name__)


PLANNER_SYSTEM_PROMPT = """\
You are a domain-agnostic Android UI automation **strategist**.

Your job: read the user's HIGH-LEVEL GOAL — whatever it is — and break
it into 3-8 MILESTONES (broad *objectives*), not atomic UI taps.

Critical rules:
  • Tailor the milestones to THIS specific goal. Do NOT default to any
    particular template (account creation, login, etc.). A goal about
    sending a message has different milestones than a goal about
    changing a setting or buying an item.
  • Each milestone describes "what should be true when this is done",
    NOT "tap which button". The runtime agent decides the actual taps
    by looking at the live screen.
  • Use the smallest number of milestones that captures the real
    structure of the task — usually 3-6.
  • Mark `is_optional=true` for milestones that may not appear in
    every flow (confirmation popups, optional verification, etc.).
  • Use placeholders for any data the user did not provide:
    <USER_NAME>, <USER_EMAIL>, <USER_PASSWORD>, <PRODUCT_NAME>,
    <RECIPIENT>, <MESSAGE>, <AMOUNT>, etc.

Good milestone shape: imperative, single objective, one short sentence.
  ✓ "Open the target app and reach the area where the task is performed"
  ✓ "Provide any required inputs the screen asks for"
  ✓ "Trigger the final action (Save / Send / Submit / Apply / Order)"
  ✓ "Handle confirmation prompts only if they appear"

NOT a milestone (too low-level — these are runtime decisions):
  ✗ "Tap the 'Next' button"
  ✗ "Type 'John' in the first name field"
  ✗ "Scroll down 200 px"

Output — ONE JSON object, no prose, no markdown fences:
{
  "goal": "<echo of the user's goal>",
  "steps": [
    {
      "step_id": 1,
      "action_description": "<one-sentence OBJECTIVE>",
      "expected_outcome": "<how to recognize this milestone is done>",
      "is_optional": false,
      "status": "pending"
    },
    ...
  ]
}

Rules:
  • step_id starts at 1, increments by 1, no gaps.
  • status is ALWAYS "pending".
  • If the goal is impossible / unsafe, return
    {"goal":"<goal>","steps":[],"error":"<short reason>"}.

Tiny examples (DIFFERENT domains — DO NOT copy these verbatim, infer
the right shape for the user's actual goal):

Goal: "Turn on the device's dark mode"
  1. Open the Settings app
  2. Navigate to the Display section
  3. Enable the dark / night theme toggle
  4. Confirm the new theme is applied (optional)

Goal: "Search for <PRODUCT_NAME> on the shopping app and add the first result to cart"
  1. Open the shopping app
  2. Use the search field to query <PRODUCT_NAME>
  3. Open the first matching product
  4. Add it to cart
  5. Confirm the item is in the cart (optional)

Goal: "Send <MESSAGE> to <RECIPIENT> on the messaging app"
  1. Open the messaging app
  2. Open or start a conversation with <RECIPIENT>
  3. Type <MESSAGE> into the input field
  4. Send the message
  5. Verify the message appears in the conversation (optional)
"""


class PlanStep(BaseModel):
    """A single milestone in the strategy.

    NOTE: despite the legacy name, this represents a high-level OBJECTIVE
    (e.g. "Reach the Create Account screen"), not an atomic UI tap.
    """

    step_id: int = Field(ge=1)
    action_description: str = Field(min_length=3)
    expected_outcome: str = Field(default="", min_length=0)
    is_optional: bool = False
    status: str = "pending"

    @field_validator("status")
    @classmethod
    def _normalize_status(cls, v: str) -> str:
        v = (v or "").lower().strip() or "pending"
        if v not in {"pending", "running", "done", "failed", "skipped"}:
            return "pending"
        return v


class Plan(BaseModel):
    goal: str
    steps: List[PlanStep] = Field(default_factory=list)
    error: Optional[str] = None

    def renumber(self) -> "Plan":
        """Force step_ids to be 1..N regardless of what the LLM returned."""
        for i, s in enumerate(self.steps, start=1):
            s.step_id = i
        return self

    def to_serializable(self) -> Dict[str, Any]:
        return self.model_dump()


class Planner:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def build_plan(self, goal: str) -> Plan:
        if not goal or not goal.strip():
            raise ValueError("Goal must be a non-empty string.")

        user_prompt = (
            f"USER GOAL: {goal.strip()}\n\n"
            "Return the JSON plan now."
        )

        try:
            raw = self.llm.complete_json(
                system_prompt=PLANNER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.2,
            )
        except LLMResponseError as exc:
            raise LLMResponseError(f"Planner LLM call failed: {exc}") from exc

        if isinstance(raw, list):
            raw = {"goal": goal, "steps": raw}

        try:
            plan = Plan.model_validate(raw).renumber()
        except ValidationError as exc:
            raise LLMResponseError(
                f"Planner produced an invalid plan schema: {exc}"
            ) from exc

        if not plan.steps and not plan.error:
            raise LLMResponseError("Planner returned an empty plan.")

        log.info("Planner produced %d step(s) for goal=%r", len(plan.steps), goal)
        return plan
