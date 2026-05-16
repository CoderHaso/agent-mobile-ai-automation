"""Executor Agent — adaptive ReAct loop.

Unlike a rigid step-runner, this executor does NOT iterate the milestone
list one entry at a time. It drives a single Observe → Reason → Act loop
that, on every tick, asks the LLM:

    "Given the GOAL, the MILESTONES (with their current status), and the
     LIVE SCREEN you see right now, what is the SINGLE next thing I should
     do — and which milestone, if any, did the previous action just complete?"

This means:
  - The agent skips milestones that the actual flow doesn't ask for
    (e.g. recovery email when Google didn't prompt).
  - The agent inserts ad-hoc RECOVERY actions when an unexpected popup
    blocks progress (e.g. a Galaxy Store overlay over Gmail).
  - The agent stops the moment it concludes the GOAL is complete —
    not when the milestone list is exhausted.
  - The same plan works across Samsung / Xiaomi / Huawei / stock because
    each tap is decided from the actual XML on screen.

Safeguards:
  - Hard cap on iterations (`config.max_iterations`).
  - Stall detection: same screen hash N times in a row → escalate / abort.
  - Per-decision retry on malformed JSON / LLM timeouts.
  - Cooperative cancellation via `request_stop()`.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Set

from pydantic import BaseModel, Field, ValidationError, field_validator

from .device_manager import (
    ActionExecutionError,
    DeviceManager,
    Observation,
)
from .llm_client import LLMClient, LLMResponseError
from .planner import Plan, PlanStep
from .ui_parser import parse_hierarchy, summarize_for_prompt
from .watchers import WatcherManager

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #

EXECUTOR_SYSTEM_PROMPT = """\
You are a domain-agnostic autonomous Android UI agent running an
Observe → Reason → Act loop. Your purpose is to achieve WHATEVER goal
the user gave you, no matter the app or domain.

═════════════════════════ PRIMARY DIRECTIVE ═════════════════════════
The CURRENT_SCREEN and the provided SCREENSHOT are your GROUND TRUTH. 
Pick the SINGLE next action that most directly progresses the GOAL based 
on WHAT YOU CAN SEE RIGHT NOW. Do NOT cling to a milestone if the screen 
is asking for something different — REACT TO THE SCREEN.

Use the provided screenshot to VERIFY the state of elements (e.g., if a 
checkbox is already checked, or a text field is already focused/filled) 
that might not be fully clear from the JSON CURRENT_SCREEN list alone.

How to think every iteration:
  1. Read the GOAL.
  2. Scan CURRENT_SCREEN and the screenshot: what elements are actually available?
  3. Ask "does any visible element directly advance the GOAL?"
     → If yes, take that action. Don't look further.
  4. Ask "does any visible element move me deeper into the flow?"
     (forward navigation, "Next", "Continue", search, etc.)
     → Take it.
  5. Otherwise: scroll to reveal more, or press back to escape.

Adaptive behavior examples (abstract — apply to any domain):
  • The active milestone is "Pick a value X" but the screen is asking
    for a DIFFERENT input first → provide that input. The flow has
    reordered; the milestone will still get done later.
  • You see a shortcut button that jumps straight to the GOAL's end
    state (e.g. "Send now", "Order", "Save") → click it instead of
    walking through every intermediate milestone.
  • A screen offers multiple paths (a list of options) → pick the one
    whose label most clearly matches the GOAL's subject. If none match,
    pick the most generic forward option and back out if it's wrong.
  • MISSING APPS: If your goal is to open or use an app, but it is not
    installed on the device, do NOT repeatedly search for it or try to
    open its package. Instead, actively click 'Install', 'Get', or Google Play
    results to install it, or use a web browser alternative if applicable.
    Recognize app store listings in search results and click them.

MILESTONES are a destination MAP (where we're going), NOT a SCRIPT
(how to get there). Mark them done in any order as you complete them.
Mark optional milestones as `skipped` if the actual flow doesn't ask
for them.

═════════════════════════ ANTI-LOOP RULES ═══════════════════════════
THESE ARE CRITICAL. Violating them causes the agent to hang.

1. FORBIDDEN_TARGETS lists EXACT element labels that have already been
   tried on THIS screen and led nowhere (screen didn't change, or you
   came back from a dead-end). NEVER click anything in this list.
   Pick a DIFFERENT element on the same screen, or scroll, or press back.

2. Look at RECENT_ACTIONS. If you JUST pressed `back` to escape a
   useless screen, do NOT immediately re-enter that screen by clicking
   the same thing that took you there.

3. NEVER repeat the SAME action with the SAME target twice in a row.

4. STALL_HINT is set when the screen has been the same for several
   iterations. That means YOU ARE LOOPING. Try, in this priority:
     a. A completely different element on screen (often the second-most
        promising one).
     b. Scroll to reveal off-screen elements.
     c. Press back to escape.

═════════════════════════ POPUP / OVERLAY ═══════════════════════════
If CURRENT_APP is NOT the app your GOAL is about (e.g. a store /
launcher / ad / system update banner / OEM nag pops over the target
app), set `is_recovery=true` and dismiss the overlay:
  - close button / X / "Cancel" / "İptal" / "Not now" / "Sonra" /
    "Later" / press back.
Do NOT update milestones during recovery.

═════════════════════════ ELEMENT PRIORITY ══════════════════════════
When scanning CURRENT_SCREEN, prefer in this order:
  1. Elements whose label DIRECTLY matches the GOAL's subject
     (the search term, the recipient name, the setting being toggled,
      the product, the menu entry, etc.).
  2. Empty input fields that the GOAL implies you should fill — use
     real values when provided, otherwise placeholders like
     <USER_NAME>, <SEARCH_TERM>, <MESSAGE>.
  3. Generic forward-navigation verbs (multilingual):
       EN: Continue, Next, OK, Allow, Done, Submit, Send, Save,
           Confirm, Apply, Search, Get started
       TR: Devam, İleri, Tamam, İzin ver, Bitir, Gönder, Kaydet,
           Onayla, Uygula, Ara, Başla
  4. Scroll / navigation drawers to reveal more elements.
Use resource_id when present, otherwise text, otherwise content_desc.

═════════════════════════ INPUTS PER ITERATION ══════════════════════
  • GOAL              the high-level user objective (THIS is what matters)
  • MILESTONES        objectives with status — guideline only, not a script
  • CURRENT_APP       foreground package name
  • CURRENT_SCREEN    interactable elements actually on screen now
  • RECENT_ACTIONS    last few actions with outcome (and whether the
                      screen changed)
  • FORBIDDEN_TARGETS targets that already failed on THIS screen — never
                      click these
  • STALL_HINT        non-empty if you have been looping

═════════════════════════ OUTPUT (strict JSON) ══════════════════════
Respond with ONE JSON object — no prose, no markdown fences:

{
  "thought": "<one short sentence — why THIS action right now>",
  "screen_summary": "<3-7 words describing what this screen is>",
  "goal_complete": false,
  "active_milestone_id": <int or null>,
  "milestone_updates": [
    {"id": 1, "status": "done"},
    {"id": 6, "status": "skipped"}
  ],
  "is_recovery": false,
  "action": {
    "kind": "click | type | press | open_app | scroll_to | wait | back | done | give_up",
    "target": "<text | resource_id | content_desc | index | system key | package | empty>",
    "target_kind": "text | resource_id | content_desc | index | key | package | none",
    "input_value": "<only when kind=='type'>"
  }
}

Action rules:
  • For `press`, target ∈ {back, home, enter, menu, recent}.
  • For `open_app`, target is an Android package name.
  • For `index` target_kind, target is the integer index of the element from CURRENT_SCREEN. Use this as a bulletproof fallback when an element has no clear text label or content_desc.
  • Use `done` when goal_complete=true.
  • Use `give_up` only after multiple failed attempts AND no plausible
    element exists AND scrolling/back also failed.
"""


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

_VALID_ACTIONS = {
    "click", "type", "press", "open_app", "scroll_to",
    "wait", "back", "done", "give_up",
}
_VALID_TARGET_KINDS = {"text", "resource_id", "content_desc", "key", "package", "none", "index"}
_VALID_MILESTONE_STATUSES = {"done", "skipped", "failed", "pending", "running"}


class ActionSpec(BaseModel):
    kind: str
    target: str = ""
    target_kind: str = "text"
    input_value: str = ""

    @field_validator("kind")
    @classmethod
    def _vk(cls, v: str) -> str:
        v = (v or "").lower().strip()
        if v not in _VALID_ACTIONS:
            raise ValueError(f"action.kind must be one of {sorted(_VALID_ACTIONS)}, got {v!r}")
        return v

    @field_validator("target_kind")
    @classmethod
    def _vt(cls, v: str) -> str:
        v = (v or "text").lower().strip()
        return v if v in _VALID_TARGET_KINDS else "text"


class MilestoneUpdate(BaseModel):
    id: int
    status: str

    @field_validator("status")
    @classmethod
    def _vs(cls, v: str) -> str:
        v = (v or "").lower().strip()
        return v if v in _VALID_MILESTONE_STATUSES else "pending"


class AgentDecision(BaseModel):
    thought: str = ""
    screen_summary: str = ""
    goal_complete: bool = False
    active_milestone_id: Optional[int] = None
    milestone_updates: List[MilestoneUpdate] = Field(default_factory=list)
    is_recovery: bool = False
    action: ActionSpec


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #

@dataclass
class ActionRecord:
    decision: AgentDecision
    success: bool
    note: str = ""

    def short(self) -> Dict[str, Any]:
        return {
            "thought": self.decision.thought,
            "kind": self.decision.action.kind,
            "target": self.decision.action.target,
            "target_kind": self.decision.action.target_kind,
            "is_recovery": self.decision.is_recovery,
            "ok": self.success,
            "note": self.note,
        }


@dataclass
class StepResult:
    """Per-milestone summary surfaced to the GUI / CLI."""

    step: PlanStep
    success: bool
    attempts: int = 0
    actions: List[ActionRecord] = field(default_factory=list)
    note: str = ""


@dataclass
class ExecutorConfig:
    max_iterations: int = 80           # hard cap on Observe→Reason→Act ticks
    stall_threshold: int = 4           # same screen hash N times => stall hint
    stall_abort_threshold: int = 8     # ... after this many => give up
    settle_seconds: float = 1.2        # wait between act and observe
    history_window: int = 6            # recent actions sent to the LLM
    max_elements_in_prompt: int = 70


# --------------------------------------------------------------------------- #
# Executor
# --------------------------------------------------------------------------- #

ProgressCallback = Callable[[PlanStep, str], None]
LogCallback = Callable[[str], None]


class Executor:
    def __init__(
        self,
        device: DeviceManager,
        llm: LLMClient,
        watchers: WatcherManager,
        config: Optional[ExecutorConfig] = None,
        on_progress: Optional[ProgressCallback] = None,
        on_log: Optional[LogCallback] = None,
    ):
        self.device = device
        self.llm = llm
        self.watchers = watchers
        self.config = config or ExecutorConfig()
        self.on_progress = on_progress or (lambda step, msg: None)
        self.on_log = on_log or (lambda msg: None)
        self._stop_requested = False

    def request_stop(self) -> None:
        """Cooperative cancel — safe from any thread."""
        self._stop_requested = True

    # --------------------------------------------------------------------- #
    # Main loop
    # --------------------------------------------------------------------- #

    def run(self, plan: Plan) -> List[StepResult]:
        self._stop_requested = False
        recent: Deque[ActionRecord] = deque(maxlen=self.config.history_window)

        # Per-screen dead-end memory:
        #   screen_hash -> {target labels that did NOT advance from this screen}
        # We feed this back to the LLM as FORBIDDEN_TARGETS so it must pick
        # something else when it returns to a screen it has seen before.
        dead_ends: Dict[str, Set[str]] = defaultdict(set)

        last_obs: Optional[Observation] = None
        last_action: Optional[ActionSpec] = None
        same_screen_count = 0

        per_step: Dict[int, StepResult] = {
            s.step_id: StepResult(step=s, success=False) for s in plan.steps
        }

        self.watchers.start()
        try:
            for it in range(1, self.config.max_iterations + 1):
                if self._stop_requested:
                    self.on_log("● Stopped by user.")
                    break

                # --- 1. OBSERVE ---
                self.device.wait(self.config.settle_seconds)
                try:
                    obs = self.device.observe()
                except Exception as exc:
                    self.on_log(f"observe failed: {exc}")
                    continue

                # --- 1b. DEAD-END LEARNING -------------------------------- #
                # If our last action was a click and the screen DID NOT
                # change, that target is a dead end on the screen we were on.
                if (last_obs is not None and last_action is not None
                        and last_action.kind == "click" and last_action.target
                        and obs.screen_hash == last_obs.screen_hash):
                    dead_ends[last_obs.screen_hash].add(last_action.target)
                    self.on_log(
                        f"  ⊘ marked dead-end on this screen: {last_action.target!r}"
                    )

                # If our last action was `back`, whatever click took us into
                # the screen we just escaped from is a dead end on the screen
                # we are now on.
                if (last_obs is not None and last_action is not None
                        and last_action.kind in ("back", "press")
                        and obs.screen_hash != last_obs.screen_hash
                        and len(recent) >= 2):
                    prior = recent[-2].decision.action
                    if prior.kind == "click" and prior.target:
                        dead_ends[obs.screen_hash].add(prior.target)
                        self.on_log(
                            f"  ⊘ marked dead-end after back: {prior.target!r}"
                        )

                # Stall counter
                if last_obs is not None and obs.screen_hash == last_obs.screen_hash:
                    same_screen_count += 1
                else:
                    same_screen_count = 0

                # --- 2. REASON ---
                stall_hint = ""
                if same_screen_count >= self.config.stall_threshold:
                    stall_hint = (
                        f"You have been on the SAME screen for "
                        f"{same_screen_count} iterations — STOP repeating "
                        "yourself. Pick a DIFFERENT element, scroll, or "
                        "press back."
                    )

                forbidden = sorted(dead_ends.get(obs.screen_hash, set()))

                try:
                    decision = self._reason(
                        plan, obs, list(recent), stall_hint, forbidden,
                    )
                except LLMResponseError as exc:
                    self.on_log(f"reasoner error: {exc}")
                    continue

                # Hard guardrail: if the LLM tries a forbidden target anyway,
                # patch the action to a recovery (press back) and record it.
                if (decision.action.kind == "click"
                        and decision.action.target in forbidden):
                    self.on_log(
                        f"  ⚠ LLM picked a forbidden target {decision.action.target!r}; "
                        "forcing back instead."
                    )
                    decision.action = ActionSpec(kind="back", target="back",
                                                target_kind="key")

                self._log_decision(it, decision, obs)

                # --- 3. APPLY MILESTONE UPDATES ---
                self._apply_milestone_updates(plan, decision, per_step)

                if decision.active_milestone_id is not None:
                    self._mark_running(plan, decision.active_milestone_id, per_step)

                # --- 4. GOAL COMPLETION CHECK ---
                if decision.goal_complete or decision.action.kind == "done":
                    self.on_log("✓ Agent reports the GOAL is complete.")
                    self._finalize_remaining(plan, per_step, success=True)
                    break

                if decision.action.kind == "give_up":
                    self.on_log("✗ Agent gave up.")
                    self._finalize_remaining(plan, per_step, success=False, note="gave up")
                    break

                # --- 5. ACT ---
                rec = self._act_with_record(decision)
                recent.append(rec)
                if rec.decision.active_milestone_id is not None:
                    per_step[rec.decision.active_milestone_id].actions.append(rec)
                    per_step[rec.decision.active_milestone_id].attempts += 1

                last_obs = obs
                last_action = decision.action

                # --- 6. STALL ABORT ---
                if same_screen_count >= self.config.stall_abort_threshold:
                    self.on_log(
                        f"✗ Aborting — stuck on the same screen for "
                        f"{same_screen_count} iterations."
                    )
                    self._finalize_remaining(
                        plan, per_step, success=False, note="stalled"
                    )
                    break
            else:
                self.on_log(
                    f"✗ Hit max_iterations={self.config.max_iterations} "
                    "without completing the goal."
                )
                self._finalize_remaining(
                    plan, per_step, success=False,
                    note="max iterations reached",
                )
        finally:
            self.watchers.stop()

        return list(per_step.values())

    # --------------------------------------------------------------------- #
    # Reasoning
    # --------------------------------------------------------------------- #

    def _reason(
        self,
        plan: Plan,
        obs: Observation,
        recent: List[ActionRecord],
        stall_hint: str,
        forbidden_targets: List[str],
    ) -> AgentDecision:
        elements = parse_hierarchy(obs.xml, include_text_labels=True)
        slim = summarize_for_prompt(
            elements, max_elements=self.config.max_elements_in_prompt
        )

        milestones_payload = [
            {
                "id": s.step_id,
                "objective": s.action_description,
                "expected_outcome": s.expected_outcome,
                "is_optional": s.is_optional,
                "status": s.status,
            }
            for s in plan.steps
        ]

        recent_payload = [r.short() for r in recent]

        # Capture screenshot for vision support
        image_base64 = None
        try:
            img = self.device.d.screenshot()
            import io
            import base64
            import os
            
            # Save for GUI live view
            os.makedirs("ui_dumps", exist_ok=True)
            img.save("ui_dumps/current.png")
            self.on_log("Captured screenshot: ui_dumps/current.png")
            
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode()
        except Exception as exc:
            log.warning("Failed to capture screenshot for reasoning: %s", exc)

        user_prompt = (
            f"GOAL: {plan.goal}\n"
            f"MILESTONES: {json.dumps(milestones_payload, ensure_ascii=False)}\n"
            f"CURRENT_APP: {obs.current_app or 'unknown'}\n"
            f"CURRENT_SCREEN: {json.dumps(slim, ensure_ascii=False)}\n"
            f"RECENT_ACTIONS: {json.dumps(recent_payload, ensure_ascii=False)}\n"
            f"FORBIDDEN_TARGETS: {json.dumps(forbidden_targets, ensure_ascii=False)}\n"
            f"STALL_HINT: {stall_hint}\n\n"
            "Remember: react to WHAT YOU SEE on this screen. Do NOT click "
            "anything in FORBIDDEN_TARGETS. Respond with ONE JSON decision now."
        )

        raw = self.llm.complete_json(
            system_prompt=EXECUTOR_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            image_base64=image_base64,
            temperature=0.15,
        )
        if isinstance(raw, list):
            raw = raw[0] if raw else {}

        try:
            return AgentDecision.model_validate(raw)
        except ValidationError as exc:
            raise LLMResponseError(f"Invalid decision schema: {exc}") from exc

    # --------------------------------------------------------------------- #
    # Acting
    # --------------------------------------------------------------------- #

    def _act_with_record(self, decision: AgentDecision) -> ActionRecord:
        try:
            self._act(decision.action)
            return ActionRecord(decision=decision, success=True, note="ok")
        except ActionExecutionError as exc:
            self.on_log(f"  action failed: {exc}")
            return ActionRecord(decision=decision, success=False, note=str(exc))

    def _act(self, action: ActionSpec) -> None:
        kind, t, kt, val = (
            action.kind, action.target.strip(), action.target_kind, action.input_value
        )

        if kind == "wait":
            self.device.wait(2.0)
            return
        if kind in ("press", "back"):
            key = "back" if kind == "back" else (t or "back").lower()
            if not self.device.press(key):
                raise ActionExecutionError(f"press({key}) failed")
            return
        if kind == "open_app":
            if not t:
                raise ActionExecutionError("open_app requires a package target")
            if not self.device.open_app(t):
                raise ActionExecutionError(f"open_app({t}) failed")
            return
        if kind == "scroll_to":
            if not t:
                raise ActionExecutionError("scroll_to requires a target text")
            if not self.device.scroll_to_text(t):
                raise ActionExecutionError(f"scroll_to({t}) failed")
            return
        if kind == "click":
            if not t:
                raise ActionExecutionError("click requires a target")
            if not self._dispatch_click(t, kt):
                raise ActionExecutionError(f"click target not found: {t!r} ({kt})")
            return
        if kind == "type":
            if not val:
                raise ActionExecutionError("type requires input_value")
            if not self.device.type_into(t or "", val):
                raise ActionExecutionError(f"type into {t!r} failed")
            return

        raise ActionExecutionError(f"unsupported action: {kind}")

    def _dispatch_click(self, target: str, kind: str) -> bool:
        d = self.device
        
        # 1. Click by parsed element index (BULLETPROOF AI FALLBACK)
        if kind == "index":
            try:
                from .ui_parser import parse_hierarchy
                idx = int(target.strip())
                obs = self.device.observe() # fresh observe
                elements = parse_hierarchy(obs.xml, include_text_labels=True)
                for e in elements:
                    if e.index == idx:
                        import re
                        match = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", e.bounds)
                        if match:
                            x1, y1, x2, y2 = map(int, match.groups())
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                            d.d.click(cx, cy)
                            return True
            except Exception as exc:
                log.warning("Click by index failed: %s", exc)
            return False

        # 2. Standard uiautomator2 selectors
        if kind == "resource_id":
            if d.click_resource_id(target) or d.click_text(target):
                return True
        elif kind == "content_desc":
            if d.click_description(target) or d.click_text(target):
                return True
        else:
            if (d.click_text(target)
                or d.click_resource_id(target)
                or d.click_description(target)):
                return True

        # BULLETPROOF FALLBACK: Parse XML and click coordinates
        try:
            import xml.etree.ElementTree as ET
            import re
            xml_data = d.d.dump_hierarchy()
            root = ET.fromstring(xml_data.encode('utf-8', 'ignore'))
            
            target_lower = target.lower()
            best_bounds = None
            for node in root.iter("node"):
                text = (node.get("text") or "").strip().lower()
                desc = (node.get("content-desc") or "").strip().lower()
                rid = (node.get("resource-id") or "").strip().lower()
                hint = (node.get("hint") or "").strip().lower()
                
                if (target_lower == text or target_lower == desc or target_lower == rid or target_lower == hint or
                    target_lower in text or target_lower in desc or target_lower in hint):
                    bounds = node.get("bounds")
                    if bounds and bounds != "[0,0][0,0]":
                        best_bounds = bounds
                        break
            
            if best_bounds:
                match = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", best_bounds)
                if match:
                    x1, y1, x2, y2 = map(int, match.groups())
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    d.d.click(cx, cy)
                    return True
        except Exception as exc:
            log.warning("Coordinate click fallback failed: %s", exc)
            
        return False

    # --------------------------------------------------------------------- #
    # Milestone bookkeeping
    # --------------------------------------------------------------------- #

    def _apply_milestone_updates(
        self,
        plan: Plan,
        decision: AgentDecision,
        per_step: Dict[int, StepResult],
    ) -> None:
        for upd in decision.milestone_updates:
            step = self._find_step(plan, upd.id)
            if step is None:
                self.on_log(f"  ⚠ unknown milestone id in update: {upd.id}")
                continue
            if upd.status == "done":
                if step.status != "done":
                    step.status = "done"
                    per_step[step.step_id].success = True
                    self.on_progress(step, "done")
                    self.on_log(f"  ✓ milestone {step.step_id} done — {step.action_description}")
            elif upd.status == "skipped":
                if step.status not in ("done", "failed"):
                    step.status = "skipped"
                    per_step[step.step_id].success = step.is_optional
                    per_step[step.step_id].note = "skipped (optional / not required by flow)"
                    self.on_progress(step, "skipped")
                    self.on_log(f"  ⤼ milestone {step.step_id} skipped — {step.action_description}")
            elif upd.status == "failed":
                step.status = "failed"
                per_step[step.step_id].success = False
                self.on_progress(step, "failed")

    def _mark_running(self, plan: Plan, mid: int, per_step: Dict[int, StepResult]) -> None:
        step = self._find_step(plan, mid)
        if step is None:
            return
        if step.status in ("done", "failed", "skipped"):
            return
        # Reset previously running milestone (if any) back to pending.
        for s in plan.steps:
            if s.step_id != mid and s.status == "running":
                s.status = "pending"
                self.on_progress(s, "pending")
        if step.status != "running":
            step.status = "running"
            self.on_progress(step, "running")

    @staticmethod
    def _find_step(plan: Plan, step_id: int) -> Optional[PlanStep]:
        for s in plan.steps:
            if s.step_id == step_id:
                return s
        return None

    def _finalize_remaining(
        self,
        plan: Plan,
        per_step: Dict[int, StepResult],
        success: bool,
        note: str = "",
    ) -> None:
        """Close the books on milestones that never got a terminal status."""
        for s in plan.steps:
            if s.status in ("done", "failed", "skipped"):
                continue
            if success:
                s.status = "done"
                per_step[s.step_id].success = True
                per_step[s.step_id].note = note or "implicitly satisfied"
            else:
                s.status = "skipped" if s.is_optional else "failed"
                per_step[s.step_id].success = s.is_optional
                per_step[s.step_id].note = note or "not reached"
            self.on_progress(s, s.status)

    # --------------------------------------------------------------------- #
    # Logging helpers
    # --------------------------------------------------------------------- #

    def _log_decision(self, it: int, d: AgentDecision, obs: Observation) -> None:
        recovery_tag = "[recovery] " if d.is_recovery else ""
        active = (
            f"M{d.active_milestone_id}"
            if d.active_milestone_id is not None else "—"
        )
        thought = (d.thought or "").strip().replace("\n", " ")
        screen = (d.screen_summary or "").strip().replace("\n", " ")
        screen_tag = f"[{screen}] " if screen else ""
        self.on_log(
            f"#{it:02d}  app={obs.current_app or '?'}  active={active}  "
            f"{screen_tag}{recovery_tag}{d.action.kind}→{d.action.target!r}  "
            f"({thought})"
        )
