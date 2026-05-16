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
from .models import find as find_model
from .planner import Plan, PlanStep
from .recorder import TaskRecorder
from .screen_context import ScreenContext, analyze_screen
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
The CURRENT_SCREEN is your GROUND TRUTH. Pick the SINGLE next action that
most directly progresses the GOAL based on WHAT YOU CAN SEE RIGHT NOW.
Do NOT cling to a milestone if the screen is asking for something
different — REACT TO THE SCREEN.

When VISION_ENABLED is true, a SCREENSHOT is also attached — use it only
then to verify checkbox state, colors, or layout not obvious from XML.
When VISION_ENABLED is false, rely on CURRENT_SCREEN + INSTALLED_APPS only.

How to think every iteration:
  1. Read the GOAL.
  2. Scan CURRENT_SCREEN and the screenshot: what elements are actually available?
  3. Ask "does any visible element directly advance the GOAL?"
     → If yes, take that action. Don't look further.
  4. Ask "does any visible element move me deeper into the flow?"
     (forward navigation, "Next", "Continue", search, etc.)
     → Take it.
  5. Otherwise: swipe launcher pages, open the app drawer, or press HOME
     (not BACK) to reset — never thrash on the notification shade.

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
  • OPENING APPS — ALWAYS consult INSTALLED_APPS first:
    - If the goal mentions an app (Gmail, WhatsApp, Settings, …), find its
      exact `package` in INSTALLED_APPS or SUGGESTED_PACKAGES.
    - Use `open_app` with that exact package string (e.g. com.google.android.gm).
    - NEVER claim an app is missing if its package appears in INSTALLED_APPS.
    - Only pursue Play Store / Install flows when the package is genuinely
      absent from INSTALLED_APPS AND you are on a store/search screen.
    - You may also launch via the launcher: tap the app icon label from
      CURRENT_SCREEN when that is faster than open_app.

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

═════════════════════════ LAUNCHER & NOTIFICATION SHADE ═══════════════
Read SCREEN_CONTEXT every turn.

  • If `is_notification_shade` is true:
    - NEVER press BACK repeatedly — that toggles the shade and wastes time.
    - Press HOME once, OR swipe UP from the bottom center to close it.
  • If `is_launcher` is true and you need an app:
    - FIRST: `open_app` with the exact package from INSTALLED_APPS /
      SUGGESTED_PACKAGES (e.g. com.google.android.gm for Gmail).
    - SECOND: click the app icon label visible on screen.
    - THIRD: click "Apps" / "Uygulamalar" to open the app drawer, OR
      `swipe` left/right for another home-screen page.
    - NEVER use `scroll_to` with a resource-id (e.g. …:id/workspace)
      or a bare number — use `swipe` left/right instead.
  • If `open_app` failed for a package listed in INSTALLED_APPS, the
    launch really failed — try another browser from INSTALLED_APPS or
    Play Store, do NOT repeat the same package endlessly.

═════════════════════════ POPUP / OVERLAY ═══════════════════════════
If CURRENT_APP is NOT the app your GOAL is about (e.g. a store /
launcher / ad / system update banner / OEM nag pops over the target
app), set `is_recovery=true` and dismiss the overlay:
  - close button / X / "Cancel" / "İptal" / "Not now" / "Sonra" /
    "Later" / press HOME (prefer HOME over BACK on launcher/shade).
Do NOT update milestones during recovery.

═════════════════════════ ELEMENT PRIORITY ══════════════════════════
When scanning CURRENT_SCREEN, prefer in this order:
  1. Elements whose label DIRECTLY matches the GOAL's subject
     (the search term, the recipient name, the setting being toggled,
      the product, the menu entry, etc.).
  2. Empty input fields that the GOAL implies you should fill — use
     real values when provided. If you must generate values (like names,
     emails, or passwords), do NOT use generic terms like 'John Doe' or
     'test123'. Generate realistic, unique, and diverse values instead!
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
  • SCREEN_CONTEXT    {is_launcher, is_notification_shade, is_app_drawer, hint}
  • INSTALLED_APPS    apps on this device relevant to the GOAL (package + label)
  • SUGGESTED_PACKAGES alias hints (keyword → package) when known
  • VISION_ENABLED    whether a screenshot is attached this turn
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
    "kind": "click | type | press | open_app | scroll_to | swipe | wait | back | done | give_up",
    "target": "<text | resource_id | content_desc | index | system key | package | empty>",
    "target_kind": "text | resource_id | content_desc | index | key | package | none",
    "input_value": "<only when kind=='type'>"
  }
}

Action rules:
  • For `press`, target ∈ {back, home, enter, menu, recent}.
  • For `swipe`, target ∈ {left, right, up, down} — launcher page turns.
  • For `open_app`, target is an Android package name.
  • For `force_stop`, target is an Android package name. Use this when an
    app is stuck, frozen, or you need to restart it from scratch. The app
    will be killed and you can re-launch it cleanly with `open_app`.
  • For `index` target_kind, target is the integer index of the element from CURRENT_SCREEN. Use this as a bulletproof fallback when an element has no clear text label or content_desc.
  • Use `done` when goal_complete=true.
  • Use `give_up` only after multiple failed attempts AND no plausible
    element exists AND scrolling/back also failed.
"""


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

_VALID_ACTIONS = {
    "click", "type", "press", "open_app", "force_stop", "scroll_to", "swipe",
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
    use_vision: bool = False           # send screenshot to LLM (user toggle)


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
        recorder: Optional[TaskRecorder] = None,
    ):
        self.device = device
        self.llm = llm
        self.watchers = watchers
        self.config = config or ExecutorConfig()
        self.on_progress = on_progress or (lambda step, msg: None)
        self.on_log = on_log or (lambda msg: None)
        self.recorder = recorder
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

        # --- STEP 0: CLEAN STATE ---
        # Always start from home launcher with related apps force-stopped.
        # This ensures both normal and macro-replay runs start identically.
        try:
            related_pkgs = self._extract_related_packages(plan)
            if related_pkgs:
                self.on_log(
                    f"▸ Step 0: Resetting to home & force-stopping "
                    f"{len(related_pkgs)} related app(s): {', '.join(related_pkgs)}"
                )
            else:
                self.on_log("▸ Step 0: Resetting to home launcher.")
            self.device.go_home_and_clean(related_pkgs)
        except Exception as exc:
            self.on_log(f"⚠ Clean-start failed (continuing anyway): {exc}")

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

                screen_ctx = analyze_screen(obs)
                if it == 1:
                    self._maybe_bootstrap_goal_app(plan, screen_ctx)

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
                        "yourself. Use open_app, swipe, app drawer, or HOME."
                    )
                if screen_ctx.is_notification_shade:
                    stall_hint = (
                        (stall_hint + " ") if stall_hint else ""
                    ) + screen_ctx.hint

                forbidden = sorted(dead_ends.get(obs.screen_hash, set()))

                try:
                    decision = self._reason(
                        plan, obs, list(recent), stall_hint, forbidden,
                        screen_ctx,
                    )
                except LLMResponseError as exc:
                    self.on_log(f"reasoner error: {exc}")
                    continue

                decision = self._guard_decision(
                    decision, screen_ctx, list(recent),
                )

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
                    if self.recorder is not None:
                        self.recorder.close()
                    break

                if decision.action.kind == "give_up":
                    self.on_log("✗ Agent gave up.")
                    self._finalize_remaining(plan, per_step, success=False, note="gave up")
                    break

                # --- 5. ACT ---
                # Tell the recorder what we're about to do (pre-action snapshot).
                # Skip recovery / wait / terminal actions — they pollute macros.
                if self.recorder is not None and decision.action.kind not in (
                    "done", "give_up", "wait"
                ) and not decision.is_recovery:
                    milestone_text = ""
                    if decision.active_milestone_id is not None:
                        ms = self._find_step(plan, decision.active_milestone_id)
                        if ms is not None:
                            milestone_text = ms.action_description
                    rec_target = decision.action.target
                    rec_target_kind = decision.action.target_kind
                    if decision.action.kind == "open_app":
                        resolved_pkg = self.device.resolve_package(rec_target)
                        if resolved_pkg:
                            rec_target = resolved_pkg
                            rec_target_kind = "package"
                    self.recorder.record_attempt(
                        pre_obs=obs,
                        action_kind=decision.action.kind,
                        target=rec_target,
                        target_kind=rec_target_kind,
                        input_value=decision.action.input_value,
                        milestone_id=decision.active_milestone_id,
                        milestone_text=milestone_text,
                        screen_summary=decision.screen_summary,
                        is_recovery=decision.is_recovery,
                    )

                rec = self._act_with_record(decision)
                recent.append(rec)
                if rec.decision.active_milestone_id is not None:
                    per_step[rec.decision.active_milestone_id].actions.append(rec)
                    per_step[rec.decision.active_milestone_id].attempts += 1

                # Settle then peek at the post-action screen so the recorder
                # can compute verification anchors and finalize the step.
                if self.recorder is not None:
                    try:
                        self.device.wait(self.config.settle_seconds * 0.5)
                        post_obs = self.device.observe()
                        progressed = (
                            post_obs.screen_hash != obs.screen_hash
                            or (post_obs.current_app or "")
                            != (obs.current_app or "")
                        )
                        if decision.action.kind == "open_app":
                            pkg = self.device.resolve_package(
                                decision.action.target,
                            ) or decision.action.target
                            cur = post_obs.current_app or ""
                            if pkg and (cur == pkg or cur.startswith(pkg)):
                                progressed = True
                        self.recorder.record_outcome(
                            post_obs,
                            success=rec.success,
                            progressed=progressed,
                        )
                    except Exception as exc:
                        log.debug("recorder post-observe failed: %s", exc)
                        self.recorder.discard_pending()

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
        screen_ctx: ScreenContext,
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

        # Installed apps (ADB) — refreshed periodically, filtered by goal.
        try:
            app_block = self.device.apps.format_for_prompt(plan.goal)
        except Exception as exc:
            log.warning("Could not read installed apps: %s", exc)
            app_block = {
                "relevant_installed_apps": [],
                "suggested_packages_for_goal": [],
                "total_installed_count": 0,
            }

        # Screenshot for GUI always; LLM only when user enabled vision AND
        # the model supports it.
        image_base64 = None
        use_vision = bool(self.config.use_vision)
        try:
            import base64
            import io
            import os

            img = self.device.d.screenshot()
            os.makedirs("ui_dumps", exist_ok=True)
            img.save("ui_dumps/current.png")
            self.on_log("Captured screenshot: ui_dumps/current.png")

            if use_vision:
                model_info = find_model(
                    self.llm.config.provider, self.llm.config.model,
                )
                if model_info is not None and model_info.supports_vision:
                    buffered = io.BytesIO()
                    img.save(buffered, format="PNG")
                    image_base64 = base64.b64encode(buffered.getvalue()).decode()
                else:
                    self.on_log(
                        "Vision is ON but this model has no vision support — "
                        "using XML only."
                    )
        except Exception as exc:
            log.warning("Failed to capture screenshot for reasoning: %s", exc)

        user_prompt = (
            f"GOAL: {plan.goal}\n"
            f"MILESTONES: {json.dumps(milestones_payload, ensure_ascii=False)}\n"
            f"CURRENT_APP: {obs.current_app or 'unknown'}\n"
            f"SCREEN_CONTEXT: {json.dumps(screen_ctx.to_prompt_dict(), ensure_ascii=False)}\n"
            f"VISION_ENABLED: {bool(image_base64)}\n"
            f"INSTALLED_APPS: "
            f"{json.dumps(app_block.get('relevant_installed_apps', []), ensure_ascii=False)}\n"
            f"SUGGESTED_PACKAGES: "
            f"{json.dumps(app_block.get('suggested_packages_for_goal', []), ensure_ascii=False)}\n"
            f"TOTAL_INSTALLED_ON_DEVICE: {app_block.get('total_installed_count', 0)}\n"
            f"CURRENT_SCREEN: {json.dumps(slim, ensure_ascii=False)}\n"
            f"RECENT_ACTIONS: {json.dumps(recent_payload, ensure_ascii=False)}\n"
            f"FORBIDDEN_TARGETS: {json.dumps(forbidden_targets, ensure_ascii=False)}\n"
            f"STALL_HINT: {stall_hint}\n\n"
            "Remember: react to WHAT YOU SEE on this screen. Check INSTALLED_APPS "
            "before claiming an app is missing. Do NOT click anything in "
            "FORBIDDEN_TARGETS. Respond with ONE JSON decision now."
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
    # Guards & bootstrap
    # --------------------------------------------------------------------- #

    def _maybe_bootstrap_goal_app(
        self, plan: Plan, screen_ctx: ScreenContext,
    ) -> None:
        """On iteration 1 at the launcher, try to open the goal app directly."""
        if not screen_ctx.is_launcher:
            return
        goal_l = plan.goal.lower()
        keywords: List[str] = []
        if "gmail" in goal_l or "google mail" in goal_l:
            keywords = ["gmail", "google mail"]
        elif "whatsapp" in goal_l:
            keywords = ["whatsapp"]
        elif "chrome" in goal_l:
            keywords = ["chrome"]
        if not keywords:
            return

        for name in keywords:
            pkg = self.device.resolve_package(name)
            if not pkg:
                continue
            if self.device.open_app(pkg):
                self.on_log(f"▸ Bootstrapped: launched {name} ({pkg})")
                return
            self.on_log(
                f"▸ {pkg} is listed but launch failed — will try UI navigation."
            )
        if "gmail" in goal_l:
            self.on_log(
                "▸ Gmail not launchable — may be absent on this device; "
                "use Samsung Internet / Chrome from INSTALLED_APPS or Play Store."
            )

    def _extract_related_packages(self, plan: Plan) -> List[str]:
        """Identify packages related to the goal that should be force-stopped."""
        from .app_catalog import KNOWN_ALIASES
        goal_l = plan.goal.lower()
        pkgs: List[str] = []
        
        # Check known aliases
        for alias, candidates in KNOWN_ALIASES.items():
            if alias in goal_l:
                for pkg in candidates:
                    if self.device.is_package_installed(pkg) and pkg not in pkgs:
                        pkgs.append(pkg)
        
        # Also check if the goal mentions Google account creation
        if "google" in goal_l or "gmail" in goal_l:
            google_pkgs = [
                "com.google.android.gms",       # Google Play Services (account flow)
                "com.google.android.gm",        # Gmail
                "com.google.android.gsf.login", # Google sign-in
            ]
            for pkg in google_pkgs:
                if self.device.is_package_installed(pkg) and pkg not in pkgs:
                    pkgs.append(pkg)
        
        return pkgs

    def _guard_decision(
        self,
        decision: AgentDecision,
        screen_ctx: ScreenContext,
        recent: List[ActionRecord],
    ) -> AgentDecision:
        """Patch obviously wasteful LLM choices before we act."""
        act = decision.action
        kind = act.kind
        target = (act.target or "").lower()

        # Notification shade: BACK toggles it — use HOME.
        if screen_ctx.is_notification_shade and kind in ("back", "press"):
            if kind == "back" or target == "back":
                self.on_log("  ↻ notification shade → HOME (not BACK)")
                decision.action = ActionSpec(
                    kind="press", target="home", target_kind="key",
                )
                return decision

        # Launcher: consecutive BACK does nothing useful — use HOME once.
        if screen_ctx.is_launcher and kind in ("back", "press") and target == "back":
            if recent and recent[-1].decision.action.kind in ("back", "press"):
                self.on_log("  ↻ launcher: repeated BACK → HOME")
                decision.action = ActionSpec(
                    kind="press", target="home", target_kind="key",
                )
                return decision

        # Launcher: invalid scroll_to → swipe.
        if screen_ctx.is_launcher and kind == "scroll_to":
            t = (act.target or "").strip()
            if ":" in t or t.isdigit() or "workspace" in t.lower():
                self.on_log("  ↻ launcher scroll_to → swipe left")
                decision.action = ActionSpec(
                    kind="swipe", target="left", target_kind="key",
                )
                return decision

        # Repeated BACK/PRESS on launcher/shade → HOME.
        if len(recent) >= 3 and (screen_ctx.is_launcher or screen_ctx.is_notification_shade):
            last3 = recent[-3:]
            if all(
                r.decision.action.kind in ("back", "press")
                for r in last3
            ):
                self.on_log("  ↻ back-loop detected → HOME")
                decision.action = ActionSpec(
                    kind="press", target="home", target_kind="key",
                )
                return decision

        # Failed open_app retries: suggest drawer once.
        failed_opens = [
            r for r in recent[-5:]
            if r.decision.action.kind == "open_app" and not r.success
        ]
        if (
            screen_ctx.is_launcher
            and len(failed_opens) >= 2
            and kind == "open_app"
        ):
            self.on_log("  ↻ repeated open_app failures → open app drawer")
            if self.device.open_app_drawer():
                decision.action = ActionSpec(
                    kind="wait", target="", target_kind="none",
                )
            else:
                decision.action = ActionSpec(
                    kind="swipe", target="left", target_kind="key",
                )
            return decision

        return decision

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
        if kind == "force_stop":
            if not t:
                raise ActionExecutionError("force_stop requires a package target")
            if not self.device.force_stop(t):
                raise ActionExecutionError(f"force_stop({t}) failed")
            return
        if kind == "scroll_to":
            if not t:
                raise ActionExecutionError("scroll_to requires a target text")
            if not self.device.smart_scroll(t, kt):
                raise ActionExecutionError(f"scroll_to({t}) failed")
            return
        if kind == "swipe":
            direction = (t or "left").lower()
            if not self.device.swipe(direction):
                raise ActionExecutionError(f"swipe({direction}) failed")
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
            if t and kt == "index":
                self._dispatch_click(t, kt)
                self.device.wait(0.5)
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
        
        action_str = f"{d.action.kind}→{d.action.target!r}"
        if d.action.kind == "type":
            action_str += f" val={d.action.input_value!r}"
            
        self.on_log(
            f"#{it:02d}  app={obs.current_app or '?'}  active={active}  "
            f"{screen_tag}{recovery_tag}{action_str}  "
            f"({thought})"
        )
