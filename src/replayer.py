"""Replay a recorded task without invoking the LLM.

For each step we:
  1. Wait for the screen to settle.
  2. Confirm the screen still matches the recorded `screen_signature`.
     - If it doesn't, raise `ReplayBroken` so the orchestrator can fall
       back to the LLM-driven Executor.
  3. Resolve any dynamic placeholders in `input_value`
     (e.g. `<USER_NAME>` → fresh value via the LLM or the offline fallback).
  4. Execute the action via `DeviceManager`.
  5. Verify success: at least one of the recorded `verification_anchors`
     must appear, OR the screen signature must change. Otherwise:
     `ReplayBroken`.

`is_recovery` actions (popup dismissals) are conditional: we only run
them if their pre-action signature matches the *current* screen.
Otherwise we silently skip them — the popup didn't appear this time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .device_manager import ActionExecutionError, DeviceManager, Observation
from .dynamic_fields import DynamicValueResolver
from .recorder import RecordedAction, RecordedTask, screen_signature
from .ui_parser import parse_hierarchy

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #

class ReplayBroken(Exception):
    """The recorded macro no longer matches reality.

    Carries enough context for the orchestrator to hand off to the LLM:
      - `step_index`     : the action that failed (1-based)
      - `reason`         : "signature_mismatch" | "verify_failed" | "act_failed"
      - `completed_milestones` : milestone_ids the replayer DID get done.
    """

    def __init__(
        self,
        *,
        step_index: int,
        reason: str,
        message: str,
        completed_milestones: List[int],
    ) -> None:
        super().__init__(message)
        self.step_index = step_index
        self.reason = reason
        self.completed_milestones = completed_milestones


@dataclass
class ReplayResult:
    success: bool
    steps_executed: int = 0
    steps_skipped: int = 0
    completed_milestones: List[int] = field(default_factory=list)
    note: str = ""


# --------------------------------------------------------------------------- #
# Replayer
# --------------------------------------------------------------------------- #

LogCallback = Callable[[str], None]
StepCallback = Callable[[int, RecordedAction, str], None]   # (idx, action, status)


class TaskReplayer:
    def __init__(
        self,
        device: DeviceManager,
        task: RecordedTask,
        *,
        resolver: Optional[DynamicValueResolver] = None,
        on_log: Optional[LogCallback] = None,
        on_step: Optional[StepCallback] = None,
        settle_seconds: float = 0.6,
        verify_seconds: float = 0.8,
        max_signature_wait: float = 4.0,
    ) -> None:
        self.device = device
        self.task = task
        self.resolver = resolver or DynamicValueResolver()
        self.on_log = on_log or (lambda msg: None)
        self.on_step = on_step or (lambda i, a, s: None)
        self.settle_seconds = settle_seconds
        self.verify_seconds = verify_seconds
        self.max_signature_wait = max_signature_wait
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    # ---- Main entry ----------------------------------------------------- #

    def run(self) -> ReplayResult:
        completed: List[int] = []
        executed = 0
        skipped = 0

        self.on_log(
            f"▶ Replaying macro [{self.task.task_id()}] — "
            f"{len(self.task.actions)} step(s), no LLM"
        )

        for action in self.task.actions:
            if self._stop_requested:
                self.on_log("● Replay stopped by user.")
                return ReplayResult(
                    success=False, steps_executed=executed, steps_skipped=skipped,
                    completed_milestones=completed, note="stopped",
                )

            self.device.wait(self.settle_seconds)
            cur = self._observe_or_fail(action.step_index, completed)

            cur_sig = screen_signature(cur)

            # ---- Recovery actions: only run if their screen actually appeared
            if action.is_recovery:
                if cur_sig != action.screen_signature:
                    self.on_log(
                        f"  ↩ skipping recovery step {action.step_index} "
                        f"({action.action_kind}→{action.target!r}); "
                        "popup not present this time"
                    )
                    self.on_step(action.step_index, action, "skipped")
                    skipped += 1
                    continue
            else:
                # ---- Signature gate ----
                if cur_sig != action.screen_signature:
                    # Brief wait — sometimes the prior step's UI is still settling.
                    cur, cur_sig = self._await_signature(action.screen_signature)
                    if cur_sig != action.screen_signature:
                        self.on_step(action.step_index, action, "broken")
                        raise ReplayBroken(
                            step_index=action.step_index,
                            reason="signature_mismatch",
                            message=(
                                f"Screen no longer matches recorded step "
                                f"{action.step_index} "
                                f"(expected sig={action.screen_signature[:6]}, "
                                f"got {cur_sig[:6]})"
                            ),
                            completed_milestones=completed,
                        )

            # ---- Resolve dynamic input values ----
            input_value = self.resolver.fill(action.input_value)
            if action.is_dynamic and input_value != action.input_value:
                self.on_log(
                    f"  ◇ resolved dynamic fields for step {action.step_index}: "
                    f"{action.input_value!r} → {input_value!r}"
                )

            self.on_step(action.step_index, action, "running")
            self.on_log(
                f"  ▸ step {action.step_index}: "
                f"{action.action_kind}→{action.target!r}"
                f"{f' value={input_value!r}' if input_value else ''}"
            )

            # ---- Act ----
            try:
                self._act(action, input_value)
            except ActionExecutionError as exc:
                self.on_step(action.step_index, action, "broken")
                raise ReplayBroken(
                    step_index=action.step_index,
                    reason="act_failed",
                    message=f"Step {action.step_index} action failed: {exc}",
                    completed_milestones=completed,
                ) from exc
            executed += 1

            # ---- Verify ----
            self.device.wait(self.verify_seconds)
            ok, verify_msg = self._verify(action, cur)
            if not ok:
                self.on_step(action.step_index, action, "broken")
                raise ReplayBroken(
                    step_index=action.step_index,
                    reason="verify_failed",
                    message=(
                        f"Step {action.step_index} verification failed: "
                        f"{verify_msg}"
                    ),
                    completed_milestones=completed,
                )

            self.on_step(action.step_index, action, "done")
            if action.milestone_id is not None and action.milestone_id not in completed:
                completed.append(action.milestone_id)

        self.on_log("✓ Replay finished — every recorded step verified.")
        return ReplayResult(
            success=True,
            steps_executed=executed,
            steps_skipped=skipped,
            completed_milestones=completed,
            note="ok",
        )

    # ---- Helpers -------------------------------------------------------- #

    def _observe_or_fail(self, step_index: int, completed: List[int]) -> Observation:
        try:
            return self.device.observe()
        except Exception as exc:
            raise ReplayBroken(
                step_index=step_index,
                reason="observe_failed",
                message=f"Could not observe screen: {exc}",
                completed_milestones=completed,
            ) from exc

    def _await_signature(self, target_sig: str):
        """Brief poll loop while the previous action's UI may still be settling."""
        import time
        deadline = time.time() + self.max_signature_wait
        cur = self.device.observe()
        cur_sig = screen_signature(cur)
        while cur_sig != target_sig and time.time() < deadline:
            self.device.wait(0.4)
            cur = self.device.observe()
            cur_sig = screen_signature(cur)
        return cur, cur_sig

    def _act(self, action: RecordedAction, input_value: str) -> None:
        d = self.device
        kind = action.action_kind
        t = (action.target or "").strip()
        kt = action.target_kind

        if kind == "wait":
            d.wait(2.0)
            return
        if kind in ("press", "back"):
            key = "back" if kind == "back" else (t or "back").lower()
            if not d.press(key):
                raise ActionExecutionError(f"press({key}) failed")
            return
        if kind == "open_app":
            if not t:
                raise ActionExecutionError("open_app requires a package")
            if not d.open_app(t):
                raise ActionExecutionError(f"open_app({t}) failed")
            return
        if kind == "scroll_to":
            if not t or not d.smart_scroll(t, kt):
                raise ActionExecutionError(f"scroll_to({t!r}) failed")
            return
        if kind == "swipe":
            if not d.swipe((t or "left").lower()):
                raise ActionExecutionError(f"swipe({t!r}) failed")
            return
        if kind == "click":
            if not t:
                raise ActionExecutionError("click requires a target")
            if kt == "resource_id":
                ok = d.click_resource_id(t) or d.click_text(t)
            elif kt == "content_desc":
                ok = d.click_description(t) or d.click_text(t)
            else:
                ok = (
                    d.click_text(t)
                    or d.click_resource_id(t)
                    or d.click_description(t)
                )
            if not ok:
                raise ActionExecutionError(f"click target not found: {t!r}")
            return
        if kind == "type":
            if not input_value:
                raise ActionExecutionError("type requires input_value")
            if not d.type_into(t, input_value):
                raise ActionExecutionError(f"type into {t!r} failed")
            return

        raise ActionExecutionError(f"unsupported recorded action: {kind!r}")

    def _verify(self, action: RecordedAction, pre_obs: Observation):
        """Return (ok, message). Either anchors appear, or signature changed."""
        try:
            post_obs = self.device.observe()
        except Exception as exc:
            return False, f"post-observe failed: {exc}"

        post_sig = screen_signature(post_obs)
        if post_sig != action.screen_signature:
            # Screen advanced — that alone is good enough when we have no anchors.
            if not action.verification_anchors:
                return True, "signature changed"

        if action.verification_anchors:
            elements = parse_hierarchy(post_obs.xml, include_text_labels=True)
            haystack = []
            for e in elements:
                for v in (e.text, e.resource_id, e.content_desc):
                    if v:
                        haystack.append(v.strip().lower())
            haystack_blob = "\n".join(haystack)
            for anchor in action.verification_anchors:
                if anchor.strip().lower() in haystack_blob:
                    return True, f"anchor {anchor!r} matched"
            return False, (
                f"none of the anchors {action.verification_anchors} "
                "appeared on the new screen"
            )

        # No anchors and signature didn't change: we couldn't confirm
        # anything happened. If the action was intentionally a 'wait',
        # accept it; otherwise fail.
        if action.action_kind == "wait":
            return True, "wait completed"
        return False, "screen unchanged and no anchors recorded"
