"""Macro-first / AI-fallback orchestrator.

Combines `TaskReplayer` (fast, no LLM) with `Executor` (slow, adaptive,
LLM-driven) into a single hybrid run:

    1.  If the library has a macro for this goal, REPLAY it.
    2.  If replay completes successfully → done. The macro is reused.
    3.  If replay raises `ReplayBroken`, log it, mark the milestones the
        replayer DID complete, and HAND OFF to the LLM-driven Executor
        from the current screen. The Executor will pick up wherever the
        replay stopped.
    4.  If no macro existed in the first place, just run the Executor.
       In every successful run the Executor's `TaskRecorder` captures
       a fresh macro that the GUI can offer to save.

The orchestrator emits a uniform set of callbacks so the GUI can render
both phases without caring which engine drove the device.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional

from .device_manager import DeviceManager
from .dynamic_fields import DynamicValueResolver
from .executor import Executor, ExecutorConfig, StepResult
from .llm_client import LLMClient
from .planner import Plan
from .recorder import RecordedTask, TaskRecorder
from .replayer import ReplayBroken, ReplayResult, TaskReplayer
from .task_library import TaskLibrary
from .watchers import WatcherManager

log = logging.getLogger(__name__)


class RunPhase(str, Enum):
    REPLAY = "replay"
    AI = "ai"


@dataclass
class HybridResult:
    success: bool
    phases: List[RunPhase] = field(default_factory=list)
    replay_result: Optional[ReplayResult] = None
    ai_results: List[StepResult] = field(default_factory=list)
    new_recording: Optional[TaskRecorder] = None
    used_macro: Optional[RecordedTask] = None
    note: str = ""


LogCallback = Callable[[str], None]
PhaseCallback = Callable[[RunPhase, str], None]
StepStatusCallback = Callable[[int, str], None]   # (milestone_id, status)


class HybridRunner:
    def __init__(
        self,
        device: DeviceManager,
        llm: LLMClient,
        watchers: WatcherManager,
        library: TaskLibrary,
        *,
        config: Optional[ExecutorConfig] = None,
        on_log: Optional[LogCallback] = None,
        on_phase: Optional[PhaseCallback] = None,
        on_milestone_status: Optional[StepStatusCallback] = None,
    ) -> None:
        self.device = device
        self.llm = llm
        self.watchers = watchers
        self.library = library
        self.config = config or ExecutorConfig()
        self.on_log = on_log or (lambda msg: None)
        self.on_phase = on_phase or (lambda ph, msg: None)
        self.on_milestone_status = on_milestone_status or (lambda mid, st: None)
        self._stop_requested = False
        self._replayer: Optional[TaskReplayer] = None
        self._executor: Optional[Executor] = None

    def request_stop(self) -> None:
        self._stop_requested = True
        if self._replayer is not None:
            self._replayer.request_stop()
        if self._executor is not None:
            self._executor.request_stop()

    # ------------------------------------------------------------------ #

    def run(self, plan: Plan) -> HybridResult:
        result = HybridResult(success=False)

        # ----- Phase 1: try library replay -----
        macro = self.library.find_for_goal(plan.goal)
        if macro is not None and macro.actions:
            result.used_macro = macro
            self.on_phase(
                RunPhase.REPLAY,
                f"macro [{macro.task_id()}] matches — replaying "
                f"{len(macro.actions)} step(s)",
            )
            result.phases.append(RunPhase.REPLAY)

            resolver = DynamicValueResolver(self.llm, goal=plan.goal)

            def _on_replay_step(idx, action, status):
                if action.milestone_id is not None:
                    self.on_milestone_status(action.milestone_id, status)

            self._replayer = TaskReplayer(
                self.device,
                macro,
                resolver=resolver,
                on_log=self.on_log,
                on_step=_on_replay_step,
            )

            self.watchers.start()
            try:
                replay_res = self._replayer.run()
            except ReplayBroken as broken:
                self.on_log(
                    f"⚠ Replay broke at step {broken.step_index}: "
                    f"{broken.reason} — handing off to AI."
                )
                self.on_phase(RunPhase.AI,
                              f"recovering from replay break ({broken.reason})")

                # Mark the milestones the replayer DID get done as 'done'
                # so the executor doesn't redo them.
                for mid in broken.completed_milestones:
                    step = next((s for s in plan.steps if s.step_id == mid), None)
                    if step is not None and step.status != "done":
                        step.status = "done"
                        self.on_milestone_status(mid, "done")

                replay_res = ReplayResult(
                    success=False,
                    completed_milestones=broken.completed_milestones,
                    note=f"broken at step {broken.step_index}: {broken.reason}",
                )
                result.replay_result = replay_res
                # Fall through to the AI phase (watchers stay active).
            else:
                result.replay_result = replay_res
                if replay_res.success:
                    self.watchers.stop()
                    self.on_log("✓ Macro replay succeeded — skipped the LLM.")
                    result.success = True
                    result.note = (
                        f"replay-only ({replay_res.steps_executed} steps, "
                        f"{replay_res.steps_skipped} skipped)"
                    )
                    return result
                self.watchers.stop()
            finally:
                self._replayer = None
        else:
            self.on_log("No matching macro in the library — running the AI agent.")

        if self._stop_requested:
            return result

        # ----- Phase 2: AI-driven execution (also records a fresh macro) -----
        recorder = TaskRecorder(
            goal=plan.goal,
            llm_provider=self.llm.config.provider,
            llm_model=self.llm.config.model,
            plan_milestones=plan.steps,
        )
        self._executor = Executor(
            device=self.device,
            llm=self.llm,
            watchers=self.watchers,
            config=self.config,
            on_progress=lambda step, msg: self.on_milestone_status(step.step_id, msg),
            on_log=self.on_log,
            recorder=recorder,
        )
        result.phases.append(RunPhase.AI)
        self.on_phase(RunPhase.AI, "AI agent driving the device")

        ai_results = self._executor.run(plan)
        self._executor = None

        result.ai_results = ai_results
        result.new_recording = recorder

        all_done = (
            ai_results
            and all(r.success for r in ai_results)
            and recorder.has_actions()
        )
        if all_done:
            result.success = True
            result.note = (
                f"AI completed; {len(recorder._actions)} step(s) recorded "
                "and ready to save"
            )
        else:
            result.note = "AI did not complete the goal; nothing to save"
        return result
