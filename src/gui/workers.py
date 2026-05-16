"""Background QThread workers.

The planner and executor are blocking I/O-heavy operations. We push them
onto worker threads so the Qt event loop stays responsive (UI repaints,
scrolling, the Stop button, etc. keep working while the agent thinks
or taps the device).

Communication is one-way via Qt Signals (thread-safe by design).
"""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import QObject, QThread, Signal

from ..device_manager import DeviceManager, list_adb_devices, AdbDeviceInfo
from ..executor import Executor, ExecutorConfig, StepResult
from ..llm_client import LLMClient
from ..planner import Plan, PlanStep, Planner
from ..watchers import WatcherManager


class DeviceScanWorker(QThread):
    """Run `adb devices -l` off the GUI thread (it can take seconds)."""

    finished_with_devices = Signal(list)
    failed = Signal(str)

    def run(self) -> None:
        try:
            devs: List[AdbDeviceInfo] = list_adb_devices()
            self.finished_with_devices.emit(devs)
        except Exception as exc:
            self.failed.emit(str(exc))


class PlannerWorker(QThread):
    """Ask the LLM to decompose a goal into a structured plan."""

    finished_with_plan = Signal(object)   # Plan
    failed = Signal(str)

    def __init__(self, llm: LLMClient, goal: str) -> None:
        super().__init__()
        self._llm = llm
        self._goal = goal

    def run(self) -> None:
        try:
            plan = Planner(self._llm).build_plan(self._goal)
            self.finished_with_plan.emit(plan)
        except Exception as exc:
            self.failed.emit(str(exc))


class ExecutorWorker(QThread):
    """Drive the approved plan through the device.

    Emits granular signals so the GUI can update a per-step status table
    and a streaming log without polling.
    """

    log_message = Signal(str)
    step_status_changed = Signal(int, str)   # step_id, status (pending|running|done|failed|skipped)
    step_progress = Signal(int, str)         # step_id, freeform progress message
    finished_with_results = Signal(list)     # List[StepResult]
    failed = Signal(str)

    def __init__(
        self,
        device: DeviceManager,
        llm: LLMClient,
        plan: Plan,
        config: Optional[ExecutorConfig] = None,
    ) -> None:
        super().__init__()
        self._device = device
        self._llm = llm
        self._plan = plan
        self._config = config or ExecutorConfig()
        self._executor: Optional[Executor] = None

    def request_stop(self) -> None:
        if self._executor is not None:
            self._executor.request_stop()
        self.log_message.emit("Stop requested — agent will halt at the next safe boundary…")

    def _on_progress(self, step: PlanStep, message: str) -> None:
        # Bridge plain Python callback into Qt signals (thread-safe).
        self.step_progress.emit(step.step_id, message)
        if message in {"pending", "running", "done", "failed", "skipped", "stopped"}:
            normalized = "skipped" if message == "stopped" else message
            self.step_status_changed.emit(step.step_id, normalized)

    def _on_log(self, msg: str) -> None:
        self.log_message.emit(msg)

    def run(self) -> None:
        try:
            watchers = WatcherManager(self._device.d)
            self._executor = Executor(
                device=self._device,
                llm=self._llm,
                watchers=watchers,
                config=self._config,
                on_progress=self._on_progress,
                on_log=self._on_log,
            )
            self.log_message.emit("Background watchers registered.")
            results = self._executor.run(self._plan)
            self.finished_with_results.emit(results)
        except Exception as exc:
            self.failed.emit(str(exc))


class ConnectDeviceWorker(QThread):
    """Open a u2 session to a chosen serial without freezing the GUI."""

    finished_with_device = Signal(object)   # DeviceManager
    failed = Signal(str)

    def __init__(self, serial: str) -> None:
        super().__init__()
        self._serial = serial

    def run(self) -> None:
        try:
            dm = DeviceManager(serial=self._serial)
            dm.connect()
            self.finished_with_device.emit(dm)
        except Exception as exc:
            self.failed.emit(str(exc))


class _LLMConnectWorker(QThread):
    """Initialize the LLM client (validates env / network) off the GUI thread."""

    finished_with_client = Signal(object)
    failed = Signal(str)

    def run(self) -> None:
        try:
            client = LLMClient.from_env()
            self.finished_with_client.emit(client)
        except Exception as exc:
            self.failed.emit(str(exc))


# Re-export for convenience.
LLMConnectWorker = _LLMConnectWorker
