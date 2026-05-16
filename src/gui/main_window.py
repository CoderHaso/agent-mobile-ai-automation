"""Main window: ties the three tabs together and manages global state.

Flow:
    Tab 1 (Devices)  →  device_connected signal
            │              │
            │              ▼
            │        Tab 2 (Plan) becomes enabled
            │              │ plan_approved
            │              ▼
            │        Tab 3 (Run) starts the executor
            ▼
       Status bar always shows: device · LLM · current step
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QLabel,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..device_manager import DeviceManager
from ..llm_client import LLMClient
from ..planner import Plan
from ..task_library import TaskLibrary
from .devices_tab import DevicesTab
from .plan_tab import PlanTab
from .run_tab import RunTab
from .teach_tab import TeachTab
from .style import DARK_QSS
from .workers import LLMConnectWorker


APP_TITLE = "Agentic Mobile AI Automation"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1180, 760)
        self.setStyleSheet(DARK_QSS)

        self._device: Optional[DeviceManager] = None
        self._llm: Optional[LLMClient] = None
        self._llm_worker: Optional[LLMConnectWorker] = None
        self._library = TaskLibrary()
        from ..taught_task import TaughtTaskLibrary
        self._taught_library = TaughtTaskLibrary()

        self._build_ui()
        self._build_statusbar()
        self._wire_signals()
        self._kick_off_llm()

    # ---- layout -------------------------------------------------------- #

    def _build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        self.devices_tab = DevicesTab()
        self.plan_tab = PlanTab(library=self._library)
        self.run_tab = RunTab(library=self._library)
        self.teach_tab = TeachTab()

        self.tabs.addTab(self.devices_tab, "1 · Devices")
        self.tabs.addTab(self.plan_tab,    "2 · Plan")
        self.tabs.addTab(self.run_tab,     "3 · Run")
        self.tabs.addTab(self.teach_tab,   "4 · Teach")
        self.tabs.setTabEnabled(1, False)
        self.tabs.setTabEnabled(2, False)
        self.tabs.setTabEnabled(3, False)

        layout.addWidget(self.tabs)
        self.setCentralWidget(central)

    def _build_statusbar(self) -> None:
        sb = QStatusBar()
        self.setStatusBar(sb)

        self.lbl_device = QLabel("Device: —")
        self.lbl_llm = QLabel("LLM: connecting…")
        self.lbl_msg = QLabel("Ready")

        sb.addWidget(self.lbl_device, 0)
        sb.addPermanentWidget(self.lbl_llm, 0)
        sb.addPermanentWidget(self.lbl_msg, 1)

    def _wire_signals(self) -> None:
        self.devices_tab.device_connected.connect(self._on_device_ready)
        self.devices_tab.log.connect(self._set_status)

        self.plan_tab.plan_approved.connect(self._on_plan_approved)
        self.plan_tab.log.connect(self._set_status)
        self.plan_tab.model_changed.connect(self._on_model_changed)

        self.run_tab.log.connect(self._set_status)
        self.run_tab.finished_with_results.connect(self._on_run_finished)
        self.run_tab.macro_saved.connect(self._on_macro_saved)

        self.teach_tab.log.connect(self._set_status)
        self.teach_tab.task_saved.connect(self._on_taught_task_saved)
        self.teach_tab.run_taught_task.connect(self._on_run_taught_task)

    # ---- bootstrapping ------------------------------------------------- #

    def _kick_off_llm(self) -> None:
        try:
            from ..llm_client import LLMClient
            client = LLMClient.from_env()
            self._on_llm_ready(client)
        except Exception as exc:
            self._on_llm_failed(str(exc))

    def _on_llm_ready(self, llm: LLMClient) -> None:
        self._llm = llm
        self.lbl_llm.setText(f"LLM: {llm.describe()}")
        self.plan_tab.attach_llm(llm)
        if self._device is not None:
            self.run_tab.attach_runtime(self._device, llm)
            self.teach_tab.attach_runtime(self._device, llm)

    def _on_llm_failed(self, msg: str) -> None:
        self.lbl_llm.setText("LLM: ERROR")
        QMessageBox.critical(
            self,
            "LLM provider error",
            f"Could not initialize the LLM client:\n\n{msg}\n\n"
            "Open .env and verify your provider keys, then restart the app.",
        )

    def _on_model_changed(self, provider: str, slug: str) -> None:
        """User picked a different (provider, model) in the picker."""
        from ..llm_client import LLMClient, LLMConfigError, LLMResponseError

        try:
            llm = LLMClient.from_choice(provider, slug)
        except (LLMConfigError, LLMResponseError) as exc:
            QMessageBox.warning(
                self,
                "Cannot switch model",
                f"Failed to switch to {provider}:{slug}\n\n{exc}\n\n"
                "Make sure the relevant *_API_KEY is set in your .env.",
            )
            return
        self._llm = llm
        self.lbl_llm.setText(f"LLM: {llm.describe()}")
        self.plan_tab.attach_llm(llm)
        if self._device is not None:
            self.run_tab.attach_runtime(self._device, llm)
            self.teach_tab.attach_runtime(self._device, llm)
        self._set_status(f"Switched LLM → {llm.describe()}")

    # ---- transitions --------------------------------------------------- #

    def _on_device_ready(self, dm: DeviceManager) -> None:
        self._device = dm
        info = dm.info()
        label = (
            f"{info.get('productName', 'unknown')}  "
            f"({info.get('displayWidth')}x{info.get('displayHeight')}, "
            f"sdk {info.get('sdkInt')})"
        )
        self.lbl_device.setText(f"Device: {label}")
        self.tabs.setTabEnabled(1, True)
        self.tabs.setTabEnabled(3, True)  # Teach tab
        self.tabs.setCurrentIndex(1)
        if self._llm is not None:
            self.run_tab.attach_runtime(dm, self._llm)
            self.teach_tab.attach_runtime(dm, self._llm)
        try:
            n = len(dm.list_installed_apps(force=True))
            self._set_status(f"Device ready — {n} installed app(s) indexed via ADB.")
        except Exception:
            pass

    def _on_plan_approved(self, plan: Plan) -> None:
        if self._device is None or self._llm is None:
            QMessageBox.warning(
                self, "Not ready",
                "Device or LLM is not initialized yet.",
            )
            return
        self.tabs.setTabEnabled(2, True)
        self.tabs.setCurrentIndex(2)
        self.run_tab.start_with_plan(
            plan,
            use_vision=self.plan_tab.use_vision_enabled(),
        )

    def _on_run_finished(self, results: list) -> None:
        ok = sum(1 for r in results if r.success)
        total = len(results)
        self._set_status(f"Run finished: {ok}/{total} succeeded.")

    def _on_macro_saved(self, _task) -> None:
        # Refresh the saved-macros panel on the Plan tab so the user
        # can see (and reuse) the macro they just kept.
        self.plan_tab.refresh_library()
        self._set_status("Macro saved to library.")

    def _on_taught_task_saved(self, _task) -> None:
        self._set_status(f"Taught task saved: {_task.name}")

    def _on_run_taught_task(self, task) -> None:
        """Build a Plan from a TaughtTask and start execution."""
        if self._device is None or self._llm is None:
            QMessageBox.warning(
                self, "Not ready",
                "Device or LLM is not initialized yet.",
            )
            return

        from ..planner import Plan, PlanStep
        steps = []
        for s in task.steps:
            steps.append(PlanStep(
                step_id=s.step_index,
                action_description=s.instruction,
                expected_outcome=f"Step {s.step_index} is completed",
                is_optional=False,
                status="pending",
            ))

        plan = Plan(
            goal=task.name,
            task_notes=task.notes or None,
            steps=steps,
        )

        self.tabs.setTabEnabled(2, True)
        self.tabs.setCurrentIndex(2)
        self.run_tab.start_with_plan(plan, use_vision=False)

    # ---- helpers ------------------------------------------------------- #

    def _set_status(self, msg: str) -> None:
        # Truncate so the status bar stays one line.
        if len(msg) > 160:
            msg = msg[:157] + "…"
        self.lbl_msg.setText(msg)

    # ---- shutdown ------------------------------------------------------ #

    def closeEvent(self, event: QCloseEvent) -> None:
        """Stop background workers gracefully so we don't crash on exit."""
        # Tell the executor (if any) to halt, then give every worker
        # a brief grace period to wind down.
        try:
            run_worker = getattr(self.run_tab, "_worker", None)
            if isinstance(run_worker, QThread) and run_worker.isRunning():
                run_worker.request_stop()  # type: ignore[attr-defined]
                run_worker.wait(2000)
        except Exception:
            pass

        for owner_attr in (
            (self.devices_tab, "_scan_worker"),
            (self.devices_tab, "_connect_worker"),
            (self.plan_tab, "_worker"),
            (self, "_llm_worker"),
        ):
            owner, attr = owner_attr
            worker = getattr(owner, attr, None)
            if isinstance(worker, QThread) and worker.isRunning():
                worker.quit()
                worker.wait(2000)
        super().closeEvent(event)
