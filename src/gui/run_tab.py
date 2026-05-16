"""Tab 3 — Live execution view.

Shows the approved plan with per-step status that updates in real time
(Pending → Running → Done / Failed / Skipped) plus a streaming log of
LLM thoughts and watcher events. Includes a Stop button that signals
the Executor to halt at the next safe boundary.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..device_manager import DeviceManager
from ..executor import ExecutorConfig, StepResult
from ..llm_client import LLMClient
from ..planner import Plan
from .style import STATUS_COLORS
from .workers import ExecutorWorker


_HEADERS = ("#", "Milestone", "Recognize when done", "Opt", "Status")


class RunTab(QWidget):
    finished_with_results = Signal(list)
    log = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._device: Optional[DeviceManager] = None
        self._llm: Optional[LLMClient] = None
        self._plan: Optional[Plan] = None
        self._worker: Optional[ExecutorWorker] = None
        self._build()

    # ---- public API ----------------------------------------------------- #

    def attach_runtime(self, device: DeviceManager, llm: LLMClient) -> None:
        self._device = device
        self._llm = llm

    def start_with_plan(self, plan: Plan) -> None:
        if self._device is None or self._llm is None:
            self._append_log("⚠ Device or LLM is not ready.")
            return
        self._plan = plan
        self._render_plan()
        self.log_view.clear()
        self.results_label.setText("")
        self.now_doing.setText("Starting…")
        self.progress.setValue(0)
        self.progress.setMaximum(max(1, len(plan.steps)))
        self.btn_stop.setEnabled(True)

        self._append_log(f"▶ Starting adaptive execution. Goal: {plan.goal!r}")
        self._append_log(f"  {len(plan.steps)} milestone(s) — agent will adapt to each screen.")
        self._worker = ExecutorWorker(
            device=self._device,
            llm=self._llm,
            plan=plan,
            config=ExecutorConfig(),
        )
        self._worker.log_message.connect(self._append_log)
        self._worker.step_status_changed.connect(self._on_step_status)
        self._worker.step_progress.connect(self._on_step_progress)
        self._worker.finished_with_results.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    # ---- UI ------------------------------------------------------------- #

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        title = QLabel("Step 3 — Adaptive run & monitor")
        title.setObjectName("h1")
        layout.addWidget(title)

        self.now_doing = QLabel("Idle")
        self.now_doing.setObjectName("h2")
        self.now_doing.setWordWrap(True)
        layout.addWidget(self.now_doing)

        # --- top row: progress + stop ---
        top = QHBoxLayout()
        top.setSpacing(8)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        top.addWidget(self.progress, stretch=1)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setObjectName("danger")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop)
        top.addWidget(self.btn_stop)
        layout.addLayout(top)

        # --- splitter: plan table | log ---
        splitter = QSplitter(Qt.Horizontal)

        # plan table
        self.table = QTableWidget(0, len(_HEADERS))
        self.table.setHorizontalHeaderLabels(_HEADERS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(1, QHeaderView.Stretch)
        h.setSectionResizeMode(2, QHeaderView.Stretch)
        h.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        splitter.addWidget(self.table)

        # log view
        log_box = QWidget()
        log_layout = QVBoxLayout(log_box)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.setSpacing(6)
        log_label = QLabel("Live agent log")
        log_label.setObjectName("h2")
        log_layout.addWidget(log_label)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.Monospace)
        mono.setPointSize(9)
        self.log_view.setFont(mono)
        log_layout.addWidget(self.log_view, stretch=1)
        splitter.addWidget(log_box)

        splitter.setSizes([520, 520])
        layout.addWidget(splitter, stretch=1)

        # --- footer: results summary ---
        self.results_label = QLabel("")
        self.results_label.setObjectName("h2")
        layout.addWidget(self.results_label)

    # ---- table rendering ----------------------------------------------- #

    def _render_plan(self) -> None:
        if self._plan is None:
            self.table.setRowCount(0)
            return
        self.table.setRowCount(len(self._plan.steps))
        for row, step in enumerate(self._plan.steps):
            id_item = QTableWidgetItem(str(step.step_id))
            id_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 0, id_item)
            self.table.setItem(row, 1, QTableWidgetItem(step.action_description))
            self.table.setItem(row, 2, QTableWidgetItem(step.expected_outcome or ""))
            opt_item = QTableWidgetItem("•" if step.is_optional else "")
            opt_item.setTextAlignment(Qt.AlignCenter)
            if step.is_optional:
                opt_item.setForeground(QColor("#fdcb6e"))
            self.table.setItem(row, 3, opt_item)
            self._set_status_cell(row, step.status)

    def _set_status_cell(self, row: int, status: str) -> None:
        item = QTableWidgetItem(status.upper())
        item.setTextAlignment(Qt.AlignCenter)
        bg, fg = STATUS_COLORS.get(status, ("#2a2b3a", "#e6e6e6"))
        item.setBackground(QColor(bg))
        item.setForeground(QColor(fg))
        self.table.setItem(row, 4, item)

    # ---- worker slots --------------------------------------------------- #

    def _row_for_step(self, step_id: int) -> int:
        if self._plan is None:
            return -1
        for i, s in enumerate(self._plan.steps):
            if s.step_id == step_id:
                return i
        return -1

    def _on_step_status(self, step_id: int, status: str) -> None:
        row = self._row_for_step(step_id)
        if row >= 0:
            self._set_status_cell(row, status)
            self.table.scrollToItem(self.table.item(row, 0))
            # Update progress = #milestones in a terminal state.
            if self._plan is not None:
                done_like = sum(
                    1 for s in self._plan.steps
                    if s.status in ("done", "failed", "skipped")
                )
                self.progress.setValue(min(self.progress.maximum(), done_like))

            if status == "running" and self._plan is not None:
                step = self._plan.steps[row]
                self.now_doing.setText(
                    f"▸ Working on milestone {step.step_id}: {step.action_description}"
                )
            elif status == "done" and self._plan is not None:
                step = self._plan.steps[row]
                self.now_doing.setText(f"✓ Milestone {step.step_id} done.")
            elif status == "skipped" and self._plan is not None:
                step = self._plan.steps[row]
                self.now_doing.setText(
                    f"⤼ Milestone {step.step_id} skipped (not needed by this flow)."
                )

    def _on_step_progress(self, step_id: int, msg: str) -> None:
        # Already logged by the worker; this hook exists in case we want
        # per-step inline annotations later (e.g. a tooltip).
        pass

    def _on_finished(self, results: list) -> None:
        self.btn_stop.setEnabled(False)
        ok = sum(1 for r in results if r.success)
        total_planned = len(self._plan.steps) if self._plan else len(results)
        self.results_label.setText(
            f"Completed: {ok}/{total_planned} step(s) succeeded "
            f"({len(results)} executed)."
        )
        self._append_log(f"■ Execution finished — {ok}/{total_planned} succeeded.")
        self.finished_with_results.emit(results)

    def _on_failed(self, msg: str) -> None:
        self.btn_stop.setEnabled(False)
        self.results_label.setText(f"❌ Crash: {msg}")
        self._append_log(f"✗ Executor crashed: {msg}")

    # ---- log + stop ----------------------------------------------------- #

    def _append_log(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_view.appendPlainText(f"[{ts}] {msg}")
        sb = self.log_view.verticalScrollBar()
        sb.setValue(sb.maximum())
        self.log.emit(msg)

    def _on_stop(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_stop()
        self.btn_stop.setEnabled(False)
