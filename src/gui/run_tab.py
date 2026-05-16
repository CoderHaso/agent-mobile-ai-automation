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
from PySide6.QtGui import QColor, QFont, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
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
from ..hybrid_runner import HybridResult
from ..llm_client import LLMClient
from ..planner import Plan
from ..task_library import TaskLibrary
from .style import STATUS_COLORS
from .workers import ExecutorWorker


_HEADERS = ("#", "Milestone", "Recognize when done", "Opt", "Status")


class RunTab(QWidget):
    finished_with_results = Signal(list)
    log = Signal(str)
    macro_saved = Signal(object)   # RecordedTask

    def __init__(self, library: Optional[TaskLibrary] = None) -> None:
        super().__init__()
        self._device: Optional[DeviceManager] = None
        self._llm: Optional[LLMClient] = None
        self._plan: Optional[Plan] = None
        self._worker: Optional[ExecutorWorker] = None
        self._library = library or TaskLibrary()
        self._build()

    # ---- public API ----------------------------------------------------- #

    def attach_runtime(self, device: DeviceManager, llm: LLMClient) -> None:
        self._device = device
        self._llm = llm

    def start_with_plan(self, plan: Plan, *, use_vision: bool = False) -> None:
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

        vision_note = "vision ON" if use_vision else "vision OFF (XML only)"
        self._append_log(f"▶ Starting hybrid execution. Goal: {plan.goal!r}")
        self._append_log(
            f"  {len(plan.steps)} milestone(s) — replay first, AI on fallback · {vision_note}"
        )
        self.phase_badge.setText("checking macro library…")
        self.phase_badge.setStyleSheet(self._badge_style("library"))
        self._worker = ExecutorWorker(
            device=self._device,
            llm=self._llm,
            plan=plan,
            config=ExecutorConfig(use_vision=use_vision),
            library=self._library,
        )
        self._worker.log_message.connect(self._append_log)
        self._worker.step_status_changed.connect(self._on_step_status)
        self._worker.step_progress.connect(self._on_step_progress)
        self._worker.phase_changed.connect(self._on_phase_changed)
        self._worker.finished_with_hybrid.connect(self._on_hybrid_finished)
        self._worker.finished_with_results.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    # ---- UI ------------------------------------------------------------- #

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        # --- title row with phase badge ---
        title_row = QHBoxLayout()
        title_row.setSpacing(10)
        title = QLabel("Step 3 — Hybrid run & monitor")
        title.setObjectName("h1")
        title_row.addWidget(title)
        title_row.addStretch(1)
        self.phase_badge = QLabel("idle")
        self.phase_badge.setObjectName("statusBadge")
        self.phase_badge.setStyleSheet(self._badge_style("idle"))
        title_row.addWidget(self.phase_badge)
        layout.addLayout(title_row)

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

        # screenshot view
        img_box = QWidget()
        img_layout = QVBoxLayout(img_box)
        img_layout.setContentsMargins(0, 0, 0, 0)
        img_layout.setSpacing(6)
        img_label = QLabel("Live screenshot")
        img_label.setObjectName("h2")
        img_layout.addWidget(img_label)
        
        self.screenshot_label = QLabel("No screenshot yet")
        self.screenshot_label.setAlignment(Qt.AlignCenter)
        self.screenshot_label.setStyleSheet("border: 1px solid #2c2d3a; background: #14151c; border-radius: 6px;")
        img_layout.addWidget(self.screenshot_label, stretch=1)
        splitter.addWidget(img_box)

        splitter.setSizes([340, 340, 340])
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
        if results:
            self.results_label.setText(
                f"Completed: {ok}/{total_planned} step(s) succeeded "
                f"({len(results)} executed)."
            )
            self._append_log(f"■ Execution finished — {ok}/{total_planned} succeeded.")
        self.finished_with_results.emit(results)

    @staticmethod
    def _badge_style(phase: str) -> str:
        colors = {
            "idle":    ("#2a2b3a", "#8e8ea0"),
            "library": ("#2c2d3a", "#fdcb6e"),
            "replay":  ("#1f8a3a", "#ffffff"),
            "ai":      ("#6c5ce7", "#ffffff"),
            "done":    ("#1f8a3a", "#ffffff"),
            "fail":    ("#d63031", "#ffffff"),
        }
        bg, fg = colors.get(phase, colors["idle"])
        return (
            f"QLabel#statusBadge{{background:{bg};color:{fg};"
            "padding:4px 10px;border-radius:10px;font-weight:600;}}"
        )

    def _on_phase_changed(self, phase: str, info: str) -> None:
        labels = {
            "replay": "▶ Replaying macro (no LLM)",
            "ai":     "🧠 AI agent active",
        }
        self.phase_badge.setText(labels.get(phase, phase))
        self.phase_badge.setStyleSheet(self._badge_style(phase))
        self._append_log(f"⇢ Phase: {phase} — {info}")

    def _on_hybrid_finished(self, result: HybridResult) -> None:
        """Update final UI state and offer to save a fresh macro."""
        self.btn_stop.setEnabled(False)
        if result.success:
            self.phase_badge.setText("✓ done")
            self.phase_badge.setStyleSheet(self._badge_style("done"))
            note = result.note or "completed"
            self.results_label.setText(f"✓ Goal achieved — {note}")
        else:
            self.phase_badge.setText("✗ failed / partial")
            self.phase_badge.setStyleSheet(self._badge_style("fail"))
            self.results_label.setText(
                f"Run did not fully complete. {result.note}"
            )

        # Offer to save a NEW macro only if the AI phase finished a fresh
        # recording successfully (the replay-only path doesn't create one).
        if (
            result.success
            and result.new_recording is not None
            and result.new_recording.has_actions()
        ):
            self._offer_save_macro(result)

    def _offer_save_macro(self, result: HybridResult) -> None:
        rec = result.new_recording
        if rec is None:
            return

        # Build the canonical (clean) task once so we can show the count.
        clean = rec.build_task(drop_recovery=True)
        n = len(clean.actions)
        n_dyn = sum(1 for a in clean.actions if a.is_dynamic)

        replaced_existing = (
            result.used_macro is not None
            and result.used_macro.task_id() == clean.task_id()
        )
        verb = "Update existing macro" if replaced_existing else "Save this run as a reusable macro"

        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Question)
        msg.setWindowTitle("Save macro?")
        msg.setText(
            f"<b>{verb}?</b><br><br>"
            f"Goal: <i>{clean.goal}</i><br>"
            f"Steps: <b>{n}</b> "
            f"({n_dyn} with dynamic placeholder fields)<br><br>"
            "Saving lets future runs of this goal SKIP the LLM and "
            "replay the recorded steps directly. The agent will only "
            "fall back to AI if the screen no longer matches."
        )
        msg.setStandardButtons(QMessageBox.Save | QMessageBox.Discard)
        if msg.exec() == QMessageBox.Save:
            try:
                path = self._library.save(clean)
                self._append_log(f"💾 Macro saved → {path}")
                self.macro_saved.emit(clean)
            except Exception as exc:
                self._append_log(f"✗ Failed to save macro: {exc}")
                QMessageBox.critical(self, "Save failed", str(exc))
        else:
            self._append_log("Macro discarded — not saved.")

    def _on_failed(self, msg: str) -> None:
        self.btn_stop.setEnabled(False)
        self.phase_badge.setText("✗ crash")
        self.phase_badge.setStyleSheet(self._badge_style("fail"))
        self.results_label.setText(f"❌ Crash: {msg}")
        self._append_log(f"✗ Executor crashed: {msg}")

    # ---- log + stop ----------------------------------------------------- #

    def _append_log(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_view.appendPlainText(f"[{ts}] {msg}")
        sb = self.log_view.verticalScrollBar()
        sb.setValue(sb.maximum())
        self.log.emit(msg)
        
        # Live screenshot update
        if "Captured screenshot: " in msg:
            path = msg.split("Captured screenshot: ")[1].strip()
            pixmap = QPixmap(path)
            if not pixmap.isNull():
                # Scale to fit label while maintaining aspect ratio
                scaled = pixmap.scaled(
                    self.screenshot_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.screenshot_label.setPixmap(scaled)

    def _on_stop(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_stop()
        self.btn_stop.setEnabled(False)
