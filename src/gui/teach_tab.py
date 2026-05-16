"""Tab 4 — Teach: build custom tasks step-by-step.

Users can:
    1. Name their task and write notes (variables the AI should use).
    2. Add step-by-step natural-language instructions.
    3. Test each step individually against the live device.
    4. Mark steps as verified once they work correctly.
    5. Save the full task and later run it from the Plan tab.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..device_manager import DeviceManager
from ..llm_client import LLMClient
from ..taught_task import TaughtStep, TaughtTask, TaughtTaskLibrary
from .style import STATUS_COLORS


_STEP_HEADERS = ("#", "Instruction", "Status", "Test", "✓")


# --------------------------------------------------------------------------- #
# Background worker for step testing
# --------------------------------------------------------------------------- #

class _StepTestWorker(QThread):
    """Run a single taught-step through the Executor on a background thread."""

    log_message = Signal(str)
    finished_ok = Signal()          # step passed
    finished_fail = Signal(str)     # step failed (reason)

    def __init__(
        self,
        device: DeviceManager,
        llm: LLMClient,
        instruction: str,
        task_notes: str,
        do_home_first: bool,
    ) -> None:
        super().__init__()
        self._device = device
        self._llm = llm
        self._instruction = instruction
        self._task_notes = task_notes
        self._do_home_first = do_home_first

    def run(self) -> None:
        from ..planner import Plan, PlanStep
        from ..executor import Executor, ExecutorConfig
        from ..watchers import WatcherManager

        try:
            self.log_message.emit(
                f"Using model: {self._llm.config.provider} / {self._llm.config.model}"
            )

            if self._do_home_first:
                self.log_message.emit("→ Step 1: going home + clean state…")
                self._device.go_home_and_clean()
                self.log_message.emit("  Home + clean done.")

            plan = Plan(
                goal=f"Execute this single instruction: {self._instruction}",
                task_notes=self._task_notes or "None",
                steps=[
                    PlanStep(
                        step_id=1,
                        action_description=self._instruction,
                        expected_outcome="The instruction is completed successfully",
                        is_optional=False,
                        status="pending",
                    )
                ],
            )

            config = ExecutorConfig(max_iterations=15, use_vision=False)
            watchers = WatcherManager(self._device.d)
            executor = Executor(
                device=self._device,
                llm=self._llm,
                watchers=watchers,
                config=config,
                on_log=lambda msg: self.log_message.emit(msg),
            )

            results = executor.run(plan)
            if results and results[0].success:
                self.finished_ok.emit()
            else:
                note = results[0].note if results else "unknown error"
                self.finished_fail.emit(note)
        except Exception as exc:
            self.finished_fail.emit(str(exc))


# --------------------------------------------------------------------------- #
# Teach Tab
# --------------------------------------------------------------------------- #

class TeachTab(QWidget):
    task_saved = Signal(object)        # TaughtTask
    run_taught_task = Signal(object)   # TaughtTask — request to run it
    log = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._device: Optional[DeviceManager] = None
        self._llm: Optional[LLMClient] = None
        self._library = TaughtTaskLibrary()
        self._current_task: Optional[TaughtTask] = None
        self._test_worker: Optional[_StepTestWorker] = None
        self._testing_row: int = -1
        self._build()
        self._refresh_saved_list()

    # ---- public API ----------------------------------------------------- #

    def attach_runtime(self, device: DeviceManager, llm: LLMClient) -> None:
        self._device = device
        self._llm = llm

    # ---- UI ------------------------------------------------------------- #

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(14)

        title = QLabel("Teach — Build Custom Tasks")
        title.setObjectName("h1")
        root.addWidget(title)

        hint = QLabel(
            "Create a task step by step. Each step is a natural-language "
            "instruction that the AI will execute on the device. Test each "
            "step individually, verify it works, then save the task. "
            "You can run saved tasks from the Run menu with AI fallback."
        )
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        root.addWidget(hint)

        # --- Top: Saved tasks list + New/Delete ---
        saved_row = QHBoxLayout()
        saved_row.setSpacing(8)
        saved_label = QLabel("Saved Tasks:")
        saved_label.setObjectName("h2")
        saved_row.addWidget(saved_label)

        self.btn_new = QPushButton("＋ New Task")
        self.btn_new.setObjectName("primary")
        self.btn_new.clicked.connect(self._on_new_task)
        saved_row.addWidget(self.btn_new)

        self.btn_delete = QPushButton("🗑 Delete")
        self.btn_delete.setObjectName("danger")
        self.btn_delete.setEnabled(False)
        self.btn_delete.clicked.connect(self._on_delete_task)
        saved_row.addWidget(self.btn_delete)

        self.btn_run_task = QPushButton("▶ Run Task")
        self.btn_run_task.setObjectName("primary")
        self.btn_run_task.setEnabled(False)
        self.btn_run_task.clicked.connect(self._on_run_task)
        saved_row.addWidget(self.btn_run_task)

        saved_row.addStretch(1)
        root.addLayout(saved_row)

        # Saved tasks table
        self.saved_table = QTableWidget(0, 3)
        self.saved_table.setHorizontalHeaderLabels(["Name", "Steps", "Verified"])
        self.saved_table.horizontalHeader().setStretchLastSection(True)
        self.saved_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.saved_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.saved_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.saved_table.setMaximumHeight(140)
        self.saved_table.itemSelectionChanged.connect(self._on_saved_selected)
        root.addWidget(self.saved_table)

        # --- Splitter: Left = task config, Right = step editor ---
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # LEFT PANEL — Task metadata
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 10, 0)
        left_layout.setSpacing(10)

        left_layout.addWidget(QLabel("Task Name:"))
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g., Create Google Account")
        left_layout.addWidget(self.name_input)

        left_layout.addWidget(QLabel("Notes (the AI reads these every turn):"))
        self.notes_input = QPlainTextEdit()
        self.notes_input.setPlaceholderText(
            "Variables and context for the AI.\n"
            "Example:\n"
            "Ad: Emre\n"
            "Soyad: Yılmaz\n"
            "Mail: emreyilmaz8932@gmail.com\n"
            "Şifre: Guclu.Sifre.2024!\n"
            "Doğum tarihi: 15/03/1995"
        )
        self.notes_input.setMaximumHeight(180)
        left_layout.addWidget(self.notes_input)

        loop_row = QHBoxLayout()
        loop_row.addWidget(QLabel("Loop Count:"))
        self.loop_spin = QSpinBox()
        self.loop_spin.setRange(1, 100)
        self.loop_spin.setValue(1)
        self.loop_spin.setToolTip("How many times to repeat this task")
        loop_row.addWidget(self.loop_spin)
        loop_row.addStretch(1)
        left_layout.addLayout(loop_row)

        self.btn_save = QPushButton("💾 Save Task")
        self.btn_save.setObjectName("primary")
        self.btn_save.clicked.connect(self._on_save)
        left_layout.addWidget(self.btn_save)

        left_layout.addStretch(1)
        splitter.addWidget(left)

        # RIGHT PANEL — Step list
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(10, 0, 0, 0)
        right_layout.setSpacing(10)

        step_header = QHBoxLayout()
        step_header.addWidget(QLabel("Steps (first step is always 'Go Home'):"))
        step_header.addStretch(1)

        self.btn_add_step = QPushButton("＋ Add Step")
        self.btn_add_step.clicked.connect(self._on_add_step)
        step_header.addWidget(self.btn_add_step)

        self.btn_remove_step = QPushButton("− Remove")
        self.btn_remove_step.clicked.connect(self._on_remove_step)
        step_header.addWidget(self.btn_remove_step)

        right_layout.addLayout(step_header)

        # Step table
        self.step_table = QTableWidget(0, len(_STEP_HEADERS))
        self.step_table.setHorizontalHeaderLabels(_STEP_HEADERS)
        hdr = self.step_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.step_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        right_layout.addWidget(self.step_table)

        # Step instruction editor
        right_layout.addWidget(QLabel("Edit selected step instruction:"))
        self.step_edit = QPlainTextEdit()
        self.step_edit.setMaximumHeight(80)
        self.step_edit.setPlaceholderText(
            "Natural-language instruction, e.g.:\n"
            "Open Gmail, tap 'Create account', select 'For my personal use'"
        )
        right_layout.addWidget(self.step_edit)

        btn_update_row = QHBoxLayout()
        self.btn_update_step = QPushButton("Update Step Text")
        self.btn_update_step.clicked.connect(self._on_update_step_text)
        btn_update_row.addWidget(self.btn_update_step)
        btn_update_row.addStretch(1)
        right_layout.addLayout(btn_update_row)

        # Test log
        right_layout.addWidget(QLabel("Test Log:"))
        self.test_log = QPlainTextEdit()
        self.test_log.setReadOnly(True)
        self.test_log.setMaximumHeight(160)
        right_layout.addWidget(self.test_log)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, stretch=1)

    # ---- Saved tasks management ----------------------------------------- #

    def _refresh_saved_list(self) -> None:
        tasks = self._library.list_tasks()
        self.saved_table.setRowCount(len(tasks))
        for row, t in enumerate(tasks):
            name_item = QTableWidgetItem(t.name)
            name_item.setData(Qt.ItemDataRole.UserRole, t)
            self.saved_table.setItem(row, 0, name_item)
            self.saved_table.setItem(row, 1, QTableWidgetItem(str(len(t.steps))))
            verified = "✓ Yes" if t.all_verified() else "✗ No"
            v_item = QTableWidgetItem(verified)
            if t.all_verified():
                v_item.setForeground(QColor("#1f8a3a"))
            else:
                v_item.setForeground(QColor("#d63031"))
            self.saved_table.setItem(row, 2, v_item)

    def _on_saved_selected(self) -> None:
        rows = self.saved_table.selectionModel().selectedRows()
        if not rows:
            self.btn_delete.setEnabled(False)
            self.btn_run_task.setEnabled(False)
            return
        row = rows[0].row()
        item = self.saved_table.item(row, 0)
        task: TaughtTask = item.data(Qt.ItemDataRole.UserRole)
        self._load_task_into_editor(task)
        self.btn_delete.setEnabled(True)
        self.btn_run_task.setEnabled(True)

    def _load_task_into_editor(self, task: TaughtTask) -> None:
        self._current_task = task
        self.name_input.setText(task.name)
        self.notes_input.setPlainText(task.notes)
        self.loop_spin.setValue(task.loop_count)
        self._render_steps()

    def _on_new_task(self) -> None:
        task = TaughtTask(name="New Task")
        # Always start with "Go to Home screen"
        task.add_step("Go to home screen and clean state (force-stop relevant apps)")
        self._current_task = task
        self.name_input.setText(task.name)
        self.notes_input.setPlainText("")
        self.loop_spin.setValue(1)
        self._render_steps()
        self.test_log.clear()
        self.saved_table.clearSelection()
        self.btn_delete.setEnabled(False)
        self.btn_run_task.setEnabled(False)

    def _on_delete_task(self) -> None:
        if self._current_task is None:
            return
        reply = QMessageBox.question(
            self, "Delete Task",
            f"Delete '{self._current_task.name}'? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._library.delete(self._current_task)
            self._current_task = None
            self.name_input.clear()
            self.notes_input.clear()
            self.step_table.setRowCount(0)
            self._refresh_saved_list()

    # ---- Step management ------------------------------------------------ #

    def _render_steps(self) -> None:
        is_testing = self._test_worker is not None and self._test_worker.isRunning()

        if self._current_task is None:
            self.step_table.setRowCount(0)
            return
        steps = self._current_task.steps
        self.step_table.setRowCount(len(steps))
        for row, step in enumerate(steps):
            # #
            idx_item = QTableWidgetItem(str(step.step_index))
            idx_item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            self.step_table.setItem(row, 0, idx_item)

            # Instruction
            instr_item = QTableWidgetItem(step.instruction[:100])
            instr_item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            instr_item.setToolTip(step.instruction)
            self.step_table.setItem(row, 1, instr_item)

            # Status
            if is_testing and row == self._testing_row:
                status_text = "⏳ Testing…"
                bg, fg = STATUS_COLORS["running"]
            elif step.verified:
                status_text = "✓ Verified"
                bg, fg = STATUS_COLORS["done"]
            elif step.last_result == "success":
                status_text = "✓ Passed"
                bg, fg = "#2a4a2a", "#8fdf8f"
            elif step.last_result == "failed":
                status_text = "✗ Failed"
                bg, fg = STATUS_COLORS["failed"]
            else:
                status_text = "— Untested"
                bg, fg = STATUS_COLORS["pending"]
            status_item = QTableWidgetItem(status_text)
            status_item.setBackground(QColor(bg))
            status_item.setForeground(QColor(fg))
            status_item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            self.step_table.setItem(row, 2, status_item)

            # Test button
            test_btn = QPushButton("▶ Test")
            test_btn.setFixedWidth(70)
            if is_testing:
                test_btn.setEnabled(False)
                if row == self._testing_row:
                    test_btn.setText("⏳")
            test_btn.clicked.connect(lambda _c=False, r=row: self._on_test_step(r))
            self.step_table.setCellWidget(row, 3, test_btn)

            # Verify checkbox button
            if step.last_result == "success" and not step.verified:
                verify_btn = QPushButton("✓")
                verify_btn.setFixedWidth(40)
                verify_btn.setStyleSheet("background: #1f8a3a; color: white; font-weight: bold;")
                verify_btn.clicked.connect(lambda _c=False, r=row: self._on_verify_step(r))
                self.step_table.setCellWidget(row, 4, verify_btn)
            elif step.verified:
                check_label = QLabel("✓")
                check_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                check_label.setStyleSheet("color: #1f8a3a; font-weight: bold; font-size: 14pt;")
                self.step_table.setCellWidget(row, 4, check_label)
            else:
                self.step_table.setCellWidget(row, 4, QLabel(""))

    def _on_add_step(self) -> None:
        if self._current_task is None:
            self._on_new_task()
        self._current_task.add_step("")
        self._render_steps()
        # Select the new row for editing
        last_row = self.step_table.rowCount() - 1
        self.step_table.selectRow(last_row)
        self.step_edit.setFocus()

    def _on_remove_step(self) -> None:
        if self._current_task is None:
            return
        rows = self.step_table.selectionModel().selectedRows()
        if not rows:
            return
        row = rows[0].row()
        if row == 0:
            QMessageBox.warning(
                self, "Cannot Remove",
                "The first step (Go Home) cannot be removed.",
            )
            return
        self._current_task.remove_step(row)
        self._render_steps()

    def _on_update_step_text(self) -> None:
        if self._current_task is None:
            return
        rows = self.step_table.selectionModel().selectedRows()
        if not rows:
            QMessageBox.information(self, "No Selection", "Select a step to update.")
            return
        row = rows[0].row()
        new_text = self.step_edit.toPlainText().strip()
        if not new_text:
            return
        if 0 <= row < len(self._current_task.steps):
            self._current_task.steps[row].instruction = new_text
            self._current_task.steps[row].verified = False
            self._current_task.steps[row].last_result = ""
            self._render_steps()

    # ---- Testing steps (background thread) ------------------------------ #

    def _on_test_step(self, row: int) -> None:
        if self._test_worker is not None and self._test_worker.isRunning():
            QMessageBox.information(
                self, "Busy", "A test is already running. Wait for it to finish."
            )
            return
        if self._device is None or self._llm is None:
            QMessageBox.warning(
                self, "Not Ready",
                "Connect a device and LLM first (Devices tab).",
            )
            return
        if self._current_task is None or row >= len(self._current_task.steps):
            return

        step = self._current_task.steps[row]
        if not step.instruction.strip():
            QMessageBox.warning(self, "Empty Step", "Write an instruction first.")
            return

        # Start the test
        self._testing_row = row
        self.test_log.clear()
        self._append_test_log(f"▶ Testing step {step.step_index}: {step.instruction}")
        self._render_steps()  # show "Testing…" status

        self._test_worker = _StepTestWorker(
            device=self._device,
            llm=self._llm,
            instruction=step.instruction,
            task_notes=self.notes_input.toPlainText().strip(),
            do_home_first=(step.step_index == 1),
        )
        self._test_worker.log_message.connect(self._append_test_log)
        self._test_worker.finished_ok.connect(self._on_test_passed)
        self._test_worker.finished_fail.connect(self._on_test_failed)
        self._test_worker.start()

    def _on_test_passed(self) -> None:
        if self._current_task and 0 <= self._testing_row < len(self._current_task.steps):
            step = self._current_task.steps[self._testing_row]
            step.last_result = "success"
            step.last_error = ""
        self._append_test_log("✓ Step completed successfully!")
        self._append_test_log("→ Did the device do the right thing? Click ✓ to verify.")
        self._testing_row = -1
        self._render_steps()

    def _on_test_failed(self, reason: str) -> None:
        if self._current_task and 0 <= self._testing_row < len(self._current_task.steps):
            step = self._current_task.steps[self._testing_row]
            step.last_result = "failed"
            step.last_error = reason
        self._append_test_log(f"✗ Step failed: {reason}")
        self._append_test_log("→ Fix the instruction and try again.")
        self._testing_row = -1
        self._render_steps()

    def _on_verify_step(self, row: int) -> None:
        if self._current_task is None or row >= len(self._current_task.steps):
            return
        self._current_task.steps[row].verified = True
        self._render_steps()
        self._append_test_log(f"✓ Step {row + 1} marked as verified.")

    # ---- Save ----------------------------------------------------------- #

    def _on_save(self) -> None:
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "No Name", "Give the task a name.")
            return

        if self._current_task is None:
            self._current_task = TaughtTask()

        self._current_task.name = name
        self._current_task.notes = self.notes_input.toPlainText().strip()
        self._current_task.loop_count = self.loop_spin.value()

        # Ensure first step is always "go home"
        if not self._current_task.steps:
            self._current_task.add_step(
                "Go to home screen and clean state (force-stop relevant apps)"
            )

        self._library.save(self._current_task)
        self._refresh_saved_list()
        self.task_saved.emit(self._current_task)
        self._append_test_log(f"💾 Task '{name}' saved ({len(self._current_task.steps)} steps).")

    # ---- Run task ------------------------------------------------------- #

    def _on_run_task(self) -> None:
        if self._current_task is None:
            return
        if not self._current_task.all_verified():
            reply = QMessageBox.question(
                self, "Unverified Steps",
                "Not all steps are verified. Run anyway with AI fallback?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        self.run_taught_task.emit(self._current_task)

    # ---- Log ------------------------------------------------------------ #

    def _append_test_log(self, msg: str) -> None:
        ts = datetime.now().strftime("[%H:%M:%S]")
        self.test_log.appendPlainText(f"{ts} {msg}")
        self.log.emit(msg)
