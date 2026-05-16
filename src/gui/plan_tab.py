"""Tab 2 — Goal entry + Plan review (Human-in-the-Loop).

Lets the user:
  1. Type a high-level goal.
  2. Generate a plan via the LLM (off-thread).
  3. Edit / add / delete / reorder steps inline.
  4. Approve the plan, which triggers execution on tab 3.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..llm_client import LLMClient
from ..planner import Plan, PlanStep
from ..task_library import TaskLibrary
from .model_picker import ModelPicker
from .style import STATUS_COLORS
from .workers import PlannerWorker


_HEADERS = (
    "#",
    "Milestone / objective (editable)",
    "Recognize-when-done hint (editable)",
    "Optional",
    "Status",
)


class PlanTab(QWidget):
    plan_approved = Signal(object)        # Plan
    log = Signal(str)
    model_changed = Signal(str, str)      # (provider, model_slug)

    def __init__(self, library: Optional[TaskLibrary] = None) -> None:
        super().__init__()
        self._llm: Optional[LLMClient] = None
        self._plan: Optional[Plan] = None
        self._worker: Optional[PlannerWorker] = None
        self._library = library or TaskLibrary()
        self._build()
        self.refresh_library()

    # ---- public API ----------------------------------------------------- #

    def attach_llm(self, llm: LLMClient) -> None:
        """Called by MainWindow once the LLM client is (re)initialized."""
        self._llm = llm
        self.btn_generate.setEnabled(True)
        self.lbl_provider.setText(f"Active LLM: {llm.describe()}")
        # Sync the model picker if the active client differs.
        self.model_picker.set_selection(llm.config.provider, llm.config.model)

    def initial_model_choice(self) -> tuple:
        """Provider/slug currently selected in the picker (for app boot)."""
        return self.model_picker.current_provider(), self.model_picker.current_slug()

    def use_vision_enabled(self) -> bool:
        return self.chk_vision.isChecked()

    def reset(self) -> None:
        self._plan = None
        self.table.setRowCount(0)
        self.btn_approve.setEnabled(False)

    # ---- UI ------------------------------------------------------------- #

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        title = QLabel("Step 2 — Goal & adaptive milestones")
        title.setObjectName("h1")
        layout.addWidget(title)

        hint = QLabel(
            "Type the high-level task you want to achieve. The Planner Agent "
            "produces a list of MILESTONES (broad objectives, not exact taps). "
            "At runtime, the Executor watches the live screen and decides each "
            "individual UI action — so the same plan adapts across Samsung / "
            "Xiaomi / Huawei and skips steps the flow doesn't actually ask for. "
            "Mark a milestone as Optional if it may not appear in every flow."
        )
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # --- model picker (above goal input) ---
        self.model_picker = ModelPicker()
        self.model_picker.model_changed.connect(self.model_changed.emit)
        layout.addWidget(self.model_picker)

        vision_row = QHBoxLayout()
        self.chk_vision = QCheckBox("Vision — ekran görüntüsünü LLM'e gönder")
        self.chk_vision.setChecked(False)
        self.chk_vision.setToolTip(
            "Kapalı (önerilen): yalnızca UI XML kullanılır; Groq modelleriyle uyumlu.\n"
            "Açık: screenshot LLM'e eklenir — yalnızca DeepSeek V4 Flash/Pro gibi "
            "vision destekleyen modellerde işe yarar."
        )
        vision_row.addWidget(self.chk_vision)
        vision_row.addStretch(1)
        layout.addLayout(vision_row)

        # --- goal row ---
        goal_row = QHBoxLayout()
        goal_row.setSpacing(8)
        self.goal_input = QLineEdit()
        self.goal_input.setPlaceholderText(
            "e.g. Create a new Gmail account on this device"
        )
        self.goal_input.returnPressed.connect(self._on_generate)
        self.goal_input.textChanged.connect(self._on_goal_changed)
        goal_row.addWidget(self.goal_input, stretch=1)

        self.btn_generate = QPushButton("Generate plan")
        self.btn_generate.setObjectName("primary")
        self.btn_generate.setEnabled(False)
        self.btn_generate.clicked.connect(self._on_generate)
        goal_row.addWidget(self.btn_generate)
        layout.addLayout(goal_row)

        self.lbl_provider = QLabel("Active LLM: (not connected)")
        self.lbl_provider.setObjectName("muted")
        layout.addWidget(self.lbl_provider)

        self.lbl_macro_hint = QLabel("")
        self.lbl_macro_hint.setObjectName("muted")
        self.lbl_macro_hint.setWordWrap(True)
        layout.addWidget(self.lbl_macro_hint)

        # --- plan table ---
        self.table = QTableWidget(0, len(_HEADERS), self)
        self.table.setHorizontalHeaderLabels(_HEADERS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.table, stretch=1)

        # --- saved macros panel ---
        macros_frame = QFrame()
        macros_frame.setObjectName("macroFrame")
        macros_frame.setStyleSheet(
            "QFrame#macroFrame{background:#181923;border:1px solid #2c2d3a;"
            "border-radius:8px;padding:8px;}"
        )
        macros_layout = QVBoxLayout(macros_frame)
        macros_layout.setContentsMargins(8, 6, 8, 6)
        macros_layout.setSpacing(6)

        macros_header = QHBoxLayout()
        macros_label = QLabel("📚 Saved macros (zero-LLM replays)")
        macros_label.setObjectName("h2")
        macros_header.addWidget(macros_label)
        macros_header.addStretch(1)
        self.btn_macro_refresh = QPushButton("Refresh")
        self.btn_macro_refresh.clicked.connect(self.refresh_library)
        macros_header.addWidget(self.btn_macro_refresh)
        self.btn_macro_delete = QPushButton("Delete")
        self.btn_macro_delete.setObjectName("danger")
        self.btn_macro_delete.clicked.connect(self._on_delete_macro)
        macros_header.addWidget(self.btn_macro_delete)
        macros_layout.addLayout(macros_header)

        self.macro_list = QListWidget()
        self.macro_list.setMaximumHeight(110)
        self.macro_list.itemDoubleClicked.connect(self._on_macro_double_clicked)
        macros_layout.addWidget(self.macro_list)

        layout.addWidget(macros_frame)

        # --- edit controls ---
        edit_row = QHBoxLayout()
        edit_row.setSpacing(8)

        self.btn_add = QPushButton("Add milestone")
        self.btn_add.clicked.connect(self._on_add_step)
        edit_row.addWidget(self.btn_add)

        self.btn_delete = QPushButton("Delete selected")
        self.btn_delete.clicked.connect(self._on_delete_step)
        edit_row.addWidget(self.btn_delete)

        self.btn_toggle_opt = QPushButton("Toggle Optional")
        self.btn_toggle_opt.clicked.connect(self._on_toggle_optional)
        edit_row.addWidget(self.btn_toggle_opt)

        self.btn_up = QPushButton("Move up")
        self.btn_up.clicked.connect(lambda: self._move_step(-1))
        edit_row.addWidget(self.btn_up)

        self.btn_down = QPushButton("Move down")
        self.btn_down.clicked.connect(lambda: self._move_step(+1))
        edit_row.addWidget(self.btn_down)

        edit_row.addStretch(1)

        self.btn_approve = QPushButton("Approve plan & start execution →")
        self.btn_approve.setObjectName("primary")
        self.btn_approve.setEnabled(False)
        self.btn_approve.clicked.connect(self._on_approve)
        edit_row.addWidget(self.btn_approve)

        layout.addLayout(edit_row)

    # ---- generation ----------------------------------------------------- #

    def _on_generate(self) -> None:
        if self._llm is None:
            QMessageBox.warning(self, "LLM not ready",
                                "Wait for the LLM provider to initialize first.")
            return
        goal = self.goal_input.text().strip()
        if not goal:
            QMessageBox.information(self, "Empty goal",
                                    "Please type a goal first.")
            return
        self.btn_generate.setEnabled(False)
        self.btn_generate.setText("Generating…")
        self.log.emit(f"Asking the planner to break down: {goal!r}")

        self._worker = PlannerWorker(self._llm, goal)
        self._worker.finished_with_plan.connect(self._on_plan_ready)
        self._worker.failed.connect(self._on_plan_failed)
        self._worker.start()

    def _on_plan_ready(self, plan: Plan) -> None:
        self.btn_generate.setEnabled(True)
        self.btn_generate.setText("Generate plan")
        self._plan = plan.renumber()
        self._render_plan()
        self.btn_approve.setEnabled(True)
        self.log.emit(f"Planner returned {len(plan.steps)} step(s).")

    def _on_plan_failed(self, msg: str) -> None:
        self.btn_generate.setEnabled(True)
        self.btn_generate.setText("Generate plan")
        QMessageBox.critical(self, "Planner failed", msg)
        self.log.emit(f"Planner failed: {msg}")

    # ---- table rendering & editing ------------------------------------- #

    def _render_plan(self) -> None:
        if self._plan is None:
            self.table.setRowCount(0)
            return
        self.table.blockSignals(True)
        try:
            self.table.setRowCount(len(self._plan.steps))
            for row, step in enumerate(self._plan.steps):
                id_item = QTableWidgetItem(str(step.step_id))
                id_item.setFlags(id_item.flags() & ~Qt.ItemIsEditable)
                id_item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row, 0, id_item)

                self.table.setItem(row, 1, QTableWidgetItem(step.action_description))
                self.table.setItem(row, 2, QTableWidgetItem(step.expected_outcome or ""))

                opt_item = QTableWidgetItem("Yes" if step.is_optional else "No")
                opt_item.setFlags(opt_item.flags() & ~Qt.ItemIsEditable)
                opt_item.setTextAlignment(Qt.AlignCenter)
                if step.is_optional:
                    opt_item.setForeground(QColor("#fdcb6e"))
                self.table.setItem(row, 3, opt_item)

                status_item = QTableWidgetItem(step.status.upper())
                status_item.setFlags(status_item.flags() & ~Qt.ItemIsEditable)
                status_item.setTextAlignment(Qt.AlignCenter)
                bg, fg = STATUS_COLORS.get(step.status, ("#2a2b3a", "#e6e6e6"))
                status_item.setBackground(QColor(bg))
                status_item.setForeground(QColor(fg))
                self.table.setItem(row, 4, status_item)
        finally:
            self.table.blockSignals(False)

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._plan is None:
            return
        row, col = item.row(), item.column()
        if not (0 <= row < len(self._plan.steps)):
            return
        step = self._plan.steps[row]
        if col == 1:
            step.action_description = item.text().strip() or step.action_description
        elif col == 2:
            step.expected_outcome = item.text().strip()

    def _selected_row(self) -> int:
        rows = self.table.selectionModel().selectedRows()
        return rows[0].row() if rows else -1

    def _on_add_step(self) -> None:
        if self._plan is None:
            self._plan = Plan(goal=self.goal_input.text().strip() or "Manual plan")
        pos = self._selected_row()
        new_step = PlanStep(
            step_id=len(self._plan.steps) + 1,
            action_description="(describe the new milestone)",
            expected_outcome="",
            is_optional=False,
            status="pending",
        )
        if pos < 0:
            self._plan.steps.append(new_step)
        else:
            self._plan.steps.insert(pos + 1, new_step)
        self._plan.renumber()
        self._render_plan()
        self.btn_approve.setEnabled(True)

    def _on_toggle_optional(self) -> None:
        if self._plan is None:
            return
        pos = self._selected_row()
        if pos < 0:
            return
        self._plan.steps[pos].is_optional = not self._plan.steps[pos].is_optional
        self._render_plan()

    def _on_delete_step(self) -> None:
        if self._plan is None:
            return
        pos = self._selected_row()
        if pos < 0:
            return
        del self._plan.steps[pos]
        self._plan.renumber()
        self._render_plan()
        self.btn_approve.setEnabled(bool(self._plan.steps))

    def _move_step(self, delta: int) -> None:
        if self._plan is None:
            return
        pos = self._selected_row()
        if pos < 0:
            return
        new_pos = pos + delta
        if not (0 <= new_pos < len(self._plan.steps)):
            return
        steps = self._plan.steps
        steps[pos], steps[new_pos] = steps[new_pos], steps[pos]
        self._plan.renumber()
        self._render_plan()
        self.table.selectRow(new_pos)

    # ---- approval ------------------------------------------------------- #

    def _on_approve(self) -> None:
        if self._plan is None or not self._plan.steps:
            return
        self.plan_approved.emit(self._plan.renumber())
        self.log.emit(f"Plan approved ({len(self._plan.steps)} step(s)). Starting execution…")

    # ---- macro library -------------------------------------------------- #

    def refresh_library(self) -> None:
        """Reload the saved-macros panel from disk."""
        self.macro_list.clear()
        for task in self._library.list_tasks():
            text = (
                f"{task.goal}  ·  {len(task.actions)} steps  "
                f"·  id={task.task_id()}"
            )
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, task.task_id())
            item.setData(Qt.UserRole + 1, task.goal)
            self.macro_list.addItem(item)
        if self.macro_list.count() == 0:
            placeholder = QListWidgetItem(
                "No macros saved yet. Run a goal to completion and accept "
                "the 'Save macro?' dialog."
            )
            placeholder.setFlags(Qt.NoItemFlags)
            placeholder.setForeground(QColor("#8e8ea0"))
            self.macro_list.addItem(placeholder)
        self._refresh_macro_hint()

    def _on_macro_double_clicked(self, item: QListWidgetItem) -> None:
        goal = item.data(Qt.UserRole + 1)
        if not goal:
            return
        self.goal_input.setText(goal)
        self._refresh_macro_hint()
        QMessageBox.information(
            self, "Macro queued",
            "Goal copied. Click 'Generate plan' (or just edit & approve) — "
            "at run time the matching macro will be replayed automatically.",
        )

    def _on_delete_macro(self) -> None:
        item = self.macro_list.currentItem()
        if item is None:
            return
        task_id = item.data(Qt.UserRole)
        if not task_id:
            return
        if QMessageBox.question(
            self, "Delete macro?",
            f"Permanently delete macro [{task_id}]?",
        ) != QMessageBox.Yes:
            return
        if self._library.delete(task_id):
            self.log.emit(f"Deleted macro {task_id}")
        self.refresh_library()

    def _on_goal_changed(self, _text: str) -> None:
        self._refresh_macro_hint()

    def _refresh_macro_hint(self) -> None:
        goal = self.goal_input.text().strip()
        if not goal:
            self.lbl_macro_hint.setText("")
            return
        macro = self._library.find_for_goal(goal)
        if macro is None:
            self.lbl_macro_hint.setText(
                "No matching macro — this run will use the AI agent and "
                "record one if it succeeds."
            )
            self.lbl_macro_hint.setStyleSheet("color:#8e8ea0;")
        else:
            self.lbl_macro_hint.setText(
                f"⚡ Library match: [{macro.task_id()}] · "
                f"{len(macro.actions)} steps will be replayed without the LLM."
            )
            self.lbl_macro_hint.setStyleSheet("color:#1f8a3a;font-weight:600;")
