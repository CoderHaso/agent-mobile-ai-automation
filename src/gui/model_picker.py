"""LLM provider + model picker widget.

Compact bar that lives at the top of the Plan tab. Lets the user switch
between Groq and DeepSeek and pick any of the registered models. Each
entry is annotated with quality/speed stars and per-1M-token pricing,
so the cost/performance trade-off is visible at a glance.

Emits `model_changed(provider: str, model_slug: str)` whenever the user
selects a different combination. The MainWindow listens to this signal
and rebuilds the LLMClient.
"""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from ..models import (
    ALL_MODELS,
    PROVIDERS,
    ModelInfo,
    by_provider,
    default_for,
    find,
)


class ModelPicker(QWidget):
    model_changed = Signal(str, str)   # (provider, slug)

    def __init__(
        self,
        initial_provider: str = "groq",
        initial_slug: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._suppress = False
        self._build()
        self.set_selection(initial_provider, initial_slug)

    # ---- public API ----------------------------------------------------- #

    def current_provider(self) -> str:
        return self.cmb_provider.currentText()

    def current_slug(self) -> str:
        data = self.cmb_model.currentData()
        if isinstance(data, ModelInfo):
            return data.slug
        return ""

    def current_model(self) -> Optional[ModelInfo]:
        data = self.cmb_model.currentData()
        return data if isinstance(data, ModelInfo) else None

    def set_selection(self, provider: str, slug: Optional[str]) -> None:
        was_suppressed = self._suppress
        self._suppress = True
        try:
            idx_p = max(0, self.cmb_provider.findText(provider))
            self.cmb_provider.setCurrentIndex(idx_p)
            self._populate_models(self.cmb_provider.currentText())
            if slug:
                for i in range(self.cmb_model.count()):
                    m = self.cmb_model.itemData(i)
                    if isinstance(m, ModelInfo) and m.slug == slug:
                        self.cmb_model.setCurrentIndex(i)
                        break
        finally:
            self._suppress = was_suppressed
        self._refresh_info()

    # ---- UI ------------------------------------------------------------- #

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        frame = QFrame()
        frame.setObjectName("modelPickerFrame")
        frame.setStyleSheet(
            "#modelPickerFrame { background: #14151c; border: 1px solid "
            "#2c2d3a; border-radius: 6px; padding: 8px; }"
        )
        outer.addWidget(frame)

        row = QHBoxLayout(frame)
        row.setSpacing(10)

        # Title
        title = QLabel("LLM Model")
        title.setObjectName("h2")
        row.addWidget(title)

        # Provider combo
        row.addWidget(QLabel("Provider:"))
        self.cmb_provider = QComboBox()
        for p in PROVIDERS:
            self.cmb_provider.addItem(p)
        self.cmb_provider.currentTextChanged.connect(self._on_provider_changed)
        row.addWidget(self.cmb_provider)

        # Model combo (wide)
        row.addWidget(QLabel("Model:"))
        self.cmb_model = QComboBox()
        self.cmb_model.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.cmb_model.setMinimumWidth(420)
        self.cmb_model.currentIndexChanged.connect(self._on_model_changed)
        row.addWidget(self.cmb_model, stretch=1)

        # Info column (quality/speed/cost)
        info_col = QVBoxLayout()
        info_col.setSpacing(2)
        self.lbl_quality = QLabel("")
        self.lbl_speed = QLabel("")
        self.lbl_cost = QLabel("")
        for w in (self.lbl_quality, self.lbl_speed, self.lbl_cost):
            w.setObjectName("muted")
        info_col.addWidget(self.lbl_quality)
        info_col.addWidget(self.lbl_speed)
        info_col.addWidget(self.lbl_cost)
        row.addLayout(info_col)

        # Notes line under the bar
        self.lbl_notes = QLabel("")
        self.lbl_notes.setObjectName("muted")
        self.lbl_notes.setWordWrap(True)
        outer.addWidget(self.lbl_notes)

    # ---- slots ---------------------------------------------------------- #

    def _populate_models(self, provider: str) -> None:
        was_suppressed = self._suppress
        self._suppress = True
        try:
            self.cmb_model.clear()
            for m in by_provider(provider):
                # Show the short label in the dropdown.
                self.cmb_model.addItem(m.short_label(), m)
                # Tooltip = full detail
                idx = self.cmb_model.count() - 1
                self.cmb_model.setItemData(idx, m.detail_label(), Qt.ToolTipRole)
            # Select default for this provider
            d = default_for(provider)
            for i in range(self.cmb_model.count()):
                if self.cmb_model.itemData(i) is d:
                    self.cmb_model.setCurrentIndex(i)
                    break
        finally:
            self._suppress = was_suppressed

    def _on_provider_changed(self, provider: str) -> None:
        if self._suppress:
            return
        self._populate_models(provider)
        self._refresh_info()
        self._emit_change()

    def _on_model_changed(self, _idx: int) -> None:
        if self._suppress:
            return
        self._refresh_info()
        self._emit_change()

    def _emit_change(self) -> None:
        m = self.current_model()
        if m is not None:
            self.model_changed.emit(m.provider, m.slug)

    def _refresh_info(self) -> None:
        m = self.current_model()
        if m is None:
            self.lbl_quality.setText("")
            self.lbl_speed.setText("")
            self.lbl_cost.setText("")
            self.lbl_notes.setText("")
            return
        self.lbl_quality.setText(f"Quality: {m.quality_stars}")
        self.lbl_speed.setText(f"Speed:   {m.speed_stars}")
        self.lbl_cost.setText(f"Cost:    {m.cost_label}")
        self.lbl_notes.setText(
            f"{m.notes}  •  context window: {m.context_k}K tokens  "
            f"•  API slug: {m.slug}"
        )
