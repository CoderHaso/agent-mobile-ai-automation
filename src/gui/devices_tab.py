"""Tab 1 — Device picker.

Shows every device `adb devices -l` knows about and lets the user:
  - Refresh the list
  - Connect (`adb connect host:port`) a wireless device
  - Disconnect (`adb disconnect …`) a TCP/wireless device
  - Activate exactly one device for the session (CLI's "active" flag)
  - Open the actual uiautomator2 session for the active device
"""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..device_manager import (
    AdbDeviceInfo,
    DeviceManager,
    adb_connect,
    adb_disconnect,
)
from .style import DEVICE_STATE_COLORS
from .workers import ConnectDeviceWorker, DeviceScanWorker


_HEADERS = ("Active", "Serial", "Model", "SDK", "Transport", "State")


class DevicesTab(QWidget):
    device_connected = Signal(object)   # DeviceManager
    log = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._devices: List[AdbDeviceInfo] = []
        self._scan_worker: Optional[DeviceScanWorker] = None
        self._connect_worker: Optional[ConnectDeviceWorker] = None
        self._build()
        self.refresh()

    # ---- UI ------------------------------------------------------------- #

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        title = QLabel("Step 1 — Choose your Android device")
        title.setObjectName("h1")
        layout.addWidget(title)

        hint = QLabel(
            "Pick which device the agent will operate on. You can also add a "
            "wireless device (adb connect) or remove one (adb disconnect) without "
            "leaving the app."
        )
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # --- table ---
        self.table = QTableWidget(0, len(_HEADERS), self)
        self.table.setHorizontalHeaderLabels(_HEADERS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        for i in range(2, len(_HEADERS)):
            header.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        self.table.itemDoubleClicked.connect(self._on_double_click)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.table, stretch=1)

        # --- wireless connect row ---
        wireless_row = QHBoxLayout()
        wireless_row.setSpacing(8)
        self.host_input = QLineEdit()
        self.host_input.setPlaceholderText("Wireless host:port  (e.g. 192.168.1.5:5555)")
        wireless_row.addWidget(self.host_input, stretch=1)
        self.btn_connect_tcp = QPushButton("Connect (adb connect)")
        self.btn_connect_tcp.clicked.connect(self._on_connect_tcp)
        wireless_row.addWidget(self.btn_connect_tcp)
        layout.addLayout(wireless_row)

        # --- action row ---
        actions = QHBoxLayout()
        actions.setSpacing(8)
        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self.refresh)
        actions.addWidget(self.btn_refresh)

        self.btn_activate = QPushButton("Activate selected")
        self.btn_activate.clicked.connect(self._on_activate_selected)
        actions.addWidget(self.btn_activate)

        self.btn_disconnect = QPushButton("Disconnect selected")
        self.btn_disconnect.clicked.connect(self._on_disconnect_selected)
        actions.addWidget(self.btn_disconnect)

        actions.addStretch(1)

        self.active_label = QLabel("No device active.")
        self.active_label.setObjectName("statusBadge")
        actions.addWidget(self.active_label)

        self.btn_use = QPushButton("Use this device →")
        self.btn_use.setObjectName("primary")
        self.btn_use.setEnabled(False)
        self.btn_use.clicked.connect(self._on_use)
        actions.addWidget(self.btn_use)

        layout.addLayout(actions)

    # ---- helpers -------------------------------------------------------- #

    def _active(self) -> Optional[AdbDeviceInfo]:
        for d in self._devices:
            if d.active:
                return d
        return None

    def _selected_index(self) -> int:
        rows = self.table.selectionModel().selectedRows()
        return rows[0].row() if rows else -1

    def _set_active(self, idx: int) -> None:
        for i, d in enumerate(self._devices):
            d.active = (i == idx)
        self._render_table()
        self._refresh_active_label()

    def _refresh_active_label(self) -> None:
        a = self._active()
        if a is None:
            self.active_label.setText("No device active.")
            self.btn_use.setEnabled(False)
        else:
            self.active_label.setText(f"Active: {a.serial}  ({a.model or '?'})")
            self.btn_use.setEnabled(a.is_usable)

    def _render_table(self) -> None:
        self.table.setRowCount(len(self._devices))
        for row, d in enumerate(self._devices):
            check = QTableWidgetItem("✓" if d.active else "")
            check.setTextAlignment(Qt.AlignCenter)
            if d.active:
                check.setForeground(QColor("#6c5ce7"))
            self.table.setItem(row, 0, check)
            self.table.setItem(row, 1, QTableWidgetItem(d.serial))
            self.table.setItem(row, 2, QTableWidgetItem(d.model or "—"))
            self.table.setItem(row, 3, QTableWidgetItem(d.sdk or "—"))
            self.table.setItem(row, 4, QTableWidgetItem(d.transport))
            state_item = QTableWidgetItem(d.state)
            state_item.setForeground(QColor(DEVICE_STATE_COLORS.get(d.state, "#e6e6e6")))
            self.table.setItem(row, 5, state_item)

    # ---- slots ---------------------------------------------------------- #

    def refresh(self) -> None:
        self.btn_refresh.setEnabled(False)
        self.btn_refresh.setText("Scanning…")
        self.log.emit("Scanning for ADB devices…")

        self._scan_worker = DeviceScanWorker()
        self._scan_worker.finished_with_devices.connect(self._on_scan_done)
        self._scan_worker.failed.connect(self._on_scan_failed)
        self._scan_worker.start()

    def _on_scan_done(self, devices: list) -> None:
        self.btn_refresh.setEnabled(True)
        self.btn_refresh.setText("Refresh")
        prev_active = self._active().serial if self._active() else None
        self._devices = list(devices)
        # Restore previous selection or auto-pick if exactly one usable.
        if prev_active:
            for d in self._devices:
                if d.serial == prev_active:
                    d.active = True
                    break
        if not any(d.active for d in self._devices):
            usable = [d for d in self._devices if d.is_usable]
            if len(usable) == 1:
                usable[0].active = True
        self._render_table()
        self._refresh_active_label()
        self.log.emit(f"Found {len(self._devices)} device(s).")

    def _on_scan_failed(self, msg: str) -> None:
        self.btn_refresh.setEnabled(True)
        self.btn_refresh.setText("Refresh")
        QMessageBox.critical(self, "ADB scan failed", msg)
        self.log.emit(f"ADB scan failed: {msg}")

    def _on_double_click(self, _item) -> None:
        idx = self._selected_index()
        if idx >= 0:
            self._set_active(idx)

    def _on_selection_changed(self) -> None:
        idx = self._selected_index()
        self.btn_disconnect.setEnabled(
            idx >= 0 and self._devices[idx].is_tcp
        )
        self.btn_activate.setEnabled(idx >= 0)

    def _on_activate_selected(self) -> None:
        idx = self._selected_index()
        if idx < 0:
            QMessageBox.information(self, "Nothing selected",
                                    "Select a device row first.")
            return
        self._set_active(idx)

    def _on_connect_tcp(self) -> None:
        host = self.host_input.text().strip()
        if not host:
            QMessageBox.information(self, "Missing host",
                                    "Enter the device IP and port "
                                    "(e.g. 192.168.1.5:5555).")
            return
        self.log.emit(f"adb connect {host} …")
        msg = adb_connect(host)
        self.log.emit(f"adb: {msg}")
        QMessageBox.information(self, "adb connect", msg)
        self.host_input.clear()
        self.refresh()

    def _on_disconnect_selected(self) -> None:
        idx = self._selected_index()
        if idx < 0:
            return
        d = self._devices[idx]
        if not d.is_tcp:
            QMessageBox.information(self, "Cannot disconnect",
                                    f"{d.serial} is a USB device. Unplug it "
                                    "physically or run `adb kill-server`.")
            return
        self.log.emit(f"adb disconnect {d.serial} …")
        msg = adb_disconnect(d.serial)
        self.log.emit(f"adb: {msg}")
        self.refresh()

    def _on_use(self) -> None:
        active = self._active()
        if active is None or not active.is_usable:
            return
        self.btn_use.setEnabled(False)
        self.btn_use.setText("Connecting…")
        self.log.emit(f"Opening uiautomator2 session on {active.serial} …")

        self._connect_worker = ConnectDeviceWorker(active.serial)
        self._connect_worker.finished_with_device.connect(self._on_connected)
        self._connect_worker.failed.connect(self._on_connect_failed)
        self._connect_worker.start()

    def _on_connected(self, dm: DeviceManager) -> None:
        self.btn_use.setText("Use this device →")
        self.btn_use.setEnabled(True)
        info = dm.info()
        self.log.emit(
            f"Connected: {info.get('productName', '?')} "
            f"({info.get('displayWidth')}x{info.get('displayHeight')}, "
            f"sdk {info.get('sdkInt')})"
        )
        self.device_connected.emit(dm)

    def _on_connect_failed(self, msg: str) -> None:
        self.btn_use.setText("Use this device →")
        self.btn_use.setEnabled(True)
        QMessageBox.critical(self, "Cannot connect", msg)
        self.log.emit(f"Connect failed: {msg}")
