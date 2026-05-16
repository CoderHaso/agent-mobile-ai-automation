"""Dark Qt stylesheet shared across all widgets.

Kept tiny on purpose — Qt's default Fusion style already looks good in
dark mode, so we only override the few places that affect readability:
table headers, tab strip, button affordances, and status colors.
"""

from __future__ import annotations

DARK_QSS = """
QMainWindow, QWidget {
    background-color: #1e1f29;
    color: #e6e6e6;
    font-family: "Segoe UI", "Inter", "Helvetica Neue", sans-serif;
    font-size: 10pt;
}

QLabel#h1 {
    font-size: 16pt;
    font-weight: 600;
    color: #ffffff;
}

QLabel#h2 {
    font-size: 11pt;
    font-weight: 600;
    color: #b8b8c8;
}

QLabel#muted {
    color: #8e8ea0;
}

QLabel#statusBadge {
    padding: 2px 8px;
    border-radius: 8px;
    background: #2a2b3a;
    color: #b8b8c8;
}

QPushButton {
    background-color: #2a2b3a;
    color: #e6e6e6;
    border: 1px solid #3a3b4d;
    padding: 6px 14px;
    border-radius: 6px;
    min-height: 24px;
}
QPushButton:hover  { background-color: #353749; border-color: #4a4d63; }
QPushButton:pressed{ background-color: #1f2030; }
QPushButton:disabled{ color: #666; border-color: #2c2d3a; background: #232431; }

QPushButton#primary {
    background-color: #6c5ce7;
    color: white;
    border: none;
    font-weight: 600;
}
QPushButton#primary:hover    { background-color: #7d6ff0; }
QPushButton#primary:disabled { background-color: #3d3a55; color: #888; }

QPushButton#danger {
    background-color: #d63031;
    color: white;
    border: none;
    font-weight: 600;
}
QPushButton#danger:hover { background-color: #e0494a; }

QLineEdit, QPlainTextEdit, QTextEdit, QComboBox {
    background-color: #14151c;
    color: #e6e6e6;
    border: 1px solid #2c2d3a;
    border-radius: 6px;
    padding: 6px 8px;
    selection-background-color: #6c5ce7;
}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QComboBox:focus {
    border: 1px solid #6c5ce7;
}

QTableWidget, QTableView {
    background-color: #14151c;
    alternate-background-color: #181925;
    gridline-color: #2c2d3a;
    border: 1px solid #2c2d3a;
    border-radius: 6px;
    selection-background-color: #6c5ce7;
    selection-color: #ffffff;
}
QHeaderView::section {
    background-color: #232431;
    color: #b8b8c8;
    padding: 6px;
    border: none;
    border-right: 1px solid #2c2d3a;
    font-weight: 600;
}

QTabWidget::pane {
    border: 1px solid #2c2d3a;
    border-radius: 6px;
    top: -1px;
    background: #1e1f29;
}
QTabBar::tab {
    background: #232431;
    color: #8e8ea0;
    padding: 8px 18px;
    border: 1px solid #2c2d3a;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background: #1e1f29;
    color: #ffffff;
    border-bottom: 2px solid #6c5ce7;
}
QTabBar::tab:disabled { color: #555; }

QStatusBar {
    background: #14151c;
    color: #b8b8c8;
    border-top: 1px solid #2c2d3a;
}

QProgressBar {
    background: #14151c;
    border: 1px solid #2c2d3a;
    border-radius: 6px;
    color: #e6e6e6;
    text-align: center;
}
QProgressBar::chunk { background-color: #6c5ce7; border-radius: 6px; }

QScrollBar:vertical {
    background: #1e1f29; width: 10px; border: none; margin: 0;
}
QScrollBar::handle:vertical {
    background: #3a3b4d; border-radius: 5px; min-height: 30px;
}
QScrollBar::handle:vertical:hover { background: #4a4d63; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
"""


# Status -> (background, foreground) for table cell highlights.
STATUS_COLORS = {
    "pending":  ("#2a2b3a", "#8e8ea0"),
    "running":  ("#6c5ce7", "#ffffff"),
    "done":     ("#1f8a3a", "#ffffff"),
    "failed":   ("#d63031", "#ffffff"),
    "skipped":  ("#3a3b4d", "#b8b8c8"),
}


# Device state -> color for the picker table.
DEVICE_STATE_COLORS = {
    "device":       "#1f8a3a",
    "offline":      "#d63031",
    "unauthorized": "#fdcb6e",
}
