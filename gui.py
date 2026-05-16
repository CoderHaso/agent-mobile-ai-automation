"""Desktop GUI entry point.

    python gui.py

Opens a PySide6 window with three tabs:
    1. Devices  — list / connect / disconnect / activate one device
    2. Plan     — type a goal, generate a plan, edit/approve it
    3. Run      — live execution with per-step status and log

The CLI in `main.py` still works; this is just the visual face of the
same agent (same planner, executor, watchers, device manager).
"""

from __future__ import annotations

import logging
import os
import sys

from dotenv import load_dotenv
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from src.gui.main_window import MainWindow


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "urllib3", "uiautomator2"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main() -> int:
    load_dotenv(override=False)
    _configure_logging()

    app = QApplication(sys.argv)
    app.setApplicationName("Agentic Mobile AI Automation")
    app.setOrganizationName("agentic-mobile")

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")
