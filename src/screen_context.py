"""Classify the current Android screen for smarter agent decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .device_manager import Observation


@dataclass(frozen=True)
class ScreenContext:
    current_app: str
    is_launcher: bool
    is_notification_shade: bool
    is_app_drawer: bool
    hint: str = ""

    def to_prompt_dict(self) -> dict:
        return {
            "is_launcher": self.is_launcher,
            "is_notification_shade": self.is_notification_shade,
            "is_app_drawer": self.is_app_drawer,
            "hint": self.hint,
        }


def analyze_screen(obs: Observation) -> ScreenContext:
    app = (obs.current_app or "").lower()
    xml = (obs.xml or "").lower()

    is_launcher = "launcher" in app
    # Only when SystemUI is foreground OR the expanded panel nodes exist.
    # Do NOT trigger on launcher XML that merely mentions status bar widgets.
    is_notification = (not is_launcher) and (
        app.startswith("com.android.systemui")
        or "notification_panel" in xml
        or "notification_stack_scroller" in xml
        or "expanded_public" in xml
    )
    is_drawer = (
        is_launcher
        and any(
            tok in xml
            for tok in (
                "all apps",
                "tüm uygulamalar",
                "app_list",
                "apps_list",
                "iconview",
            )
        )
    )

    hint = ""
    if is_notification:
        hint = (
            "Notification/quick-settings shade is open. Do NOT press back "
            "repeatedly — press HOME once, or swipe UP from the bottom center "
            "to close the shade."
        )
    elif is_drawer:
        hint = "App drawer / all-apps list is visible — click the target app by label."
    elif is_launcher:
        hint = (
            "On home launcher: prefer open_app with exact package from "
            "INSTALLED_APPS, or tap 'Apps' / swipe horizontally for more pages. "
            "Never scroll_to a resource-id."
        )

    return ScreenContext(
        current_app=app,
        is_launcher=is_launcher,
        is_notification_shade=is_notification,
        is_app_drawer=is_drawer,
        hint=hint,
    )
