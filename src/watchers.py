"""Background uiautomator2 Watchers.

A `Watcher` is a name + condition + action that uiautomator2 evaluates on
its own thread every time the device polls the UI. We use them to *silently*
dismiss the kinds of popups that would otherwise derail the executor:

    - Runtime permission prompts (Allow / While using the app / OK)
    - "App keeps stopping" / ANR dialogs (Close / Wait)
    - System update / "Set up your device" suggestions
    - Google Play update / sign-in nags
    - Battery optimization, notification, and storage prompts

Watchers are best-effort. If a popup is not in our list, the executor will
still see it on the next observe() and the LLM can decide what to do.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import uiautomator2 as u2

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class WatcherRule:
    """A single 'when X is on screen, click Y' rule."""

    name: str
    when_text: str           # textMatches regex
    click_text: str          # textMatches regex of the button to click


# Multilingual snippets that frequently mean "dismiss" / "later" / "no":
_DISMISS_RX = (
    r"(?i)^("
    r"no thanks|not now|later|skip|cancel|close( app)?|dismiss|"
    r"i.?ptal|sonra|atla|kapat|ge.?ç|reddet|"            # Turkish
    r"daha sonra|hay.?r|"                                  # Turkish (alt)
    r"\u4ee5\u540e|\u53d6\u6d88"                          # Chinese 以后/取消
    r")$"
)
_ACCEPT_RX = (
    r"(?i)^("
    r"ok|okay|got it|continue|next|allow( all the time)?|"
    r"while using the app|allow only while using the app|"
    r"tamam|izin ver|devam( et)?|kabul"                    # Turkish
    r")$"
)


# The order matters: the first matching watcher wins on a given screen.
DEFAULT_RULES: tuple[WatcherRule, ...] = (
    # ─── OEM cross-app popups that hijack the foreground ───────────────── #
    # These are the most painful: e.g. Galaxy Store opens *over* Gmail and
    # the agent gets stuck typing into the wrong app. We aggressively
    # dismiss anything that looks like a store / sign-in nag.
    WatcherRule(
        "samsung_account_signin",
        when_text=r"(?i).*(samsung account|galaxy store|samsung pass).*",
        click_text=_DISMISS_RX,
    ),
    WatcherRule(
        "miui_ad_or_safety",
        when_text=r"(?i).*(security scan|miui|recommended|app vault).*",
        click_text=_DISMISS_RX,
    ),
    WatcherRule(
        "huawei_petal_appgallery",
        when_text=r"(?i).*(petal search|appgallery|hms|huawei id).*",
        click_text=_DISMISS_RX,
    ),
    WatcherRule(
        "play_protect_or_update",
        when_text=r"(?i).*(play protect|update available|new version available).*",
        click_text=_DISMISS_RX,
    ),

    # ─── Runtime permission prompts (Android 11+ flavors) ───────────────── #
    WatcherRule(
        "perm_while_using",
        when_text=r"(?i).*(allow|permission|access|izin).*",
        click_text=r"(?i)^(while using the app|allow only while using the app|sadece uygulamay\u0131 kulland\u0131\u011f\u0131m s\u00fcrede)$",
    ),
    WatcherRule(
        "perm_allow",
        when_text=r"(?i).*(allow|permission|access|izin).*",
        click_text=r"(?i)^(allow|allow all the time|izin ver|her zaman izin ver)$",
    ),

    # ─── Save password / autofill ─────────────────────────────────────── #
    WatcherRule(
        "autofill_no_thanks",
        when_text=r"(?i).*(save password|use autofill|\u015fifreyi kaydet|otomatik doldur).*",
        click_text=_DISMISS_RX,
    ),

    # ─── ANR / crash dialogs ──────────────────────────────────────────── #
    WatcherRule(
        "anr_close",
        when_text=r"(?i).*(isn'?t responding|keeps stopping|has stopped|yan\u0131t verm|durduruldu|\u00e7\u00f6kt\u00fc).*",
        click_text=r"(?i)^(close app|close|ok|tamam|kapat)$",
    ),

    # ─── System suggestions / setup banners ───────────────────────────── #
    WatcherRule(
        "setup_later",
        when_text=r"(?i).*(set up|finish setting up|update available|kurulum|g\u00fcncelle).*",
        click_text=_DISMISS_RX,
    ),

    # ─── Battery / data usage / notification access ───────────────────── #
    WatcherRule(
        "battery_optimization",
        when_text=r"(?i).*(battery optimization|pil optimizasyonu|background.*restrict).*",
        click_text=_ACCEPT_RX,
    ),

    # ─── Generic info dialogs (last resort, narrow trigger) ───────────── #
    WatcherRule(
        "generic_got_it",
        when_text=r"(?i).*(notice|info|tip|ipucu|bilgi).*",
        click_text=r"(?i)^(ok|got it|tamam|anlad\u0131m)$",
    ),
)


class WatcherManager:
    """Registers, starts, stops, and reports background watcher activity."""

    def __init__(self, device: u2.Device, rules: Iterable[WatcherRule] | None = None):
        self.device = device
        self.rules: tuple[WatcherRule, ...] = tuple(rules) if rules else DEFAULT_RULES
        self._registered = False

    def register(self) -> None:
        if self._registered:
            return
        w = self.device.watcher
        try:
            w.reset()
        except Exception:
            pass

        for rule in self.rules:
            try:
                (
                    w.when(rule.when_text)
                    .when(rule.click_text)
                    .click()
                )
                log.debug("Registered watcher: %s", rule.name)
            except Exception as exc:
                log.warning("Failed to register watcher %s: %s", rule.name, exc)
        self._registered = True

    def start(self, interval: float = 2.0) -> None:
        if not self._registered:
            self.register()
        try:
            self.device.watcher.start(interval=interval)
            log.info(
                "Started %d background watcher(s) (interval=%.1fs)",
                len(self.rules), interval,
            )
        except Exception as exc:
            log.warning("Failed to start watchers: %s", exc)

    def stop(self) -> None:
        try:
            self.device.watcher.stop()
            self.device.watcher.remove()
            log.info("Stopped background watchers")
        except Exception as exc:
            log.warning("Failed to stop watchers: %s", exc)
        self._registered = False

    def triggered(self) -> list[str]:
        """Names of watchers that fired since the last call (best-effort)."""
        try:
            return [w for w in self.device.watcher.triggered]  # type: ignore[attr-defined]
        except Exception:
            return []

    def __enter__(self) -> "WatcherManager":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
