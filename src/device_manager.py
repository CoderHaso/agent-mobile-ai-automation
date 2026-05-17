"""Thin wrapper around uiautomator2 that exposes only the verbs the
Executor Agent needs: connect, observe (XML dump + screen hash), and act
(click / type / press / wait).

Keeping this file small and side-effect free makes the rest of the agent
easy to unit test without a real device (you can swap in a mock).
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any, List, Optional

import adbutils
import uiautomator2 as u2

from .app_catalog import AppCatalog, InstalledApp

log = logging.getLogger(__name__)


class DeviceConnectionError(RuntimeError):
    """Raised when we cannot reach the Android device over ADB."""


class ActionExecutionError(RuntimeError):
    """Raised when an LLM-requested action cannot be performed on screen."""


@dataclass
class AdbDeviceInfo:
    """A row in the adb device picker."""

    serial: str
    state: str                 # device | offline | unauthorized | ...
    transport: str             # usb | tcp | emulator | unknown
    model: str = ""
    product: str = ""
    sdk: str = ""
    active: bool = False       # selected for this session

    @property
    def is_usable(self) -> bool:
        return self.state == "device"

    @property
    def is_tcp(self) -> bool:
        return self.transport in ("tcp", "emulator") or ":" in self.serial


def list_adb_devices() -> List[AdbDeviceInfo]:
    """Return everything `adb devices -l` can see, with light metadata.

    Resilient: if `adb` is missing, returns an empty list rather than raising,
    so the caller can surface a friendly error.
    """
    try:
        raw = adbutils.adb.list()
    except Exception as exc:
        log.warning("adbutils.adb.list() failed: %s", exc)
        return []

    out: List[AdbDeviceInfo] = []
    for d in raw:
        serial = getattr(d, "serial", "") or ""
        state = getattr(d, "state", "unknown") or "unknown"
        if serial.startswith("emulator-"):
            transport = "emulator"
        elif ":" in serial:
            transport = "tcp"
        else:
            transport = "usb"

        model = product = sdk = ""
        if state == "device":
            try:
                dev = adbutils.adb.device(serial=serial)
                props = dev.prop
                model = (props.model or "").strip()
                # adbutils exposes a few common props as attributes; fall back
                # to a raw shell call when they are not populated.
                product = (getattr(props, "name", "") or "").strip()
                sdk = str(getattr(props, "sdk", "") or "")
                if not sdk:
                    sdk = dev.shell("getprop ro.build.version.sdk").strip()
            except Exception as exc:
                log.debug("Could not read props for %s: %s", serial, exc)

        out.append(AdbDeviceInfo(
            serial=serial,
            state=state,
            transport=transport,
            model=model,
            product=product,
            sdk=sdk,
        ))
    return out


def adb_connect(host_port: str) -> str:
    """`adb connect host:port` — returns the raw status string."""
    try:
        result = adbutils.adb.connect(host_port)
        return str(result).strip() or f"connected to {host_port}"
    except Exception as exc:
        return f"connect failed: {exc}"


def adb_disconnect(serial_or_host_port: str) -> str:
    """`adb disconnect <serial>` — only meaningful for TCP devices."""
    try:
        result = adbutils.adb.disconnect(serial_or_host_port)
        return str(result).strip() or f"disconnected {serial_or_host_port}"
    except Exception as exc:
        return f"disconnect failed: {exc}"


@dataclass(frozen=True)
class Observation:
    """A snapshot of the device at a point in time."""

    xml: str
    screen_hash: str
    current_app: Optional[str]
    width: int
    height: int

    def changed_from(self, other: "Observation | None") -> bool:
        """Cheap heuristic for 'did the screen change after my action?'."""
        if other is None:
            return True
        return self.screen_hash != other.screen_hash


class DeviceManager:
    """uiautomator2 facade with retry-friendly primitives."""

    def __init__(self, serial: Optional[str] = None, default_wait: float = 1.5):
        self.serial = serial or None
        self.default_wait = default_wait
        self._d: Optional[u2.Device] = None
        self._app_catalog: Optional[AppCatalog] = None

    @property
    def d(self) -> u2.Device:
        if self._d is None:
            raise DeviceConnectionError(
                "Device not connected. Call DeviceManager.connect() first."
            )
        return self._d

    def connect(self) -> u2.Device:
        """Connect to the device. Raises DeviceConnectionError on failure."""
        try:
            self._d = u2.connect(self.serial) if self.serial else u2.connect()
            self._app_catalog = AppCatalog(self._d)
            info = self._d.info
            log.info(
                "Connected to device: %s (%sx%s, sdk=%s)",
                info.get("productName"),
                info.get("displayWidth"),
                info.get("displayHeight"),
                info.get("sdkInt"),
            )
            return self._d
        except Exception as exc:
            raise DeviceConnectionError(
                f"Failed to connect to ADB device "
                f"({self.serial or 'auto-detect'}): {exc}"
            ) from exc

    def observe(self) -> Observation:
        """Dump the current UI hierarchy and return a structured snapshot."""
        d = self.d
        xml = d.dump_hierarchy(compressed=False, pretty=False)
        info = d.info
        try:
            current_app = d.app_current().get("package")
        except Exception:
            current_app = None
        screen_hash = hashlib.sha1(xml.encode("utf-8", "ignore")).hexdigest()[:16]
        return Observation(
            xml=xml,
            screen_hash=screen_hash,
            current_app=current_app,
            width=int(info.get("displayWidth", 0)),
            height=int(info.get("displayHeight", 0)),
        )

    def click_text(self, text: str, timeout: float = 5.0) -> bool:
        import re
        if self.d(text=text).wait(timeout=timeout):
            self.d(text=text).click()
            return True
        if self.d(textContains=text).wait(timeout=1.0):
            self.d(textContains=text).click()
            return True
        # Case-insensitive fallback (uiautomator2 expects a string, not a re.Pattern)
        rx_str = f"(?i).*{re.escape(text)}.*"
        if self.d(textMatches=rx_str).wait(timeout=1.0):
            self.d(textMatches=rx_str).click()
            return True
        return False

    def click_resource_id(self, rid: str, timeout: float = 5.0) -> bool:
        if self.d(resourceId=rid).wait(timeout=timeout):
            self.d(resourceId=rid).click()
            return True
        return False

    def click_description(self, desc: str, timeout: float = 5.0) -> bool:
        import re
        if self.d(description=desc).wait(timeout=timeout):
            self.d(description=desc).click()
            return True
        rx_str = f"(?i).*{re.escape(desc)}.*"
        if self.d(descriptionMatches=rx_str).wait(timeout=1.0):
            self.d(descriptionMatches=rx_str).click()
            return True
        return False

    def type_into_focused(self, value: str) -> bool:
        """Type into the currently focused EditText (clear first)."""
        if not value:
            return False
        try:
            focused = self.d(focused=True)
            if focused.exists:
                try:
                    focused.clear_text()
                except Exception:
                    pass
                focused.set_text(value)
                return True
        except Exception as exc:
            log.debug("type_into_focused (focused=True) failed: %s", exc)

        try:
            edit = self.d(className="android.widget.EditText", focused=True)
            if edit.exists:
                try:
                    edit.clear_text()
                except Exception:
                    pass
                edit.set_text(value)
                return True
        except Exception as exc:
            log.debug("type_into_focused (EditText) failed: %s", exc)

        try:
            self.d.clear_text()
            self.d.send_keys(value)
            return True
        except Exception as exc:
            log.warning("type_into_focused fallback failed: %s", exc)
            return False

    def type_into(
        self,
        target: str,
        value: str,
        timeout: float = 5.0,
        *,
        target_kind: str = "text",
    ) -> bool:
        """Type into a field. Prefer focused field when target is empty."""
        if not value:
            return False

        t = (target or "").strip()
        kind = (target_kind or "text").lower().strip()

        # Field already focused — do not match spurious text like "8" or "type".
        if not t or kind in ("none", "focused"):
            return self.type_into_focused(value)

        if kind == "index":
            return False  # caller must click index first, then type_into_focused

        # Single character / digit as text target is almost always wrong.
        if len(t) <= 2 and t.isdigit():
            return self.type_into_focused(value)

        if t.lower() in ("type", "click", "text", "input"):
            return self.type_into_focused(value)

        selectors = [
            self.d(resourceId=t),
            self.d(description=t),
        ]
        if len(t) > 2:
            selectors.append(self.d(text=t))

        for sel in selectors:
            try:
                if sel.wait(timeout=timeout):
                    try:
                        sel.clear_text()
                    except Exception:
                        pass
                    sel.set_text(value)
                    return True
            except Exception as exc:
                log.warning("type_into selector failed: %s", exc)

        return self.type_into_focused(value)

    def press(self, key: str) -> bool:
        """Press a hardware/system key (back, home, enter, …)."""
        try:
            self.d.press(key)
            return True
        except Exception as exc:
            log.warning("press(%s) failed: %s", key, exc)
            return False

    def shell(self, command: str, timeout: float = 30.0) -> str:
        """Run an arbitrary `adb shell` command and return stdout."""
        try:
            try:
                return str(self.d.shell(command, timeout=timeout)).strip()
            except TypeError:
                return str(self.d.shell(command)).strip()
        except Exception as exc:
            log.warning("shell(%r) failed: %s", command[:80], exc)
            return ""

    @property
    def apps(self) -> AppCatalog:
        if self._app_catalog is None:
            self._app_catalog = AppCatalog(self.d)
        return self._app_catalog

    def list_installed_apps(self, force: bool = False) -> List[InstalledApp]:
        return self.apps.refresh(force=force)

    def resolve_package(self, name_or_package: str) -> Optional[str]:
        return self.apps.resolve(name_or_package)

    def is_package_installed(self, package: str) -> bool:
        return self.apps.is_installed(package)

    def _foreground_package(self) -> str:
        try:
            return str((self.d.app_current() or {}).get("package") or "")
        except Exception:
            return ""

    def _foreground_matches(self, package: str) -> bool:
        cur = self._foreground_package()
        return bool(cur) and (cur == package or cur.startswith(package))

    def _am_start_main_activity(self, package: str) -> bool:
        """Launch via `am start` using the resolved MAIN/LAUNCHER activity."""
        out = self.shell(
            "cmd package resolve-activity --brief "
            "-a android.intent.action.MAIN "
            "-c android.intent.category.LAUNCHER "
            f"{package}"
        )
        component = ""
        for line in reversed(out.splitlines()):
            line = line.strip()
            if "/" in line and package in line:
                component = line.split()[-1]
                break
        if not component:
            return False
        result = self.shell(f"am start -W -n {component}")
        return "Error" not in result and "Exception" not in result

    def launch_package(self, package: str) -> bool:
        """Try several ADB / u2 strategies to foreground an installed app."""
        strategies = (
            ("u2_monkey", lambda: self.d.app_start(package, use_monkey=True)),
            ("u2_direct", lambda: self.d.app_start(package, use_monkey=False)),
            ("am_start", lambda: self._am_start_main_activity(package)),
            ("monkey_shell", lambda: self.shell(
                f"monkey -p {package} -c android.intent.category.LAUNCHER 1"
            ) or True),
        )
        for name, fn in strategies:
            try:
                fn()
                self.wait(1.0)
                if self._foreground_matches(package):
                    log.info("launch_package(%s) ok via %s", package, name)
                    return True
                self.wait(0.6)
                if self._foreground_matches(package):
                    log.info("launch_package(%s) ok via %s (delayed)", package, name)
                    return True
            except Exception as exc:
                log.debug("launch_package(%s) %s failed: %s", package, name, exc)
        return False

    def open_app(self, package_or_name: str) -> bool:
        """Launch by package id OR human name (e.g. 'Gmail' → com.google.android.gm)."""
        raw = (package_or_name or "").strip()
        if not raw:
            return False

        pkg = raw
        if "." not in raw or not self.is_package_installed(raw):
            resolved = self.resolve_package(raw)
            if resolved:
                pkg = resolved
                log.info("Resolved app %r → %s", raw, pkg)
            elif "." not in raw:
                log.warning("Could not resolve app name %r to a package", raw)
                return False

        if not self.is_package_installed(pkg):
            log.warning("Package %s is not installed on this device", pkg)
            return False

        if self.launch_package(pkg):
            return True
        log.warning("open_app(%s): all launch strategies failed", pkg)
        return False

    def go_home(self) -> bool:
        """Reset to launcher — safer than back when stuck in shade/overlay."""
        return self.press("home")

    def force_stop(self, package: str) -> bool:
        """Force-stop an app via `am force-stop`. Returns True if command ran."""
        if not package:
            return False
        try:
            self.shell(f"am force-stop {package}")
            log.info("force_stop(%s): done", package)
            return True
        except Exception as exc:
            log.warning("force_stop(%s) failed: %s", package, exc)
            return False

    def get_current_focus(self) -> str:
        """Return the currently focused window/activity via dumpsys."""
        try:
            out = self.shell("dumpsys window | grep mCurrentFocus")
            # Format: mCurrentFocus=Window{hash u0 com.pkg/.Activity}
            if "mCurrentFocus" in out:
                return out.strip()
            return ""
        except Exception:
            return ""

    def is_app_running(self, package: str) -> bool:
        """Check if an app process is currently running."""
        if not package:
            return False
        try:
            out = self.shell(f"pidof {package}")
            return bool(out.strip())
        except Exception:
            return False

    def go_home_and_clean(self, packages: list[str] | None = None) -> None:
        """Reset to home launcher and force-stop specified packages.
        
        This is the FIRST step of every task — ensures a clean state.
        """
        self.press("home")
        self.wait(0.5)
        if packages:
            for pkg in packages:
                self.force_stop(pkg)
            self.wait(0.3)
        # Press home again to ensure we're on the launcher
        self.press("home")
        self.wait(0.5)
        log.info("go_home_and_clean: reset complete (stopped %d app(s))",
                 len(packages or []))

    def swipe(self, direction: str, *, scale: float = 0.75) -> bool:
        """Swipe on screen (left/right/up/down). Used for launcher pages."""
        direction = (direction or "").lower().strip()
        try:
            info = self.d.info
            w = int(info.get("displayWidth", 720))
            h = int(info.get("displayHeight", 1600))
        except Exception:
            w, h = 720, 1600

        cx, cy = w // 2, h // 2
        dx = int(w * scale * 0.35)
        dy = int(h * scale * 0.35)
        try:
            if direction == "left":
                self.d.swipe(cx + dx, cy, cx - dx, cy, 0.25)
            elif direction == "right":
                self.d.swipe(cx - dx, cy, cx + dx, cy, 0.25)
            elif direction == "up":
                self.d.swipe(cx, h - dy, cx, h - 3 * dy, 0.25)
            elif direction == "down":
                self.d.swipe(cx, dy, cx, 3 * dy, 0.25)
            else:
                return False
            return True
        except Exception as exc:
            log.warning("swipe(%s) failed: %s", direction, exc)
            return False

    def open_app_drawer(self) -> bool:
        """Open the all-apps drawer (Samsung / AOSP launchers)."""
        for label in (
            "Apps", "Uygulamalar", "Tüm uygulamalar", "All apps",
            "Aplicaciones",
        ):
            if self.click_text(label):
                return True
        if self.swipe("up"):
            self.wait(0.8)
            return True
        return False

    def smart_scroll(self, target: str, target_kind: str = "text") -> bool:
        """scroll_to handler: text search, launcher swipe, or resource scroll."""
        t = (target or "").strip()
        if not t:
            return False

        app = self._foreground_package().lower()

        # LLM sometimes passes page numbers — swipe launcher horizontally.
        if t.isdigit() and "launcher" in app:
            direction = "left" if int(t) > 1 else "right"
            return self.swipe(direction)

        # Resource-id scroll on launcher workspace rarely works — swipe instead.
        if (":" in t or target_kind == "resource_id") and "launcher" in app:
            return self.swipe("left")

        if ":" in t or target_kind == "resource_id":
            try:
                sel = self.d(resourceId=t)
                if sel.exists:
                    sel.scroll.horiz.forward(steps=3)
                    return True
            except Exception:
                pass
            return self.swipe("left")

        return self.scroll_to_text(t)

    def scroll_to_text(self, text: str) -> bool:
        try:
            return bool(
                self.d(scrollable=True).scroll.to(textContains=text)
            )
        except Exception:
            return False

    def wait(self, seconds: float | None = None) -> None:
        time.sleep(seconds if seconds is not None else self.default_wait)

    def screenshot(self, path: str) -> None:
        try:
            self.d.screenshot(path)
        except Exception as exc:
            log.warning("screenshot(%s) failed: %s", path, exc)

    def info(self) -> dict[str, Any]:
        return dict(self.d.info)
