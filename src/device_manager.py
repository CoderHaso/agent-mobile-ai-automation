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

    def type_into(
        self,
        target: str,
        value: str,
        timeout: float = 5.0,
    ) -> bool:
        """Type into a field identified by resourceId or visible text/desc."""
        selectors = [
            self.d(resourceId=target),
            self.d(text=target),
            self.d(description=target),
        ]
        for sel in selectors:
            try:
                if sel.wait(timeout=timeout):
                    sel.set_text(value)
                    return True
            except Exception as exc:
                log.warning("type_into selector failed: %s", exc)
        # Fallback: type into the currently focused field.
        try:
            self.d.send_keys(value)
            return True
        except Exception:
            return False

    def press(self, key: str) -> bool:
        """Press a hardware/system key (back, home, enter, …)."""
        try:
            self.d.press(key)
            return True
        except Exception as exc:
            log.warning("press(%s) failed: %s", key, exc)
            return False

    def open_app(self, package: str) -> bool:
        try:
            self.d.app_start(package, use_monkey=True)
            return True
        except Exception as exc:
            log.warning("open_app(%s) failed: %s", package, exc)
            return False

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
