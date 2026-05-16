"""Installed-app discovery and package resolution via ADB.

The executor uses this so it never claims "Gmail not installed" when
`com.google.android.gm` is already on the device. Resolution order:

  1. Well-known alias table (gmail → com.google.android.gm, …)
  2. Fuzzy match on package name + human-readable label from `app_info`
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

# Common names → canonical package(s), most likely first.
KNOWN_ALIASES: Dict[str, List[str]] = {
    "gmail": ["com.google.android.gm"],
    "google mail": ["com.google.android.gm"],
    "mail": ["com.google.android.gm", "com.samsung.android.email.provider"],
    "chrome": ["com.android.chrome", "com.google.android.apps.chrome"],
    "google chrome": ["com.android.chrome"],
    "samsung internet": ["com.sec.android.app.sbrowser"],
    "internet": ["com.sec.android.app.sbrowser", "com.android.chrome"],
    "browser": ["com.android.chrome", "com.sec.android.app.sbrowser"],
    "play store": ["com.android.vending"],
    "google play": ["com.android.vending"],
    "play": ["com.android.vending"],
    "settings": ["com.android.settings"],
    "ayarlar": ["com.android.settings"],
    "whatsapp": ["com.whatsapp"],
    "telegram": ["org.telegram.messenger"],
    "instagram": ["com.instagram.android"],
    "facebook": ["com.facebook.katana"],
    "youtube": ["com.google.android.youtube"],
    "maps": ["com.google.android.apps.maps"],
    "google maps": ["com.google.android.apps.maps"],
    "photos": ["com.google.android.apps.photos"],
    "camera": ["com.android.camera", "com.android.camera2"],
    "contacts": ["com.android.contacts", "com.google.android.contacts"],
    "phone": ["com.android.dialer", "com.google.android.dialer"],
    "messages": ["com.google.android.apps.messaging", "com.android.mms"],
    "sms": ["com.google.android.apps.messaging", "com.android.mms"],
    "calendar": ["com.google.android.calendar", "com.android.calendar"],
    "drive": ["com.google.android.apps.docs"],
    "google drive": ["com.google.android.apps.docs"],
    "soyo": ["com.soyo"],  # device-specific; resolved if installed
}

_STOPWORDS = frozenset({
    "a", "an", "the", "on", "in", "to", "for", "and", "or", "new", "open",
    "create", "use", "this", "device", "account", "app", "application",
    "bir", "yeni", "hesap", "oluştur", "aç", "için", "ile", "ve",
})


@dataclass(frozen=True)
class InstalledApp:
    package: str
    label: str

    def to_dict(self) -> dict:
        return {"package": self.package, "label": self.label}


def _tokenize_goal(goal: str) -> List[str]:
    raw = re.findall(r"[a-z0-9ğüşıöç]+", (goal or "").lower())
    return [t for t in raw if len(t) >= 3 and t not in _STOPWORDS]


class AppCatalog:
    """Cached inventory of packages on the connected device."""

    def __init__(self, device: object, *, cache_ttl: float = 120.0) -> None:
        # `device` is a uiautomator2.Device — typed loosely to avoid import cycle.
        self._d = device
        self._cache_ttl = cache_ttl
        self._apps: List[InstalledApp] = []
        self._fetched_at: float = 0.0

    def refresh(self, force: bool = False) -> List[InstalledApp]:
        now = time.time()
        if not force and self._apps and (now - self._fetched_at) < self._cache_ttl:
            return self._apps

        packages = self._list_package_names()
        apps: List[InstalledApp] = []
        for pkg in packages:
            label = self._label_for(pkg)
            apps.append(InstalledApp(package=pkg, label=label))

        apps.sort(key=lambda a: a.label.lower())
        self._apps = apps
        self._fetched_at = now
        log.info("App catalog refreshed: %d package(s)", len(apps))
        return self._apps

    def _list_package_names(self) -> List[str]:
        """All enabled packages (`pm list packages -e`)."""
        try:
            raw = self._d.shell("pm list packages -e", timeout=30)
        except TypeError:
            raw = self._d.shell("pm list packages -e")
        except Exception as exc:
            log.warning("pm list packages failed: %s", exc)
            return []

        out: List[str] = []
        for line in str(raw).splitlines():
            line = line.strip()
            if line.startswith("package:"):
                out.append(line.split(":", 1)[1].strip())
        return out

    def _label_for(self, package: str) -> str:
        try:
            info = self._d.app_info(package)
            label = (info.get("label") or "").strip()
            if label:
                return label
        except Exception:
            pass
        # Fallback: last segment of the package name.
        return package.rsplit(".", 1)[-1]

    def is_installed(self, package: str) -> bool:
        if not package:
            return False
        try:
            raw = self._d.shell(f"pm path {package}", timeout=10)
        except TypeError:
            raw = self._d.shell(f"pm path {package}")
        except Exception:
            raw = ""
        if str(raw).strip().startswith("package:"):
            return True
        # Fallback: scan cached package list (some OEMs flake on `pm path`).
        return package in {a.package for a in self.refresh()}

    def resolve(self, query: str, apps: Optional[List[InstalledApp]] = None) -> Optional[str]:
        """Map a human name or partial package to an installed package."""
        q = (query or "").strip()
        if not q:
            return None

        # Already looks like a package id.
        if "." in q and self.is_installed(q):
            return q

        apps = apps if apps is not None else self.refresh()
        installed = {a.package for a in apps}
        q_lower = q.lower()

        # 1) Known aliases
        for alias, candidates in KNOWN_ALIASES.items():
            if alias in q_lower or q_lower in alias:
                for pkg in candidates:
                    if pkg in installed:
                        return pkg

        # 2) Exact label match
        for app in apps:
            if app.label.lower() == q_lower:
                return app.package

        # 3) Substring in label or package
        hits: List[Tuple[int, str]] = []
        for app in apps:
            pkg_l = app.package.lower()
            lbl_l = app.label.lower()
            score = 0
            if q_lower in lbl_l:
                score += 10 + len(q_lower)
            if q_lower in pkg_l:
                score += 6 + len(q_lower)
            for tok in _tokenize_goal(q):
                if tok in lbl_l or tok in pkg_l:
                    score += 4
            if score:
                hits.append((score, app.package))
        if hits:
            hits.sort(reverse=True)
            return hits[0][1]

        return None

    def relevant_for_goal(
        self,
        goal: str,
        *,
        limit: int = 35,
    ) -> List[InstalledApp]:
        """Subset of installed apps likely relevant to `goal`."""
        apps = self.refresh()
        if not apps:
            return []

        tokens = _tokenize_goal(goal)
        scored: List[Tuple[int, InstalledApp]] = []

        for app in apps:
            score = 0
            blob = f"{app.label} {app.package}".lower()
            for tok in tokens:
                if tok in blob:
                    score += 8
            for alias, pkgs in KNOWN_ALIASES.items():
                if any(t in alias for t in tokens) and app.package in pkgs:
                    score += 20
            if score:
                scored.append((score, app))

        scored.sort(key=lambda x: x[0], reverse=True)
        picked = [app for _, app in scored[:limit]]

        # Always inject alias hits even if fuzzy score missed them.
        seen = {a.package for a in picked}
        for alias, pkgs in KNOWN_ALIASES.items():
            if not any(t in alias for t in tokens):
                continue
            for pkg in pkgs:
                if pkg in seen:
                    continue
                for app in apps:
                    if app.package == pkg:
                        picked.append(app)
                        seen.add(pkg)
                        break

        return picked[:limit]

    def format_for_prompt(self, goal: str) -> dict:
        """JSON-serializable block for the executor prompt."""
        relevant = self.relevant_for_goal(goal)
        suggestions: List[dict] = []
        for tok in _tokenize_goal(goal):
            for alias, pkgs in KNOWN_ALIASES.items():
                if tok in alias or alias.startswith(tok):
                    for pkg in pkgs:
                        if self.is_installed(pkg):
                            suggestions.append({
                                "keyword": tok,
                                "package": pkg,
                                "label": self._label_for(pkg),
                            })
        return {
            "relevant_installed_apps": [a.to_dict() for a in relevant],
            "suggested_packages_for_goal": suggestions[:12],
            "total_installed_count": len(self.refresh()),
        }
