"""Generate fresh, realistic values for `<PLACEHOLDER>` fields at replay.

When a recorded macro types `<USER_NAME>`, we don't want it to paste
the literal string `<USER_NAME>` into the device. Instead, on every
replay we ask the LLM for a single realistic value for that field,
keyed by placeholder + goal context.

The first time `<USER_NAME>` appears in a run we generate "Mehmet" (or
whatever the LLM produces). Subsequent occurrences in the SAME run get
the cached value, so first-name and last-name fields stay coherent.
"""

from __future__ import annotations

import logging
import random
import re
import string
from typing import Callable, Dict, Optional

from .llm_client import LLMClient, LLMResponseError

log = logging.getLogger(__name__)

_PLACEHOLDER_RX = re.compile(r"<([A-Z][A-Z0-9_]+)>")


_FALLBACKS: Dict[str, Callable[[], str]] = {
    "USER_NAME":     lambda: random.choice(["Mehmet", "Ayşe", "Ali", "Zeynep", "Emre", "Elif"]),
    "FIRST_NAME":    lambda: random.choice(["Mehmet", "Ayşe", "Ali", "Zeynep", "Emre", "Elif"]),
    "LAST_NAME":     lambda: random.choice(["Yılmaz", "Demir", "Kaya", "Şahin", "Çelik"]),
    "USER_EMAIL":    lambda: f"user{random.randint(1000,9999)}@example.com",
    "USER_PASSWORD": lambda: "Tmp" + "".join(random.choices(string.ascii_letters + string.digits, k=10)) + "!",
    "USER_PHONE":    lambda: "+9055" + "".join(random.choices(string.digits, k=8)),
    "RECIPIENT":     lambda: "Test Kişi",
    "MESSAGE":       lambda: "Merhaba!",
    "SEARCH_TERM":   lambda: "telefon",
    "PRODUCT_NAME":  lambda: "telefon",
    "AMOUNT":        lambda: "1",
    "OTP":           lambda: "".join(random.choices(string.digits, k=6)),
}


class DynamicValueResolver:
    """LLM-backed generator with deterministic fallbacks."""

    def __init__(self, llm: Optional[LLMClient] = None, *, goal: str = "") -> None:
        self.llm = llm
        self.goal = goal
        self._cache: Dict[str, str] = {}

    def reset(self) -> None:
        self._cache.clear()

    def fill(self, template: str) -> str:
        """Replace every `<PLACEHOLDER>` in `template` with a fresh value."""
        if not template or "<" not in template:
            return template
        out = template
        for name in _PLACEHOLDER_RX.findall(template):
            value = self._cache.get(name) or self._generate(name)
            self._cache[name] = value
            out = out.replace(f"<{name}>", value)
        return out

    # ------------------------------------------------------------------ #

    def _generate(self, placeholder: str) -> str:
        # Try the LLM first when available
        if self.llm is not None:
            try:
                return self._llm_value(placeholder)
            except (LLMResponseError, Exception) as exc:
                log.debug("LLM dynamic generation failed for %s: %s",
                          placeholder, exc)
        # Fallback: hard-coded sensible defaults
        fn = _FALLBACKS.get(placeholder)
        if fn is not None:
            return fn()
        # Last resort: lowercased placeholder name
        return placeholder.lower().replace("_", " ")

    def _llm_value(self, placeholder: str) -> str:
        system = (
            "You generate realistic single values for placeholders inside "
            "an Android UI macro. Respond with strict JSON only."
        )
        user = (
            f"Goal context: {self.goal!r}\n"
            f"Placeholder: <{placeholder}>\n"
            "Return JSON: {\"value\": \"...\"}\n\n"
            "Rules:\n"
            "  - One short value, no quotes or punctuation around it.\n"
            "  - For names use a realistic Turkish first OR last name.\n"
            "  - For emails use a plausible-looking address using example.com.\n"
            "  - For passwords output 12+ chars mixing letters, digits, and a symbol.\n"
            "  - For phone numbers use Turkish-format mobile (+905...).\n"
            "  - For OTPs return exactly 6 digits.\n"
            "  - For messages keep them short and harmless.\n"
            "  - NEVER echo the placeholder back."
        )
        raw = self.llm.complete_json(system, user, temperature=0.7)
        if isinstance(raw, dict):
            value = str(raw.get("value", "")).strip()
            if value and value != f"<{placeholder}>":
                return value
        raise LLMResponseError(f"LLM returned no usable value for <{placeholder}>")
