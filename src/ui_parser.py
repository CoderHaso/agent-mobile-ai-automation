"""Convert raw uiautomator2 XML hierarchies into a compact, LLM-friendly
representation.

A real Android XML dump is huge (tens of KB) and full of noise: layout
containers, drawing flags, bounds, etc. Sending all of it to the LLM is
slow, expensive, and confuses reasoning. This module keeps only the
*interactable* nodes (clickable, long-clickable, scrollable, editable,
focusable text fields) and exposes them as small dicts.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

from lxml import etree


_INTERACTABLE_ATTRS = ("clickable", "long-clickable", "scrollable", "focusable")


@dataclass
class UIElement:
    """A single interactable element on the current screen."""

    index: int
    cls: str
    text: str
    resource_id: str
    content_desc: str
    hint: str
    package: str
    clickable: bool
    long_clickable: bool
    scrollable: bool
    editable: bool
    focused: bool
    selected: bool
    enabled: bool
    bounds: str

    def to_llm_dict(self) -> dict:
        """Slim version for the prompt. Drops keys the LLM doesn't need."""
        d = asdict(self)
        for key in ("package", "long_clickable", "enabled"):
            d.pop(key, None)
        return {k: v for k, v in d.items() if v not in ("", False)}


def _b(value: str | None) -> bool:
    return str(value).lower() == "true"


def _is_interactable(node: etree._Element) -> bool:
    """True if the node is something a user could realistically interact with."""
    if any(_b(node.get(a)) for a in _INTERACTABLE_ATTRS):
        return True
    cls = (node.get("class") or "").lower()
    if "edittext" in cls:
        return True
    if "button" in cls and (node.get("text") or node.get("content-desc")):
        return True
    return False


def _is_meaningful_text_node(node: etree._Element) -> bool:
    """Non-interactable but informative text (titles, labels, errors)."""
    cls = (node.get("class") or "").lower()
    if "textview" not in cls:
        return False
    text = (node.get("text") or "").strip()
    return bool(text) and len(text) <= 120


def parse_hierarchy(xml: str, include_text_labels: bool = True) -> list[UIElement]:
    """Walk the XML once and return only the elements worth showing to the LLM."""
    if not xml:
        return []
    try:
        root = etree.fromstring(xml.encode("utf-8"))
    except etree.XMLSyntaxError:
        return []

    elements: list[UIElement] = []
    idx = 0
    for node in root.iter():
        if node.tag != "node":
            continue
        keep = _is_interactable(node)
        if not keep and include_text_labels:
            keep = _is_meaningful_text_node(node)
        if not keep:
            continue
        elements.append(
            UIElement(
                index=idx,
                cls=(node.get("class") or "").split(".")[-1],
                text=(node.get("text") or "").strip(),
                resource_id=node.get("resource-id") or "",
                content_desc=(node.get("content-desc") or "").strip(),
                hint=(node.get("hint") or "").strip(),
                package=node.get("package") or "",
                clickable=_b(node.get("clickable")),
                long_clickable=_b(node.get("long-clickable")),
                scrollable=_b(node.get("scrollable")),
                editable="edittext" in (node.get("class") or "").lower(),
                focused=_b(node.get("focused")),
                selected=_b(node.get("selected")),
                enabled=_b(node.get("enabled")),
                bounds=node.get("bounds") or "",
            )
        )
        idx += 1
    return elements


def summarize_for_prompt(
    elements: Iterable[UIElement],
    max_elements: int = 60,
) -> list[dict]:
    """Trim and serialize for inclusion in the LLM prompt."""
    trimmed = list(elements)[:max_elements]
    return [e.to_llm_dict() for e in trimmed]


def find_focused_app(xml: str) -> str | None:
    """Best-effort guess at which package the user is currently looking at."""
    try:
        root = etree.fromstring(xml.encode("utf-8"))
    except etree.XMLSyntaxError:
        return None
    for node in root.iter("node"):
        pkg = node.get("package")
        if pkg and pkg not in ("android", "com.android.systemui"):
            return pkg
    return None
