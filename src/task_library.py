"""On-disk library of recorded macros.

Each macro lives in `recordings/<task_id>__<slug>.json`. The library
exposes:

  • `list_tasks()`         — every saved macro (sorted, newest first)
  • `save(task)`           — atomic write
  • `delete(task_id)`      — remove
  • `find_for_goal(goal)`  — find a macro whose goal text is similar
                              enough to skip planning + LLM altogether
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional, Tuple

from .recorder import RecordedTask

log = logging.getLogger(__name__)

DEFAULT_DIR = Path("recordings")


def _slug(text: str, limit: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return s[:limit] or "task"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


class TaskLibrary:
    def __init__(self, base_dir: os.PathLike | str = DEFAULT_DIR) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ---- I/O ------------------------------------------------------------ #

    def _path_for(self, task: RecordedTask) -> Path:
        return self.base_dir / f"{task.task_id()}__{_slug(task.goal)}.json"

    def save(self, task: RecordedTask) -> Path:
        path = self._path_for(task)
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(task.model_dump(), f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
        log.info("Saved macro %s (%d actions) → %s",
                 task.task_id(), len(task.actions), path)
        return path

    def load(self, path: os.PathLike | str) -> Optional[RecordedTask]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return RecordedTask.model_validate(data)
        except Exception as exc:
            log.warning("Failed to load %s: %s", path, exc)
            return None

    def delete(self, task_id: str) -> bool:
        for p in self.base_dir.glob("*.json"):
            if p.name.startswith(f"{task_id}__"):
                try:
                    p.unlink()
                    return True
                except OSError as exc:
                    log.warning("delete %s failed: %s", p, exc)
        return False

    # ---- Querying ------------------------------------------------------- #

    def list_tasks(self) -> List[RecordedTask]:
        tasks: List[Tuple[float, RecordedTask]] = []
        for p in self.base_dir.glob("*.json"):
            t = self.load(p)
            if t is not None:
                tasks.append((t.created_at, t))
        tasks.sort(key=lambda kv: kv[0], reverse=True)
        return [t for _, t in tasks]

    def find_for_goal(
        self,
        goal: str,
        *,
        threshold: float = 0.78,
    ) -> Optional[RecordedTask]:
        """Return the best-matching recorded task whose goal looks like `goal`.

        Uses a normalized-string ratio on top of an exact match. The
        threshold is tuned to allow for trivial wording differences
        ("create gmail account" vs "create a new gmail account") but to
        reject genuinely different goals.
        """
        target = _norm(goal)
        if not target:
            return None
        best: Optional[Tuple[float, RecordedTask]] = None
        for t in self.list_tasks():
            cand = _norm(t.goal)
            if cand == target:
                return t
            score = SequenceMatcher(None, target, cand).ratio()
            if score >= threshold and (best is None or score > best[0]):
                best = (score, t)
        return best[1] if best else None

    # ---- Pretty-printing ------------------------------------------------ #

    @staticmethod
    def describe(task: RecordedTask) -> str:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(task.created_at))
        return (
            f"[{task.task_id()}] {task.goal}  "
            f"({len(task.actions)} steps · {when} · "
            f"model={task.llm_provider}:{task.llm_model})"
        )
