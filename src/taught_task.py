"""Taught Task data model.

A TaughtTask is a user-defined, step-by-step recipe that the agent follows
deterministically. Each step is a plain-language instruction (e.g., "Open
Gmail app") that the AI interprets at runtime. Unlike auto-recorded macros,
these are hand-crafted by the user through the Teach tab.

Structure:
    TaughtTask
        ├── name          — human label ("Create Google Account")
        ├── notes         — free-form context the AI reads every turn
        │                   (e.g., "Ad: Emre, Soyad: Yılmaz, Mail: ...")
        ├── loop_count    — how many times to repeat the full task
        ├── steps[]       — ordered TaughtStep objects
        │     ├── instruction   — what to do (natural language)
        │     ├── verified      — user confirmed this step works
        │     └── last_result   — outcome of the last test run
        └── created_at / updated_at
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

TAUGHT_DIR = Path("taught_tasks")


class TaughtStep(BaseModel):
    """One step in a user-taught task."""

    step_index: int = 1
    instruction: str = ""           # e.g., "Open Gmail and tap 'Create account'"
    verified: bool = False          # user confirmed it works
    last_result: str = ""           # "success" | "failed" | "" (untested)
    last_error: str = ""            # error details if failed


class TaughtTask(BaseModel):
    """A complete user-taught task recipe."""

    name: str = ""
    notes: str = ""                 # free-form context for the AI
    loop_count: int = 1             # how many times to repeat
    steps: List[TaughtStep] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    def task_id(self) -> str:
        """Stable 8-char id derived from the task name."""
        return hashlib.sha1(
            self.name.lower().strip().encode("utf-8")
        ).hexdigest()[:8]

    def add_step(self, instruction: str) -> TaughtStep:
        """Add a new step at the end."""
        step = TaughtStep(
            step_index=len(self.steps) + 1,
            instruction=instruction,
        )
        self.steps.append(step)
        self.updated_at = time.time()
        return step

    def remove_step(self, index: int) -> None:
        """Remove step at position (0-based) and renumber."""
        if 0 <= index < len(self.steps):
            self.steps.pop(index)
            for i, s in enumerate(self.steps):
                s.step_index = i + 1
            self.updated_at = time.time()

    def move_step(self, from_idx: int, to_idx: int) -> None:
        """Move a step from one position to another."""
        if 0 <= from_idx < len(self.steps) and 0 <= to_idx < len(self.steps):
            step = self.steps.pop(from_idx)
            self.steps.insert(to_idx, step)
            for i, s in enumerate(self.steps):
                s.step_index = i + 1
            self.updated_at = time.time()

    def all_verified(self) -> bool:
        """True if every step has been verified by the user."""
        return bool(self.steps) and all(s.verified for s in self.steps)


# --------------------------------------------------------------------------- #
# On-disk persistence
# --------------------------------------------------------------------------- #

class TaughtTaskLibrary:
    """Manages taught tasks on disk (taught_tasks/*.json)."""

    def __init__(self, base_dir: os.PathLike | str = TAUGHT_DIR) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, task: TaughtTask) -> Path:
        safe = task.task_id()
        return self.base_dir / f"{safe}.json"

    def save(self, task: TaughtTask) -> Path:
        task.updated_at = time.time()
        path = self._path_for(task)
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(task.model_dump(), f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
        log.info("Saved taught task %r → %s", task.name, path)
        return path

    def load(self, path: os.PathLike | str) -> Optional[TaughtTask]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return TaughtTask.model_validate(data)
        except Exception as exc:
            log.warning("Failed to load taught task %s: %s", path, exc)
            return None

    def delete(self, task: TaughtTask) -> bool:
        path = self._path_for(task)
        try:
            if path.exists():
                path.unlink()
                return True
        except OSError as exc:
            log.warning("delete taught task failed: %s", exc)
        return False

    def list_tasks(self) -> List[TaughtTask]:
        tasks: list[tuple[float, TaughtTask]] = []
        for p in self.base_dir.glob("*.json"):
            t = self.load(p)
            if t is not None:
                tasks.append((t.updated_at, t))
        tasks.sort(key=lambda kv: kv[0], reverse=True)
        return [t for _, t in tasks]

    def find_by_name(self, name: str) -> Optional[TaughtTask]:
        target = name.strip().lower()
        for t in self.list_tasks():
            if t.name.strip().lower() == target:
                return t
        return None
