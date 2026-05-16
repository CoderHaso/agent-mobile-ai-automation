"""CLI entry point for the Human-in-the-Loop Autonomous Android UI Agent.

Workflow:
    1. Connect to the LLM provider (Groq or DeepSeek) and the Android device.
    2. Ask the user for a high-level goal.
    3. Planner Agent → JSON task list.
    4. Human-in-the-loop review (approve / edit / add / delete / replan).
    5. Register & start background uiautomator2 Watchers.
    6. Executor Agent runs the plan step-by-step (Observe → Reason → Act → Verify).
    7. Render a final summary table.

Run:
    python main.py
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Windows: force UTF-8 on stdout/stderr so Rich can render Unicode (★ box
# drawing chars, etc.) without crashing on cp1252/cp1254 legacy consoles.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

from dotenv import load_dotenv

from src import cli_ui
from src.device_manager import (
    DeviceConnectionError,
    DeviceManager,
    list_adb_devices,
)
from src.executor import Executor, ExecutorConfig
from src.llm_client import LLMClient, LLMConfigError, LLMResponseError
from src.models import ALL_MODELS
from src.planner import Planner
from src.watchers import WatcherManager


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet down noisy third-party loggers unless --verbose was passed.
    if not verbose:
        for noisy in ("httpx", "urllib3", "uiautomator2"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="agentic-android",
        description="Human-in-the-Loop Autonomous Android UI Agent",
    )
    p.add_argument(
        "--serial", "-s",
        help="ADB device serial (overrides ANDROID_SERIAL env var). "
             "If omitted and multiple devices are found, the picker UI opens.",
    )
    p.add_argument(
        "--no-picker",
        action="store_true",
        help="Skip the interactive device picker even when ambiguous; "
             "rely solely on auto-detect / --serial.",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    p.add_argument(
        "--max-attempts",
        type=int,
        default=5,
        help="Max LLM/UI attempts per step before marking it failed (default: 5).",
    )
    p.add_argument(
        "--settle",
        type=float,
        default=1.5,
        help="Seconds to wait between observe/act cycles (default: 1.5).",
    )
    p.add_argument(
        "--provider",
        choices=("groq", "deepseek"),
        help="LLM provider override (defaults to LLM_PROVIDER from .env).",
    )
    p.add_argument(
        "--model",
        help="LLM model slug override (e.g. "
             "'meta-llama/llama-4-maverick-17b-128e-instruct' or 'deepseek-v4-pro').",
    )
    p.add_argument(
        "--list-models",
        action="store_true",
        help="Print the registered model catalog (with stars + pricing) and exit.",
    )
    return p.parse_args()


def _print_model_catalog() -> None:
    """Pretty-print the model registry to stdout (ASCII-safe for Windows)."""
    from rich.box import ROUNDED
    from rich.table import Table
    from rich.text import Text

    table = Table(
        title="Registered LLM models",
        title_style="bold magenta",
        box=ROUNDED,
        header_style="bold cyan",
        expand=True,
    )
    table.add_column("Provider", style="cyan", no_wrap=True)
    table.add_column("Model slug (use --model)", overflow="fold")
    table.add_column("Quality", no_wrap=True)
    table.add_column("Speed", no_wrap=True)
    table.add_column("$ in/1M", justify="right", no_wrap=True)
    table.add_column("$ out/1M", justify="right", no_wrap=True)
    table.add_column("Ctx", justify="right", no_wrap=True)
    table.add_column("Notes", overflow="fold", style="dim")

    for m in ALL_MODELS:
        # ASCII stars so Windows cp1252/cp1254 consoles don't crash.
        table.add_row(
            m.provider,
            m.slug,
            Text(m.quality_stars_ascii, style="yellow"),
            Text(m.speed_stars_ascii, style="green"),
            f"${m.input_per_m:.3f}",
            f"${m.output_per_m:.3f}",
            f"{m.context_k}K",
            m.notes,
        )
    cli_ui.console.print(table)


def _connect_llm(args: argparse.Namespace) -> LLMClient | None:
    try:
        if args.provider or args.model:
            provider = args.provider or os.getenv("LLM_PROVIDER", "groq")
            return LLMClient.from_choice(provider, args.model)
        return LLMClient.from_env()
    except LLMConfigError as exc:
        cli_ui.fatal(
            f"LLM configuration error: {exc}\n\n"
            "Copy .env.example to .env and fill in your provider keys."
        )
        return None


def _resolve_serial(args: argparse.Namespace) -> str | None:
    """Decide which device serial to connect to.

    Order:
      1. --serial / ANDROID_SERIAL
      2. If exactly one usable device is online, use it silently
      3. Otherwise (or whenever ambiguous and --no-picker not set), open
         the interactive picker so the user can connect/disconnect/select.
    """
    explicit = args.serial or os.getenv("ANDROID_SERIAL") or None
    if explicit:
        return explicit.strip() or None

    devices = list_adb_devices()
    usable = [d for d in devices if d.is_usable]

    if args.no_picker:
        if len(usable) == 1:
            return usable[0].serial
        return None  # let DeviceManager auto-detect (and likely fail loudly)

    if len(usable) == 1 and len(devices) == 1:
        return usable[0].serial

    return cli_ui.pick_device(initial_serial=None)


def _connect_device(serial: str | None) -> DeviceManager | None:
    dm = DeviceManager(serial=serial)
    try:
        dm.connect()
        return dm
    except DeviceConnectionError as exc:
        cli_ui.fatal(
            f"{exc}\n\n"
            "Make sure: \n"
            "  1. The device is plugged in / emulator is booted.\n"
            "  2. `adb devices` lists it as 'device' (not 'unauthorized').\n"
            "  3. You ran `python -m uiautomator2 init` once."
        )
        return None


def _run_once(args: argparse.Namespace, llm: LLMClient, device: DeviceManager) -> int:
    planner = Planner(llm)
    watchers = WatcherManager(device.d)

    while True:
        try:
            goal = cli_ui.ask_goal()
        except KeyboardInterrupt:
            cli_ui.console.print("\n[dim]Bye.[/dim]")
            return 0

        cli_ui.console.print(
            f"\n[dim]Asking the planner ({llm.describe()}) to break this down…[/dim]"
        )

        try:
            plan = planner.build_plan(goal)
        except (LLMResponseError, ValueError) as exc:
            cli_ui.fatal(f"Planner failed: {exc}")
            continue

        if not plan.steps:
            cli_ui.fatal(
                f"Planner refused to produce a plan: {plan.error or 'no reason given'}."
            )
            continue

        try:
            approved = cli_ui.review_loop(plan)
        except KeyboardInterrupt:
            cli_ui.console.print("\n[dim]Aborted by user. Bye.[/dim]")
            return 0
        except Exception as exc:
            if cli_ui.is_replan_requested(exc):
                cli_ui.console.print("[cyan]Restarting with a new goal…[/cyan]")
                continue
            raise

        executor = Executor(
            device=device,
            llm=llm,
            watchers=watchers,
            config=ExecutorConfig(
                settle_seconds=args.settle,
            ),
            on_progress=cli_ui.step_progress,
        )

        cli_ui.console.print(
            f"\n[bold green]► Executing {len(approved.steps)} step(s)…[/bold green]\n"
        )

        try:
            results = executor.run(approved)
        except KeyboardInterrupt:
            cli_ui.console.print("\n[red]Execution interrupted by user.[/red]")
            return 130
        except Exception as exc:
            logging.getLogger(__name__).exception("Unhandled executor error")
            cli_ui.fatal(f"Executor crashed: {exc}")
            return 1

        cli_ui.console.print()
        cli_ui.render_results(results)

        all_ok = all(r.success for r in results) and len(results) == len(approved.steps)
        return 0 if all_ok else 2


def main() -> int:
    load_dotenv(override=False)
    args = _parse_args()
    _configure_logging(args.verbose)

    if args.list_models:
        _print_model_catalog()
        return 0

    llm = _connect_llm(args)
    if llm is None:
        return 1

    serial = _resolve_serial(args)
    if serial is None and not args.no_picker:
        cli_ui.console.print("[dim]No device selected. Bye.[/dim]")
        return 0

    device = _connect_device(serial)
    if device is None:
        return 1

    info = device.info()
    device_label = (
        f"{info.get('productName', 'unknown')} "
        f"({info.get('displayWidth')}x{info.get('displayHeight')}, "
        f"sdk {info.get('sdkInt')})"
    )
    cli_ui.banner(provider_label=llm.describe(), device_label=device_label)

    return _run_once(args, llm, device)


if __name__ == "__main__":
    sys.exit(main())
