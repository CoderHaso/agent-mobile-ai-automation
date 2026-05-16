"""Rich-powered CLI helpers.

All user-facing rendering and the human-in-the-loop approval REPL live here,
so the rest of the codebase stays I/O-clean.
"""

from __future__ import annotations

from typing import Iterable

from rich.box import ROUNDED
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from .device_manager import (
    AdbDeviceInfo,
    adb_connect,
    adb_disconnect,
    list_adb_devices,
)
from .executor import StepResult
from .planner import Plan, PlanStep


console = Console()


_STATUS_STYLE = {
    "pending": "dim",
    "running": "yellow bold",
    "done": "green bold",
    "failed": "red bold",
    "skipped": "blue",
}


def banner(provider_label: str, device_label: str) -> None:
    title = Text("Agentic Mobile AI Automation", style="bold magenta")
    body = Text.assemble(
        ("LLM provider : ", "dim"), (provider_label, "cyan"), "\n",
        ("Device       : ", "dim"), (device_label, "cyan"), "\n",
        ("Mode         : ", "dim"), ("Human-in-the-Loop", "green"),
    )
    console.print(Panel(body, title=title, border_style="magenta", box=ROUNDED))


def ask_goal() -> str:
    console.print()
    console.print(Panel.fit(
        "[bold]Step 1 — Describe your goal[/bold]\n"
        "Tell the agent what you want it to do, in plain English.\n"
        "[dim]Example: Open the Calculator app and compute 23 * 7[/dim]",
        border_style="cyan",
    ))
    while True:
        goal = Prompt.ask("[bold cyan]Goal[/bold cyan]").strip()
        if goal:
            return goal
        console.print("[red]Please enter a non-empty goal.[/red]")


def render_plan(plan: Plan, *, title: str = "Proposed Milestones") -> None:
    table = Table(
        title=f"{title}  —  {plan.goal}",
        title_style="bold magenta",
        box=ROUNDED,
        expand=True,
        header_style="bold cyan",
    )
    table.add_column("#", justify="right", width=4)
    table.add_column("Milestone (objective)", overflow="fold")
    table.add_column("Recognize when done", overflow="fold", style="dim")
    table.add_column("Opt", justify="center", width=4)
    table.add_column("Status", justify="center", width=10)

    for step in plan.steps:
        style = _STATUS_STYLE.get(step.status, "white")
        opt = Text("•", style="yellow") if step.is_optional else Text("")
        table.add_row(
            str(step.step_id),
            step.action_description,
            step.expected_outcome or "—",
            opt,
            Text(step.status.upper(), style=style),
        )
    console.print(table)


def review_loop(plan: Plan) -> Plan:
    """Interactive human-in-the-loop edit/approve session.

    Note: the plan now contains MILESTONES (high-level objectives), not
    atomic UI taps. The Executor decides each individual tap at runtime
    by looking at the live screen, so the same plan adapts across
    different OEMs and skips milestones the actual flow doesn't ask for.
    """
    help_md = Markdown(
        "**Commands**\n"
        "- `START`              — accept the milestones and begin adaptive execution\n"
        "- `edit <id>`          — rewrite milestone <id>\n"
        "- `add <position>`     — insert a new milestone at <position> (e.g. `add 3`)\n"
        "- `delete <id>`        — remove milestone <id>\n"
        "- `move <id> <pos>`    — move milestone <id> to position <pos>\n"
        "- `optional <id>`      — toggle the Optional flag (e.g. recovery email)\n"
        "- `replan`             — abort and start over with a new goal\n"
        "- `quit`               — exit without doing anything"
    )

    while True:
        render_plan(plan)
        console.print(Panel(help_md, border_style="cyan", title="Review"))
        cmd = Prompt.ask("[bold cyan]> [/bold cyan]").strip()
        if not cmd:
            continue
        lower = cmd.lower()

        if lower == "start":
            if not plan.steps:
                console.print("[red]Plan is empty — add at least one step first.[/red]")
                continue
            return plan.renumber()

        if lower == "quit":
            raise KeyboardInterrupt("User aborted at review.")

        if lower == "replan":
            raise _ReplanRequested()

        parts = cmd.split(maxsplit=2)
        op = parts[0].lower()

        try:
            if op == "edit" and len(parts) >= 2:
                _edit_step(plan, int(parts[1]))
            elif op == "add" and len(parts) >= 2:
                _add_step(plan, int(parts[1]))
            elif op == "delete" and len(parts) >= 2:
                _delete_step(plan, int(parts[1]))
            elif op == "move" and len(parts) >= 3:
                _move_step(plan, int(parts[1]), int(parts[2]))
            elif op == "optional" and len(parts) >= 2:
                _toggle_optional(plan, int(parts[1]))
            else:
                console.print("[yellow]Unknown command. Type one of: START / edit / add / delete / move / optional / replan / quit.[/yellow]")
        except (ValueError, IndexError) as exc:
            console.print(f"[red]Invalid command: {exc}[/red]")
        plan.renumber()


class _ReplanRequested(Exception):
    """Internal signal: user wants to start over with a new goal."""


def is_replan_requested(exc: BaseException) -> bool:
    return isinstance(exc, _ReplanRequested)


def _find_index(plan: Plan, step_id: int) -> int:
    for i, s in enumerate(plan.steps):
        if s.step_id == step_id:
            return i
    raise ValueError(f"step_id {step_id} not found")


def _edit_step(plan: Plan, step_id: int) -> None:
    i = _find_index(plan, step_id)
    s = plan.steps[i]
    new_action = Prompt.ask(
        "  New action description",
        default=s.action_description,
    ).strip()
    new_outcome = Prompt.ask(
        "  New expected outcome",
        default=s.expected_outcome or "",
    ).strip()
    if new_action:
        s.action_description = new_action
    s.expected_outcome = new_outcome


def _add_step(plan: Plan, position: int) -> None:
    action = Prompt.ask("  Milestone objective").strip()
    if not action:
        console.print("[red]Empty milestone — aborting add.[/red]")
        return
    outcome = Prompt.ask(
        "  Recognize-when-done hint (optional)", default=""
    ).strip()
    is_opt = Prompt.ask(
        "  Mark as Optional? (y/N)", default="n"
    ).strip().lower().startswith("y")
    new_step = PlanStep(
        step_id=position,
        action_description=action,
        expected_outcome=outcome,
        is_optional=is_opt,
        status="pending",
    )
    insert_at = max(0, min(position - 1, len(plan.steps)))
    plan.steps.insert(insert_at, new_step)


def _toggle_optional(plan: Plan, step_id: int) -> None:
    i = _find_index(plan, step_id)
    plan.steps[i].is_optional = not plan.steps[i].is_optional


def _delete_step(plan: Plan, step_id: int) -> None:
    i = _find_index(plan, step_id)
    plan.steps.pop(i)


def _move_step(plan: Plan, step_id: int, new_pos: int) -> None:
    i = _find_index(plan, step_id)
    s = plan.steps.pop(i)
    insert_at = max(0, min(new_pos - 1, len(plan.steps)))
    plan.steps.insert(insert_at, s)


def step_progress(step: PlanStep, message: str) -> None:
    style = _STATUS_STYLE.get(message, "cyan")
    console.print(
        f"[dim]·[/dim] [bold]Step {step.step_id}[/bold]: "
        f"[{style}]{message}[/{style}]"
    )


def render_results(results: Iterable[StepResult]) -> None:
    table = Table(
        title="Execution Summary",
        title_style="bold magenta",
        box=ROUNDED,
        header_style="bold cyan",
    )
    table.add_column("#", justify="right", width=4)
    table.add_column("Action", overflow="fold")
    table.add_column("Attempts", justify="right", width=10)
    table.add_column("Result", justify="center", width=10)
    table.add_column("Note", overflow="fold", style="dim")

    for r in results:
        result_style = "green bold" if r.success else "red bold"
        result_text = "DONE" if r.success else "FAILED"
        table.add_row(
            str(r.step.step_id),
            r.step.action_description,
            str(r.attempts),
            Text(result_text, style=result_style),
            r.note or "—",
        )
    console.print(table)


def _state_style(state: str) -> str:
    return {
        "device": "green bold",
        "offline": "red",
        "unauthorized": "yellow bold",
    }.get(state, "white")


def _render_devices(devices: list[AdbDeviceInfo]) -> None:
    table = Table(
        title="ADB Devices",
        title_style="bold magenta",
        box=ROUNDED,
        header_style="bold cyan",
    )
    table.add_column("#", justify="right", width=4)
    table.add_column("Active", justify="center", width=8)
    table.add_column("Serial", overflow="fold")
    table.add_column("Model", overflow="fold")
    table.add_column("SDK", justify="right", width=5)
    table.add_column("Transport", justify="center", width=10)
    table.add_column("State", justify="center", width=14)

    if not devices:
        table.add_row("—", "—", "[dim]no devices found[/dim]", "—", "—", "—", "—")
    else:
        for i, d in enumerate(devices, start=1):
            check = "[green bold][X][/green bold]" if d.active else "[ ]"
            table.add_row(
                str(i),
                check,
                d.serial,
                d.model or "[dim]?[/dim]",
                d.sdk or "[dim]?[/dim]",
                d.transport,
                Text(d.state, style=_state_style(d.state)),
            )
    console.print(table)


def pick_device(initial_serial: str | None = None) -> str | None:
    """Interactive ADB device picker.

    Returns the serial of the activated device, or None if the user quits.
    """
    devices = list_adb_devices()

    # Pre-activate either the env-supplied serial or the only usable device.
    def _autoselect():
        if initial_serial:
            for d in devices:
                if d.serial == initial_serial:
                    d.active = True
                    return
        usable = [d for d in devices if d.is_usable]
        if len(usable) == 1:
            usable[0].active = True

    _autoselect()

    help_md = Markdown(
        "**Commands**\n"
        "- `use <#>` / `<#>`         — set device as the ACTIVE one for this session\n"
        "- `toggle <#>`              — flip active flag (only one device may be active)\n"
        "- `connect <host:port>`     — `adb connect` a wireless device (e.g. `192.168.1.5:5555`)\n"
        "- `disconnect <#>`          — `adb disconnect` (TCP/wireless devices only)\n"
        "- `refresh`                 — re-scan ADB\n"
        "- `OK`                      — proceed with the active device\n"
        "- `quit`                    — exit"
    )

    while True:
        console.print()
        _render_devices(devices)
        console.print(Panel(help_md, border_style="cyan", title="Device picker"))

        active = next((d for d in devices if d.active), None)
        hint = (
            f"[green]active:[/green] {active.serial}"
            if active else "[yellow]no active device selected[/yellow]"
        )
        console.print(hint)

        cmd = Prompt.ask("[bold cyan]device> [/bold cyan]").strip()
        if not cmd:
            continue
        lower = cmd.lower()

        if lower in ("ok", "start", "go", "y", "yes"):
            if active is None:
                console.print("[red]Pick a device first (e.g. `use 1`).[/red]")
                continue
            if not active.is_usable:
                console.print(
                    f"[red]Device {active.serial!r} state is "
                    f"{active.state!r} — fix it on the device, then `refresh`.[/red]"
                )
                continue
            return active.serial

        if lower in ("quit", "exit", "q"):
            return None

        if lower == "refresh":
            previously_active = active.serial if active else None
            devices = list_adb_devices()
            for d in devices:
                if d.serial == previously_active:
                    d.active = True
            if not any(d.active for d in devices):
                _autoselect()
            continue

        parts = cmd.split(maxsplit=1)
        op = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if op.isdigit():
            arg = op
            op = "use"

        try:
            if op in ("use", "select", "toggle"):
                idx = int(arg) - 1
                if not (0 <= idx < len(devices)):
                    console.print(f"[red]No device #{arg}.[/red]")
                    continue
                target = devices[idx]
                if op == "toggle" and target.active:
                    target.active = False
                else:
                    for d in devices:
                        d.active = False
                    target.active = True

            elif op == "connect":
                if not arg:
                    console.print("[red]Usage: connect <host:port>[/red]")
                    continue
                msg = adb_connect(arg)
                console.print(f"[cyan]adb:[/cyan] {msg}")
                devices = list_adb_devices()
                for d in devices:
                    if d.serial == arg or d.serial.startswith(arg.split(":")[0]):
                        d.active = True
                        break

            elif op == "disconnect":
                idx = int(arg) - 1
                if not (0 <= idx < len(devices)):
                    console.print(f"[red]No device #{arg}.[/red]")
                    continue
                target = devices[idx]
                if not target.is_tcp:
                    console.print(
                        f"[yellow]{target.serial} is a USB device — unplug it physically "
                        "or use `adb kill-server` if you want it gone.[/yellow]"
                    )
                    continue
                msg = adb_disconnect(target.serial)
                console.print(f"[cyan]adb:[/cyan] {msg}")
                devices = list_adb_devices()
                if not any(d.active for d in devices):
                    _autoselect()

            else:
                console.print(
                    "[yellow]Unknown command. Try: <#>, use <#>, toggle <#>, "
                    "connect <host:port>, disconnect <#>, refresh, OK, quit.[/yellow]"
                )
        except (ValueError, IndexError) as exc:
            console.print(f"[red]Invalid command: {exc}[/red]")


def fatal(message: str) -> None:
    console.print(Panel(
        Text(message, style="bold red"),
        title="Fatal error",
        border_style="red",
    ))
