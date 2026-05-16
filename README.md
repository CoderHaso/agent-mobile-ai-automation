# Agentic Mobile AI Automation

A **Human-in-the-Loop Autonomous Android UI Agent** that converts a high-level
natural-language goal (e.g. *"Create a new Gmail account"*) into a live,
**adaptive** automation on a real device or emulator.

This is **not** a rigid macro recorder. The Planner produces high-level
**milestones** (broad objectives), and the Executor runs a continuous
**ReAct loop** that, on every iteration, looks at the actual screen and
decides the single next UI action. As a result the same plan:

- adapts across **Samsung / Xiaomi / Huawei / stock Android** without changes,
- **skips** milestones the real flow doesn't ask for (e.g. recovery email),
- handles **unexpected popups** (e.g. a Galaxy Store overlay over Gmail) as
  ad-hoc *recovery* actions without losing its place,
- **stops the moment the goal is achieved**, not when a step list is exhausted.

```
                ┌──────────────┐
   user goal -->│   Planner    │── JSON milestones (objectives, not taps)
                └──────┬───────┘
                       │ approve / edit / mark-optional (you)
                       ▼
   ┌────────────────────────────────────────────────────────┐
   │              Executor — adaptive ReAct loop            │
   │   Observe (XML)  →  Reason (LLM, with milestones)      │
   │        ▲                       │                       │
   │        │                       ▼                       │
   │        └────  Act (uiautomator2)  ←── recovery?        │
   └─────────────────────┬──────────────────────────────────┘
                         │
                         ▼
                ┌──────────────┐    background
                │   Android    │◀── Watchers (popups, perms, OEM nags)
                └──────────────┘
```

## Tech Stack

- **Python 3.8+** (3.10+ strongly recommended — 3.8 is end-of-life since Oct 2024).
  The Pydantic models are written with `typing.List` / `Optional` so the agent
  also runs cleanly on 3.8 / 3.9.
- **[uiautomator2](https://github.com/openatx/uiautomator2)** – ADB device control,
  XML hierarchy dumps, and background `Watchers`.
- **Groq** *or* **DeepSeek** (OpenAI-compatible chat-completions API) – fast LLM
  reasoning for both planning and per-step decisions.
- **[Rich](https://github.com/Textualize/rich)** – CLI tables, prompts, panels,
  syntax highlighting.

## Project Layout

```
agentic-mobile-ai-automation/
├── main.py                 # CLI entry point (Rich)
├── gui.py                  # Desktop GUI entry point (PySide6)
├── requirements.txt
├── .env.example
└── src/
    ├── __init__.py
    ├── cli_ui.py           # Rich-powered console helpers + approval flow
    ├── device_manager.py   # uiautomator2 wrapper (connect, dump, act, adb list)
    ├── executor.py         # Per-step Observe→Reason→Act→Verify loop (cancellable)
    ├── llm_client.py       # OpenAI-compatible client (Groq/DeepSeek)
    ├── planner.py          # Goal → JSON task list (Planner Agent)
    ├── ui_parser.py        # XML cleaning / interactable-element extraction
    ├── watchers.py         # Background popup/permission handlers
    └── gui/                # PySide6 desktop app
        ├── main_window.py  # 3-tab MainWindow + status bar
        ├── devices_tab.py  # Tab 1: device picker (USB + wireless)
        ├── plan_tab.py     # Tab 2: goal + editable plan table (HITL)
        ├── run_tab.py      # Tab 3: live execution + log + Stop button
        ├── workers.py      # QThread workers (Planner/Executor/Scan/Connect)
        └── style.py        # Dark Qt stylesheet
```

## Setup

### 1. Install dependencies

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure your LLM provider

```bash
cp .env.example .env
# then edit .env and set GROQ_API_KEY (or DEEPSEEK_API_KEY)
```

### 3. Prepare your Android device / emulator

1. Enable **Developer Options** → **USB debugging**.
2. Plug in via USB (or boot an emulator) and confirm with `adb devices`.
3. The first run installs the `uiautomator2` server on the device:

```bash
python -m uiautomator2 init
```

## Usage

### Desktop GUI (recommended)

```bash
python gui.py
```

Opens a PySide6 window with three tabs that mirror the CLI flow:

1. **Devices** — every ADB device in a table, with `Activate selected`,
   `Refresh`, `Disconnect selected` buttons and a `host:port` field for
   `adb connect` (wireless devices). Click **Use this device →** to open
   the uiautomator2 session.
2. **Plan** — goal input + `Generate plan` button. The plan table is
   inline-editable; use **Add step / Delete selected / Move up/down** to
   tweak it, then **Approve plan & start execution →**.
3. **Run** — live per-step status (Pending → Running → Done/Failed/Skipped)
   plus a streaming agent log. **Stop** signals the executor to halt at
   the next safe boundary.

The status bar always shows the connected device and LLM provider.

### CLI (Rich)

```bash
python main.py
```

You'll be guided through:

0. **Device picker** – if more than one device is visible to ADB (or you didn't
   pass `--serial`), an interactive Rich table lets you:
   - `use <#>` / `<#>` — set a device as the **active** one for this session
   - `toggle <#>` — flip the active flag
   - `connect <host:port>` — `adb connect` a wireless device
   - `disconnect <#>` — `adb disconnect` (TCP/wireless devices only)
   - `refresh` — re-scan ADB
   - `OK` — proceed with the active device
   - `quit` — exit
   Pass `--serial <S>` (or set `ANDROID_SERIAL=`) to skip the picker, or
   `--no-picker` to disable it altogether.
1. **Goal input** – type your high-level task.
2. **Plan review** – the Planner Agent returns a JSON task list. You can:
   - `approve` – accept as is
   - `edit <id>` – rewrite a step
   - `add <position>` – insert a new step
   - `delete <id>` – remove a step
   - `START` – begin execution
3. **Execution** – watchers run in the background while the executor processes
   each step. Per-step XML is sent to the LLM, which returns a strict JSON
   action `{action, target, input_value}` that gets executed on the device.
4. **Verification & summary** – after every step the agent verifies the screen
   changed; a final Rich table summarizes which steps passed/failed.

## Safety

- Run against a **test account / emulator first.** The agent will literally do
  what the LLM tells it to.
- Sensitive fields (passwords, OTPs) should be passed as `input_value` only when
  you've explicitly placed them in the plan during the human-in-the-loop step.
- All LLM responses are validated against a Pydantic schema before any tap.
