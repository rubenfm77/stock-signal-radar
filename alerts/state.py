"""
Tiny JSON-backed state store so the alert script only fires on a *change*
of combined signal, not on every scheduled run. The workflow commits this
file back to the repo after each run (see .github/workflows/signal-alerts.yml).
"""
from __future__ import annotations

import json
from pathlib import Path

STATE_PATH = Path(__file__).parent / "state.json"


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    return json.loads(STATE_PATH.read_text())


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2, default=str))
