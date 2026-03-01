"""
Macro / skills system — lets users define named movement sequences.
Macros are persisted in macros.json at the project root.

Step format (as a string):
  wave, move_to(0.5, 0, 0.5), grab, move_to(0.6, -0.2, 0.3), release, reset

Parsed into a list of dicts:
  [{"action": "wave", "args": []},
   {"action": "move_to", "args": [0.5, 0, 0.5]}, ...]
"""
import json
import re
from pathlib import Path

MACROS_FILE = Path(__file__).parent.parent / "macros.json"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def load_macros() -> dict:
    if MACROS_FILE.exists():
        try:
            return json.loads(MACROS_FILE.read_text())
        except Exception:
            pass
    return {}


def save_macros(macros: dict) -> None:
    MACROS_FILE.write_text(json.dumps(macros, indent=2))


def add_macro(name: str, steps_str: str) -> dict:
    """Parse steps_str, store under name, return updated macros dict."""
    steps = parse_steps(steps_str)
    macros = load_macros()
    macros[name.strip().lower()] = {"steps": steps, "raw": steps_str.strip()}
    save_macros(macros)
    return macros


def delete_macro(name: str) -> dict:
    macros = load_macros()
    macros.pop(name.strip().lower(), None)
    save_macros(macros)
    return macros


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_steps(steps_str: str) -> list:
    """
    Parse a comma-separated step string into a list of action dicts.
    Respects commas inside parentheses (for args).
    Returns: [{"action": str, "args": [float|int, ...]}, ...]
    """
    # Split by commas NOT inside parentheses
    tokens = re.split(r",\s*(?![^(]*\))", steps_str)
    steps = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        m = re.match(r"^(\w+)\s*\(([^)]*)\)\s*$", token)
        if m:
            action = m.group(1).strip()
            raw_args = m.group(2).strip()
            args = [float(a.strip()) for a in raw_args.split(",") if a.strip()]
        else:
            action = token
            args = []
        steps.append({"action": action, "args": args})
    return steps


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def execute_macro(name: str, sim) -> str:
    """Execute a named macro against the simulator. Returns a status string."""
    macros = load_macros()
    key = name.strip().lower()
    if key not in macros:
        available = ", ".join(macros.keys()) or "none"
        return f"Macro '{name}' not found. Available: {available}"

    steps = macros[key]["steps"]
    log = []
    for step in steps:
        action = step["action"]
        args = step["args"]
        if action == "move_to" and len(args) >= 3:
            sim.move_to(args[0], args[1], args[2])
            log.append(f"move_to({args[0]}, {args[1]}, {args[2]})")
        elif action == "grab":
            sim.grab()
            log.append("grab()")
        elif action == "release":
            sim.release()
            log.append("release()")
        elif action == "reset":
            sim.reset()
            log.append("reset()")
        elif action == "wave":
            sim.wave()
            log.append("wave()")
        elif action == "dance":
            sim.dance()
            log.append("dance()")
        elif action == "sweep":
            sim.sweep()
            log.append("sweep()")
        elif action == "helicopter":
            sim.helicopter()
            log.append("helicopter()")
        elif action == "salute":
            sim.salute()
            log.append("salute()")
        elif action == "push" and len(args) >= 3:
            sim.push(args[0], args[1], args[2])
            log.append(f"push({args[0]}, {args[1]}, {args[2]})")
        else:
            log.append(f"[skipped unknown: {action}]")

    return f"Macro '{name}' done: {' → '.join(log)}"


def macros_for_prompt(macros: dict) -> str:
    """Build a short description of available macros for the system prompt."""
    if not macros:
        return ""
    lines = ["User-defined macros (call with execute_macro):"]
    for name, data in macros.items():
        lines.append(f"  - {name}: {data.get('raw', '')}")
    return "\n".join(lines)
