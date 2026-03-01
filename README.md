---
title: RoboVibe
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# RoboVibe

> *Natural language → physical robot actions, powered by Mistral AI*

RoboVibe lets you control a robotic arm in a physics simulation using plain language. Type or speak a command — Mistral plans the action sequence, PyBullet executes it, and a live 3D viewer animates the result in real time as each action streams back.

---

## Demo

```
"wave hello"                            → elbow out sideways, forearm sweeps horizontally
"dance"                                 → polyrhythm choreography (4 joints, different frequencies)
"helicopter"                            → forearm roll spins like rotor blades, arm sways
"salute"                                → military salute with bow finish
"grab the blue box and place it there"  → IK approach → grip → lift → move → release
"sweep the workspace"                   → slow arc scan left → right → center
"do my pickup macro"                    → executes user-defined skill by name
[mic button]  speak in any language     → Voxtral STT → auto-sends as command
```

---

## Architecture

```
Browser (Three.js)
  ├─ 3D viewer — OrbitControls, rAF + LERP interpolation, real-time streaming animation
  ├─ Chat panel — text + voice (mic → Voxtral STT → command)
  └─ Macros panel — define custom movement sequences at runtime

FastAPI server (SSE streaming)
  ├─ POST /api/command  → SSE stream: tool_start | frames | response events
  ├─ POST /api/voice    → Voxtral transcribes raw PCM → returns text
  ├─ GET  /api/scene    → current robot pose (initial render)
  ├─ GET  /api/logs     → in-memory log buffer (debug)
  └─ CRUD /api/macros   → user-defined skills

Mistral stack
  ├─ mistral-small-latest  — agentic tool-calling planner (streaming, per-tool frames)
  ├─ pixtral-12b-2409      — scene perception (VLM describes robot's field of view)
  └─ voxtral-mini          — voice input STT (browser mic, raw PCM16)

PyBullet
  ├─ KUKA iiwa 7-DOF arm, headless physics
  ├─ Sinusoidal joint control (wave, dance) — organic fluid motion
  ├─ IK-based move_to — PyBullet calculateInverseKinematics
  ├─ Constraint-based grab — JOINT_FIXED constraint attaches object to EE, lifts with arm
  └─ Keyframe recording: link positions every 4 steps → streamed to browser
```

### Streaming flow

```
user types command
  → POST /api/command (SSE)
    → Pixtral describes scene
    → Mistral plans (tool_choice="any" first call, "auto" after)
    → for each tool call:
        → "tool_start" event → UI shows "Running: wave…"
        → PyBullet executes + records frames
        → "frames" event → browser starts animating immediately
    → "response" event → final text in chat
```

The browser animation loop runs continuously — new frame batches are appended as they arrive, so chained actions (wave → move → grab) animate without pauses.

---

## Quick start

### 1. Environment

```bash
# Python 3.12 via micromamba
micromamba create -n mistral-py312 python=3.12
micromamba activate mistral-py312

# PyBullet (requires conda-forge — pip build fails on macOS 15)
micromamba install -c conda-forge pybullet -n mistral-py312

# Python deps
pip install -r requirements.txt
```

### 2. API key

```bash
cp .env.example .env
# Edit .env:
#   MISTRAL_API_KEY=...   (required — planning, vision, voice)
```

### 3. Run

```bash
python server.py
# → http://localhost:7860
```

---

## Robot actions

| Command | Tool | What happens |
|---|---|---|
| wave | `wave()` | Arm raises, j2 twists elbow sideways, j3 oscillates — classic horizontal wave |
| dance | `dance()` | 4 joints at different frequencies simultaneously (polyrhythm), bow finale |
| sweep | `sweep()` | Slow arc scan left → right → center |
| helicopter | `helicopter()` | Arm extends, forearm roll spins ±2.4 rad rapidly while base sways |
| salute | `salute()` | Military salute — arm raises to the side, crisp hold, bow finish |
| grab the box | `move_to` → `grab()` | IK approach + JOINT_FIXED constraint lifts object |
| push the sphere | `push(x,y,z)` | Approach from behind, push through |
| go home / reset | `reset()` | Smooth motor-controlled return to home pose |

---

## Macros / Skills

Define named sequences through the UI at runtime:

```
Name:  pickup_and_place
Steps: move_to(0.5, 0.1, 0.25), grab, move_to(0.6, -0.2, 0.25), release, reset
```

Saved to `macros.json`. Mistral sees them in its system prompt — just say the macro name.

---

## Perception pipeline

Each command triggers a scene snapshot:

1. **Pixtral-12B** — VLM describes objects, positions, and reachability in plain text
2. **sim.get_scene_state()** — exact coordinates as fallback if Pixtral is unavailable

---

## Debug

```bash
# See live logs (last 100 lines)
curl -s localhost:7860/api/logs | python3 -c "import sys,json; [print(l) for l in json.load(sys.stdin)['lines']]"

# Clear log buffer
curl -s -X DELETE localhost:7860/api/logs
```

---

## Project structure

```
Mistral-robot/
├── server.py           FastAPI server — SSE streaming, all routes
├── robot/
│   └── simulator.py    PyBullet KUKA wrapper, sinusoidal motions, constraint grab
├── agent/
│   ├── planner.py      Mistral agentic loop — run_streaming() generator
│   ├── perception.py   Pixtral-12B scene description
│   ├── macros.py       Macro storage, parsing, execution
│   └── voice.py        Voxtral STT (CLI + browser PCM16)
├── ui/
│   ├── index.html      Three.js web UI
│   └── app.js          SSE client, rAF+LERP animation, streaming frame buffer
└── macros.json         Persisted user macros
```
