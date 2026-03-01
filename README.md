# RoboVibe 🤖

> *Natural language → physical robot actions, powered by Mistral AI*

RoboVibe lets you control a robotic arm in a physics simulation using plain language. Type or speak a command — Mistral plans the action sequence, PyBullet executes it, and a live 3D viewer shows the result in real time.

---

## Demo

```
"wave hello"              → robot raises arm and waves
"grab the blue box"       → IK path → approach → grip
"pick the ball and place it on the right"  → multi-step plan
"do the dance"            → executes your saved macro
```

---

## Architecture

```
Browser (Three.js)
  ├─ 3D viewer — OrbitControls, real-time animation from keyframes
  ├─ Chat panel — text + voice (mic → Voxtral STT → command)
  └─ Macros panel — define custom movement sequences

FastAPI server
  ├─ POST /api/command  → Mistral plans → PyBullet executes → returns 3D frames
  ├─ POST /api/voice    → Voxtral transcribes → returns text
  ├─ GET  /api/scene    → current robot pose
  └─ CRUD /api/macros   → user-defined skills

Mistral stack
  ├─ mistral-small-latest  — agentic tool-calling planner
  ├─ pixtral-12b-2409      — scene perception (VLM, describes what the robot sees)
  └─ voxtral-mini          — voice input STT (browser mic)

PyBullet
  └─ KUKA iiwa 7-DOF arm, headless, records link positions every 4 steps
```

---

## Quick start

### 1. Environment

```bash
# Python 3.12 via micromamba
micromamba create -n mistral-py312 python=3.12
micromamba activate mistral-py312

# PyBullet (requires conda-forge, pip build fails on macOS 15)
micromamba install -c conda-forge pybullet -n mistral-py312

# Python deps
pip install -r requirements.txt
```

### 2. API keys

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

## Perception pipeline

Each command triggers a scene snapshot sent to the planner:

1. **Pixtral-12B** (Mistral) — VLM describes objects, positions, and reachability
2. **sim.get_scene_state()** — raw coordinates as fallback

---

## Macros / Skills

Users can define named movement sequences through the UI:

```
Name:  pickup_and_place
Steps: move_to(0.5, 0.1, 0.25), grab, move_to(0.6, -0.2, 0.25), release, reset
```

Saved to `macros.json`. Mistral knows about them automatically — just say the name.

---

## Project structure

```
Mistral-robot/
├── server.py           FastAPI server (main entry point)
├── main.py             CLI mode (text + voice, no browser)
├── robot/
│   └── simulator.py    PyBullet KUKA wrapper + 3D keyframe recording
├── agent/
│   ├── planner.py      Mistral tool-calling agentic loop
│   ├── perception.py   Pixtral-12B scene description
│   ├── macros.py       Macro storage, parsing, execution
│   └── voice.py        Voxtral STT (CLI + browser)
├── ui/
│   ├── index.html      Three.js web UI
│   └── app.py          Gradio UI (legacy fallback)
└── macros.json         Persisted user macros
```

---

## Roadmap

- [x] Voice input in browser (mic → Voxtral STT → command)
- [ ] Franka Panda robot (real gripper URDF)
- [ ] ElevenLabs TTS responses
