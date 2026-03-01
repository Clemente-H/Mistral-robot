"""
RoboVibe — FastAPI server
Serves the Three.js web UI and handles Mistral AI + PyBullet simulation.

Run: python server.py
     → open http://localhost:7860
"""
import asyncio
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

load_dotenv()

MISTRAL_KEY = os.environ["MISTRAL_API_KEY"]

from robot.simulator import RobotSimulator
from agent.planner import RobotPlanner
from agent.perception import ScenePerception
from agent.macros import add_macro, delete_macro, load_macros

# ---------------------------------------------------------------------------
# Global state — single worker thread for PyBullet (not thread-safe)
# ---------------------------------------------------------------------------
sim        = RobotSimulator(headless=True)
sim.start()
perception = ScenePerception(mistral_key=MISTRAL_KEY)
planner    = RobotPlanner(api_key=MISTRAL_KEY, sim=sim)
executor   = ThreadPoolExecutor(max_workers=1)

UI_DIR = Path(__file__).parent / "ui"

app = FastAPI(title="RoboVibe")


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class CommandRequest(BaseModel):
    text: str


class MacroRequest(BaseModel):
    name: str
    steps: str


# ---------------------------------------------------------------------------
# Sync simulation logic (runs inside ThreadPoolExecutor)
# ---------------------------------------------------------------------------
def _run_command_sync(text: str) -> dict:
    """Execute a user command: Mistral plans → PyBullet executes → return 3D frames."""
    frame = sim.get_screenshot()
    scene_desc = perception.describe(frame)
    if not scene_desc:
        scene_desc = str(sim.get_scene_state())

    sim.start_recording_joints()
    response = planner.run(text, scene_description=scene_desc)
    frames = sim.stop_recording_joints()

    # Always include at least the final pose as one frame
    if not frames:
        frames = [sim._get_3d_frame()]

    return {"response": response, "frames": frames}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    index = UI_DIR / "index.html"
    if not index.exists():
        return JSONResponse({"error": "index.html not found"}, status_code=404)
    return FileResponse(index)


@app.get("/app.js")
async def get_appjs():
    return FileResponse(UI_DIR / "app.js", media_type="application/javascript")


@app.get("/api/scene")
async def get_scene():
    """Current robot pose — used for initial Three.js render."""
    return sim._get_3d_frame()


@app.post("/api/command")
async def handle_command(req: CommandRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty command")
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, _run_command_sync, req.text.strip())
    return result


# Macros CRUD
@app.get("/api/macros")
async def get_macros():
    return load_macros()


@app.post("/api/macros")
async def create_macro(req: MacroRequest):
    try:
        macros = add_macro(req.name, req.steps)
        return macros
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/macros/{name}")
async def remove_macro(name: str):
    return delete_macro(name)


@app.post("/api/voice")
async def handle_voice(request: Request):
    """Receive raw PCM16-LE audio from browser, transcribe via Voxtral, return text."""
    sample_rate = int(request.query_params.get("sample_rate", 16000))
    body = await request.body()
    if len(body) < 3200:  # < 0.1 s at 16 kHz
        raise HTTPException(status_code=400, detail="Audio too short")

    from agent.voice import transcribe_pcm_bytes
    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(
        executor, lambda: transcribe_pcm_bytes(MISTRAL_KEY, body, sample_rate)
    )
    return {"text": text}


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("RoboVibe starting → http://localhost:7860")
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")
