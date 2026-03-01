"""
RoboVibe — FastAPI server
Serves the Three.js web UI and handles Mistral AI + PyBullet simulation.

Run: python server.py
     → open http://localhost:7860
"""
import asyncio
import json
import os
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ---------------------------------------------------------------------------
# Debug log capture — keeps last 400 lines in memory, exposed at /api/logs
# ---------------------------------------------------------------------------
_LOG_BUFFER: deque = deque(maxlen=400)

class _LogCapture:
    def __init__(self, orig): self._orig = orig
    def write(self, data):
        self._orig.write(data)
        stripped = data.rstrip()
        if stripped:
            _LOG_BUFFER.append(stripped)
    def flush(self): self._orig.flush()
    def isatty(self): return False

sys.stdout = _LogCapture(sys.stdout)
sys.stderr = _LogCapture(sys.stderr)

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
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
    """
    SSE endpoint — streams each tool execution as it happens.
    Events: thinking | tool_start | frames | response | error
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty command")

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def sync_worker():
        try:
            frame = sim.get_screenshot()
            scene_desc = perception.describe(frame) or str(sim.get_scene_state())

            asyncio.run_coroutine_threadsafe(
                queue.put(json.dumps({"type": "thinking"})), loop
            ).result()

            for ev_type, ev_data in planner.run_streaming(req.text.strip(), scene_desc):
                asyncio.run_coroutine_threadsafe(
                    queue.put(json.dumps({"type": ev_type, "data": ev_data})), loop
                ).result()
        except Exception as e:
            asyncio.run_coroutine_threadsafe(
                queue.put(json.dumps({"type": "error", "data": {"message": str(e)}})), loop
            ).result()
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

    executor.submit(sync_worker)

    async def generate():
        while True:
            item = await queue.get()
            if item is None:
                break
            yield f"data: {item}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Debug logs
@app.get("/api/logs")
async def get_logs(n: int = 100):
    """Return last N captured log lines. Use /api/logs?n=200 for more."""
    lines = list(_LOG_BUFFER)
    return {"lines": lines[-n:], "total": len(lines)}


@app.delete("/api/logs")
async def clear_logs():
    _LOG_BUFFER.clear()
    return {"cleared": True}


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
