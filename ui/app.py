"""
RoboVibe — Gradio UI
- Tabbed layout: Control tab + Macros tab
- Interactive camera: azimuth, elevation, distance sliders
- Macro/skills system: define, save, delete custom movement sequences
Run: python ui/app.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import gradio as gr
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

MISTRAL_KEY = os.environ["MISTRAL_API_KEY"]
NVIDIA_KEY   = os.environ.get("NVIDIA_API_KEY", "")

from robot.simulator import RobotSimulator
from agent.planner import RobotPlanner
from agent.perception import ScenePerception
from agent.macros import load_macros, add_macro, delete_macro

# --- Global state ---
sim        = RobotSimulator(headless=True)
sim.start()
perception = ScenePerception(nvidia_key=NVIDIA_KEY, mistral_key=MISTRAL_KEY)
planner    = RobotPlanner(api_key=MISTRAL_KEY, sim=sim)

STILL_PATH = "last_frame.jpg"
GIF_PATH   = "last_action.gif"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_still() -> str:
    Image.fromarray(sim.get_screenshot()).save(STILL_PATH)
    return STILL_PATH


def _frames_to_gif(frames: list, path: str, fps: int = 15) -> str:
    if not frames:
        return _save_still()
    pil = [Image.fromarray(f) for f in frames]
    pil[0].save(
        path,
        save_all=True,
        append_images=pil[1:],
        duration=int(1000 / fps),
        loop=0,
    )
    return path


def _get_scene_desc() -> str:
    import numpy as np
    frame = sim.get_screenshot()
    desc = perception.describe(frame)
    return desc if desc else str(sim.get_scene_state())


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

def update_camera(azimuth: float, elevation: float, distance: float) -> str:
    """Called when any camera slider changes — re-render and return new image."""
    sim.set_camera(azimuth=azimuth, elevation=elevation, distance=distance)
    return _save_still()


# ---------------------------------------------------------------------------
# Chat / commands
# ---------------------------------------------------------------------------

def run_command(user_input: str, history: list):
    if not user_input.strip():
        return history, _save_still()

    scene_desc = _get_scene_desc()

    sim.start_recording()
    response = planner.run(user_input.strip(), scene_description=scene_desc)
    frames = sim.stop_recording()

    output_path = _frames_to_gif(frames, GIF_PATH) if frames else _save_still()

    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": response})
    return history, output_path


def on_send(msg, history):
    new_history, media = run_command(msg, history)
    state = str(sim.get_scene_state())
    return new_history, media, state, ""


# ---------------------------------------------------------------------------
# Macros
# ---------------------------------------------------------------------------

def _macros_as_markdown(macros: dict) -> str:
    if not macros:
        return "*No macros saved yet.*"
    lines = []
    for name, data in macros.items():
        lines.append(f"**{name}** — `{data.get('raw', '')}`")
    return "\n\n".join(lines)


def on_save_macro(name: str, steps: str):
    if not name.strip():
        return "⚠️ Macro name cannot be empty.", _macros_as_markdown(load_macros()), "", ""
    if not steps.strip():
        return "⚠️ Steps cannot be empty.", _macros_as_markdown(load_macros()), "", ""
    try:
        macros = add_macro(name, steps)
        return f"✅ Macro '{name.strip().lower()}' saved!", _macros_as_markdown(macros), "", ""
    except Exception as e:
        return f"❌ Error: {e}", _macros_as_markdown(load_macros()), name, steps


def on_delete_macro(name: str):
    if not name.strip():
        return "⚠️ Enter a macro name to delete.", _macros_as_markdown(load_macros())
    macros = delete_macro(name)
    return f"🗑️ Macro '{name.strip().lower()}' deleted (if it existed).", _macros_as_markdown(macros)


# ---------------------------------------------------------------------------
# Gradio layout
# ---------------------------------------------------------------------------
CSS = """
.camera-sliders { background: #1a1a1a; border-radius: 8px; padding: 12px; margin-top: 8px; }
.macro-steps textarea { font-family: monospace; }
footer { display: none !important; }
"""

with gr.Blocks(title="RoboVibe", css=CSS) as demo:
    gr.Markdown(
        "# RoboVibe\n"
        "*Natural language → Mistral AI → Physical robot actions*"
    )

    with gr.Tabs():

        # ------------------------------------------------------------------ #
        #  TAB 1 — CONTROL                                                   #
        # ------------------------------------------------------------------ #
        with gr.TabItem("Control"):
            with gr.Row():
                # Left column — robot view + camera sliders
                with gr.Column(scale=1):
                    robot_view = gr.Image(
                        value=_save_still(),
                        label="Robot View",
                        show_label=True,
                    )
                    with gr.Group(elem_classes="camera-sliders"):
                        gr.Markdown("**Camera**")
                        with gr.Row():
                            cam_az = gr.Slider(
                                0, 360, value=sim.camera_azimuth,
                                step=1, label="Azimuth °",
                                interactive=True,
                            )
                            cam_el = gr.Slider(
                                -10, 85, value=sim.camera_elevation,
                                step=1, label="Elevation °",
                                interactive=True,
                            )
                            cam_dist = gr.Slider(
                                0.5, 5.0, value=sim.camera_distance,
                                step=0.05, label="Distance m",
                                interactive=True,
                            )

                # Right column — chat
                with gr.Column(scale=1):
                    chatbot = gr.Chatbot(label="Conversation", height=380)
                    with gr.Row():
                        text_input = gr.Textbox(
                            placeholder="e.g. 'wave hello', 'grab the blue box', 'do the dance'",
                            show_label=False,
                            scale=4,
                        )
                        send_btn = gr.Button("Send", scale=1, variant="primary")

            scene_info = gr.Textbox(label="Scene state", interactive=False, lines=2)

            # Camera slider events — re-render on release (faster)
            for slider in [cam_az, cam_el, cam_dist]:
                slider.release(
                    fn=update_camera,
                    inputs=[cam_az, cam_el, cam_dist],
                    outputs=[robot_view],
                )

            # Chat events
            send_btn.click(
                on_send,
                inputs=[text_input, chatbot],
                outputs=[chatbot, robot_view, scene_info, text_input],
            )
            text_input.submit(
                on_send,
                inputs=[text_input, chatbot],
                outputs=[chatbot, robot_view, scene_info, text_input],
            )

        # ------------------------------------------------------------------ #
        #  TAB 2 — MACROS / SKILLS                                           #
        # ------------------------------------------------------------------ #
        with gr.TabItem("Macros / Skills"):
            gr.Markdown(
                "Define reusable movement sequences. "
                "Once saved, just say the macro name in the chat and the robot will execute it.\n\n"
                "**Step format:** `action, action(args), ...`  "
                "Valid actions: `wave`, `grab`, `release`, `reset`, `move_to(x, y, z)`\n\n"
                "**Example:** `move_to(0.5, 0.1, 0.3), grab, move_to(0.6, -0.2, 0.3), release, reset`"
            )

            with gr.Row():
                with gr.Column(scale=1):
                    macro_name_in = gr.Textbox(label="Macro name", placeholder="e.g. pickup_and_place")
                    macro_steps_in = gr.Textbox(
                        label="Steps",
                        placeholder="wave, move_to(0.5, 0.1, 0.25), grab, move_to(0.6, -0.2, 0.25), release, reset",
                        lines=3,
                        elem_classes="macro-steps",
                    )
                    with gr.Row():
                        save_btn = gr.Button("Save Macro", variant="primary")
                        delete_name_in = gr.Textbox(
                            label="Delete macro (name)",
                            placeholder="macro name",
                            scale=2,
                        )
                        delete_btn = gr.Button("Delete", variant="stop")

                    macro_status = gr.Textbox(label="Status", interactive=False)

                with gr.Column(scale=1):
                    macros_display = gr.Markdown(
                        value=_macros_as_markdown(load_macros()),
                        label="Saved Macros",
                    )

            save_btn.click(
                on_save_macro,
                inputs=[macro_name_in, macro_steps_in],
                outputs=[macro_status, macros_display, macro_name_in, macro_steps_in],
            )
            delete_btn.click(
                on_delete_macro,
                inputs=[delete_name_in],
                outputs=[macro_status, macros_display],
            )


if __name__ == "__main__":
    demo.launch(share=False, theme=gr.themes.Monochrome())
