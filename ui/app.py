"""
RoboVibe — Gradio UI
Shows the PyBullet scene as an animated GIF after each command.
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

# --- Global state ---
sim        = RobotSimulator(headless=True)
sim.start()
perception = ScenePerception(nvidia_key=NVIDIA_KEY, mistral_key=MISTRAL_KEY)
planner    = RobotPlanner(api_key=MISTRAL_KEY, sim=sim)

STILL_PATH = "last_frame.jpg"
GIF_PATH   = "last_action.gif"


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


def run_command(user_input: str, history: list):
    if not user_input.strip():
        return history, _save_still()

    scene_desc = _get_scene_desc()

    # Record frames while planner executes tools
    sim.start_recording()
    response = planner.run(user_input.strip(), scene_description=scene_desc)
    frames = sim.stop_recording()

    output_path = _frames_to_gif(frames, GIF_PATH) if frames else _save_still()

    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": response})
    return history, output_path


# --- Gradio layout ---
with gr.Blocks(title="RoboVibe") as demo:
    gr.Markdown("# RoboVibe\n*Mistral Vibe, but for the physical world.*")

    with gr.Row():
        with gr.Column(scale=1):
            robot_view = gr.Image(
                value=_save_still(),
                label="Robot View",
                show_label=True,
            )

        with gr.Column(scale=1):
            chatbot = gr.Chatbot(label="Conversation", height=380)
            with gr.Row():
                text_input = gr.Textbox(
                    placeholder="e.g. 'wave hello', 'grab the blue box'",
                    show_label=False,
                    scale=4,
                )
                send_btn = gr.Button("Send", scale=1, variant="primary")

    scene_info = gr.Textbox(label="Scene state", interactive=False, lines=2)

    def on_send(msg, history):
        new_history, media = run_command(msg, history)
        state = str(sim.get_scene_state())
        return new_history, media, state, ""

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

if __name__ == "__main__":
    demo.launch(share=False, theme=gr.themes.Monochrome())
