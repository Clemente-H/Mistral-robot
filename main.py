"""
RoboVibe — main entry point.
Modes:
  python main.py --text       → text input, headless (for testing)
  python main.py --text --gui → text input + PyBullet GUI window
  python main.py --gui        → voice input + PyBullet GUI window
  python main.py              → full voice mode, headless
"""
import os
import argparse
from dotenv import load_dotenv

load_dotenv()

MISTRAL_KEY = os.environ["MISTRAL_API_KEY"]
NVIDIA_KEY   = os.environ.get("NVIDIA_API_KEY", "")

from robot.simulator import RobotSimulator
from agent.planner import RobotPlanner
from agent.perception import ScenePerception


def save_screenshot(sim: RobotSimulator, path: str = "last_frame.jpg"):
    """Save current scene as JPEG (useful to inspect what Cosmos sees)."""
    from PIL import Image
    frame = sim.get_screenshot()
    Image.fromarray(frame).save(path)
    print(f"  [screenshot] saved → {path}")


def run_loop(text_mode: bool = False, gui: bool = False):
    print("Starting RoboVibe...")
    sim = RobotSimulator(headless=not gui)
    sim.start()
    mode_str = "GUI" if gui else "headless"
    print(f"PyBullet simulator started ({mode_str}).")

    perception = ScenePerception(nvidia_key=NVIDIA_KEY, mistral_key=MISTRAL_KEY)
    planner = RobotPlanner(api_key=MISTRAL_KEY, sim=sim)

    if not text_mode:
        from agent.voice import VoiceListener
        listener = VoiceListener(api_key=MISTRAL_KEY)

    print("\nRoboVibe ready. Type 'quit' to exit.\n")

    try:
        while True:
            try:
                if text_mode:
                    command = input("You: ").strip()
                    if command.lower() in ("quit", "exit", "q"):
                        break
                    if not command:
                        continue
                else:
                    command = listener.listen()
                    if not command:
                        continue
            except (KeyboardInterrupt, EOFError):
                print("\nExiting...")
                break

            print(f"\n> Command: {command}")

            # Capture scene — try Cosmos first, fall back to sim state
            scene_desc = ""
            frame = sim.get_screenshot()
            save_screenshot(sim)  # always save last_frame.jpg to inspect

            if NVIDIA_KEY:
                print("  [perception] Sending to Cosmos Reason2...")
                scene_desc = perception.describe(frame)

            if not scene_desc:
                scene_desc = str(sim.get_scene_state())

            print(f"  [scene] {scene_desc[:150]}")

            # Mistral plans + executes
            print("  [planner] Planning actions...")
            response = planner.run(command, scene_description=scene_desc)
            print(f"\nRobot: {response}\n")

    finally:
        sim.stop()
        print("Simulator stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", action="store_true",
                        help="Use text input instead of voice")
    parser.add_argument("--gui", action="store_true",
                        help="Show PyBullet GUI window")
    args = parser.parse_args()
    run_loop(text_mode=args.text, gui=args.gui)
