"""
RoboVibe — main entry point.
Modes:
  python main.py           → voice loop (mic → Voxtral → Mistral → PyBullet)
  python main.py --text    → text input loop (no mic, for testing)
"""
import os
import sys
import argparse
from dotenv import load_dotenv

load_dotenv()

MISTRAL_KEY = os.environ["MISTRAL_API_KEY"]
NVIDIA_KEY   = os.environ["NVIDIA_API_KEY"]

from robot.simulator import RobotSimulator
from agent.planner import RobotPlanner
from agent.perception import ScenePerception


def run_loop(text_mode: bool = False):
    print("Starting RoboVibe...")
    sim = RobotSimulator(headless=True)
    sim.start()
    print("PyBullet simulator started.")

    perception = ScenePerception(api_key=NVIDIA_KEY)
    planner = RobotPlanner(api_key=MISTRAL_KEY, sim=sim)

    if not text_mode:
        from agent.voice import VoiceListener
        listener = VoiceListener(api_key=MISTRAL_KEY)

    print("\nRoboVibe ready. Type 'quit' to exit.\n")

    try:
        while True:
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

            print(f"\n> Command: {command}")

            # Get scene state from Cosmos Reason2
            print("  [perception] Capturing scene...")
            frame = sim.get_screenshot()
            scene_desc = perception.describe(frame)
            print(f"  [cosmos] {scene_desc[:120]}...")

            # Run Mistral planner
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
    args = parser.parse_args()
    run_loop(text_mode=args.text)
