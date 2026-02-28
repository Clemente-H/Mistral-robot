"""
Mistral agent loop with tool-calling.
Receives a user command + scene description and calls robot actions.
"""
import json
from mistralai import Mistral
from robot.simulator import RobotSimulator

# Tool definitions sent to Mistral
ROBOT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "move_to",
            "description": "Move the robot arm end-effector to a position in world coordinates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "X coordinate in meters"},
                    "y": {"type": "number", "description": "Y coordinate in meters"},
                    "z": {"type": "number", "description": "Z coordinate in meters (height)"},
                },
                "required": ["x", "y", "z"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grab",
            "description": "Close the gripper to grab an object near the end-effector.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "release",
            "description": "Open the gripper to release the held object.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reset",
            "description": "Return the robot arm to its home/rest position.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wave",
            "description": "Wave hello — raise the arm and oscillate the wrist as a greeting gesture.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

SYSTEM_PROMPT = """You are the controller of a KUKA iiwa robotic arm in a PyBullet simulation.
You ALWAYS respond by calling one or more tools — never just reply with text unless there is truly nothing physical to do.

Available actions and when to use them:
- wave(): greetings, "say hi", "hello", "wave"
- move_to(x,y,z): move end-effector to a position
- grab(): close gripper after positioning over an object
- release(): open gripper to drop object
- reset(): return to home position, "go home", "reset"

Scene coordinates:
- Robot base fixed at (0, 0, 0) — it cannot rotate or translate
- Reachable workspace: x=[0.2, 0.8], y=[-0.5, 0.5], z=[0.0, 0.8]
- Blue box: around (0.5, 0.1, 0.05)
- Target zone: around (0.6, -0.2, 0.04)

Grab sequence: move_to(x, y, z+0.2) → move_to(x, y, z) → grab()
Place sequence: move_to(dest_x, dest_y, z+0.2) → move_to(dest_x, dest_y, z) → release()

If the command is physically impossible (e.g. "turn around" — the base is fixed),
briefly explain why and suggest what you CAN do instead. Keep responses short."""


class RobotPlanner:
    def __init__(self, api_key: str, sim: RobotSimulator):
        self.client = Mistral(api_key=api_key)
        self.sim = sim
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def _execute_tool(self, name: str, args: dict) -> str:
        if name == "move_to":
            self.sim.move_to(args["x"], args["y"], args["z"])
            return f"Moved to ({args['x']}, {args['y']}, {args['z']})"
        elif name == "grab":
            self.sim.grab()
            return "Gripper closed"
        elif name == "release":
            self.sim.release()
            return "Gripper opened"
        elif name == "reset":
            self.sim.reset()
            return "Arm reset to home"
        elif name == "wave":
            self.sim.wave()
            return "Waved hello"
        return f"Unknown tool: {name}"

    def run(self, user_command: str, scene_description: str = "") -> str:
        """
        Run one turn of the agent loop.
        Returns the final text response from Mistral.
        """
        content = user_command
        if scene_description:
            content += f"\n\nCurrent scene (from Cosmos Reason2):\n{scene_description}"

        self.messages.append({"role": "user", "content": content})

        # Agentic loop — keep calling until no more tool calls
        while True:
            response = self.client.chat.complete(
                model="mistral-small-latest",
                messages=self.messages,
                tools=ROBOT_TOOLS,
                tool_choice="auto",
            )
            msg = response.choices[0].message

            # Always serialize to dict — never store SDK objects in history
            assistant_msg = {"role": "assistant", "content": msg.content}
            if msg.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            self.messages.append(assistant_msg)

            if not msg.tool_calls:
                return msg.content

            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                result = self._execute_tool(tc.function.name, args)
                print(f"  [tool] {tc.function.name}({args}) → {result}")
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
