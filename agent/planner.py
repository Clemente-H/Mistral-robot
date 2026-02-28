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
]

SYSTEM_PROMPT = """You are a robotic arm controller for a KUKA iiwa arm in a PyBullet simulation.

Scene coordinates (approximate):
- Robot base: (0, 0, 0)
- Reachable workspace: x=[0.2, 0.8], y=[-0.5, 0.5], z=[0.0, 0.8]
- Blue box: around (0.5, 0.1, 0.05)
- Target/table zone: around (0.6, -0.2, 0.04)

When the user gives a command, use the available tools to execute it step by step.
To grab an object: first move_to above it (z+0.2), then move_to its position, then grab.
To place: move_to destination, then release.
Always end with a brief confirmation of what you did."""


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
            self.messages.append(msg)

            if not msg.tool_calls:
                # Final text response
                return msg.content

            # Execute all tool calls in this turn
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                result = self._execute_tool(tc.function.name, args)
                print(f"  [tool] {tc.function.name}({args}) → {result}")
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
