"""
Mistral agent loop with tool-calling.
Receives a user command + scene description and calls robot actions.
"""
import json
from mistralai import Mistral
from robot.simulator import RobotSimulator
from agent.macros import load_macros, execute_macro, macros_for_prompt

# Base tool definitions sent to Mistral
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
    {
        "type": "function",
        "function": {
            "name": "dance",
            "description": "Perform a choreographed multi-phase dance: raises arm, swings left/right, spins wrist, bows.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sweep",
            "description": "Scan the workspace with a slow arc — extends arm and rotates base left to right.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "push",
            "description": "Push an object at the given position away from the robot along the +X axis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "X coordinate of the object"},
                    "y": {"type": "number", "description": "Y coordinate of the object"},
                    "z": {"type": "number", "description": "Z coordinate of the object (height)"},
                },
                "required": ["x", "y", "z"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_macro",
            "description": "Execute a user-defined movement macro (skill) by name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The macro name to execute"},
                },
                "required": ["name"],
            },
        },
    },
]

BASE_SYSTEM_PROMPT = """You are the controller of a KUKA iiwa robotic arm in a PyBullet simulation.
You ALWAYS respond by calling one or more tools — never just reply with text unless there is truly nothing physical to do.

Available actions and when to use them:
- wave(): greetings, "say hi", "hello", "wave" — long dramatic wave sequence
- dance(): "dance", "show me a dance", "celebrate" — multi-phase choreography
- sweep(): "scan", "look around", "survey the area" — slow arc scan of workspace
- push(x,y,z): "push the box", "knock over", "shove" — pushes object away from robot
- move_to(x,y,z): move end-effector to a position
- grab(): close gripper after positioning over an object
- release(): open gripper to drop object
- reset(): return to home position, "go home", "reset"
- execute_macro(name): run a user-defined skill/macro

Scene objects and coordinates:
- Robot base fixed at (0, 0, 0) — it cannot rotate or translate
- Reachable workspace: x=[0.2, 0.8], y=[-0.5, 0.5], z=[0.0, 0.8]
- Blue box (also called "blue cube", "box"): around (0.5, 0.1, 0.05)
- Red sphere (also called "red ball", "ball", "sphere"): around (0.6, -0.2, 0.04)
- Yellow cylinder (also called "yellow tube", "cylinder"): around (0.3, 0.4, 0.05)
- Green cube (also called "green block", "small cube"): around (0.4, -0.38, 0.04)

Grab sequence: move_to(x, y, z+0.2) → move_to(x, y, z) → grab()
Place sequence: move_to(dest_x, dest_y, z+0.2) → move_to(dest_x, dest_y, z) → release()
Push: push(x, y, z) — handles the full approach and push automatically

If the command is physically impossible (e.g. "turn around" — the base is fixed),
briefly explain why and suggest what you CAN do instead. Keep responses short."""


class RobotPlanner:
    def __init__(self, api_key: str, sim: RobotSimulator):
        self.client = Mistral(api_key=api_key)
        self.sim = sim
        self.messages = []  # system prompt injected fresh each run (includes macros)

    def _build_system_prompt(self) -> str:
        macros = load_macros()
        extra = macros_for_prompt(macros)
        if extra:
            return BASE_SYSTEM_PROMPT + "\n\n" + extra
        return BASE_SYSTEM_PROMPT

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
        elif name == "dance":
            self.sim.dance()
            return "Performed dance sequence"
        elif name == "sweep":
            self.sim.sweep()
            return "Swept the workspace"
        elif name == "push":
            self.sim.push(args["x"], args["y"], args["z"])
            return f"Pushed object at ({args['x']}, {args['y']}, {args['z']})"
        elif name == "execute_macro":
            result = execute_macro(args.get("name", ""), self.sim)
            return result
        return f"Unknown tool: {name}"

    def run(self, user_command: str, scene_description: str = "") -> str:
        """
        Run one turn of the agent loop.
        Returns the final text response from Mistral.
        """
        # Rebuild system prompt each turn so new macros are visible
        system_msg = {"role": "system", "content": self._build_system_prompt()}

        content = user_command
        if scene_description:
            content += f"\n\nCurrent scene (from Pixtral):\n{scene_description}"

        # Keep conversation history but always refresh system prompt
        if not self.messages:
            self.messages = [system_msg]
        else:
            self.messages[0] = system_msg  # update system in-place

        self.messages.append({"role": "user", "content": content})

        # Agentic loop — keep calling until no more tool calls
        # First call uses "any" to force at least one action.
        # Subsequent calls use "auto" so the model can choose to stop.
        max_iterations = 10
        for iteration in range(max_iterations):
            tool_choice = "any" if iteration == 0 else "auto"
            response = self.client.chat.complete(
                model="mistral-small-latest",
                messages=self.messages,
                tools=ROBOT_TOOLS,
                tool_choice=tool_choice,
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
                return msg.content or "Done."

            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                result = self._execute_tool(tc.function.name, args)
                print(f"  [tool] {tc.function.name}({args}) → {result}")
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        return "Actions completed."
