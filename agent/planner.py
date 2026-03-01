"""
Mistral agent loop with tool-calling.
Receives a user command + scene description and calls robot actions.
"""
import json
from mistralai import Mistral
from robot.simulator import RobotSimulator
from agent.macros import load_macros, execute_macro, macros_for_prompt

# Base tool definitions sent to Mistral
_VALID_TOOL_NAMES: set[str] = set()  # populated after ROBOT_TOOLS is defined

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
            "name": "helicopter",
            "description": "Spin the forearm like a helicopter rotor while the arm sways — entertaining whirring motion.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "salute",
            "description": "Military salute — arm raises to the side with a crisp hold, then lowers with a bow.",
            "parameters": {"type": "object", "properties": {}, "required": []},
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

_VALID_TOOL_NAMES = {t["function"]["name"] for t in ROBOT_TOOLS}

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
- helicopter(): "helicopter", "spin", "rotor" — forearm spins like rotor blades, arm sways
- salute(): "salute", "attention", "yes sir" — crisp military salute with bow finish
- execute_macro(name): run a user-defined skill/macro

Scene objects and coordinates:
- Robot base fixed at (0, 0, 0) — it cannot rotate or translate
- Reachable workspace: x=[0.2, 0.8], y=[-0.5, 0.5], z=[0.0, 0.8]
- Blue box (also called "blue cube", "box"): around (0.5, 0.1, 0.05)
- Red sphere (also called "red ball", "ball"): around (0.6, -0.2, 0.04)
- Yellow cylinder (also called "yellow tube"): around (0.3, 0.4, 0.05)
- Green cube (also called "green block"): around (0.4, -0.38, 0.04)
- Purple box (also called "purple cube", "purple"): around (0.7, 0.25, 0.04)
- Cyan sphere (also called "cyan ball", "teal"): around (0.22, 0.05, 0.04)
- Pink cylinder (also called "pink tube", "pink"): around (0.72, -0.35, 0.04)

The current scene (exact coordinates) will be provided with each command — use those coordinates.
When placing "next to" or "near" an object, offset ~0.15 m in x or y from that object's position.

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
        elif name == "helicopter":
            self.sim.helicopter()
            return "Helicopter rotor sequence done"
        elif name == "salute":
            self.sim.salute()
            return "Saluted"
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
            # Filter out malformed tool calls (Mistral occasionally puts JSON as the name)
            valid_calls = [
                tc for tc in (msg.tool_calls or [])
                if tc.function.name in _VALID_TOOL_NAMES
            ]
            invalid_calls = [
                tc for tc in (msg.tool_calls or [])
                if tc.function.name not in _VALID_TOOL_NAMES
            ]
            for tc in invalid_calls:
                print(f"  [planner] ignored malformed tool call: {tc.function.name!r:.60}")

            if valid_calls:
                assistant_msg = {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in valid_calls
                    ],
                }
                self.messages.append(assistant_msg)
            else:
                if msg.content:
                    self.messages.append({"role": "assistant", "content": msg.content})
                return msg.content or "Done."

            for tc in valid_calls:
                args = json.loads(tc.function.arguments)
                result = self._execute_tool(tc.function.name, args)
                print(f"  [tool] {tc.function.name}({args}) → {result}")
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        return "Actions completed."

    def run_streaming(self, user_command: str, scene_description: str = ""):
        """
        Generator version of run(). Yields (event_type, data) tuples so the
        server can stream each tool execution to the client in real time.

        Events:
          ("tool_start", {"name": str, "args": dict})
          ("frames",     {"tool": str, "frames": list})
          ("response",   {"text": str})
        """
        system_msg = {"role": "system", "content": self._build_system_prompt()}
        content = user_command
        if scene_description:
            content += f"\n\nCurrent scene (from Pixtral):\n{scene_description}"

        if not self.messages:
            self.messages = [system_msg]
        else:
            self.messages[0] = system_msg

        self.messages.append({"role": "user", "content": content})

        for iteration in range(10):
            tool_choice = "any" if iteration == 0 else "auto"
            response = self.client.chat.complete(
                model="mistral-small-latest",
                messages=self.messages,
                tools=ROBOT_TOOLS,
                tool_choice=tool_choice,
            )
            msg = response.choices[0].message

            valid_calls = [
                tc for tc in (msg.tool_calls or [])
                if tc.function.name in _VALID_TOOL_NAMES
            ]
            for tc in (msg.tool_calls or []):
                if tc.function.name not in _VALID_TOOL_NAMES:
                    print(f"  [planner] ignored malformed tool: {tc.function.name!r:.60}")

            if valid_calls:
                assistant_msg = {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in valid_calls
                    ],
                }
                self.messages.append(assistant_msg)
            else:
                if msg.content:
                    self.messages.append({"role": "assistant", "content": msg.content})
                yield "response", {"text": msg.content or "Done."}
                return

            for tc in valid_calls:
                args = json.loads(tc.function.arguments)
                yield "tool_start", {"name": tc.function.name, "args": args}

                # Record frames just for this one tool call
                self.sim.start_recording_joints()
                result = self._execute_tool(tc.function.name, args)
                frames = self.sim.stop_recording_joints()
                if not frames:
                    frames = [self.sim._get_3d_frame()]

                print(f"  [tool] {tc.function.name}({args}) → {result}")
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
                yield "frames", {"tool": tc.function.name, "frames": frames}

        yield "response", {"text": "Actions completed."}
