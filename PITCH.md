# RoboVibe — Pitch & Demo Guide

## The idea in one sentence

> **Mistral as the bridge between human language and physical robot action.**

Today, programming a robot requires knowing its coordinate system, its kinematics, its joint limits. RoboVibe removes all of that. You just talk to it.

---

## The problem

Robots are physically capable of incredible things. But the gap between *what you want* and *what the robot understands* has always required an engineer in the middle — someone who translates human intent into motor commands, trajectories, and control loops.

---

## The solution

What if that translation layer was a language model?

Mistral doesn't just respond with text. It **calls functions**. It can look at a scene, understand what's there, plan a sequence of physical actions, and execute them — all from a single sentence.

RoboVibe is the proof of concept:
- You say *"grab the blue box and move it to the right"*
- Mistral sees the scene (via Pixtral vision), plans the move sequence, calls `move_to()` → `grab()` → `move_to()` → `release()`
- The robot arm executes it in real time

---

## What makes it interesting

**1. The model is the controller**
No hard-coded behavior trees. No scripted responses. Mistral decides the action sequence based on the scene and the intent. The same model that writes code and answers questions now operates physical hardware.

**2. Skills grow with usage**
Users can define macros — "remember this sequence as *dance*". The model learns the macro vocabulary. Next time you say "do the dance", it knows. This is how robots learn from their operators, not from engineers.

**3. The full Mistral stack**
- `mistral-small` — plans and executes via tool-calling
- `pixtral-12b` — eyes of the robot, describes the scene
- `voxtral-mini` — ears of the robot, understands voice commands
- All on one API key

---

## Demo flow (live)

```
1. Open http://localhost:7860
   → Show the 3D KUKA arm, rotatable with mouse

2. "wave hello"
   → Robot raises arm and waves 5 times

3. "grab the blue box"
   → Mistral calls: move_to(0.5, 0.1, 0.25) → move_to(0.5, 0.1, 0.05) → grab()
   → Watch the arm plan and execute

4. Define a macro live:
   Name: patrol
   Steps: move_to(0.5, 0.3, 0.4), move_to(0.5, -0.3, 0.4), move_to(0.5, 0, 0.2), reset
   → "do the patrol" — robot executes the custom sequence

5. Show voice: click mic, speak a command, watch it execute
```

---

## Key talking points

- **Not a chatbot.** Mistral here is an action planner — its outputs are physical movements, not words.
- **Generalizable.** The same architecture works for any robot, any task, any language. Swap the URDF, keep the stack.
- **Teachable at runtime.** Macros are the beginning of a skill library that users build through conversation, not code.
- **Mistral-native.** Planning, vision, and voice all run through the same API. One key, one platform.

---

## One-liner for judges

*"We used Mistral's tool-calling, vision, and voice APIs to turn a language model into a robot controller. You talk, it moves."*
