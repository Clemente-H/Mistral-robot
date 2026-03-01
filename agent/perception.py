"""
Scene perception — Pixtral-12B (Mistral API).
Describes the robot scene from a screenshot to inform the planner.
"""
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from mistralai import Mistral

PROMPT = """Describe the physical scene in the robot simulation.
Focus on:
- Position of each visible object (left/right/center, near/far, on the ground or elevated)
- Color and shape of each object
- Position of the robot arm end-effector
- What actions seem physically possible (e.g., the arm can reach the blue box)
Be concise and factual. Use spatial coordinates if visible."""


def _rgb_to_base64(frame: np.ndarray) -> str:
    img = Image.fromarray(frame)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


class ScenePerception:
    def __init__(self, mistral_key: str):
        self.client = Mistral(api_key=mistral_key)

    def describe(self, frame: np.ndarray) -> str:
        """Describe the scene via Pixtral-12B. Returns empty string on failure."""
        img_b64 = _rgb_to_base64(frame)
        try:
            response = self.client.chat.complete(
                model="pixtral-12b-2409",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url",
                         "image_url": f"data:image/jpeg;base64,{img_b64}"},
                    ],
                }],
                max_tokens=512,
            )
            print("  [pixtral] OK")
            return response.choices[0].message.content
        except Exception as e:
            print(f"  [pixtral] failed: {e}")
            return ""
