"""
Scene perception via NVIDIA Cosmos Reason2 (NIM API).
Takes a screenshot from PyBullet and returns a physical scene description.
"""
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from openai import OpenAI

COSMOS_PROMPT = """Describe the physical scene in the robot simulation.
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
    def __init__(self, api_key: str):
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key,
        )

    def describe(self, frame: np.ndarray) -> str:
        """Send a screenshot to Cosmos Reason2 and return scene description."""
        img_b64 = _rgb_to_base64(frame)
        response = self.client.chat.completions.create(
            model="nvidia/cosmos-reason2-8b",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": COSMOS_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                        },
                    ],
                }
            ],
            max_tokens=512,
        )
        return response.choices[0].message.content
