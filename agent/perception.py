"""
Scene perception — tries Cosmos Reason2 (NVIDIA NIM) first,
falls back to Pixtral-12B (Mistral API) if unavailable.
Both use the same prompt and return a physical scene description.
"""
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from openai import OpenAI
from mistralai import Mistral

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
    def __init__(self, nvidia_key: str = "", mistral_key: str = ""):
        self.nvidia_key = nvidia_key
        self.mistral_key = mistral_key

        if nvidia_key:
            self.cosmos_client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=nvidia_key,
            )
        if mistral_key:
            self.pixtral_client = Mistral(api_key=mistral_key)

    def _describe_cosmos(self, img_b64: str) -> str:
        response = self.cosmos_client.chat.completions.create(
            model="nvidia/cosmos-reason2-8b",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": COSMOS_PROMPT},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                ],
            }],
            max_tokens=512,
        )
        return response.choices[0].message.content

    def _describe_pixtral(self, img_b64: str) -> str:
        response = self.pixtral_client.chat.complete(
            model="pixtral-12b-2409",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": COSMOS_PROMPT},
                    {"type": "image_url",
                     "image_url": f"data:image/jpeg;base64,{img_b64}"},
                ],
            }],
            max_tokens=512,
        )
        return response.choices[0].message.content

    def describe(self, frame: np.ndarray) -> str:
        """
        Describe the scene. Tries Cosmos Reason2 first, falls back to Pixtral.
        Returns empty string if both fail (main.py uses sim state as last resort).
        """
        img_b64 = _rgb_to_base64(frame)

        if self.nvidia_key:
            try:
                result = self._describe_cosmos(img_b64)
                print("  [cosmos] OK")
                return result
            except Exception as e:
                print(f"  [cosmos] failed ({type(e).__name__}), trying Pixtral...")

        if self.mistral_key:
            try:
                result = self._describe_pixtral(img_b64)
                print("  [pixtral] OK")
                return result
            except Exception as e:
                print(f"  [pixtral] failed: {e}")

        return ""
