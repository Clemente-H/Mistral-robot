"""
Quick check: list available NVIDIA NIM models with current API key.
"""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"],
)

try:
    models = client.models.list()
    print("Available models:")
    for m in models.data:
        print(f"  - {m.id}")
except Exception as e:
    print(f"Error: {e}")
