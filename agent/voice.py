"""
Voice input via Voxtral Realtime (Mistral API).
Records from microphone until silence, returns transcribed text.
"""
import asyncio
import queue
import numpy as np
import sounddevice as sd
from mistralai import Mistral
from mistralai.models import AudioFormat

SAMPLE_RATE = 16000
CHANNELS = 1
SILENCE_THRESHOLD = 0.01   # RMS below this = silence
SILENCE_DURATION = 1.5     # seconds of silence to stop recording
CHUNK_SIZE = 1024           # frames per callback


def record_until_silence() -> np.ndarray:
    """
    Record from microphone until SILENCE_DURATION seconds of silence.
    Returns int16 PCM array.
    """
    audio_chunks = []
    silence_frames = 0
    silence_limit = int(SAMPLE_RATE * SILENCE_DURATION / CHUNK_SIZE)
    recording = [True]

    def callback(indata, frames, time_info, status):
        rms = np.sqrt(np.mean(indata ** 2))
        nonlocal silence_frames
        if rms < SILENCE_THRESHOLD:
            silence_frames += 1
            if silence_frames >= silence_limit:
                recording[0] = False
        else:
            silence_frames = 0
        audio_chunks.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        dtype="float32", blocksize=CHUNK_SIZE,
                        callback=callback):
        print("🎙 Listening... (speak now)")
        while recording[0]:
            sd.sleep(50)

    audio = np.concatenate(audio_chunks, axis=0)
    return (audio * 32767).astype(np.int16)


async def _transcribe_async(client: Mistral, pcm: np.ndarray) -> str:
    """Send PCM audio to Voxtral Realtime and collect full transcript."""
    audio_bytes = pcm.tobytes()
    chunk_bytes = 16000 * 2  # 0.5s chunks

    async def audio_gen():
        for i in range(0, len(audio_bytes), chunk_bytes):
            yield audio_bytes[i:i + chunk_bytes]

    final_text = ""
    async for event in client.audio.realtime.transcribe_stream(
        audio_stream=audio_gen(),
        model="voxtral-mini-transcribe-realtime-2602",
        audio_format=AudioFormat(encoding="pcm_s16le", sample_rate=SAMPLE_RATE),
    ):
        if hasattr(event, "text") and event.text:
            final_text = event.text  # overwrite — last event is the complete transcript

    return final_text.strip()


class VoiceListener:
    def __init__(self, api_key: str):
        self.client = Mistral(api_key=api_key)

    def listen(self) -> str:
        """Record audio and return transcribed text (blocking)."""
        pcm = record_until_silence()
        text = asyncio.run(_transcribe_async(self.client, pcm))
        print(f"Transcribed: {text!r}")
        return text


def transcribe_pcm_bytes(api_key: str, pcm_bytes: bytes, sample_rate: int = 16000) -> str:
    """
    Transcribe raw PCM16-LE bytes received from the browser.
    Resamples to SAMPLE_RATE (16 kHz) if the browser ran at a different rate.
    """
    pcm = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    if sample_rate != SAMPLE_RATE:
        # Linear interpolation to target sample rate
        n = int(len(pcm) * SAMPLE_RATE / sample_rate)
        xs_old = np.arange(len(pcm))
        xs_new = np.linspace(0, len(pcm) - 1, n)
        pcm = np.interp(xs_new, xs_old, pcm)

    pcm_int16 = (np.clip(pcm, -1.0, 1.0) * 32767).astype(np.int16)
    client = Mistral(api_key=api_key)
    return asyncio.run(_transcribe_async(client, pcm_int16))
