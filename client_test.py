import argparse
import asyncio
import base64
import json
import numpy as np
import soundfile as sf
import websockets

async def transcribe(audio_path, endpoint="ws://localhost:8000/v1/realtime"):
    # Load and normalize the audio
    audio, sr = sf.read(audio_path, dtype="float32")
    pcm16 = (audio * 32768.0).clip(-32768, 32767).astype(np.int16)

    async with websockets.connect(endpoint) as ws:
        await ws.recv()  # Wait for session.created acknowledgment
        await ws.send(json.dumps({"type": "session.update"}))

        # Simulate real-time streaming by sending 500ms chunks (8000 samples at 16kHz)
        for i in range(0, len(pcm16), 8000):
            chunk = base64.b64encode(pcm16[i:i+8000].tobytes()).decode()
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append", 
                "audio": chunk
            }))
            # Optional: add asyncio.sleep(0.5) here to truly simulate live microphone pacing

        # Signal the end of the audio stream
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        text = ""
        while True:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=60))
            if msg["type"] == "transcription.delta":
                text += msg["delta"]
                print(msg["delta"], end="", flush=True) # Print tokens as they arrive
            elif msg["type"] == "transcription.done":
                print("\n\nTranscription complete.")
                break
        return text

# Run the test
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio via WebSocket")
    parser.add_argument(
        "--endpoint",
        default="ws://localhost:8000/v1/realtime",
        help="WebSocket server endpoint (default: ws://localhost:8000/v1/realtime)"
    )
    parser.add_argument(
        "audio_file",
        help="Path to the audio file to transcribe"
    )
    args = parser.parse_args()
    
    asyncio.run(transcribe(args.audio_file, args.endpoint))