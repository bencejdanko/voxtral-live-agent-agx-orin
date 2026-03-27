# Voxtral Live Agent on the AGX Orin

Scripts to create a live, low-latency and low memory voice-agent system using the AGX Orin

## Set up Voxtral Transcribe 2

[Follow the setup instructions at the documentation here](https://docs.openwear.ai/s/7fb400e8-f7f4-408f-bb3f-a7db9a531c47/doc/voxtral-KpbgfunaiW)

## Client testing script

```
python3 client_test.py --endpoint ws://ubuntu:8000/v1/realtime test_audio.wav
```

```
Hello, world.

Transcription complete.
```

## Voxtral TTS

To do: must modify `quantize_marlin` for `mistralai/Voxtral-4B-TTS-2603`.

The script currently uses RTN (Round-To-Nearest) with Marlin-packed INT4.