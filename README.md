# ArcVoice

OpenAI-compatible TTS and speech transcription API powered by [Liquid AI LFM2.5-Audio-1.5B](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B).

Drop-in replacement for OpenAI's `/v1/audio/speech` and `/v1/audio/transcriptions` endpoints.

## Requirements

- Python 3.10+
- CUDA GPU recommended

## Installation

```bash
uv pip install -e .

# For development (ruff, pytest, httpx)
uv pip install -e ".[dev]"

# Recommended: install flash-attn for faster inference (requires CUDA)
uv pip install --no-build-isolation flash-attn>=2.7
```

## Quick Start

```bash
# Start the server
python -m arcvoice.app
```

The server runs on `http://0.0.0.0:8000` by default.

### Text-to-Speech

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, world!", "model": "lfm2-audio-1.5b"}' \
  --output speech.wav
```

### Transcription

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=lfm2-audio-1.5b
```

### Using the OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="your-key")

# TTS
response = client.audio.speech.create(
    model="lfm2-audio-1.5b",
    input="Hello from ArcVoice!",
    voice="default",
)
response.stream_to_file("output.wav")

# Transcription
transcript = client.audio.transcriptions.create(
    model="lfm2-audio-1.5b",
    file=open("audio.wav", "rb"),
)
print(transcript.text)
```

## Configuration

All settings are configured via environment variables with the `ARCVOICE_` prefix:

| Variable | Default | Description |
|---|---|---|
| `ARCVOICE_MODEL_ID` | `LiquidAI/LFM2.5-Audio-1.5B` | HuggingFace model ID |
| `ARCVOICE_HOST` | `0.0.0.0` | Server bind address |
| `ARCVOICE_PORT` | `8000` | Server port |
| `ARCVOICE_API_KEY` | *(none)* | Bearer token for auth; unset disables auth |
| `ARCVOICE_DEVICE` | `cuda` | Torch device (`cuda`, `cpu`) |
| `ARCVOICE_MAX_NEW_TOKENS` | `512` | Max tokens per generation |
| `ARCVOICE_AUDIO_TEMPERATURE` | `1.0` | Audio generation temperature |
| `ARCVOICE_AUDIO_TOP_K` | `4` | Audio generation top-k sampling |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/audio/speech` | Text-to-speech. Returns audio in wav, mp3, flac, ogg, or opus. |
| `POST` | `/v1/audio/transcriptions` | Speech-to-text. Accepts wav, mp3, flac, ogg, webm. Response format: `json`, `text`, or `verbose_json`. |
| `GET` | `/v1/models` | List available models. |
| `GET` | `/health` | Health check. |

## Development

```bash
# Lint
ruff check arcvoice/
ruff format --check arcvoice/

# Run tests
pytest
```
