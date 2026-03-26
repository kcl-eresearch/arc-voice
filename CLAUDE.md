# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ArcVoice is an OpenAI-compatible TTS and speech transcription (ASR) API server powered by the Liquid AI LFM2.5-Audio-1.5B model. It exposes `/v1/audio/speech` and `/v1/audio/transcriptions` endpoints that match the OpenAI audio API contract.

## Commands

```bash
# Install dependencies (--no-build-isolation needed for flash-attn)
uv pip install --no-build-isolation -e ".[dev]"

# Run the server
python -m arcvoice.app
# or: uvicorn arcvoice.app:app --host 0.0.0.0 --port 8000

# Lint
ruff check arcvoice/
ruff format --check arcvoice/

# Run tests
pytest
pytest arcvoice/tests/test_foo.py::test_name   # single test
```

## Configuration

All settings are in `arcvoice/config.py` via pydantic-settings. Every field can be overridden with an `ARCVOICE_` prefixed env var (e.g. `ARCVOICE_API_KEY`, `ARCVOICE_DEVICE=cpu`, `ARCVOICE_PORT=9000`). When `ARCVOICE_API_KEY` is unset, auth is disabled.

## Architecture

- **`app.py`** — FastAPI application, health check, `/v1/models` endpoint, uvicorn entrypoint.
- **`routes.py`** — `/v1/audio/speech` (TTS) and `/v1/audio/transcriptions` (ASR) route handlers. All audio routes require Bearer auth when configured.
- **`model.py`** — Thread-safe lazy-loading singleton for the LFM2AudioModel/Processor. Contains `synthesize()` (text→waveform), `transcribe()` (audio bytes→text), and `waveform_to_bytes()` (encoding). Output sample rate is 24kHz.
- **`schemas.py`** — Pydantic request/response models matching OpenAI's audio API shape.
- **`auth.py`** — Optional Bearer token dependency; skipped when no API key is configured.
- **`config.py`** — Centralized `Settings` singleton using pydantic-settings with `ARCVOICE_` env prefix.

The model interaction pattern uses `ChatState` from `liquid_audio`: a conversation turn is built with `add_text("[TTS]...")` or `add_audio(waveform) + add_text("[ASR]")`, then `model.generate_interleaved()` streams tokens. Single-element tokens are text; multi-element tokens are audio codes that get decoded via `processor.decode()`.

## Key Dependencies

- `liquid-audio` — Liquid AI model/processor (`LFM2AudioModel`, `LFM2AudioProcessor`, `ChatState`)
- `flash-attn>=2.7` — required for model inference (needs CUDA)
- `torchaudio` — audio I/O and format encoding

## Ruff Config

Target Python 3.10, line length 100.
