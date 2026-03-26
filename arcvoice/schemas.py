"""OpenAI-compatible request/response schemas for audio endpoints."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# POST /v1/audio/speech
# ---------------------------------------------------------------------------


class SpeechRequest(BaseModel):
    model: str = "lfm2-audio-1.5b"
    input: str = Field(..., description="Text to synthesize")
    voice: str = Field(default="default", description="Voice identifier (accepted but not used)")
    response_format: Literal["wav", "mp3", "flac", "ogg", "opus"] = "wav"
    speed: float = Field(
        default=1.0, ge=0.25, le=4.0, description="Accepted for compatibility; ignored"
    )


# ---------------------------------------------------------------------------
# POST /v1/audio/transcriptions
# ---------------------------------------------------------------------------


class TranscriptionResponse(BaseModel):
    text: str


class TranscriptionVerboseResponse(BaseModel):
    task: str = "transcribe"
    language: str = "english"
    duration: float | None = None
    text: str
