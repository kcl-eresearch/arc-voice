"""OpenAI-compatible audio API routes."""

from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from arcvoice.auth import verify_api_key
from arcvoice.model import SAMPLE_RATE, synthesize, transcribe, waveform_to_bytes
from arcvoice.schemas import SpeechRequest, TranscriptionResponse, TranscriptionVerboseResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/audio", dependencies=[Depends(verify_api_key)])

_AUDIO_CONTENT_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "flac": "audio/flac",
    "ogg": "audio/ogg",
    "opus": "audio/opus",
}


# ---------------------------------------------------------------------------
# POST /v1/audio/speech  –  Text-to-Speech
# ---------------------------------------------------------------------------


@router.post("/speech")
async def create_speech(req: SpeechRequest) -> Response:
    """Generate audio from text (OpenAI-compatible TTS endpoint)."""
    if not req.input.strip():
        raise HTTPException(status_code=400, detail="`input` must not be empty")

    logger.info("TTS request: %d chars, format=%s", len(req.input), req.response_format)

    try:
        waveform = synthesize(req.input)
    except Exception:
        logger.exception("TTS synthesis failed")
        raise HTTPException(status_code=500, detail="Speech synthesis failed")

    fmt = req.response_format
    # opus is encoded as ogg/opus by torchaudio
    encode_fmt = "ogg" if fmt == "opus" else fmt

    try:
        audio_bytes = waveform_to_bytes(waveform, fmt=encode_fmt)
    except Exception:
        logger.exception("Audio encoding to %s failed, falling back to wav", fmt)
        audio_bytes = waveform_to_bytes(waveform, fmt="wav")
        fmt = "wav"

    return Response(
        content=audio_bytes,
        media_type=_AUDIO_CONTENT_TYPES.get(fmt, "application/octet-stream"),
        headers={"Content-Disposition": f'attachment; filename="speech.{fmt}"'},
    )


# ---------------------------------------------------------------------------
# POST /v1/audio/transcriptions  –  Automatic Speech Recognition
# ---------------------------------------------------------------------------


@router.post("/transcriptions", response_model=None)
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form("lfm2-audio-1.5b"),
    language: str = Form("en"),
    prompt: str = Form(""),
    response_format: Literal["json", "text", "verbose_json"] = Form("json"),
    temperature: float = Form(0.0),
) -> TranscriptionResponse | TranscriptionVerboseResponse | Response:
    """Transcribe audio to text (OpenAI-compatible transcription endpoint)."""
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    logger.info(
        "Transcription request: file=%s size=%d format=%s",
        file.filename,
        len(audio_bytes),
        response_format,
    )

    try:
        text = transcribe(audio_bytes, filename=file.filename or "audio.wav")
    except Exception:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail="Transcription failed")

    if response_format == "text":
        return Response(content=text, media_type="text/plain")
    if response_format == "verbose_json":
        # Estimate duration from file size (rough; real duration would need decoding)
        import torchaudio, io

        try:
            info = torchaudio.info(io.BytesIO(audio_bytes))
            duration = info.num_frames / info.sample_rate
        except Exception:
            duration = None
        return TranscriptionVerboseResponse(text=text, language=language, duration=duration)

    return TranscriptionResponse(text=text)
