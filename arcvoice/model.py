"""Liquid AI LFM2.5-Audio-1.5B model wrapper for TTS and ASR."""

from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path
from threading import Lock

import torch
import torchaudio
from liquid_audio import ChatState, LFM2AudioModel, LFM2AudioProcessor

from arcvoice.config import settings

logger = logging.getLogger(__name__)

_model: LFM2AudioModel | None = None
_processor: LFM2AudioProcessor | None = None
_lock = Lock()

SAMPLE_RATE = 24_000


def get_model_and_processor() -> tuple[LFM2AudioModel, LFM2AudioProcessor]:
    """Lazy-load and return the model and processor (thread-safe singleton)."""
    global _model, _processor
    if _model is None:
        with _lock:
            if _model is None:
                logger.info("Loading model %s on %s ...", settings.model_id, settings.device)
                _processor = LFM2AudioProcessor.from_pretrained(settings.model_id).eval()
                _model = LFM2AudioModel.from_pretrained(settings.model_id).eval()
                if settings.attn_implementation:
                    _model.lfm.set_attn_implementation(settings.attn_implementation)
                if settings.device != "cpu":
                    _model = _model.to(settings.device)
                logger.info("Model loaded.")
    return _model, _processor


def synthesize(text: str) -> torch.Tensor:
    """Run TTS: convert text to a waveform tensor (1, T) at 24kHz."""
    model, processor = get_model_and_processor()

    chat = ChatState(processor)
    chat.new_turn("system")
    chat.add_text("Perform TTS.")
    chat.end_turn()

    chat.new_turn("user")
    chat.add_text(text)
    chat.end_turn()

    chat.new_turn("assistant")
    audio_tokens: list[torch.Tensor] = []
    for token in model.generate_sequential(
        **chat,
        max_new_tokens=settings.max_new_tokens,
        audio_temperature=settings.audio_temperature,
        audio_top_k=settings.audio_top_k,
    ):
        if token.numel() > 1:
            audio_tokens.append(token)

    if not audio_tokens:
        raise RuntimeError("Model produced no audio output")

    # Stack audio codes and decode to waveform
    audio_codes = torch.stack(audio_tokens[:-1], dim=1).unsqueeze(0)
    waveform = processor.decode(audio_codes)
    return waveform.cpu()


def transcribe(audio_bytes: bytes, filename: str) -> str:
    """Run ASR: transcribe audio bytes to text."""
    model, processor = get_model_and_processor()

    # torchcodec backend cannot load from BytesIO; use a temp file.
    fmt = _guess_format(filename) or "wav"
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / f"input.{fmt}"
        path.write_bytes(audio_bytes)
        waveform, sr = torchaudio.load(str(path))

    chat = ChatState(processor)
    chat.new_turn("system")
    chat.add_text("Perform ASR.")
    chat.end_turn()

    chat.new_turn("user")
    chat.add_audio(waveform, sr)
    chat.end_turn()

    chat.new_turn("assistant")
    text_tokens: list[torch.Tensor] = []
    for token in model.generate_sequential(
        **chat,
        max_new_tokens=settings.max_new_tokens,
    ):
        if token.numel() == 1:
            text_tokens.append(token)

    if not text_tokens:
        raise RuntimeError("Model produced no text output")

    text = processor.text.decode(torch.cat(text_tokens).tolist(), skip_special_tokens=True)
    return text.strip()


def waveform_to_bytes(waveform: torch.Tensor, fmt: str = "wav") -> bytes:
    """Encode a waveform tensor to audio bytes in the requested format."""
    # Newer torchaudio backends (torchcodec) cannot write to BytesIO directly;
    # write to a temporary file with the correct extension instead.
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / f"out.{fmt}"
        torchaudio.save(str(path), waveform, SAMPLE_RATE, format=fmt)
        return path.read_bytes()


def _guess_format(filename: str) -> str | None:
    """Return torchaudio format hint from filename extension."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext if ext in ("wav", "mp3", "flac", "ogg", "webm") else None
