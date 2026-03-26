"""End-to-end round-trip test: TTS → STT and verify the text survives."""

from __future__ import annotations

import pytest

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.e2e,
]


@pytest.fixture(params=["Hello world.", "The quick brown fox jumps over the lazy dog."])
def sample_text(request):
    return request.param


async def test_tts_then_stt_round_trip(client, sample_text):
    """Synthesise text to audio, transcribe it back, and check they match."""
    # Step 1 — TTS
    tts_resp = await client.post(
        "/v1/audio/speech",
        json={
            "input": sample_text,
            "response_format": "wav",
        },
    )
    assert tts_resp.status_code == 200, f"TTS failed: {tts_resp.text}"
    assert len(tts_resp.content) > 0

    audio_bytes = tts_resp.content

    # Step 2 — STT
    stt_resp = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("speech.wav", audio_bytes, "audio/wav")},
        data={"response_format": "json"},
    )
    assert stt_resp.status_code == 200, f"STT failed: {stt_resp.text}"

    transcribed = stt_resp.json()["text"]

    # Step 3 — Compare (case-insensitive, strip punctuation for robustness)
    def normalise(s: str) -> str:
        return "".join(ch for ch in s.lower() if ch.isalnum() or ch == " ").strip()

    expected = normalise(sample_text)
    actual = normalise(transcribed)

    assert actual == expected, (
        f"Round-trip mismatch!\n  sent:       {sample_text!r}\n  got back:   {transcribed!r}\n"
        f"  normalised: {actual!r} != {expected!r}"
    )
