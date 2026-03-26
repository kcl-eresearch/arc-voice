"""End-to-end round-trip test: TTS → STT and verify the text survives."""

from __future__ import annotations

import pytest

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.e2e,
]


@pytest.fixture(params=[
    "The rain in Spain stays mainly in the plain.",
    "The quick brown fox jumps over the lazy dog.",
])
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

    # Step 3 — Compare using word error rate (WER) to allow minor transcription differences
    def normalise(s: str) -> list[str]:
        return "".join(ch for ch in s.lower() if ch.isalnum() or ch == " ").split()

    expected_words = normalise(sample_text)
    actual_words = normalise(transcribed)

    # Compute word error rate (Levenshtein on word lists)
    def wer(ref: list[str], hyp: list[str]) -> float:
        n = len(ref)
        m = len(hyp)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if ref[i - 1] == hyp[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
        return dp[n][m] / max(n, 1)

    error_rate = wer(expected_words, actual_words)
    assert error_rate <= 0.3, (
        f"Round-trip WER too high ({error_rate:.0%})!\n"
        f"  sent:     {sample_text!r}\n  got back: {transcribed!r}"
    )
