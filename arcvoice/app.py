"""FastAPI application - OpenAI-compatible audio API backed by Liquid AI LFM2.5-Audio-1.5B."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from arcvoice.config import settings
from arcvoice.routes import router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(
    title="ArcVoice",
    description="OpenAI-compatible TTS & transcription API powered by Liquid AI LFM2.5-Audio-1.5B",
    version="0.1.0",
)

app.include_router(router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models() -> dict:
    """Minimal /v1/models response for compatibility."""
    return {
        "object": "list",
        "data": [
            {
                "id": "lfm2-audio-1.5b",
                "object": "model",
                "created": 0,
                "owned_by": "liquid-ai",
            }
        ],
    }


def main() -> None:
    import uvicorn

    uvicorn.run(
        "arcvoice.app:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
