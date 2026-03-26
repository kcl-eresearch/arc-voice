"""Shared fixtures for arcvoice tests."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from arcvoice.app import app


@pytest.fixture()
async def client():
    """Async HTTP client wired to the FastAPI app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
