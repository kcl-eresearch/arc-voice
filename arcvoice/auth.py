"""Optional Bearer-token authentication."""

from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from arcvoice.config import settings

_scheme = HTTPBearer(auto_error=False)


async def verify_api_key(
    creds: HTTPAuthorizationCredentials | None = Depends(_scheme),
) -> None:
    if settings.api_key is None:
        return  # auth disabled
    if creds is None or creds.credentials != settings.api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
