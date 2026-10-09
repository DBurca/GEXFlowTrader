"""
Load, refresh, and persist Schwab OAuth tokens.

Priority:
  1. SCHWAB_ACCESS_TOKEN environment variable
  2. tokens.json (written by keygetter.py), with auto-refresh when expired
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any

import requests

import config

SCHWAB_TOKEN_URL = "https://api.schwabapi.com/v1/oauth/token"
REFRESH_BUFFER_SECONDS = 60


def token_file_path() -> Path:
    return Path(config.TOKEN_FILE_PATH)


def load_token_file() -> dict[str, Any] | None:
    path = token_file_path()
    if not path.exists():
        return None

    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid token file format in {path}.")
    return data


def save_token_file(token_payload: dict[str, Any]) -> Path:
    expires_in = token_payload.get("expires_in")
    expires_at = None
    if expires_in is not None:
        expires_at = time.time() + float(expires_in)

    existing = load_token_file() or {}
    refresh_token = token_payload.get("refresh_token") or existing.get("refresh_token")

    record = {
        "access_token": token_payload["access_token"],
        "refresh_token": refresh_token,
        "expires_at": expires_at,
        "token_type": token_payload.get("token_type", "Bearer"),
        "updated_at": time.time(),
    }

    path = token_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)

    return path


def _token_is_valid(record: dict[str, Any]) -> bool:
    access_token = record.get("access_token")
    if not access_token:
        return False

    expires_at = record.get("expires_at")
    if expires_at is None:
        return True

    return time.time() < float(expires_at) - REFRESH_BUFFER_SECONDS


def _refresh_access_token(refresh_token: str) -> dict[str, Any]:
    if not config.SCHWAB_API_KEY or not config.SCHWAB_API_SECRET:
        raise ValueError(
            "SCHWAB_API_KEY and SCHWAB_API_SECRET are required to refresh tokens."
        )

    credentials = base64.b64encode(
        f"{config.SCHWAB_API_KEY}:{config.SCHWAB_API_SECRET}".encode()
    ).decode()

    response = requests.post(
        SCHWAB_TOKEN_URL,
        headers={
            "Authorization": f"Basic {credentials}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        },
        timeout=30,
    )
    if not response.ok:
        raise RuntimeError(
            f"Token refresh failed ({response.status_code}): {response.text}"
        )

    payload = response.json()
    save_token_file(payload)
    return payload


def get_access_token() -> str:
    """
    Return a valid Schwab access token.

    Uses SCHWAB_ACCESS_TOKEN if set, otherwise loads tokens.json and
    refreshes automatically when the access token is expired.
    """
    if config.SCHWAB_ACCESS_TOKEN:
        return config.SCHWAB_ACCESS_TOKEN

    record = load_token_file()
    if record is None:
        raise ValueError(
            "No Schwab access token found. Run `python keygetter.py` first, "
            "or export SCHWAB_ACCESS_TOKEN."
        )

    if _token_is_valid(record):
        return str(record["access_token"])

    refresh_token = record.get("refresh_token")
    if not refresh_token:
        raise ValueError(
            "Stored access token is expired and no refresh token is available. "
            "Run `python keygetter.py` again."
        )

    refreshed = _refresh_access_token(str(refresh_token))
    return str(refreshed["access_token"])
