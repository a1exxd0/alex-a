"""Shared Google OAuth credential management.

All Google skills (calendar, gmail, …) import ``get_credentials()`` from here
so the user completes a single OAuth consent flow that covers every scope the
assistant needs.

If you add a new skill with additional scopes, append to ``_SCOPES`` and
delete the cached token at ``settings.google_user_token_path`` so the user
re-authorises with the expanded grant.
"""

from __future__ import annotations

import logging
from pathlib import Path

from google.auth.transport.requests import Request  # type: ignore[import-untyped]
from google.oauth2.credentials import Credentials  # type: ignore[import-untyped]
from google_auth_oauthlib.flow import InstalledAppFlow  # type: ignore[import-untyped]

from assistant.config import settings

log = logging.getLogger(__name__)

# All scopes needed across every Google skill.
# Extend this list when adding new skills; delete the cached user token to re-auth.
_SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/gmail.modify",
]


def get_credentials() -> Credentials:
    """Return valid Google OAuth credentials, refreshing or re-authorising as needed.

    1. If a cached user token exists and is valid, use it.
    2. If it's expired but has a refresh token, refresh it.
    3. Otherwise, run the interactive ``InstalledAppFlow`` (opens a browser).

    The token is persisted to ``settings.google_user_token_path`` after any
    change so subsequent calls are fast.
    """
    creds: Credentials | None = None
    user_token_path = Path(settings.google_user_token_path)
    credentials_path = Path(settings.google_credentials_path)

    if user_token_path.exists():
        creds = Credentials.from_authorized_user_file(str(user_token_path), _SCOPES)

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        log.info("Refreshing expired Google OAuth token")
        creds.refresh(Request())
    else:
        if not credentials_path.exists():
            raise FileNotFoundError(
                f"Google OAuth client credentials not found at {credentials_path}. "
                "Download the OAuth 2.0 client JSON from the Google Cloud Console "
                "and save it to that path."
            )
        log.info("Starting Google OAuth consent flow (browser will open)...")
        flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), _SCOPES)
        creds = flow.run_local_server(port=0)

    user_token_path.parent.mkdir(parents=True, exist_ok=True)
    user_token_path.write_text(creds.to_json())
    log.info("Google OAuth token saved to %s", user_token_path)
    return creds
