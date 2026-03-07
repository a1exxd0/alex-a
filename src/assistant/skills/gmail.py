"""Gmail skill — read, search, and send emails via the Gmail API.

Requires:
  - OAuth client credentials JSON at ``settings.google_credentials_path``
  - On first run, completes an interactive browser OAuth flow and caches the
    user token at ``settings.google_user_token_path``
  - The ``gmail.modify`` scope must be included in ``auth.google._SCOPES``
    (it is by default).

Tools exposed:
  - ``list_emails``   — list recent inbox messages
  - ``search_emails`` — search using Gmail query syntax
  - ``read_email``    — read a specific email by ID
  - ``send_email``    — compose and send a new email (requires confirmation)
  - ``reply_email``   — reply to an existing thread (requires confirmation)
  - ``trash_email``   — move an email to trash (requires confirmation)
"""

from __future__ import annotations

import asyncio
import base64
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from assistant.core.agent import Agent
    from assistant.platforms.base import Platform

from googleapiclient.discovery import build  # type: ignore[import-untyped]
from pydantic import BaseModel, Field

from assistant.auth.google import get_credentials
from assistant.config import settings
from assistant.core.tool_registry import ToolRegistry

log = logging.getLogger(__name__)

# Suppress noisy discovery cache warning
logging.getLogger("googleapiclient.discovery_cache").setLevel(logging.ERROR)


# ── Service builder ───────────────────────────────────────────────────────────


def _build_service():
    """Return an authorised Gmail API v1 service client."""
    return build("gmail", "v1", credentials=get_credentials())


# ── Helpers ───────────────────────────────────────────────────────────────────


def _get_header(headers: list[dict], name: str) -> str:
    for h in headers:
        if h.get("name", "").lower() == name.lower():
            return h.get("value", "")
    return ""


def _decode_body(part: dict) -> str:
    """Recursively extract plain-text body from a message part."""
    mime_type = part.get("mimeType", "")
    if mime_type == "text/plain":
        data = part.get("body", {}).get("data", "")
        if data:
            return base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")
    for sub in part.get("parts", []):
        result = _decode_body(sub)
        if result:
            return result
    return ""


def _format_message_summary(msg: dict) -> str:
    headers = msg.get("payload", {}).get("headers", [])
    subject = _get_header(headers, "subject") or "(no subject)"
    sender = _get_header(headers, "from") or "unknown"
    date = _get_header(headers, "date") or ""
    snippet = msg.get("snippet", "")
    msg_id = msg.get("id", "")
    return f"[{msg_id}] {date}\n  From: {sender}\n  Subject: {subject}\n  {snippet}"


# ── Pydantic models ───────────────────────────────────────────────────────────


class ListEmailsParams(BaseModel):
    max_results: int = Field(
        default=10,
        description="Maximum number of emails to return (default 10, max 50).",
        ge=1,
        le=50,
    )
    label: str = Field(
        default="INBOX",
        description="Gmail label to list. Common values: INBOX, SENT, UNREAD, STARRED.",
    )


class SearchEmailsParams(BaseModel):
    query: str = Field(
        description=(
            "Gmail search query, e.g. 'from:alice@example.com subject:invoice is:unread'. "
            "Supports full Gmail search syntax."
        )
    )
    max_results: int = Field(
        default=10,
        description="Maximum number of results to return (default 10, max 50).",
        ge=1,
        le=50,
    )


class ReadEmailParams(BaseModel):
    message_id: str = Field(
        description="The Gmail message ID (from list_emails or search_emails output)."
    )
    max_body_chars: int = Field(
        default=3000,
        description="Truncate the body to this many characters. Default 3000.",
        ge=100,
        le=20000,
    )


class SendEmailParams(BaseModel):
    to: str = Field(description="Recipient email address.")
    subject: str = Field(description="Email subject line.")
    body: str = Field(description="Plain-text email body.")
    cc: Optional[str] = Field(default=None, description="Optional CC address(es), comma-separated.")


class ReplyEmailParams(BaseModel):
    message_id: str = Field(
        description="ID of the message to reply to (from list_emails or search_emails)."
    )
    body: str = Field(description="Plain-text reply body.")


class TrashEmailParams(BaseModel):
    message_id: str = Field(
        description="ID of the message to move to trash."
    )


class GetEmailStyleParams(BaseModel):
    pass


# ── Tool handlers ─────────────────────────────────────────────────────────────


async def _list_emails(params: ListEmailsParams) -> str:
    service = _build_service()
    result = (
        service.users()
        .messages()
        .list(userId="me", labelIds=[params.label], maxResults=params.max_results)
        .execute()
    )
    messages = result.get("messages", [])
    if not messages:
        return f"No messages found in {params.label}."

    lines = [f"**{params.label}** — {len(messages)} message(s):\n"]
    for m in messages:
        full = service.users().messages().get(userId="me", id=m["id"], format="metadata",
                                              metadataHeaders=["From", "Subject", "Date"]).execute()
        lines.append(_format_message_summary(full))

    return "\n\n".join(lines)


async def _search_emails(params: SearchEmailsParams) -> str:
    service = _build_service()
    result = (
        service.users()
        .messages()
        .list(userId="me", q=params.query, maxResults=params.max_results)
        .execute()
    )
    messages = result.get("messages", [])
    if not messages:
        return f"No messages found for query: {params.query!r}"

    lines = [f"**Search:** {params.query!r} — {len(messages)} result(s):\n"]
    for m in messages:
        full = service.users().messages().get(userId="me", id=m["id"], format="metadata",
                                              metadataHeaders=["From", "Subject", "Date"]).execute()
        lines.append(_format_message_summary(full))

    return "\n\n".join(lines)


async def _read_email(params: ReadEmailParams) -> str:
    service = _build_service()
    msg = service.users().messages().get(userId="me", id=params.message_id, format="full").execute()

    headers = msg.get("payload", {}).get("headers", [])
    subject = _get_header(headers, "subject") or "(no subject)"
    sender = _get_header(headers, "from") or "unknown"
    to = _get_header(headers, "to") or "unknown"
    date = _get_header(headers, "date") or ""

    body = _decode_body(msg.get("payload", {}))
    if not body:
        body = msg.get("snippet", "(body not available)")
    if len(body) > params.max_body_chars:
        body = body[: params.max_body_chars] + f"\n... [truncated at {params.max_body_chars} chars]"

    return (
        f"**From:** {sender}\n"
        f"**To:** {to}\n"
        f"**Date:** {date}\n"
        f"**Subject:** {subject}\n"
        f"**ID:** {params.message_id}\n\n"
        f"{body}"
    )


async def _send_email(params: SendEmailParams) -> str:
    service = _build_service()

    msg = MIMEMultipart()
    msg["to"] = params.to
    msg["subject"] = params.subject
    if params.cc:
        msg["cc"] = params.cc
    msg.attach(MIMEText(params.body, "plain"))

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    service.users().messages().send(userId="me", body={"raw": raw}).execute()
    return f"Email sent to {params.to} — Subject: {params.subject!r}"


async def _reply_email(params: ReplyEmailParams) -> str:
    service = _build_service()

    # Fetch the original to get thread ID and headers
    original = service.users().messages().get(
        userId="me", id=params.message_id, format="metadata",
        metadataHeaders=["From", "Subject", "Message-ID"]
    ).execute()

    headers = original.get("payload", {}).get("headers", [])
    original_from = _get_header(headers, "from")
    original_subject = _get_header(headers, "subject")
    original_message_id = _get_header(headers, "message-id")
    thread_id = original.get("threadId", "")

    reply_subject = original_subject if original_subject.lower().startswith("re:") else f"Re: {original_subject}"

    msg = MIMEMultipart()
    msg["to"] = original_from
    msg["subject"] = reply_subject
    if original_message_id:
        msg["In-Reply-To"] = original_message_id
        msg["References"] = original_message_id
    msg.attach(MIMEText(params.body, "plain"))

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    service.users().messages().send(
        userId="me", body={"raw": raw, "threadId": thread_id}
    ).execute()

    return f"Reply sent to {original_from} — Subject: {reply_subject!r}"


async def _get_email_style(params: GetEmailStyleParams) -> str:
    path = settings.email_style_path
    if not path.exists():
        return f"Email style guide not found at {path}."
    return path.read_text(encoding="utf-8")


async def _trash_email(params: TrashEmailParams) -> str:
    service = _build_service()
    service.users().messages().trash(userId="me", id=params.message_id).execute()
    return f"Message {params.message_id} moved to trash."


# ── Proactive email poller ───────────────────────────────────────────────────


def _fetch_and_mark_unread(query: str) -> list[dict]:
    """Fetch unread messages matching *query*, mark them read, return their metadata."""
    service = _build_service()
    result = (
        service.users()
        .messages()
        .list(userId="me", q=query, maxResults=20)
        .execute()
    )
    messages = result.get("messages", [])
    if not messages:
        return []

    full_messages = []
    for m in messages:
        full = service.users().messages().get(
            userId="me", id=m["id"], format="metadata",
            metadataHeaders=["From", "Subject", "Date"],
        ).execute()
        full_messages.append(full)
        # Mark as read by removing the UNREAD label
        service.users().messages().modify(
            userId="me",
            id=m["id"],
            body={"removeLabelIds": ["UNREAD"]},
        ).execute()

    return full_messages


def _format_notification(messages: list[dict], intro: str) -> str:
    lines = [intro]
    for msg in messages:
        headers = msg.get("payload", {}).get("headers", [])
        subject = _get_header(headers, "subject") or "(no subject)"
        sender = _get_header(headers, "from") or "unknown"
        date = _get_header(headers, "date") or ""
        snippet = msg.get("snippet", "")
        lines.append(f"**[{msg['id']}]** {subject}\nFrom: {sender} — {date}\n{snippet}")
    return "\n\n".join(lines)


class GmailPoller:
    """Background task that proactively checks for unread emails and notifies via Discord.

    On startup: checks for unread emails in the last 24 hours.
    Every *poll_interval* seconds: checks for unread emails in the last *poll_interval* seconds.
    All retrieved emails are marked as read so they are not reported again.
    """

    def __init__(self, poll_interval: int) -> None:
        self._interval = poll_interval

    async def run(self, platform: "Platform", channel_id: str, agent: "Agent") -> None:
        # Startup check: last 24 hours
        await self._check(platform, channel_id, agent, window="1d", intro="**Unread emails from the last 24 hours:**")

        while True:
            await asyncio.sleep(self._interval)
            minutes = self._interval // 60
            await self._check(
                platform,
                channel_id,
                agent,
                window=f"{minutes}m",
                intro="**New unread email(s):**",
            )

    async def _check(self, platform: "Platform", channel_id: str, agent: "Agent", window: str, intro: str) -> None:
        try:
            query = f"is:unread in:inbox newer_than:{window}"
            messages = _fetch_and_mark_unread(query)
            if messages:
                notification = _format_notification(messages, intro)
                await platform.send(notification, channel_id)
                agent.record_proactive(notification)
                log.info("Notified %d unread email(s) (window=%s)", len(messages), window)
        except Exception:
            log.exception("Gmail poller encountered an error (window=%s)", window)


# ── Skill registration ────────────────────────────────────────────────────────


def register(registry: ToolRegistry) -> None:
    """Register Gmail tools with the agent's tool registry."""

    @registry.tool(
        "get_email_style",
        "Returns the user's email writing style guide. "
        "Always call this before composing or replying to an email.",
    )
    async def get_email_style(params: GetEmailStyleParams) -> str:
        return await _get_email_style(params)

    @registry.tool(
        "list_emails",
        "List recent emails from Gmail. Returns sender, subject, date, and a short snippet. "
        "Use the label parameter to target INBOX, SENT, UNREAD, or STARRED.",
    )
    async def list_emails(params: ListEmailsParams) -> str:
        return await _list_emails(params)

    @registry.tool(
        "search_emails",
        "Search Gmail using Gmail's query syntax (e.g. 'from:alice subject:invoice is:unread'). "
        "Returns matching messages with sender, subject, and snippet.",
    )
    async def search_emails(params: SearchEmailsParams) -> str:
        return await _search_emails(params)

    @registry.tool(
        "read_email",
        "Read the full content of a specific email by its message ID. "
        "Use list_emails or search_emails first to get a message ID.",
    )
    async def read_email(params: ReadEmailParams) -> str:
        return await _read_email(params)

    @registry.tool(
        "send_email",
        "Compose and send a new email. Call get_email_style first to match the user's writing voice.",
        requires_confirmation=True,
    )
    async def send_email(params: SendEmailParams) -> str:
        return await _send_email(params)

    @registry.tool(
        "reply_email",
        "Reply to an existing email thread. The reply is sent to the original sender "
        "and stays in the same thread. Call get_email_style first to match the user's writing voice.",
        requires_confirmation=True,
    )
    async def reply_email(params: ReplyEmailParams) -> str:
        return await _reply_email(params)

    @registry.tool(
        "trash_email",
        "Move an email to trash by message ID.",
        requires_confirmation=True,
    )
    async def trash_email(params: TrashEmailParams) -> str:
        return await _trash_email(params)

    log.info("Gmail skill registered")
