"""Google Calendar skill — read and create events on the user's calendar.

Requires:
  - OAuth client credentials JSON at ``settings.google_credentials_path``
  - On first run, completes an interactive browser OAuth flow and caches the
    user token at ``settings.google_user_token_path``

Tools exposed:
  - ``list_calendar_events`` — list upcoming events
  - ``create_calendar_event`` — create a new event (requires confirmation)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from googleapiclient.discovery import build  # type: ignore[import-untyped]
from pydantic import BaseModel, Field

from assistant.auth.google import get_credentials
from assistant.core.tool_registry import ToolRegistry

log = logging.getLogger(__name__)

# Suppress noisy "file_cache is only supported with oauth2client<4.0.0" warning
logging.getLogger("googleapiclient.discovery_cache").setLevel(logging.ERROR)

def _build_service():
    """Return an authorised Google Calendar API v3 service client."""
    return build("calendar", "v3", credentials=get_credentials())


# ── Pydantic models for tool params ──────────────────────────────────────────


class ListEventsParams(BaseModel):
    days: int = Field(
        default=7,
        description="Number of days ahead to look for events (default 7).",
        ge=1,
        le=90,
    )
    max_results: int = Field(
        default=20,
        description="Maximum number of events to return (default 20).",
        ge=1,
        le=50,
    )
    include_details: bool = Field(
        default=False,
        description=(
            "If true, include extra details like description/notes, "
            "attendees, and conferencing links. Default false for a compact listing."
        ),
    )


class UpdateEventParams(BaseModel):
    event_id: str = Field(
        description="The event ID to update (from list_calendar_events output).",
    )
    summary: Optional[str] = Field(
        default=None,
        description="New title for the event. Omit to keep unchanged.",
    )
    start: Optional[str] = Field(
        default=None,
        description=(
            "New start date-time in ISO 8601 format. "
            "For all-day events use date only: '2026-03-05'. Omit to keep unchanged."
        ),
    )
    end: Optional[str] = Field(
        default=None,
        description=(
            "New end date-time in ISO 8601 format. "
            "For all-day events use date only: '2026-03-06'. Omit to keep unchanged."
        ),
    )
    description: Optional[str] = Field(
        default=None,
        description="New description/notes. Omit to keep unchanged.",
    )
    location: Optional[str] = Field(
        default=None,
        description="New location. Omit to keep unchanged.",
    )


class CreateEventParams(BaseModel):
    summary: str = Field(description="Title of the event.")
    start: str = Field(
        description=(
            "Start date-time in ISO 8601 format, e.g. '2026-03-05T10:00:00+00:00'. "
            "For all-day events use date only: '2026-03-05'."
        ),
    )
    end: str = Field(
        description=(
            "End date-time in ISO 8601 format, e.g. '2026-03-05T11:00:00+00:00'. "
            "For all-day events use date only: '2026-03-06'."
        ),
    )
    description: Optional[str] = Field(
        default=None,
        description="Optional longer description / notes for the event.",
    )
    location: Optional[str] = Field(
        default=None,
        description="Optional location for the event.",
    )


# ── Tool handlers ─────────────────────────────────────────────────────────────


async def _list_events(params: ListEventsParams) -> str:
    service = _build_service()

    now = datetime.now(timezone.utc)
    time_max = now + timedelta(days=params.days)

    result = (
        service.events()
        .list(
            calendarId="primary",
            timeMin=now.isoformat(),
            timeMax=time_max.isoformat(),
            maxResults=params.max_results,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )

    events = result.get("items", [])
    if not events:
        return f"No events found in the next {params.days} day(s)."

    lines: list[str] = [f"📅 **Upcoming events** (next {params.days} day(s)):\n"]
    for ev in events:
        event_id = ev.get("id", "")
        start = ev["start"].get("dateTime", ev["start"].get("date", "unknown"))
        end = ev["end"].get("dateTime", ev["end"].get("date", ""))
        summary = ev.get("summary", "(no title)")
        location = ev.get("location", "")

        line = f"• **{summary}** — {start}"
        if end:
            line += f" → {end}"
        if location:
            line += f"  📍 {location}"
        line += f"  [id: {event_id}]"

        if params.include_details:
            description = ev.get("description", "")
            if description:
                # Trim long notes to keep output manageable
                short = description[:200] + ("…" if len(description) > 200 else "")
                line += f"\n  📝 {short}"

            attendees = ev.get("attendees", [])
            if attendees:
                names = [a.get("displayName") or a.get("email", "?") for a in attendees]
                line += f"\n  👥 {', '.join(names)}"

            hangout = ev.get("hangoutLink", "")
            if hangout:
                line += f"\n  🔗 {hangout}"

        lines.append(line)

    return "\n".join(lines)


async def _create_event(params: CreateEventParams) -> str:
    service = _build_service()

    # Determine whether this is an all-day event or a timed event
    is_all_day = len(params.start) <= 10  # "2026-03-05" is 10 chars

    event_body: dict = {
        "summary": params.summary,
    }

    if is_all_day:
        event_body["start"] = {"date": params.start}
        event_body["end"] = {"date": params.end}
    else:
        event_body["start"] = {"dateTime": params.start}
        event_body["end"] = {"dateTime": params.end}

    if params.description:
        event_body["description"] = params.description
    if params.location:
        event_body["location"] = params.location

    created = service.events().insert(calendarId="primary", body=event_body).execute()

    link = created.get("htmlLink", "")
    return (
        f"✅ Event **{created.get('summary', params.summary)}** created!\n"
        f"Link: {link}"
    )


async def _update_event(params: UpdateEventParams) -> str:
    service = _build_service()

    # Fetch the existing event
    try:
        existing = service.events().get(calendarId="primary", eventId=params.event_id).execute()
    except Exception:
        return f"Error: could not find event with ID '{params.event_id}'."

    # Apply only the fields that were provided
    if params.summary is not None:
        existing["summary"] = params.summary
    if params.description is not None:
        existing["description"] = params.description
    if params.location is not None:
        existing["location"] = params.location
    if params.start is not None:
        is_all_day = len(params.start) <= 10
        existing["start"] = {"date": params.start} if is_all_day else {"dateTime": params.start}
    if params.end is not None:
        is_all_day = len(params.end) <= 10
        existing["end"] = {"date": params.end} if is_all_day else {"dateTime": params.end}

    updated = (
        service.events()
        .update(calendarId="primary", eventId=params.event_id, body=existing)
        .execute()
    )

    return f"✅ Event **{updated.get('summary', '')}** updated."


# ── Skill registration ───────────────────────────────────────────────────────


def register(registry: ToolRegistry) -> None:
    """Register Google Calendar tools with the agent's tool registry."""

    @registry.tool(
        "list_calendar_events",
        "List upcoming events from the user's Google Calendar. "
        "Returns event titles, times, and locations.",
    )
    async def list_calendar_events(params: ListEventsParams) -> str:
        return await _list_events(params)

    @registry.tool(
        "create_calendar_event",
        "Create a new event on the user's Google Calendar. "
        "Requires a title, start time, and end time.",
        requires_confirmation=True,
    )
    async def create_calendar_event(params: CreateEventParams) -> str:
        return await _create_event(params)

    @registry.tool(
        "update_calendar_event",
        "Update an existing event on the user's Google Calendar. "
        "Use list_calendar_events first to get the event ID. "
        "Only the fields you provide will be changed.",
        requires_confirmation=True,
    )
    async def update_calendar_event(params: UpdateEventParams) -> str:
        return await _update_event(params)

    log.info("Calendar skill registered")
