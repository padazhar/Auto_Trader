from __future__ import annotations

from typing import Any

from fastapi import Request, WebSocket

DASHBOARD_TOKEN_COOKIE = "dashboard_token"


def _clean_token(token: Any) -> str:
    if token is None:
        return ""
    cleaned = str(token).strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"\"", "'"}:
        cleaned = cleaned[1:-1].strip()
    return cleaned


def resolve_monitoring_token(paths) -> None:
    """Monitoring authentication is disabled for all modes."""
    return None


def is_default_monitoring_token(paths) -> bool:
    """Monitoring authentication is disabled for all modes."""
    return False


def authenticate_monitoring_token(candidate: Any, expected: str | None) -> bool:
    """Monitoring authentication is disabled for all modes."""
    return True


def request_token(request: Request) -> str | None:
    return _clean_token(request.headers.get("x-dashboard-token")) or None


def websocket_token(websocket: WebSocket) -> str | None:
    return _clean_token(websocket.headers.get("x-dashboard-token")) or None


def token_required(request: Request, paths) -> None:
    """Monitoring authentication is disabled for all modes."""
    return None


async def websocket_token_required(websocket: WebSocket, paths) -> None:
    """Monitoring authentication is disabled for all modes."""
    return None
