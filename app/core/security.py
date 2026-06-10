from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass

from fastapi import Cookie, Header, Response

from app.core.config import settings
from app.utils.errors import AppError


@dataclass(slots=True)
class AdminSession:
    username: str
    issued_at: int
    expires_at: int


def _urlsafe_b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("utf-8").rstrip("=")


def _urlsafe_b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(f"{value}{padding}".encode("utf-8"))


def ensure_admin_auth_configured() -> None:
    if settings.admin_auth_enabled:
        return
    raise AppError(
        status_code=503,
        code="ADMIN_AUTH_NOT_CONFIGURED",
        message="Admin authentication is not configured.",
    )


def _admin_secret_bytes() -> bytes:
    ensure_admin_auth_configured()
    return str(settings.admin_auth_secret).encode("utf-8")


def verify_admin_credentials(username: str, password: str) -> str:
    ensure_admin_auth_configured()

    normalized_username = username.strip()
    expected_username = str(settings.admin_username or "").strip()
    expected_password = str(settings.admin_password or "")

    if not normalized_username or not password:
        raise AppError(status_code=401, code="ADMIN_AUTH_FAILED", message="Invalid admin credentials.")

    username_ok = hmac.compare_digest(normalized_username, expected_username)
    password_ok = hmac.compare_digest(password, expected_password)
    if not (username_ok and password_ok):
        raise AppError(status_code=401, code="ADMIN_AUTH_FAILED", message="Invalid admin credentials.")

    return normalized_username


def create_admin_session_token(username: str, now: int | None = None) -> str:
    ensure_admin_auth_configured()

    issued_at = int(now or time.time())
    payload = {
        "username": username,
        "iat": issued_at,
        "exp": issued_at + settings.admin_session_ttl_seconds,
    }
    payload_segment = _urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signature = hmac.new(
        _admin_secret_bytes(),
        payload_segment.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return f"{payload_segment}.{_urlsafe_b64encode(signature)}"


def verify_admin_session_token(token: str, now: int | None = None) -> AdminSession | None:
    if not token:
        return None
    try:
        ensure_admin_auth_configured()
    except AppError:
        return None

    payload_segment, separator, signature_segment = token.partition(".")
    if not payload_segment or not separator or not signature_segment:
        return None

    expected_signature = hmac.new(
        _admin_secret_bytes(),
        payload_segment.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    try:
        provided_signature = _urlsafe_b64decode(signature_segment)
    except (ValueError, UnicodeDecodeError):
        return None
    if not hmac.compare_digest(provided_signature, expected_signature):
        return None

    try:
        payload = json.loads(_urlsafe_b64decode(payload_segment).decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None

    username = payload.get("username")
    issued_at = payload.get("iat")
    expires_at = payload.get("exp")
    if not isinstance(username, str) or not isinstance(issued_at, int) or not isinstance(expires_at, int):
        return None

    if expires_at <= int(now or time.time()):
        return None

    return AdminSession(username=username, issued_at=issued_at, expires_at=expires_at)


def set_admin_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=settings.admin_session_cookie_name,
        value=token,
        max_age=settings.admin_session_ttl_seconds,
        httponly=True,
        secure=settings.app_env.strip().lower() == "production",
        samesite="lax",
        path="/",
    )


def clear_admin_session_cookie(response: Response) -> None:
    response.delete_cookie(
        key=settings.admin_session_cookie_name,
        httponly=True,
        secure=settings.app_env.strip().lower() == "production",
        samesite="lax",
        path="/",
    )


async def require_admin_session(
    admin_session_token: str | None = Cookie(default=None, alias=settings.admin_session_cookie_name),
) -> AdminSession:
    ensure_admin_auth_configured()

    session = verify_admin_session_token(admin_session_token or "")
    if session is None:
        raise AppError(
            status_code=401,
            code="ADMIN_AUTH_REQUIRED",
            message="Admin authentication is required.",
        )
    return session


async def verify_internal_api_key(
    x_internal_api_key: str | None = Header(default=None, alias="X-Internal-API-Key"),
) -> None:
    if not settings.internal_api_key:
        raise AppError(
            status_code=503,
            code="INTERNAL_KEY_NOT_CONFIGURED",
            message="Internal API key is not configured.",
        )
    if x_internal_api_key != settings.internal_api_key:
        raise AppError(status_code=401, code="UNAUTHORIZED", message="Invalid internal API key.")
