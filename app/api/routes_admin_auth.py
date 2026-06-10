from __future__ import annotations

from fastapi import APIRouter, Depends, Response, status

from app.core.security import (
    clear_admin_session_cookie,
    create_admin_session_token,
    require_admin_session,
    set_admin_session_cookie,
    verify_admin_credentials,
)
from app.schemas.admin_auth_schema import (
    AdminLoginRequestSchema,
    AdminLoginResponseSchema,
    AdminLogoutResponseSchema,
    AdminSessionResponseSchema,
)

router = APIRouter(prefix="/admin/auth", tags=["admin-auth"])


@router.post("/login", response_model=AdminLoginResponseSchema, status_code=status.HTTP_200_OK)
async def admin_login(
    payload: AdminLoginRequestSchema,
    response: Response,
) -> AdminLoginResponseSchema:
    username = verify_admin_credentials(payload.username, payload.password)
    token = create_admin_session_token(username)
    set_admin_session_cookie(response, token)
    return AdminLoginResponseSchema(ok=True, username=username)


@router.post("/logout", response_model=AdminLogoutResponseSchema, status_code=status.HTTP_200_OK)
async def admin_logout(response: Response) -> AdminLogoutResponseSchema:
    clear_admin_session_cookie(response)
    return AdminLogoutResponseSchema(ok=True)


@router.get("/session", response_model=AdminSessionResponseSchema, status_code=status.HTTP_200_OK)
async def admin_session(
    session=Depends(require_admin_session),
) -> AdminSessionResponseSchema:
    return AdminSessionResponseSchema(
        authenticated=True,
        username=session.username,
        expires_at=session.expires_at,
    )
