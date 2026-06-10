from __future__ import annotations

from pydantic import Field, StrictBool, StrictInt, StrictStr

from app.schemas.common import StrictBaseModel


class AdminLoginRequestSchema(StrictBaseModel):
    username: StrictStr = Field(min_length=1, max_length=255)
    password: StrictStr = Field(min_length=1, max_length=255)


class AdminLoginResponseSchema(StrictBaseModel):
    ok: StrictBool
    username: StrictStr


class AdminSessionResponseSchema(StrictBaseModel):
    authenticated: StrictBool
    username: StrictStr
    expires_at: StrictInt


class AdminLogoutResponseSchema(StrictBaseModel):
    ok: StrictBool
