from __future__ import annotations

from typing import Literal

from pydantic import StrictBool, StrictStr

from app.schemas.common import StrictBaseModel


class ComponentHealthSchema(StrictBaseModel):
    name: StrictStr
    status: Literal["healthy", "degraded", "unavailable"]
    ok: StrictBool
    message: StrictStr


class HealthResponseSchema(StrictBaseModel):
    status: Literal["healthy", "degraded", "unavailable"]
    components: list[ComponentHealthSchema]
