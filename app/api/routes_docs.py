from __future__ import annotations

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import require_admin_session
from app.db.session import get_db
from app.schemas.docs_schema import (
    DocsAdminDocumentSchema,
    DocsAdminPublishRequestSchema,
    DocsAdminUpdateRequestSchema,
    DocsLegacyConfigSchema,
    DocsLegacyConfigUpdateRequestSchema,
    DocsLegacyLiveDataSchema,
    DocsLegacyPublicSchema,
    DocsLegacySectionListResponseSchema,
    DocsLegacySectionReorderRequestSchema,
    DocsLegacySectionSchema,
    DocsLegacySectionUpdateRequestSchema,
    DocsLegacyTeamListResponseSchema,
    DocsLegacyTeamMemberCreateRequestSchema,
    DocsLegacyTeamMemberSchema,
    DocsLegacyTeamMemberUpdateRequestSchema,
    DocsPublicResponseSchema,
)
from app.services.docs_service import (
    create_legacy_team_member,
    delete_legacy_team_member,
    get_admin_docs,
    get_legacy_docs_config,
    get_legacy_docs_sections,
    get_legacy_docs_team,
    get_legacy_live_data,
    get_legacy_public_docs,
    get_public_docs,
    publish_docs,
    reorder_legacy_docs_sections,
    save_docs_draft,
    update_legacy_docs_config,
    update_legacy_docs_section,
    update_legacy_team_member,
)

router = APIRouter(prefix="/docs", tags=["docs"])


@router.get("/public", response_model=DocsPublicResponseSchema, status_code=status.HTTP_200_OK)
async def public_docs(
    response: Response,
    session: AsyncSession = Depends(get_db),
) -> DocsPublicResponseSchema:
    response.headers["Cache-Control"] = "public, max-age=15, stale-while-revalidate=60"
    return await get_public_docs(session)


@router.get("/legacy-public", response_model=DocsLegacyPublicSchema, status_code=status.HTTP_200_OK)
async def public_docs_legacy(session: AsyncSession = Depends(get_db)) -> DocsLegacyPublicSchema:
    return await get_legacy_public_docs(session)


@router.get(
    "/admin",
    response_model=DocsAdminDocumentSchema,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_admin_session)],
)
async def admin_docs(session: AsyncSession = Depends(get_db)) -> DocsAdminDocumentSchema:
    return await get_admin_docs(session)


@router.put(
    "/admin",
    response_model=DocsAdminDocumentSchema,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_admin_session)],
)
async def update_admin_docs(
    payload: DocsAdminUpdateRequestSchema,
    session: AsyncSession = Depends(get_db),
) -> DocsAdminDocumentSchema:
    return await save_docs_draft(session, payload)


@router.post(
    "/admin/publish",
    response_model=DocsAdminDocumentSchema,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_admin_session)],
)
async def publish_admin_docs(
    payload: DocsAdminPublishRequestSchema,
    session: AsyncSession = Depends(get_db),
) -> DocsAdminDocumentSchema:
    return await publish_docs(session, payload.actor)


@router.get(
    "/config",
    response_model=DocsLegacyConfigSchema,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_admin_session)],
)
async def docs_config(session: AsyncSession = Depends(get_db)) -> DocsLegacyConfigSchema:
    return await get_legacy_docs_config(session)


@router.put(
    "/config",
    response_model=DocsLegacyConfigSchema,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_admin_session)],
)
async def update_docs_config(
    payload: DocsLegacyConfigUpdateRequestSchema,
    session: AsyncSession = Depends(get_db),
) -> DocsLegacyConfigSchema:
    return await update_legacy_docs_config(
        session,
        is_enabled=payload.is_enabled,
        start_at=payload.start_at,
        end_at=payload.end_at,
        clear_schedule=payload.clear_schedule,
    )


@router.get(
    "/sections",
    response_model=DocsLegacySectionListResponseSchema,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_admin_session)],
)
async def docs_sections(session: AsyncSession = Depends(get_db)) -> DocsLegacySectionListResponseSchema:
    return await get_legacy_docs_sections(session)


@router.put(
    "/sections/{section_id}",
    response_model=DocsLegacySectionSchema,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_admin_session)],
)
async def update_docs_section(
    section_id: str,
    payload: DocsLegacySectionUpdateRequestSchema,
    session: AsyncSession = Depends(get_db),
) -> DocsLegacySectionSchema:
    return await update_legacy_docs_section(
        session,
        section_id,
        title=payload.title,
        content_markdown=payload.content,
        icon=payload.icon,
        sort_order=payload.sort_order,
        is_visible=payload.is_visible,
    )


@router.put(
    "/sections-reorder",
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_admin_session)],
)
async def reorder_docs_sections(
    payload: DocsLegacySectionReorderRequestSchema,
    session: AsyncSession = Depends(get_db),
) -> dict[str, bool]:
    await reorder_legacy_docs_sections(session, [(item.id, item.sort_order) for item in payload.order])
    return {"ok": True}


@router.get(
    "/team",
    response_model=DocsLegacyTeamListResponseSchema,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_admin_session)],
)
async def docs_team(session: AsyncSession = Depends(get_db)) -> DocsLegacyTeamListResponseSchema:
    return await get_legacy_docs_team(session)


@router.post(
    "/team",
    response_model=DocsLegacyTeamMemberSchema,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_admin_session)],
)
async def create_docs_team_member_endpoint(
    payload: DocsLegacyTeamMemberCreateRequestSchema,
    session: AsyncSession = Depends(get_db),
) -> DocsLegacyTeamMemberSchema:
    return await create_legacy_team_member(
        session,
        full_name=payload.full_name,
        role=payload.role,
        email=payload.email,
        profile_picture_url=payload.profile_picture_url,
        sort_order=payload.sort_order,
    )


@router.put(
    "/team/{member_id}",
    response_model=DocsLegacyTeamMemberSchema,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_admin_session)],
)
async def update_docs_team_member_endpoint(
    member_id: str,
    payload: DocsLegacyTeamMemberUpdateRequestSchema,
    session: AsyncSession = Depends(get_db),
) -> DocsLegacyTeamMemberSchema:
    return await update_legacy_team_member(
        session,
        member_id,
        full_name=payload.full_name,
        role=payload.role,
        email=payload.email,
        profile_picture_url=payload.profile_picture_url,
        sort_order=payload.sort_order,
    )


@router.delete(
    "/team/{member_id}",
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_admin_session)],
)
async def delete_docs_team_member_endpoint(
    member_id: str,
    session: AsyncSession = Depends(get_db),
) -> dict[str, bool]:
    await delete_legacy_team_member(session, member_id)
    return {"ok": True}


@router.get(
    "/live-data",
    response_model=DocsLegacyLiveDataSchema,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_admin_session)],
)
async def docs_live_data(session: AsyncSession = Depends(get_db)) -> DocsLegacyLiveDataSchema:
    return await get_legacy_live_data(session)
