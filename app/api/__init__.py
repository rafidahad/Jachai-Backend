from fastapi import APIRouter

from app.api.routes_admin_auth import router as admin_auth_router
from app.api.routes_claims import router as claims_router
from app.api.routes_clusters import router as clusters_router
from app.api.routes_dashboard import router as dashboard_router
from app.api.routes_docs import router as docs_router
from app.api.routes_health import router as health_router
from app.api.routes_sources import router as sources_router
from app.api.routes_verification_jobs import router as jobs_router
from app.api.routes_webhooks import router as webhooks_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health_router)
api_router.include_router(admin_auth_router)
api_router.include_router(claims_router)
api_router.include_router(jobs_router)
api_router.include_router(sources_router)
api_router.include_router(dashboard_router)
api_router.include_router(docs_router)
api_router.include_router(clusters_router)
api_router.include_router(webhooks_router)
