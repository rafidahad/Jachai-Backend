from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

from app.api import api_router
from app.core.config import settings
from app.core.database import close_database, init_database
from app.core.logging import configure_logging, get_logger
from app.core.redis import close_redis, init_redis
from app.services.health_service import get_runtime_readiness
from app.utils.errors import AppError

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await init_database()
    await init_redis()
    yield
    await close_redis()
    await close_database()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)

if settings.backend_cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.backend_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(api_router)


@app.get("/health", tags=["health"])
async def root_health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz", tags=["health"])
async def readiness_health(response: Response) -> dict[str, object]:
    ready, checks = await get_runtime_readiness()
    response.status_code = 200 if ready else 503
    return {
        "status": "ready" if ready else "not_ready",
        "checks": checks,
    }


@app.exception_handler(AppError)
async def app_error_handler(_: Request, exc: AppError) -> ORJSONResponse:
    return ORJSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message}},
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError) -> ORJSONResponse:
    return ORJSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed.",
                "details": exc.errors(),
            }
        },
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(_: Request, exc: Exception) -> ORJSONResponse:
    logger.exception("unhandled_exception", exc_info=exc)
    return ORJSONResponse(
        status_code=500,
        content={"error": {"code": "INTERNAL_SERVER_ERROR", "message": "Unexpected server error."}},
    )
