FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv

COPY --from=ghcr.io/astral-sh/uv:0.7.13 /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app
COPY alembic.ini ./
COPY alembic ./alembic
COPY scripts ./scripts
COPY docker ./docker

RUN uv sync --frozen --no-dev

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app \
    HOME=/home/jachai \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/data/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/data/sentence-transformers

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system --gid 10001 jachai \
    && useradd --system --uid 10001 --gid 10001 --create-home --home-dir /home/jachai jachai \
    && mkdir -p /app /data \
    && chown -R jachai:jachai /app /data

WORKDIR /app

COPY --from=builder --chown=jachai:jachai /opt/venv /opt/venv
COPY --from=builder --chown=jachai:jachai /app /app

RUN chmod +x /app/docker/entrypoint.sh

EXPOSE 8000

USER jachai

ENTRYPOINT ["/app/docker/entrypoint.sh"]
