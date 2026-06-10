# Backend Docker Deployment

## Files

- `backend/Dockerfile`
- `backend/docker-compose.yml`
- `backend/docker-compose.pgvector.yml`

## What this setup assumes

- You already have Redis running on the VPS.
- You already have PostgreSQL running and reachable.
- The backend will run in Docker.
- Your current production shape is:
  - external Neon PostgreSQL
  - existing Redis instance

The main `docker-compose.yml` is now aligned to that setup first.
The optional `docker-compose.pgvector.yml` only exists in case you later want a local Postgres container.

## VPS recommendation for 2 cores / 4 GB RAM

Use these defaults first:

- `UVICORN_WORKERS=1`
- `EMBEDDING_DEVICE=cpu`
- prefer Gemini embeddings if the local `BAAI/bge-m3` model is too heavy

Why:

- the backend is I/O-heavy and model calls are mostly remote,
- a single worker avoids multiplying model memory,
- local `BAAI/bge-m3` can be memory-hungry on a small VPS.

## 1. Prepare `backend/.env`

Start from:

```bash
cp backend/.env.example backend/.env
```

Important values:

- `DATABASE_URL`
- `DATABASE_SYNC_URL`
- `REDIS_URL`
- `GEMINI_API_KEY`
- `GEMINI_BACKUP_API_KEYS`
- `NVIDIA_API_KEY`
- `TAVILY_API_KEY`
- `INTERNAL_API_KEY`
- `ADMIN_USERNAME`
- `ADMIN_PASSWORD`
- `ADMIN_AUTH_SECRET`

Important:

- `DATABASE_URL` and `DATABASE_SYNC_URL` should point to the same actual database.
- If you use Neon, do not leave `DATABASE_SYNC_URL` on a placeholder host. It should match your real Neon host and database name.

If your Redis is already running on the VPS host and published on port `6379`, this is the simplest Linux Docker value:

```env
REDIS_URL=redis://host.docker.internal:6379/0
```

If Redis requires a password:

```env
REDIS_URL=redis://default:yourpassword@host.docker.internal:6379/0
```

If Redis is another Docker container and is not published on the host, either:

- publish Redis on the VPS host and keep using `host.docker.internal`, or
- attach this backend compose stack to the same Docker network and use the Redis container name as the host.

## 2. Deploy with external PostgreSQL

This is your current recommended path because you are already using Neon and an existing Redis instance:

```bash
cd backend
docker compose up -d --build
```

## 3. Optional: deploy with local pgvector PostgreSQL

If you want Postgres on the same VPS, first set these in `backend/.env`:

```env
DATABASE_URL=postgresql+asyncpg://jachai:change_this_postgres_password@jachai-postgres:5432/jachai
DATABASE_SYNC_URL=postgresql://jachai:change_this_postgres_password@jachai-postgres:5432/jachai
```

Then start both services:

```bash
cd backend
docker compose -f docker-compose.yml -f docker-compose.pgvector.yml up -d --build
```

## 4. Reverse proxy note

By default, compose binds the backend only to localhost:

- `127.0.0.1:8000 -> container:8000`

That is safer for production behind Nginx or Caddy.

If you really need public binding, set:

```bash
BACKEND_BIND_ADDRESS=0.0.0.0
```

## 5. Useful commands

View logs:

```bash
cd backend
docker compose logs -f jachai-backend
```

Restart:

```bash
cd backend
docker compose restart jachai-backend
```

Rebuild after code changes:

```bash
cd backend
docker compose up -d --build
```

Stop:

```bash
cd backend
docker compose down
```

Stop including local Postgres:

```bash
cd backend
docker compose -f docker-compose.yml -f docker-compose.pgvector.yml down
```

## 6. Health checks

Backend health:

```bash
curl http://127.0.0.1:8000/health
```

System health:

```bash
curl http://127.0.0.1:8000/api/v1/health/system
```

## 7. Important current behavior

- The app already creates the `vector` and `pgcrypto` extensions on startup.
- The app auto-creates tables when `AUTO_CREATE_TABLES=true`.
- The container waits for Postgres and Redis before starting Uvicorn.
- The model cache is persisted in a Docker volume so local embedding downloads are not repeated every deploy.
