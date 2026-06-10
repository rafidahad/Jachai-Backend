from __future__ import annotations

import argparse
import asyncio
import json

import httpx


async def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the JachAI claim pipeline.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--text",
        default="Breaking news says a city-wide internet shutdown has already been confirmed for tonight.",
    )
    args = parser.parse_args()

    async with httpx.AsyncClient(base_url=args.base_url, timeout=30) as client:
        response = await client.post("/api/v1/claims/text", json={"text": args.text})
        print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
