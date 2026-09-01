"""Validate the configured Workflows API endpoint and credentials.

Run with: ``uv run python src/verify_api_connection.py``
"""

import asyncio
import os
import sys

import httpx
from dotenv import load_dotenv

load_dotenv(override=True)


async def main() -> int:
    api_key = os.environ.get("MISTRAL_API_KEY")
    server_url = os.environ.get("SERVER_URL", "https://api.mistral.ai").rstrip("/")

    if not api_key:
        print("FAILED: MISTRAL_API_KEY is not set.")
        return 1

    endpoint = f"{server_url}/v1/workflows/workers/whoami"
    print(f"Checking {server_url}…")
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            response = await client.get(endpoint, headers={"Authorization": f"Bearer {api_key}"})
    except httpx.RequestError as exc:
        print(f"FAILED: endpoint could not be reached ({type(exc).__name__}).")
        return 1

    if response.status_code != 200:
        print(f"FAILED: endpoint rejected the request (HTTP {response.status_code}).")
        return 1

    namespace = response.json().get("namespace", "unknown")
    print("OK: endpoint is reachable and the API key is accepted.")
    print(f"Worker namespace: {namespace}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
