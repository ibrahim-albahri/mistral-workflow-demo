"""Mistral file helpers for the Agents implementation.

The workflow version reaches for ``workflows_mistralai.get_mistral_client()``; here
there is no worker context, so every helper takes an explicit client or builds a
plain one from the environment.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Awaitable, Callable, TypeVar

import httpx
from mistralai.client import Mistral
from mistralai.client.errors import SDKError

logger = logging.getLogger(__name__)

T = TypeVar("T")

DOWNLOAD_TIMEOUT_SECONDS = 60

#: The SDK defaults to 30s, which a vision turn over a full-page scan routinely
#: exceeds. Comparable to the 2–5 minute activity timeouts in the workflow version.
REQUEST_TIMEOUT_MS = 300_000


def api_key() -> str:
    return os.environ["MISTRAL_API_KEY"]


def server_url() -> str:
    return os.environ.get("SERVER_URL", "https://api.mistral.ai")


def build_client() -> Mistral:
    """Build an unopened client; callers use it as an async context manager."""
    return Mistral(
        api_key=api_key(),
        server_url=server_url(),
        timeout_ms=REQUEST_TIMEOUT_MS,
    )


@asynccontextmanager
async def mistral_client():
    async with build_client() as client:
        yield client


#: Retry budget for a rate-limited or transient API call. The workflow version gets
#: this from its activities' ``retry_policy_max_attempts``; here it is explicit.
MAX_ATTEMPTS = 5
BACKOFF_BASE_SECONDS = 4


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.ReadTimeout, httpx.ConnectError)):
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(exc, SDKError) and status in {429, 500, 502, 503, 504}


async def call_with_retry(
    operation: Callable[[], Awaitable[T]], description: str = "API call"
) -> T:
    """Await ``operation``, retrying rate limits and transient failures."""
    last: BaseException | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return await operation()
        except Exception as exc:  # noqa: BLE001 - re-raised below when not retryable
            if not _is_retryable(exc) or attempt == MAX_ATTEMPTS:
                raise
            last = exc
            delay = BACKOFF_BASE_SECONDS * 2 ** (attempt - 1)
            logger.warning(
                "%s failed (attempt %d/%d): %s — retrying in %ds",
                description,
                attempt,
                MAX_ATTEMPTS,
                type(exc).__name__,
                delay,
            )
            await asyncio.sleep(delay)
    raise AssertionError(f"unreachable: {last}")


async def get_signed_url(client: Mistral, file_id: str) -> str:
    signed_url = await client.files.get_signed_url_async(file_id=file_id)
    return signed_url.url


async def download_file(client: Mistral, file_id: str) -> bytes:
    """Download a stored file through its signed URL."""
    url = await get_signed_url(client, file_id)
    async with httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT_SECONDS) as http:
        response = await http.get(url)
        response.raise_for_status()
        return response.content


async def upload_png(client: Mistral, image_bytes: bytes, filename: str) -> str:
    """Store a processed PNG and return its file id."""
    response = await client.files.upload_async(
        file={
            "file_name": filename,
            "content": image_bytes,
            "content_type": "image/png",
        },
        purpose="ocr",
    )
    return response.id


async def upload_document(
    client: Mistral, document_bytes: bytes, filename: str, content_type: str
) -> str:
    """Store an original upload and return its file id."""
    response = await client.files.upload_async(
        file={
            "file_name": filename,
            "content": document_bytes,
            "content_type": content_type,
        },
        purpose="ocr",
    )
    return response.id
