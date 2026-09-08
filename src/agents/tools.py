"""Function tools the preprocessing agent calls, plus their JSON schemas.

These are ports of the ``inspect_preprocessing_image`` and ``apply_preprocessing_tool``
activities from ``workflows.personal_doc_workflow``, minus the ``@workflows.activity``
decorator. The agent chooses *which* operation to run; ``shared.preprocessing`` keeps
ownership of every transformation parameter.
"""

import asyncio
import logging
from typing import Any

from mistralai.client import Mistral

from agents.files import download_file, get_signed_url, upload_png
from shared.preprocessing import (
    PREPROCESSING_OPERATIONS,
    apply_preprocessing_operation_bytes,
    enhanced_image_filename,
    inspect_image_bytes,
    preview_image_bytes,
)

INSPECT_TOOL_NAME = "inspect_document_image"
APPLY_TOOL_NAME = "apply_preprocessing_operation"

#: ``sauvola`` needs scikit-image, which is not a project dependency.
UNAVAILABLE_OPERATIONS = frozenset({"sauvola"})

#: Operations that can destroy content, mirroring the ones ``conservative_ocr_config``
#: turns off (see ``test_conservative_profile_disables_content_removal_steps``).
#: ``table_grid_removal`` inpaints what it takes for grid lines, and on an identity
#: document the MRZ band reads as a table region — it erased the MRZ outright in an
#: end-to-end run, costing every field the MRZ would have filled in. The agent picks
#: *which* operation to run, so it must only be offered non-destructive ones.
DESTRUCTIVE_OPERATIONS = frozenset(
    {"table_grid_removal", "stamp_removal", "perspective"}
)

#: The operations the agent may choose from.
AGENT_OPERATIONS: tuple[str, ...] = tuple(
    operation
    for operation in PREPROCESSING_OPERATIONS
    if operation not in UNAVAILABLE_OPERATIONS | DESTRUCTIVE_OPERATIONS
)

#: Previews shown to the agent are deliberately smaller than the module default:
#: an image entry stays in the stored conversation and is re-sent on every later
#: turn, and this is large enough to judge skew, lighting and orientation.
AGENT_PREVIEW_DIMENSION = 512

logger = logging.getLogger(__name__)


async def inspect_document_image(
    client: Mistral, file_id: str, filename: str
) -> dict[str, Any]:
    """Return quality metrics and a bounded preview reference for an image."""
    image_bytes = await download_file(client, file_id)
    # OpenCV work is CPU-bound; off the event loop it cannot stall API calls.
    preview = await asyncio.to_thread(
        preview_image_bytes, image_bytes, AGENT_PREVIEW_DIMENSION
    )
    metrics = await asyncio.to_thread(inspect_image_bytes, image_bytes)
    preview_id = await upload_png(
        client, preview, f"{enhanced_image_filename(filename)}.preview.png"
    )
    return {
        "file_id": file_id,
        "filename": filename,
        "metrics": metrics,
        "preview_file_id": preview_id,
        "preview_url": await get_signed_url(client, preview_id),
    }


async def apply_preprocessing_operation(
    client: Mistral, source_file_id: str, source_filename: str, operation: str
) -> dict[str, Any]:
    """Apply one agent-selected operation and persist the replacement artifact."""
    if operation not in AGENT_OPERATIONS:
        raise ValueError(f"Unsupported preprocessing operation: {operation}")
    source_bytes = await download_file(client, source_file_id)
    logger.info("applying preprocessing operation %r", operation)
    processed_bytes = await asyncio.to_thread(
        apply_preprocessing_operation_bytes, source_bytes, operation
    )
    preview = await asyncio.to_thread(
        preview_image_bytes, processed_bytes, AGENT_PREVIEW_DIMENSION
    )
    metrics = await asyncio.to_thread(inspect_image_bytes, processed_bytes)
    filename = enhanced_image_filename(source_filename)
    file_id = await upload_png(client, processed_bytes, filename)
    preview_id = await upload_png(client, preview, f"{filename}.preview.png")
    return {
        "file_id": file_id,
        "filename": filename,
        "content_type": "image/png",
        "operation": operation,
        "metrics": metrics,
        "preview_file_id": preview_id,
        "preview_url": await get_signed_url(client, preview_id),
    }


#: Dispatch table used by the orchestrator's ``function.call`` pump.
TOOL_DISPATCH = {
    INSPECT_TOOL_NAME: inspect_document_image,
    APPLY_TOOL_NAME: apply_preprocessing_operation,
}


#: Declared on the preprocessing agent at creation time.
PREPROCESSING_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": INSPECT_TOOL_NAME,
            "description": (
                "Measure an image's OCR quality. Returns width, height, blur_variance, "
                "brightness, noise_estimate, table_regions and a preview_url."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_id": {
                        "type": "string",
                        "description": "Id of the image file to inspect.",
                    },
                    "filename": {
                        "type": "string",
                        "description": "Filename of the image file to inspect.",
                    },
                },
                "required": ["file_id", "filename"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": APPLY_TOOL_NAME,
            "description": (
                "Apply one preprocessing operation and store the result as a new PNG. "
                "Returns the new file_id, filename and metrics."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source_file_id": {
                        "type": "string",
                        "description": "Id of the image to transform (the latest one).",
                    },
                    "source_filename": {
                        "type": "string",
                        "description": "Filename of the image to transform.",
                    },
                    "operation": {
                        "type": "string",
                        "enum": list(AGENT_OPERATIONS),
                        "description": "The operation to apply.",
                    },
                },
                "required": ["source_file_id", "source_filename", "operation"],
                "additionalProperties": False,
            },
        },
    },
]
