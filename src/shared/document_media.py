"""Supported media formats and Mistral content chunks for document workflows."""

CONTENT_TYPE_BY_EXTENSION = {
    "pdf": "application/pdf",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
}

SUPPORTED_DOCUMENT_EXTENSIONS = tuple(CONTENT_TYPE_BY_EXTENSION)
SUPPORTED_IMAGE_CONTENT_TYPES = frozenset(
    content_type
    for content_type in CONTENT_TYPE_BY_EXTENSION.values()
    if content_type.startswith("image/")
)


def get_document_content_type(filename: str) -> str:
    """Return the canonical MIME type for a supported document filename."""
    extension = filename.rsplit(".", maxsplit=1)[-1].lower() if "." in filename else ""
    try:
        return CONTENT_TYPE_BY_EXTENSION[extension]
    except KeyError as exc:
        supported = ", ".join(f".{item}" for item in SUPPORTED_DOCUMENT_EXTENSIONS)
        raise ValueError(
            f"Unsupported document format for '{filename}'. Supported formats: {supported}."
        ) from exc


def build_mistral_document_chunk(
    document_url: str,
    filename: str,
    content_type: str,
) -> dict[str, str]:
    """Build the format-specific content chunk expected by Mistral chat."""
    if content_type == "application/pdf":
        return {
            "type": "document_url",
            "document_url": document_url,
            "document_name": filename,
        }
    if content_type in SUPPORTED_IMAGE_CONTENT_TYPES:
        return {
            "type": "image_url",
            "image_url": document_url,
        }

    raise ValueError(f"Unsupported document content type: {content_type}.")
