"""Helpers for transcript image blob references."""

from __future__ import annotations

import base64
from pathlib import Path

from spellbook.ir_types import (
    IMAGE_MEDIA_TYPES,
    IRBlock,
    IRImageBase64Source,
    IRImageBlobSource,
    IRImageBlock,
    IRToolResultBlock,
    IRToolResultContentBlock,
)


def persist_image_blobs_in_block(block: IRBlock) -> IRBlock:
    """Replace persisted image payloads with blob references when possible."""
    match block:
        case IRImageBlock():
            return _persist_image_blob(block)
        case IRToolResultBlock():
            content = [_persist_tool_result_content(item) for item in block.content]
            return block.model_copy(update={"content": content})
        case _:
            return block


def hydrate_image_blobs_in_block(block: IRBlock, transcript_path: Path) -> IRBlock:
    """Replace transcript blob references with provider-ready base64 images."""
    match block:
        case IRImageBlock():
            return _hydrate_image_blob(block, transcript_path)
        case IRToolResultBlock():
            content = [
                _hydrate_tool_result_content(item, transcript_path)
                for item in block.content
            ]
            return block.model_copy(update={"content": content})
        case _:
            return block


def resolve_blob_path(blob_path: str, transcript_path: Path) -> Path:
    path = Path(blob_path)
    if path.is_absolute():
        return path
    return transcript_path.parent / path


def _persist_tool_result_content(
    item: IRToolResultContentBlock,
) -> IRToolResultContentBlock:
    if isinstance(item, IRImageBlock):
        return _persist_image_blob(item)
    return item


def _persist_image_blob(block: IRImageBlock) -> IRImageBlock:
    if block.blob_path is None or isinstance(block.source, IRImageBlobSource):
        return block
    return block.model_copy(update={"source": IRImageBlobSource()})


def _hydrate_tool_result_content(
    item: IRToolResultContentBlock,
    transcript_path: Path,
) -> IRToolResultContentBlock:
    if isinstance(item, IRImageBlock):
        return _hydrate_image_blob(item, transcript_path)
    return item


def _hydrate_image_blob(block: IRImageBlock, transcript_path: Path) -> IRImageBlock:
    if not isinstance(block.source, IRImageBlobSource):
        return block
    if block.blob_path is None:
        raise ValueError(
            f"Invalid image block at event {block.event_id}: "
            "blob image sources must have a `blob_path`."
        )

    blob = resolve_blob_path(block.blob_path, transcript_path)
    image_bytes = blob.read_bytes()
    media_type = IMAGE_MEDIA_TYPES.get(blob.suffix.lower(), "image/png")
    data = base64.standard_b64encode(image_bytes).decode("ascii")
    return block.model_copy(
        update={"source": IRImageBase64Source(media_type=media_type, data=data)}
    )
