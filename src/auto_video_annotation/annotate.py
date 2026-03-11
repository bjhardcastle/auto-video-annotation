"""LLM-based keypoint coordinate detection using the Anthropic API."""

from __future__ import annotations

import base64
import io
import logging

import anthropic
import cv2
import numpy as np
from PIL import Image

import auto_video_annotation.keypoints as keypoints_module

logger = logging.getLogger(__name__)

# Tool schema for structured keypoint output
_KEYPOINTS_TOOL: anthropic.types.ToolParam = {
    "name": "record_keypoints",
    "description": (
        "Record the pixel coordinates of each requested keypoint in the image. "
        "Use null for x and y when a keypoint is not visible."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "keypoints": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "x": {"type": ["number", "null"]},
                        "y": {"type": ["number", "null"]},
                    },
                    "required": ["name", "x", "y"],
                },
            }
        },
        "required": ["keypoints"],
    },
}

# Maximum dimension for images sent to the API
_MAX_IMAGE_DIM = 1568


def annotate_frame(
    client: anthropic.Anthropic,
    frame: np.ndarray,
    frame_number: int,
    keypoints: list[keypoints_module.Keypoint],
    model: str,
) -> list[keypoints_module.Annotation]:
    """Ask the LLM to locate each keypoint in *frame* and return annotations.

    The frame is resized to fit within ``_MAX_IMAGE_DIM`` before sending, and
    the returned coordinates are scaled back to the original resolution.
    """
    original_h, original_w = frame.shape[:2]
    image_bytes, display_w, display_h = _encode_frame(frame)
    scale_x = original_w / display_w
    scale_y = original_h / display_h

    keypoint_list = "\n".join(
        f"- {kp.name}: {kp.description}" if kp.description else f"- {kp.name}"
        for kp in keypoints
    )
    keypoint_names = [kp.name for kp in keypoints]
    prompt = (
        f"This image is {display_w}×{display_h} pixels (width × height).\n\n"
        f"Identify the pixel coordinates of these keypoints:\n{keypoint_list}\n\n"
        "Call the record_keypoints tool with the x (horizontal) and y (vertical) "
        "pixel coordinates for each keypoint. Use null if a keypoint is not visible."
    )

    logger.info("Annotating frame %d (%dx%d) with %d keypoints", frame_number, display_w, display_h, len(keypoint_names))

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        tools=[_KEYPOINTS_TOOL],
        tool_choice={"type": "any"},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64.standard_b64encode(image_bytes).decode(),
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )

    tool_use_block = next(
        (block for block in response.content if block.type == "tool_use"),
        None,
    )
    if tool_use_block is None:
        logger.warning("Frame %d: model did not call the tool; skipping", frame_number)
        return []

    raw_keypoints: list[dict] = tool_use_block.input.get("keypoints", [])
    annotations: list[keypoints_module.Annotation] = []
    for kp in raw_keypoints:
        if not isinstance(kp, dict):
            logger.warning("Frame %d: unexpected keypoint format %r; skipping", frame_number, kp)
            continue
        x_display = kp.get("x")
        y_display = kp.get("y")
        if x_display is None or y_display is None:
            logger.debug("Frame %d: keypoint '%s' not visible", frame_number, kp.get("name"))
            continue
        annotations.append(
            keypoints_module.Annotation(
                frame=frame_number,
                keypoint=kp["name"],
                x=round(float(x_display) * scale_x, 2),
                y=round(float(y_display) * scale_y, 2),
            )
        )
    logger.debug("Frame %d: %d / %d keypoints annotated", frame_number, len(annotations), len(keypoint_names))
    return annotations


def _encode_frame(frame: np.ndarray) -> tuple[bytes, int, int]:
    """Convert a BGR numpy frame to a PNG byte string, resizing if necessary.

    Returns:
        (png_bytes, width, height) after any resizing.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    w, h = img.size
    if max(w, h) > _MAX_IMAGE_DIM:
        scale = _MAX_IMAGE_DIM / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        w, h = img.size
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), w, h
