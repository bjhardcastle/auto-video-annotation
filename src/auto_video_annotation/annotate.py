"""Shared frame-encoding and prompt-building utilities for annotation runners."""

from __future__ import annotations

import io
import logging

import cv2
import numpy as np
from PIL import Image

import auto_video_annotation.keypoints as keypoints_module

logger = logging.getLogger(__name__)

TOOL_NAME = "record_keypoints"
TOOL_DESCRIPTION = (
    "Record the pixel coordinates of each requested keypoint in the image. "
    "Use null for x and y when a keypoint is not visible."
)

# Raw JSON Schema for the tool input — shared across providers.
TOOL_INPUT_SCHEMA: dict = {
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
}

# Anthropic image size limits: max 1568px per side, max 1.15 megapixels
_MAX_IMAGE_DIM = 1568
_MAX_IMAGE_PIXELS = 1_150_000


def encode_frame(frame: np.ndarray) -> tuple[bytes, int, int]:
    """Convert a BGR numpy frame to a PNG byte string, resizing if necessary.

    Returns:
        (png_bytes, width, height) after any resizing.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    w, h = img.size
    scale = 1.0
    if max(w, h) > _MAX_IMAGE_DIM:
        scale = min(scale, _MAX_IMAGE_DIM / max(w, h))
    if w * h * scale * scale > _MAX_IMAGE_PIXELS:
        scale = min(scale, ((_MAX_IMAGE_PIXELS / (w * h)) ** 0.5))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        w, h = img.size
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), w, h


def build_prompt(
    keypoints: list[keypoints_module.Keypoint],
    display_w: int,
    display_h: int,
    video_description: str | None = None,
) -> str:
    """Build the annotation prompt for the given keypoints and image dimensions."""
    keypoint_list = "\n".join(
        f"- {kp.name}: {kp.description}" if kp.description else f"- {kp.name}"
        for kp in keypoints
    )
    description_prefix = f"{video_description}\n\n" if video_description else ""
    return (
        f"{description_prefix}"
        f"This image is {display_w}×{display_h} pixels (width × height).\n\n"
        f"Identify the pixel coordinates of these keypoints:\n{keypoint_list}\n\n"
        "Return the [x,y] coordinates for each keypoint, where x is the horizontal "
        "coordinate and y is the vertical coordinate. Use null if a keypoint is not clearly visible."
    )


def parse_raw_keypoints(
    raw: list[dict],
    frame_number: int,
    scale_x: float,
    scale_y: float,
    display_w: int,
    display_h: int,
) -> list[keypoints_module.Annotation]:
    """Convert raw tool-call keypoint dicts to scaled ``Annotation`` objects."""
    annotations: list[keypoints_module.Annotation] = []
    for kp in raw:
        if not isinstance(kp, dict):
            logger.warning("Frame %d: unexpected keypoint format %r; skipping", frame_number, kp)
            continue
        x_display = kp.get("x")
        y_display = kp.get("y")
        if x_display is None or y_display is None:
            logger.debug("Frame %d: keypoint '%s' not visible", frame_number, kp.get("name"))
            continue
        x_display = float(x_display)
        y_display = float(y_display)
        if not (0 <= x_display <= display_w and 0 <= y_display <= display_h):
            logger.warning(
                "Frame %d: keypoint '%s' coords (%.1f, %.1f) outside display bounds %dx%d; skipping",
                frame_number, kp.get("name"), x_display, y_display, display_w, display_h,
            )
            continue
        annotations.append(
            keypoints_module.Annotation(
                frame=frame_number,
                keypoint=kp["name"],
                x=round(x_display * scale_x, 2),
                y=round(y_display * scale_y, 2),
            )
        )
    return annotations
