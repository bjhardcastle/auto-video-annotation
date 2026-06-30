"""Annotation runner using the Anthropic API."""

from __future__ import annotations

import base64
import logging
from typing import Any

import anthropic
import numpy as np

import auto_video_annotation.annotate as annotate_module
import auto_video_annotation.keypoints as keypoints_module

logger = logging.getLogger(__name__)

_TOOL: anthropic.types.ToolParam = {
    "name": annotate_module.TOOL_NAME,
    "description": annotate_module.TOOL_DESCRIPTION,
    "input_schema": annotate_module.TOOL_INPUT_SCHEMA,
}

_FORCED_TOOL_UNSUPPORTED_TEXT = "tool_choice forces tool use is not compatible with this model"
_SYSTEM_PROMPT = (
    f"Use the {annotate_module.TOOL_NAME} tool to record the requested keypoints. "
    "Return all visible and non-visible keypoints through that tool; do not answer with prose."
)


class AnthropicRunner:
    """Locates keypoints in video frames using the Anthropic messages API."""

    def __init__(self, model: str) -> None:
        self.model = model
        self._client = anthropic.Anthropic()
        self._force_tool_choice = True
        logger.debug("AnthropicRunner initialised with model=%s", model)

    def annotate_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        keypoints: list[keypoints_module.Keypoint],
        video_description: str | None = None,
    ) -> list[keypoints_module.Annotation]:
        original_h, original_w = frame.shape[:2]
        image_bytes, display_w, display_h = annotate_module.encode_frame(frame)
        scale_x = original_w / display_w
        scale_y = original_h / display_h

        prompt = annotate_module.build_prompt(keypoints, display_w, display_h, video_description)
        logger.info(
            "Annotating frame %d (%dx%d) with %d keypoints via Anthropic",
            frame_number, display_w, display_h, len(keypoints),
        )

        response = self._create_message(
            [
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
            ]
        )

        tool_use_block = next(
            (block for block in response.content if block.type == "tool_use"),
            None,
        )
        if tool_use_block is None:
            logger.warning("Frame %d: model did not call the tool; skipping", frame_number)
            return []

        raw: list[dict] = tool_use_block.input.get("keypoints", [])
        annotations = annotate_module.parse_raw_keypoints(raw, frame_number, scale_x, scale_y, display_w, display_h)
        logger.debug(
            "Frame %d: %d / %d keypoints annotated",
            frame_number, len(annotations), len(keypoints),
        )
        return annotations

    def _create_message(self, messages: list[dict[str, Any]]) -> anthropic.types.Message:
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 1024,
            "system": _SYSTEM_PROMPT,
            "tools": [_TOOL],
            "messages": messages,
        }
        if self._force_tool_choice:
            request["tool_choice"] = {"type": "tool", "name": annotate_module.TOOL_NAME}

        try:
            return self._client.messages.create(**request)
        except anthropic.BadRequestError as exc:
            if not self._force_tool_choice or not _is_forced_tool_choice_unsupported(exc):
                raise
            logger.info(
                "Anthropic model %s rejected forced tool use; retrying with default tool_choice=auto",
                self.model,
            )
            self._force_tool_choice = False
            request.pop("tool_choice", None)
            return self._client.messages.create(**request)


def _is_forced_tool_choice_unsupported(exc: anthropic.BadRequestError) -> bool:
    error = exc.body.get("error") if isinstance(exc.body, dict) else None
    message = error.get("message") if isinstance(error, dict) else exc.message
    return isinstance(message, str) and _FORCED_TOOL_UNSUPPORTED_TEXT in message
