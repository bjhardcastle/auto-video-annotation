"""Annotation runner using the Anthropic API."""

from __future__ import annotations

import base64
import logging

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


class AnthropicRunner:
    """Locates keypoints in video frames using the Anthropic messages API."""

    def __init__(self, model: str) -> None:
        self.model = model
        self._client = anthropic.Anthropic()
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

        response = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            tools=[_TOOL],
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

        raw: list[dict] = tool_use_block.input.get("keypoints", [])
        annotations = annotate_module.parse_raw_keypoints(raw, frame_number, scale_x, scale_y, display_w, display_h)
        logger.debug(
            "Frame %d: %d / %d keypoints annotated",
            frame_number, len(annotations), len(keypoints),
        )
        return annotations
