"""Annotation runner using the OpenRouter API (OpenAI-compatible)."""

from __future__ import annotations

import base64
import json
import logging

import numpy as np
import openai

import auto_video_annotation.annotate as annotate_module
import auto_video_annotation.keypoints as keypoints_module

logger = logging.getLogger(__name__)

_TOOL: dict = {
    "type": "function",
    "function": {
        "name": annotate_module.TOOL_NAME,
        "description": annotate_module.TOOL_DESCRIPTION,
        "parameters": annotate_module.TOOL_INPUT_SCHEMA,
    },
}


class OpenRouterRunner:
    """Locates keypoints in video frames using the OpenRouter API."""

    def __init__(self, model: str) -> None:
        self.model = model
        self._client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
        )
        logger.debug("OpenRouterRunner initialised with model=%s", model)

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
        b64_data = base64.standard_b64encode(image_bytes).decode()
        logger.info(
            "Annotating frame %d (%dx%d) with %d keypoints via OpenRouter",
            frame_number, display_w, display_h, len(keypoints),
        )

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_data}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            tools=[_TOOL],
            tool_choice="required",
        )

        tool_calls = response.choices[0].message.tool_calls
        if not tool_calls:
            logger.warning("Frame %d: model did not call the tool; skipping", frame_number)
            return []

        raw_args = json.loads(tool_calls[0].function.arguments)
        raw: list[dict] = raw_args.get("keypoints", [])
        annotations = annotate_module.parse_raw_keypoints(raw, frame_number, scale_x, scale_y, display_w, display_h)
        logger.debug(
            "Frame %d: %d / %d keypoints annotated",
            frame_number, len(annotations), len(keypoints),
        )
        return annotations
