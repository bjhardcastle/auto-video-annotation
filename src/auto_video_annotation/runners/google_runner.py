"""Annotation runner using the Google Gemini API (google-genai SDK)."""

from __future__ import annotations

import logging

import numpy as np
from google import genai
from google.genai import types

import auto_video_annotation.annotate as annotate_module
import auto_video_annotation.keypoints as keypoints_module

logger = logging.getLogger(__name__)

# Gemini uses its own Schema type; nullable fields use nullable=True rather
# than a union type array.
_TOOL = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name=annotate_module.TOOL_NAME,
            description=annotate_module.TOOL_DESCRIPTION,
            parameters=types.Schema(
                type="object",
                properties={
                    "keypoints": types.Schema(
                        type="array",
                        items=types.Schema(
                            type="object",
                            properties={
                                "name": types.Schema(type="string"),
                                "x": types.Schema(type="number", nullable=True),
                                "y": types.Schema(type="number", nullable=True),
                            },
                            required=["name", "x", "y"],
                        ),
                    )
                },
                required=["keypoints"],
            ),
        )
    ]
)

_TOOL_CONFIG = types.ToolConfig(
    function_calling_config=types.FunctionCallingConfig(mode="ANY")
)


class GoogleRunner:
    """Locates keypoints in video frames using the Google Gemini API."""

    def __init__(self, model: str) -> None:
        self.model = model
        self._client = genai.Client()
        logger.debug("GoogleRunner initialised with model=%s", model)

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
            "Annotating frame %d (%dx%d) with %d keypoints via Google",
            frame_number, display_w, display_h, len(keypoints),
        )

        contents = types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                types.Part.from_text(text=prompt),
            ],
        )
        response = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                tools=[_TOOL],
                tool_config=_TOOL_CONFIG,
            ),
        )

        function_call = next(
            (part.function_call for part in response.candidates[0].content.parts if part.function_call),
            None,
        )
        if function_call is None:
            logger.warning("Frame %d: model did not call the tool; skipping", frame_number)
            return []

        raw: list[dict] = list(dict(function_call.args).get("keypoints", []))
        # MapComposite values are also MapComposite; normalise to plain dicts.
        raw = [dict(kp) for kp in raw]
        # Gemini returns coordinates in 0–1000 normalized space regardless of
        # the pixel-coordinate instruction; convert to display pixels first.
        for kp in raw:
            if kp.get("x") is not None:
                kp["x"] = float(kp["x"]) / 1000.0 * display_w
            if kp.get("y") is not None:
                kp["y"] = float(kp["y"]) / 1000.0 * display_h
        annotations = annotate_module.parse_raw_keypoints(raw, frame_number, scale_x, scale_y, display_w, display_h)
        logger.debug(
            "Frame %d: %d / %d keypoints annotated",
            frame_number, len(annotations), len(keypoints),
        )
        return annotations
