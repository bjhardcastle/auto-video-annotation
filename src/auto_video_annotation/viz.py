"""Visualisation — save annotated frames with keypoints overlaid."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

import auto_video_annotation.keypoints as keypoints_module

logger = logging.getLogger(__name__)

_RADIUS = 6
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.5
_FONT_THICKNESS = 1
_CIRCLE_THICKNESS = 2
# Distinct BGR colours cycled across keypoints
_PALETTE = [
    (0, 255, 0),
    (0, 128, 255),
    (255, 0, 128),
    (0, 255, 255),
    (255, 0, 255),
    (255, 128, 0),
    (128, 0, 255),
    (0, 200, 100),
]


def save_annotated_frames(
    frames: list[tuple[int, np.ndarray]],
    annotations: list[keypoints_module.Annotation],
    output_dir: Path,
) -> None:
    """Draw keypoint markers on each frame and save as PNG files.

    Files are named ``frame_{frame_number:06d}.png`` inside *output_dir*.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group annotations by frame number for quick lookup
    by_frame: dict[int, list[keypoints_module.Annotation]] = {}
    for ann in annotations:
        by_frame.setdefault(ann.frame, []).append(ann)

    # Build a stable colour map per keypoint name
    all_names = sorted({ann.keypoint for ann in annotations})
    colour_map = {name: _PALETTE[i % len(_PALETTE)] for i, name in enumerate(all_names)}

    for frame_number, frame in frames:
        annotated = frame.copy()
        frame_anns = by_frame.get(frame_number, [])
        logger.info("Frame %d: drawing %d annotations", frame_number, len(frame_anns))
        for ann in frame_anns:
            colour = colour_map[ann.keypoint]
            cx, cy = int(round(ann.x)), int(round(ann.y))
            cv2.circle(annotated, (cx, cy), _RADIUS, colour, _CIRCLE_THICKNESS)
            # Label slightly above the circle
            cv2.putText(
                annotated,
                ann.keypoint,
                (cx + _RADIUS + 2, cy - _RADIUS),
                _FONT,
                _FONT_SCALE,
                colour,
                _FONT_THICKNESS,
                cv2.LINE_AA,
            )
        dest = output_dir / f"frame_{frame_number:06d}.png"
        cv2.imwrite(str(dest), annotated)
        logger.debug("Saved annotated frame %d → %s", frame_number, dest)

    logger.info("Saved %d annotated frames to %s", len(frames), output_dir)
