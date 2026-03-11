"""Data models and JSON I/O for keypoints and annotations."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pydantic

logger = logging.getLogger(__name__)


class Annotation(pydantic.BaseModel):
    frame: int
    keypoint: str
    x: float
    y: float


class Keypoint(pydantic.BaseModel):
    name: str
    description: str | None = None


def load_keypoints(path: Path) -> list[Keypoint]:
    """Load keypoints from a JSON file.

    Each item may be a plain string or an object with ``name`` and optional
    ``description`` fields::

        ["nose tip", {"name": "eye_top_center", "description": "centre of the top eyelid margin"}]
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON array, got {type(data)}")
    keypoints: list[Keypoint] = []
    for i, item in enumerate(data):
        if isinstance(item, str):
            keypoints.append(Keypoint(name=item))
        elif isinstance(item, dict):
            keypoints.append(Keypoint.model_validate(item))
        else:
            raise ValueError(f"{path}[{i}]: expected a string or object, got {type(item)}")
    logger.info("Loaded %d keypoints from %s", len(keypoints), path)
    return keypoints


def save_annotations(annotations: list[Annotation], path: Path) -> None:
    """Write annotations to a JSON file as an array of records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [ann.model_dump() for ann in annotations]
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    logger.info("Wrote %d annotations to %s", len(annotations), path)
