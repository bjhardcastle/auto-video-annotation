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


def load_annotations(path: Path) -> list[Annotation]:
    """Load annotations from a JSONL file (one JSON object per line)."""
    annotations: list[Annotation] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            annotations.append(Annotation.model_validate_json(line))
    logger.info("Loaded %d annotations from %s", len(annotations), path)
    return annotations


def append_annotations(annotations: list[Annotation], path: Path) -> None:
    """Append annotations to a JSONL file, one record per line.

    Creates the file and parent directories if they do not exist.
    Safe to call after each frame so partial results survive early termination.
    """
    if not annotations:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for ann in annotations:
            f.write(ann.model_dump_json() + "\n")
    logger.debug("Appended %d annotations to %s", len(annotations), path)


def save_annotations(annotations: list[Annotation], path: Path) -> None:
    """Write annotations to a JSON file as an array of records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [ann.model_dump() for ann in annotations]
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    logger.info("Wrote %d annotations to %s", len(annotations), path)
