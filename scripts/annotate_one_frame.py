"""Annotate one image frame with the auto_video_annotation runners.

Usage:
    uv run python scripts/annotate_one_frame.py
    uv run python scripts/annotate_one_frame.py --provider google --model gemini-2.5-pro
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import dotenv

import auto_video_annotation.keypoints as keypoints_module
import auto_video_annotation.runners as runners_module
import auto_video_annotation.viz as viz_module

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = PROJECT_ROOT / ".scratch" / "Screenshot 2026-03-11 141017.png"
DEFAULT_KEYPOINTS = PROJECT_ROOT / "keypoints.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run keypoint annotation on a single image frame.",
    )
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        default=DEFAULT_IMAGE,
        help=f"Image to annotate. Defaults to {DEFAULT_IMAGE}",
    )
    parser.add_argument(
        "--keypoints",
        type=Path,
        default=DEFAULT_KEYPOINTS,
        help=f"Keypoints JSON file. Defaults to {DEFAULT_KEYPOINTS}",
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "google", "openrouter"],
        default="anthropic",
        help="Annotation provider to use.",
    )
    parser.add_argument(
        "--model",
        default="claude-fable-5",
        help="Model identifier to pass through to the selected provider.",
    )
    parser.add_argument(
        "--frame-number",
        type=int,
        default=0,
        help="Frame number to attach to returned annotations.",
    )
    parser.add_argument(
        "--video-description",
        default=None,
        help="Optional extra scene/context description to include in the prompt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write annotations and visualization. Defaults to output/<image-stem>.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON file path. Defaults to <output-dir>/annotations.json.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dotenv.load_dotenv(PROJECT_ROOT / ".env")
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    image_path = args.image.resolve()
    keypoints_path = args.keypoints.resolve()
    output_dir = args.output_dir or DEFAULT_OUTPUT_ROOT / image_path.stem
    annotations_path = args.output or output_dir / "annotations.json"
    frames_dir = output_dir / "frames"

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise SystemExit(f"Could not read image: {image_path}")

    keypoints = keypoints_module.load_keypoints(keypoints_path)
    runner = runners_module.create_runner(
        provider=args.provider,
        model=args.model,
    )
    annotations = runner.annotate_frame(
        frame=frame,
        frame_number=args.frame_number,
        keypoints=keypoints,
        video_description=args.video_description,
    )

    records = [annotation.model_dump() for annotation in annotations]
    output_json = json.dumps(records, indent=2)
    print(output_json)

    annotations_path.parent.mkdir(parents=True, exist_ok=True)
    annotations_path.write_text(output_json + "\n", encoding="utf-8")
    viz_module.save_annotated_frames(
        frames=[(args.frame_number, frame)],
        annotations=annotations,
        output_dir=frames_dir,
    )
    logging.info("Wrote annotations to %s", annotations_path.resolve())
    logging.info("Wrote visualization to %s", frames_dir.resolve())


if __name__ == "__main__":
    main()
