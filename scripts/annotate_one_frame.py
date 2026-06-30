"""Annotate one image frame with the auto_video_annotation runners.

Usage:
    uv run python scripts/annotate_one_frame.py
    uv run python scripts/annotate_one_frame.py --models sonnet-4-6 fable-5
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
DEFAULT_MODELS = ["sonnet-4-6", "sonnet-5", "opus-4-7", "opus-4-8", "fable-5"]
WITH_KEYPOINT_DESCRIPTIONS = False


def _model_id(provider: str, model: str) -> str:
    if provider == "anthropic" and not model.startswith("claude-"):
        return f"claude-{model}"
    return model


def _safe_stem(name: str) -> str:
    safe = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in name
    )
    return safe.strip("._") or "model"


def _description_suffix(with_descriptions: bool) -> str:
    return "_with_descriptions" if with_descriptions else "_without_descriptions"


def _save_model_viz(
    frame,
    annotations: list[keypoints_module.Annotation],
    frame_number: int,
    frames_dir: Path,
    model_name: str,
    filename_suffix: str,
) -> Path:
    viz_module.save_annotated_frames(
        frames=[(frame_number, frame)],
        annotations=annotations,
        output_dir=frames_dir,
    )
    generated_path = frames_dir / f"frame_{frame_number:06d}.png"
    model_path = frames_dir / f"{_safe_stem(model_name)}{filename_suffix}.png"
    generated_path.replace(model_path)
    return model_path


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
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model identifiers to run. Anthropic shorthand is expanded with claude-.",
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
        "--keypoint-descriptions",
        action=argparse.BooleanOptionalAction,
        dest="with_keypoint_descriptions",
        default=WITH_KEYPOINT_DESCRIPTIONS,
        help="Include keypoint descriptions in the prompt. Use --no-keypoint-descriptions to omit them.",
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
        help="Optional JSON file path. Defaults to <output-dir>/annotations_<description-mode>.json.",
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
    description_suffix = _description_suffix(args.with_keypoint_descriptions)
    output_dir = args.output_dir or DEFAULT_OUTPUT_ROOT / image_path.stem
    annotations_path = args.output or output_dir / f"annotations{description_suffix}.json"
    frames_dir = output_dir / "frames"
    per_model_annotations_dir = output_dir / "annotations"

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise SystemExit(f"Could not read image: {image_path}")

    keypoints = keypoints_module.load_keypoints(keypoints_path)
    if args.with_keypoint_descriptions:
        prompt_keypoints = keypoints
    else:
        prompt_keypoints = [
            keypoint.model_copy(update={"description": ""})
            for keypoint in keypoints
        ]
    all_records: dict[str, list[dict]] = {}

    for model_name in args.models:
        model_id = _model_id(args.provider, model_name)
        logging.info("Annotating with %s", model_id)
        runner = runners_module.create_runner(
            provider=args.provider,
            model=model_id,
        )

        annotations = runner.annotate_frame(
            frame=frame,
            frame_number=args.frame_number,
            keypoints=prompt_keypoints,
            video_description=args.video_description,
        )

        records = [annotation.model_dump() for annotation in annotations]
        all_records[model_name] = records

        per_model_annotations_dir.mkdir(parents=True, exist_ok=True)
        per_model_path = (
            per_model_annotations_dir
            / f"{_safe_stem(model_name)}{description_suffix}.json"
        )
        per_model_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        viz_path = _save_model_viz(
            frame=frame,
            annotations=annotations,
            frame_number=args.frame_number,
            frames_dir=frames_dir,
            model_name=model_name,
            filename_suffix=description_suffix,
        )
        logging.info("Wrote %s annotations to %s", model_name, per_model_path.resolve())
        logging.info("Wrote %s visualization to %s", model_name, viz_path.resolve())

    output_json = json.dumps(all_records, indent=2)
    print(output_json)

    annotations_path.parent.mkdir(parents=True, exist_ok=True)
    annotations_path.write_text(output_json + "\n", encoding="utf-8")
    logging.info("Wrote annotations to %s", annotations_path.resolve())
    logging.info("Wrote visualizations to %s", frames_dir.resolve())


if __name__ == "__main__":
    main()
