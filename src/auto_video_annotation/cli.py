"""CLI entry point for auto-video-annotation."""

from __future__ import annotations

import logging
from pathlib import Path

import anthropic
import dotenv
import pydantic_settings

import auto_video_annotation.annotate as annotate_module
import auto_video_annotation.keypoints as keypoints_module
import auto_video_annotation.video as video_module
import auto_video_annotation.viz as viz_module

logger = logging.getLogger(__name__)


class Settings(pydantic_settings.BaseSettings):
    model_config = pydantic_settings.SettingsConfigDict(
        cli_kebab_case="all",
        cli_implicit_flags=True,
    )

    video: Path
    keypoints: Path = Path("keypoints.json")
    output_dir: Path = Path("output")
    num_frames: int = 10
    model: str = "claude-sonnet-4-6"
    log_level: str = "INFO"


def main() -> None:
    dotenv.load_dotenv()
    settings = Settings(_cli_parse_args=True)  # type: ignore[call-arg]

    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    logger.info("Starting annotation run")
    run_dir = settings.output_dir / settings.video.stem

    logger.info("Video: %s", settings.video)
    logger.info("Keypoints: %s", settings.keypoints)
    logger.info("Output dir: %s", run_dir)
    logger.info("Frames to extract: %d", settings.num_frames)
    logger.info("Model: %s", settings.model)

    keypoint_names = keypoints_module.load_keypoints(settings.keypoints)
    if not keypoint_names:
        raise SystemExit("No keypoints found in the input file.")

    frames = video_module.load_diverse_frames(settings.video, settings.num_frames)
    if not frames:
        raise SystemExit("No frames could be read from the video.")

    client = anthropic.Anthropic()
    all_annotations: list[keypoints_module.Annotation] = []
    for frame_number, frame in frames:
        annotations = annotate_module.annotate_frame(
            client=client,
            frame=frame,
            frame_number=frame_number,
            keypoints=keypoint_names,
            model=settings.model,
        )
        all_annotations.extend(annotations)

    keypoints_module.save_annotations(all_annotations, run_dir / "annotations.json")
    viz_module.save_annotated_frames(frames, all_annotations, run_dir / "frames")
    logger.info("Done. %d annotations written.", len(all_annotations))


if __name__ == "__main__":
    main()
