"""CLI entry point for auto-video-annotation."""


import logging
import urllib.parse
from pathlib import Path
from typing import Literal

import dotenv
import pydantic_settings

import auto_video_annotation.keypoints as keypoints_module
import auto_video_annotation.runners as runners_module
import auto_video_annotation.video as video_module
import auto_video_annotation.viz as viz_module

logger = logging.getLogger(__name__)


class Settings(pydantic_settings.BaseSettings):
    model_config = pydantic_settings.SettingsConfigDict(
        cli_kebab_case="all",
        cli_implicit_flags=True,
    )

    video: str
    keypoints: Path = Path("keypoints.json")
    output_dir: Path = Path("output")
    num_frames: int = 3
    provider: Literal["anthropic", "google", "openrouter"] = "anthropic"
    model: str = "claude-fable-5"
    video_description: str | None = None
    log_level: str = "INFO"
    viz_only: bool = False


def main() -> None:
    dotenv.load_dotenv()
    settings = Settings(_cli_parse_args=True)  # type: ignore[call-arg]

    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    video_stem = urllib.parse.urlparse(settings.video).path
    video_stem = Path(video_stem).stem or Path(settings.video).stem
    run_dir = settings.output_dir / video_stem

    logger.info("Video: %s", settings.video)
    logger.info("Output dir: %s", run_dir)

    frames = video_module.load_diverse_frames(settings.video, settings.num_frames)
    if not frames:
        raise SystemExit("No frames could be read from the video.")

    annotations_path = run_dir / "annotations.jsonl"

    if settings.viz_only:
        logger.info("Viz-only mode: loading annotations from %s", annotations_path)
        if not annotations_path.exists():
            raise SystemExit(f"No annotations file found at {annotations_path}")
        all_annotations = keypoints_module.load_annotations(annotations_path)
    else:
        logger.info("Keypoints: %s", settings.keypoints)
        logger.info("Frames to extract: %d", settings.num_frames)
        logger.info("Provider: %s", settings.provider)
        logger.info("Model: %s", settings.model)

        keypoint_names = keypoints_module.load_keypoints(settings.keypoints)
        if not keypoint_names:
            raise SystemExit("No keypoints found in the input file.")

        runner = runners_module.create_runner(settings.provider, settings.model)
        all_annotations: list[keypoints_module.Annotation] = []
        for frame_number, frame in frames:
            annotations = runner.annotate_frame(
                frame=frame,
                frame_number=frame_number,
                keypoints=keypoint_names,
                video_description=settings.video_description,
            )
            all_annotations.extend(annotations)
            keypoints_module.append_annotations(annotations, annotations_path)
            logger.info("Frame %d: %d annotations saved (total %d)", frame_number, len(annotations), len(all_annotations))
        logger.info("Done. %d annotations written to %s", len(all_annotations), annotations_path)

    viz_module.save_annotated_frames(frames, all_annotations, run_dir / "frames")


if __name__ == "__main__":
    main()
