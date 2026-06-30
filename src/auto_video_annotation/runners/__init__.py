"""Runner protocol and factory for annotation providers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

import auto_video_annotation.keypoints as keypoints_module

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@runtime_checkable
class AnnotationRunner(Protocol):
    """Protocol for LLM-based keypoint annotation runners."""

    def annotate_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        keypoints: list[keypoints_module.Keypoint],
        video_description: str | None = None,
    ) -> list[keypoints_module.Annotation]: ...


def create_runner(provider: str, model: str) -> AnnotationRunner:
    """Instantiate a runner for *provider* using *model*.

    Args:
        provider: One of ``"anthropic"``, ``"openrouter"``, or ``"google"``.
        model: Model identifier string passed to the provider API.
    """
    logger.info("Creating runner: provider=%s model=%s", provider, model)
    if provider == "anthropic":
        import auto_video_annotation.runners.anthropic_runner as anthropic_runner
        return anthropic_runner.AnthropicRunner(model=model)
    elif provider == "openrouter":
        import auto_video_annotation.runners.openrouter_runner as openrouter_runner
        return openrouter_runner.OpenRouterRunner(model=model)
    elif provider == "google":
        import auto_video_annotation.runners.google_runner as google_runner
        return google_runner.GoogleRunner(model=model)
    else:
        raise ValueError(
            f"Unknown provider {provider!r}. Choose from: anthropic, openrouter, google"
        )
