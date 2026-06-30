"""Video loading and diverse frame extraction."""

from __future__ import annotations

import logging
import urllib.parse
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _resolve_video_source(video_path: Path | str) -> str:
    """Return a string suitable for cv2.VideoCapture.

    Converts ``s3://bucket/key`` to ``https://bucket.s3.amazonaws.com/key``
    for open (public) S3 buckets.  Local paths are returned as POSIX strings.
    """
    s = str(video_path)
    parsed = urllib.parse.urlparse(s)
    if parsed.scheme == "s3":
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        url = f"https://{bucket}.s3.amazonaws.com/{key}"
        logger.debug("Resolved S3 path %s → %s", s, url)
        return url
    if isinstance(video_path, Path):
        return video_path.as_posix()
    return s


def load_diverse_frames(video_path: Path | str, num_frames: int) -> list[tuple[int, np.ndarray]]:
    """Open a video file or public S3 URL and return a diverse subset of frames.

    Accepts local ``Path`` objects or strings.  S3 paths (``s3://bucket/key``)
    are converted to HTTPS URLs and streamed via FFmpeg/OpenCV.

    Uses greedy histogram-based selection to maximise visual variety:
    1. Sample ``num_frames * 5`` candidate frames uniformly.
    2. Iteratively select the candidate whose HSV histogram is most dissimilar
       from those already chosen (max-min-distance criterion).

    Returns:
        List of ``(frame_number, bgr_frame)`` tuples sorted by frame number.
    """
    source = _resolve_video_source(video_path)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {source}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise ValueError(f"Video has no readable frames: {source}")

    logger.info(
        "Opened %s — total frames: %d, requesting %d diverse frames",
        source,
        total_frames,
        num_frames,
    )

    num_candidates = min(num_frames * 5, total_frames)
    candidate_indices = np.linspace(10, total_frames - 1, num_candidates, dtype=int).tolist()

    candidates: list[tuple[int, np.ndarray]] = []
    for idx in candidate_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            candidates.append((idx, frame))
    cap.release()

    logger.debug("Read %d candidate frames", len(candidates))

    if len(candidates) <= num_frames:
        return sorted(candidates, key=lambda t: t[0])

    histograms = [_hsv_histogram(frame) for _, frame in candidates]
    selected_indices = _greedy_diverse_selection(histograms, num_frames)
    result = sorted([candidates[i] for i in selected_indices], key=lambda t: t[0])
    logger.info("Selected %d diverse frames", len(result))
    return result


def _hsv_histogram(frame: np.ndarray) -> np.ndarray:
    """Compute a normalised HSV histogram for a BGR frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def _greedy_diverse_selection(histograms: list[np.ndarray], k: int) -> list[int]:
    """Greedy k-subset selection maximising pairwise histogram distance."""
    n = len(histograms)
    # Start with the frame whose histogram is most different from the mean
    mean_hist = np.mean(histograms, axis=0)
    first = int(np.argmax([
        cv2.compareHist(h.reshape(-1, 1).astype(np.float32),
                        mean_hist.reshape(-1, 1).astype(np.float32),
                        cv2.HISTCMP_BHATTACHARYYA)
        for h in histograms
    ]))
    selected = [first]
    # min distance from any selected frame for each candidate
    min_dist = np.array([
        cv2.compareHist(histograms[i].reshape(-1, 1).astype(np.float32),
                        histograms[first].reshape(-1, 1).astype(np.float32),
                        cv2.HISTCMP_BHATTACHARYYA)
        for i in range(n)
    ])
    min_dist[first] = -1.0  # mark as used

    for _ in range(k - 1):
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)
        # update min distances
        for i in range(n):
            if min_dist[i] < 0:
                continue
            d = cv2.compareHist(
                histograms[i].reshape(-1, 1).astype(np.float32),
                histograms[next_idx].reshape(-1, 1).astype(np.float32),
                cv2.HISTCMP_BHATTACHARYYA,
            )
            if d < min_dist[i]:
                min_dist[i] = d
        min_dist[next_idx] = -1.0

    return selected
