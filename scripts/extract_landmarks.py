"""Extract fixed-length hand landmarks from a subset index."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def sample_indices(frame_count: int, output_frames: int) -> np.ndarray:
    if frame_count <= 0:
        return np.zeros(output_frames, dtype=np.int64)
    return np.linspace(0, frame_count - 1, output_frames).round().astype(np.int64)


def extract_video(
    video_path: Path, detector: vision.HolisticLandmarker, output_frames: int
) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    target_indices = sample_indices(frame_count, output_frames)
    keypoints = np.zeros((output_frames, 126), dtype=np.float32)
    for output_index, frame_index in enumerate(target_indices):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(image)
        values = []
        for hand in (
            getattr(result, "left_hand_landmarks", None),
            getattr(result, "right_hand_landmarks", None),
        ):
            if not hand:
                continue
            values.extend(coordinate for landmark in hand for coordinate in (landmark.x, landmark.y, landmark.z))
        keypoints[output_index, : len(values)] = values[:126]
    capture.release()
    return keypoints


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error("--frames must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_options = python.BaseOptions(model_asset_path=str(args.model))
    options = vision.HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
    )
    failures = []
    processed = 0
    with vision.HolisticLandmarker.create_from_options(options) as detector:
        with args.index.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                destination = args.output_dir / row["split"] / row["label"] / f"{row['video_id']}.npz"
                if destination.exists() and not args.overwrite:
                    continue
                try:
                    keypoints = extract_video(
                        args.dataset_root / row["relative_path"], detector, args.frames
                    )
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        destination,
                        keypoints=keypoints,
                        label=row["label"],
                        video_id=row["video_id"],
                        split=row["split"],
                    )
                    processed += 1
                except (OSError, RuntimeError, ValueError) as error:
                    failures.append({"video_id": row["video_id"], "error": str(error)})
    if failures:
        failure_path = args.output_dir / "extraction_failures.csv"
        with failure_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["video_id", "error"])
            writer.writeheader()
            writer.writerows(failures)
        raise SystemExit(f"{len(failures)} videos failed; see {failure_path}")
    print(f"Extracted {processed} videos to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
