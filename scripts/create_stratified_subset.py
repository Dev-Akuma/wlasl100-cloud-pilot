"""Create deterministic, stratified video indices for a smaller cloud pilot."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def select_files(
    dataset_root: Path, fraction: float, seed: int
) -> list[dict[str, str]]:
    """Select videos within every split/class directory."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be greater than 0 and at most 1")

    selected: list[dict[str, str]] = []
    for split_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            videos = sorted(class_dir.glob("*.mp4"))
            if not videos:
                continue
            count = max(1, round(len(videos) * fraction))
            rng = random.Random(f"{seed}:{split_dir.name}:{class_dir.name}")
            for video in sorted(rng.sample(videos, min(count, len(videos)))):
                selected.append(
                    {
                        "split": split_dir.name,
                        "label": class_dir.name,
                        "video_id": video.stem,
                        "relative_path": video.relative_to(dataset_root).as_posix(),
                    }
                )
    return selected


def write_outputs(rows: list[dict[str, str]], output_dir: Path, metadata: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["split", "label", "video_id", "relative_path"]
    with (output_dir / "subset_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "subset_index.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "manifest.json").write_text(
        json.dumps({**metadata, "records": len(rows)}, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        parser.error(f"dataset root does not exist: {dataset_root}")
    rows = select_files(dataset_root, args.fraction, args.seed)
    write_outputs(
        rows,
        args.output_dir,
        {
            "dataset_root_name": dataset_root.name,
            "fraction": args.fraction,
            "seed": args.seed,
            "stratification": "split,class",
        },
    )
    print(f"Wrote {len(rows)} selected videos to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
