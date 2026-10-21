"""Run a configurable 3x3 pilot matrix and record its results."""

from __future__ import annotations

import argparse
import csv
import itertools
import shlex
import subprocess
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", required=True, help="Command template; use {learning_rate}, {dropout}, and {run_id}.")
    parser.add_argument("--learning-rates", nargs="+", default=["0.001", "0.0001", "0.00001"])
    parser.add_argument("--dropouts", nargs="+", default=["0.2", "0.4", "0.6"])
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, default=Path.cwd())
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_id, (learning_rate, dropout) in enumerate(
        itertools.product(args.learning_rates, args.dropouts), start=1
    ):
        command = args.command.format(
            learning_rate=learning_rate, dropout=dropout, run_id=run_id
        )
        started = time.time()
        completed = None
        error = ""
        if not args.dry_run:
            completed = subprocess.run(
                shlex.split(command), cwd=args.workdir, check=False
            )
            if completed.returncode:
                error = f"exit code {completed.returncode}"
        rows.append(
            {
                "run_id": run_id,
                "learning_rate": learning_rate,
                "dropout": dropout,
                "command": command,
                "status": "planned" if args.dry_run else ("ok" if not error else "failed"),
                "seconds": round(time.time() - started, 3),
                "error": error,
            }
        )
        if error:
            print(f"Run {run_id} failed: {error}")
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} pilot rows to {args.output_csv.resolve()}")
    if any(row["status"] == "failed" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
