"""Run the minimal Phase 2 command-line forecasting pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 2 feature building and Ray forecasting.")
    parser.add_argument("--skip-feature-build", action="store_true", help="Reuse an existing feature table.")
    parser.add_argument("--num-cpus", type=int, default=4, help="CPU count to pass to Ray.")
    return parser.parse_args()


def run_step(command: list[str]) -> None:
    print("\n$", " ".join(command))
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    python = sys.executable
    if not args.skip_feature_build:
        run_step([python, str(SCRIPT_DIR / "build_feature_table.py")])
    run_step([python, str(SCRIPT_DIR / "run_ray_forecasting.py"), "--num-cpus", str(args.num_cpus)])
    print("\nPhase 2 command-line forecasting pipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
