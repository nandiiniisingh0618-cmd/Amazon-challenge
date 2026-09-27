"""
run_all.py - End-to-end entry point.

Running this script from the project root:
    python code/business_entity_resolution/src/run_all.py

Executes both training and prediction in sequence, producing:
    output/candidate_pairs.tsv
    output/matching_results.tsv
"""

import subprocess
import sys
from pathlib import Path

SRC_DIR = Path(__file__).parent


def run_step(script_name: str):
    script_path = SRC_DIR / script_name
    print(f"\n{'='*60}")
    print(f"Running: {script_path}")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, str(script_path)],
        check=True,
    )
    return result


if __name__ == "__main__":
    run_step("train.py")
    run_step("predict.py")
    print("\n✓ All steps complete. Check output/ for results.")
