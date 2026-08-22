"""69c - Wrapper for the current best paper-facing AO v3 candidate.

Purpose:
    The strongest current AO branch is evidence-backed but currently lives only
    as scattered CLI flags and output directories. This wrapper provides a
    stable rerun entrypoint without changing `69_active_observer_v3.py`
    defaults.

Candidate configuration locked here:
    - likelihood_beta = 1.0
    - step2_q_max_cap_margin = 0.0
    - q_max_correction = 0.15
    - conformal_method = local

Current evidence anchor:
    outputs_active_observer_v3_beta1_step2nocap_qcorr015/metrics_summary.csv

Run:
    python scripts/69c_active_observer_v3_paper_candidate.py
    python scripts/69c_active_observer_v3_paper_candidate.py --dry-run
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


SCRIPT_69 = Path("scripts/69_active_observer_v3.py")
DEFAULT_OUTPUT_DIR = "outputs_active_observer_v3_beta1_step2nocap_qcorr015"


def build_command(output_dir: str, extra_args: list[str]) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT_69),
        "--output-dir",
        output_dir,
        "--likelihood-beta",
        "1.0",
        "--step2-q-max-cap-margin",
        "0.0",
        "--q-max-correction",
        "0.15",
        "--conformal-method",
        "local",
        *extra_args,
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "passthrough",
        nargs="*",
        help="additional flags forwarded to 69_active_observer_v3.py",
    )
    args = parser.parse_args()

    cmd = build_command(args.output_dir, args.passthrough)
    print("[69c] paper-facing AO candidate wrapper")
    print("[69c] command:")
    print(" ".join(cmd))
    if args.dry_run:
        return

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
