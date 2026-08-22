"""Emit the locked canonical evaluation split as a fixed file (requirement A3).

Consumes:  data/formulations.csv, data/curves_long.csv, data/theta_bank.csv
Produces:  data/canonical_split_v1.csv  (curve_id, split_name, fold_index)
Runtime:   < 1 s

Why this exists:
  paper_requirements_locked A3 requires a single fixed split file that every
  method reads, so no method can silently re-derive a different split. The
  split logic here is a byte-for-byte replica of the canonical path used by
  scripts/73_diagnostics.py and scripts/72_canonical_benchmark_v2.py:
  sorted intersection of the three tables, then two nested train_test_splits.

Why --seed defaults to 42 (not the repo-wide 0):
  ADR-028 pins the canonical data split at random_state=42, and every reported
  number was produced with it. ADR-031 moved the canonical *script* to v2 but
  explicitly kept this split. Pass --seed only to inspect alternative splits.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def build_split(seed: int) -> pd.DataFrame:
    formulations = pd.read_csv("data/formulations.csv")
    curves = pd.read_csv("data/curves_long.csv")
    theta_df = pd.read_csv("data/theta_bank.csv")

    common_ids = sorted(
        set(formulations["curve_id"])
        & set(curves["curve_id"])
        & set(theta_df["curve_id"])
    )

    idx_all = np.arange(len(common_ids))
    train_cal_idx, test_idx = train_test_split(idx_all, test_size=0.25, random_state=seed)
    train_idx, cal_idx = train_test_split(train_cal_idx, test_size=0.2, random_state=seed)

    split_name = np.empty(len(common_ids), dtype=object)
    split_name[train_idx] = "train"
    split_name[cal_idx] = "cal"
    split_name[test_idx] = "test"

    return pd.DataFrame(
        {
            "curve_id": [common_ids[i] for i in idx_all],
            "split_name": split_name,
            "fold_index": idx_all,
        }
    )


def main(seed: int) -> None:
    df = build_split(seed)
    out_path = "data/canonical_split_v1.csv"
    df.to_csv(out_path, index=False)
    counts = df["split_name"].value_counts().to_dict()
    print(f"Wrote {out_path}: n={len(df)} "
          f"(train={counts.get('train', 0)}, "
          f"cal={counts.get('cal', 0)}, "
          f"test={counts.get('test', 0)}) at seed={seed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42,
                        help="split seed; 42 reproduces the ADR-028 canonical split")
    main(parser.parse_args().seed)
