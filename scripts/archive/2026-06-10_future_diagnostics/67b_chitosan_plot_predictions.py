"""67b - Plot the prospective predictions for visual inspection.

Reads outputs/67_chitosan_prospective/predictions.csv and produces a
6-panel plot (one per formulation), each showing DEX and GCV predicted
curves with 90% PI bands.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    in_csv = Path("outputs/67_chitosan_prospective/predictions.csv")
    out_png = Path("outputs/67_chitosan_prospective/curves_plot.png")
    df = pd.read_csv(in_csv)
    forms = sorted(df["formulation"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    colors = {"DEX": "tab:blue", "GCV": "tab:orange"}
    for ax, f in zip(axes, forms):
        sub_all = df[df["formulation"] == f]
        for drug in ("DEX", "GCV"):
            sub = sub_all[sub_all["drug"] == drug].sort_values("t_hours")
            t = sub["t_hours"].to_numpy()
            yp = sub["Q_predicted_point"].to_numpy()
            lo = sub["Q_lo90"].to_numpy()
            hi = sub["Q_hi90"].to_numpy()
            ax.plot(t, yp, color=colors[drug], lw=2, label=f"{drug} predicted")
            ax.fill_between(t, lo, hi, color=colors[drug], alpha=0.18,
                            label=f"{drug} 90% PI")
        meta = sub_all.iloc[0]
        title = (f"{f}: HA={meta['HA_band']} ({meta['HA_MW_MDa']} MDa), "
                 f"PVP={'+' + str(int(meta['PVP_MW_kDa'])) + 'kDa' if meta['PVP'] else 'none'}")
        ax.set_title(title, fontsize=10)
        ax.set_xscale("log")
        ax.set_xlim(0.5, 700)
        ax.set_ylim(0, 1.05)
        ax.axhline(1.0, color="gray", lw=0.5, linestyle=":")
        ax.set_xlabel("time (h)")
        ax.set_ylabel("Q (cumulative release)")
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("YC Chitosan — Locked Prospective Predictions (FIB-CASP / Ritger-Peppas)\n"
                 "Compare experimental Q(t) against these bands when data arrives.",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"[67b] wrote {out_png}", flush=True)


if __name__ == "__main__":
    main()
