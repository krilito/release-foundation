"""
112 - Plot the external methods and venues strategy panel.

Purpose:
    Turn the refreshed external paper/journal inventory into a compact figure
    that shows:
    1. how the field is actually operating now, and
    2. which journal bands matter most for a release-intelligence program.

Consumes:
    docs/drug_release_external_paper_journal_inventory_2026-05-29.csv

Produces:
    outputs/92_release_strategy_figures/external_methods_venues/external_methods_venues_panel.png
    outputs/92_release_strategy_figures/external_methods_venues/external_methods_venues_panel.pdf
    outputs/92_release_strategy_figures/external_methods_venues/summary.txt
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch


INFILE = Path("docs/drug_release_external_paper_journal_inventory_2026-05-29.csv")
OUTDIR = Path("outputs/92_release_strategy_figures/external_methods_venues")


STYLE_ORDER = [
    "supervised prediction",
    "early-observation forecasting",
    "workflow / assay design",
    "optimization / platform / review",
]

STYLE_MAP = {
    "Bannigan et al.": "early-observation forecasting",
    "Machine learning in drug delivery": "optimization / platform / review",
    "ML integrated with in vitro experiments for PLGA nanoparticles": "supervised prediction",
    "Utilizing ML for predicting drug release from polymeric DDS": "optimization / platform / review",
    "ML workflow for liposome IVR tests": "workflow / assay design",
    "GA optimization of tree models for liposome release": "supervised prediction",
    "ML predicts tablet release profiles and kinetic parameters": "supervised prediction",
    "Polymer microparticles challenges and role of ML": "optimization / platform / review",
    "AI and ML guided optimization in drug delivery": "optimization / platform / review",
    "Interpretable two-stage ML for PLGA microspheres": "early-observation forecasting",
}

ROLE_COLOR = {
    "fight": "#C03A2B",
    "fight-adjacent": "#D97706",
    "borrow": "#2E8B57",
    "borrow-adjacent": "#0F766E",
    "defend": "#6D28D9",
}

SHORT_NAME = {
    "Bannigan et al.": "NC / Bannigan",
    "Machine learning in drug delivery": "JCR review",
    "ML integrated with in vitro experiments for PLGA nanoparticles": "PLGA nano ML",
    "Utilizing ML for predicting drug release from polymeric DDS": "CMPB review",
    "ML workflow for liposome IVR tests": "Liposome IVR",
    "GA optimization of tree models for liposome release": "Liposome release",
    "ML predicts tablet release profiles and kinetic parameters": "Tablet release",
    "Polymer microparticles challenges and role of ML": "IJP review",
    "AI and ML guided optimization in drug delivery": "ADDR opt review",
    "Interpretable two-stage ML for PLGA microspheres": "PLGA 2-stage",
}

CORE_VENUES = [
    ("Journal of Controlled Release", "flagship delivery / release framing"),
    ("International Journal of Pharmaceutics", "best broad pharmaceutics home"),
    ("Molecular Pharmaceutics", "mechanistic / object-level fit"),
    ("Drug Delivery and Translational Research", "translational delivery lane"),
    ("Eur. J. Pharm. Biopharm.", "biopharmaceutics / release systems"),
    ("J. Drug Deliv. Sci. Technol.", "practical delivery technology lane"),
]

ADJACENT_VENUES = [
    ("Scientific Reports", "concrete adjacent model papers"),
    ("Nanoscale", "nanocarrier release prediction pressure"),
    ("The AAPS Journal", "oral formulation + release profiles"),
    ("Digital Discovery", "workflow / IVR design pressure"),
    ("Advanced Drug Delivery Reviews", "optimization and field language"),
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_methods(ax: plt.Axes, df: pd.DataFrame) -> None:
    y_map = {name: idx for idx, name in enumerate(STYLE_ORDER)}
    x_offsets = {
        "fight": -0.10,
        "fight-adjacent": -0.04,
        "borrow": 0.02,
        "borrow-adjacent": 0.08,
        "defend": 0.14,
    }
    for _, row in df.iterrows():
        style = STYLE_MAP[row["work"]]
        x = float(row["year"]) + x_offsets.get(row["strategic_role"], 0.0)
        y = y_map[style]
        color = ROLE_COLOR[row["strategic_role"]]
        ax.scatter(x, y, s=150, color=color, edgecolor="black", linewidth=0.7, alpha=0.92)
        ax.text(x + 0.03, y + 0.05, SHORT_NAME[row["work"]], fontsize=8.5, ha="left", va="bottom")

    ax.set_yticks(range(len(STYLE_ORDER)))
    ax.set_yticklabels(
        [
            "Supervised\nprediction",
            "Early-observation\nforecasting",
            "Workflow /\nassay design",
            "Optimization /\nplatform / review",
        ],
        fontsize=9,
    )
    ax.set_xticks([2023, 2024, 2025, 2026])
    ax.set_xlim(2022.7, 2026.45)
    ax.set_title("A. How the current external field is actually operating", loc="left", fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Year")

    legend_items = [
        ("fight", "Fight"),
        ("fight-adjacent", "Fight-adjacent"),
        ("borrow", "Borrow"),
        ("borrow-adjacent", "Borrow-adjacent"),
        ("defend", "Defend"),
    ]
    for idx, (key, label) in enumerate(legend_items):
        ax.scatter(2022.83 + 0.58 * idx, -0.58, s=90, color=ROLE_COLOR[key], edgecolor="black", linewidth=0.6)
        ax.text(2022.90 + 0.58 * idx, -0.58, label, va="center", ha="left", fontsize=8.2)


def add_box(ax: plt.Axes, x: float, y: float, w: float, h: float, title: str, rows: list[tuple[str, str]], face: str) -> None:
    rect = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.014,rounding_size=0.02",
        linewidth=1.0,
        edgecolor="#333333",
        facecolor=face,
    )
    ax.add_patch(rect)
    ax.text(x + 0.02, y + h - 0.06, title, fontsize=11, fontweight="bold", ha="left", va="top")
    cursor = y + h - 0.14
    for name, desc in rows:
        ax.text(x + 0.02, cursor, name, fontsize=9.2, fontweight="bold", ha="left", va="top")
        ax.text(x + 0.02, cursor - 0.045, desc, fontsize=8.4, ha="left", va="top")
        cursor -= 0.12


def plot_venues(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("B. Venue bands that currently matter most", loc="left", fontweight="bold")
    add_box(
        ax,
        0.02,
        0.08,
        0.46,
        0.82,
        "Core non-pure-OA release venues",
        CORE_VENUES,
        "#EAF3FB",
    )
    add_box(
        ax,
        0.52,
        0.08,
        0.46,
        0.82,
        "Adjacent active venues / pressure lanes",
        ADJACENT_VENUES,
        "#F8F0E8",
    )


def write_summary(out_png: Path, out_pdf: Path) -> None:
    lines = [
        "=== 112 -- external methods and venues panel ===",
        "",
        f"PNG: {out_png.as_posix()}",
        f"PDF: {out_pdf.as_posix()}",
        "",
        "Panels",
        "  A. Representative external papers arranged by current operating style",
        "  B. Core vs adjacent venue bands for release-intelligence positioning",
        "",
        "Topline",
        "  The field remains split across direct prediction, early-observation forecasting, workflow systems, and optimization/platform narratives.",
        "  Our open slot remains the shared release-intelligence stack rather than the best single regressor.",
    ]
    (OUTDIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dir(OUTDIR)
    df = pd.read_csv(INFILE)
    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(15, 12),
        gridspec_kw={"height_ratios": [1.0, 1.18], "hspace": 0.24},
    )
    plot_methods(axes[0], df)
    plot_venues(axes[1])
    fig.suptitle(
        "External ML + drug-release field | current methods and venue bands",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )

    out_png = OUTDIR / "external_methods_venues_panel.png"
    out_pdf = OUTDIR / "external_methods_venues_panel.pdf"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    write_summary(out_png, out_pdf)


if __name__ == "__main__":
    main()
