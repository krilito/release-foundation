from __future__ import annotations

"""
130_audit_internal181_vs_raw181.py

Consume:
- data/curves_long.csv
- data/external/bannigan_lai_181/Dataset_17_feat.tsv

Produce:
- outputs/130_internal181_vs_raw181_audit/per_curve_summary.csv
- outputs/130_internal181_vs_raw181_audit/per_anchor_audit.csv
- outputs/130_internal181_vs_raw181_audit/transformation_summary.csv
- outputs/130_internal181_vs_raw181_audit/reconstructed_internal181.csv
- outputs/130_internal181_vs_raw181_audit/summary.md

Expected runtime:
- < 10 s
"""

import argparse
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import pandas as pd


ANCHOR_SPECS = (
    (0.25, "T=0.25"),
    (0.50, "T=0.5"),
    (1.00, "T=1.0"),
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root,
        help="repository root",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=repo_root / "outputs" / "130_internal181_vs_raw181_audit",
        help="output directory",
    )
    return parser.parse_args()


def round_half_up_2(value: float) -> float:
    return float(
        Decimal(str(float(value))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    )


def load_internal_curves(repo_root: Path) -> pd.DataFrame:
    curves = pd.read_csv(repo_root / "data" / "curves_long.csv").copy()
    curves["curve_id"] = curves["curve_id"].astype(str)
    curves["time"] = pd.to_numeric(curves["time"], errors="coerce")
    curves["release"] = pd.to_numeric(curves["release"], errors="coerce")
    curves = curves.dropna(subset=["curve_id", "time", "release"]).copy()
    return curves.sort_values(["curve_id", "time", "release"]).reset_index(drop=True)


def load_raw_curves(repo_root: Path) -> pd.DataFrame:
    raw = pd.read_excel(
        repo_root / "data" / "external" / "bannigan_lai_181" / "Dataset_17_feat.tsv",
        sheet_name="Sheet1",
    ).copy()
    raw["Experimental_index"] = raw["Experimental_index"].astype(str)
    numeric_cols = ["Time", "Release", "T=0.25", "T=0.5", "T=1.0"]
    for col in numeric_cols:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw.dropna(subset=["Experimental_index", "Time", "Release"]).copy()
    return raw


def build_reconstructed_internal(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reconstructed_rows: list[dict[str, float | str | bool]] = []
    per_curve_rows: list[dict[str, int | str | bool]] = []
    per_anchor_rows: list[dict[str, int | str | float | bool]] = []

    for curve_id, group in raw.groupby("Experimental_index", sort=False):
        rounded_raw = group[["Time", "Release"]].copy()
        rounded_raw["rounded_time"] = rounded_raw["Time"].map(round_half_up_2)
        rounded_raw["rounded_release"] = rounded_raw["Release"].map(round_half_up_2)
        rounded_raw["source"] = "raw_rounded"
        rounded_raw = rounded_raw.drop_duplicates(subset=["rounded_time"], keep="first")
        anchor_append_rows: list[dict[str, float | str]] = []

        rounded_time_set = set(rounded_raw["rounded_time"].tolist())
        raw_duplicate_drop_count = int(len(group) - len(rounded_raw))

        inserted_anchor_count = 0
        collision_count = 0
        for anchor_time, anchor_col in ANCHOR_SPECS:
            raw_has_exact_anchor = bool((group["Time"] == anchor_time).any())
            raw_has_round_collision = bool(
                group["Time"].map(round_half_up_2).eq(anchor_time).any()
            )
            anchor_release_raw = float(group.iloc[0][anchor_col])
            anchor_release_rounded = round_half_up_2(anchor_release_raw)
            anchor_inserted = anchor_time not in rounded_time_set

            if anchor_inserted:
                anchor_append_rows.append(
                    {
                        "Time": anchor_time,
                        "Release": anchor_release_raw,
                        "rounded_time": anchor_time,
                        "rounded_release": anchor_release_rounded,
                        "source": "anchor_inserted",
                    }
                )
                rounded_time_set.add(anchor_time)
                inserted_anchor_count += 1
            else:
                collision_count += 1

            per_anchor_rows.append(
                {
                    "curve_id": curve_id,
                    "anchor_time": anchor_time,
                    "anchor_column": anchor_col,
                    "anchor_release_raw": anchor_release_raw,
                    "anchor_release_rounded": anchor_release_rounded,
                    "raw_has_exact_anchor_time": raw_has_exact_anchor,
                    "raw_has_round_collision": raw_has_round_collision,
                    "anchor_inserted": anchor_inserted,
                }
            )

        if anchor_append_rows:
            rounded_raw = pd.concat(
                [rounded_raw, pd.DataFrame(anchor_append_rows)],
                ignore_index=True,
            )
        rounded_raw = rounded_raw.sort_values(
            ["rounded_time", "source", "Time", "Release"]
        ).reset_index(drop=True)
        for _, row in rounded_raw.iterrows():
            reconstructed_rows.append(
                {
                    "curve_id": curve_id,
                    "time": float(row["rounded_time"]),
                    "release": float(row["rounded_release"]),
                    "source": str(row["source"]),
                }
            )

        per_curve_rows.append(
            {
                "curve_id": curve_id,
                "n_raw_points": int(len(group)),
                "n_after_rounding_and_dedup": int(len(rounded_raw) - inserted_anchor_count),
                "raw_duplicate_drop_count": raw_duplicate_drop_count,
                "anchor_collision_count": collision_count,
                "inserted_anchor_count": inserted_anchor_count,
                "expected_n_internal_points": int(len(rounded_raw)),
            }
        )

    reconstructed = pd.DataFrame(reconstructed_rows).sort_values(
        ["curve_id", "time", "release", "source"]
    )
    per_curve = pd.DataFrame(per_curve_rows).sort_values("curve_id")
    per_anchor = pd.DataFrame(per_anchor_rows).sort_values(["curve_id", "anchor_time"])
    return reconstructed.reset_index(drop=True), per_curve.reset_index(drop=True), per_anchor.reset_index(drop=True)


def compare_internal_to_reconstruction(
    internal: pd.DataFrame,
    reconstructed: pd.DataFrame,
    per_curve: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    internal_cmp = internal[["curve_id", "time", "release"]].copy()
    internal_cmp["time"] = internal_cmp["time"].map(round_half_up_2)
    internal_cmp["release"] = internal_cmp["release"].map(round_half_up_2)
    internal_cmp = internal_cmp.sort_values(["curve_id", "time", "release"]).reset_index(drop=True)

    reconstructed_cmp = reconstructed[["curve_id", "time", "release"]].copy()
    reconstructed_cmp = reconstructed_cmp.sort_values(["curve_id", "time", "release"]).reset_index(drop=True)

    merged = internal_cmp.merge(
        reconstructed_cmp,
        on=["curve_id", "time", "release"],
        how="outer",
        indicator=True,
    )

    curve_match_rows: list[dict[str, str | int | bool]] = []
    for curve_id, group in internal_cmp.groupby("curve_id", sort=False):
        predicted = reconstructed_cmp[reconstructed_cmp["curve_id"] == curve_id].reset_index(drop=True)
        truth = group.reset_index(drop=True)
        curve_match_rows.append(
            {
                "curve_id": curve_id,
                "n_internal_points": int(len(truth)),
                "n_reconstructed_points": int(len(predicted)),
                "curve_matches_reconstruction": bool(predicted.equals(truth)),
            }
        )

    curve_matches = pd.DataFrame(curve_match_rows).sort_values("curve_id").reset_index(drop=True)
    per_curve = per_curve.merge(curve_matches, on="curve_id", how="left")
    return per_curve, merged


def build_transformation_summary(
    raw: pd.DataFrame,
    internal: pd.DataFrame,
    per_curve: pd.DataFrame,
    per_anchor: pd.DataFrame,
    comparison_merge: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "n_curves": int(raw["Experimental_index"].nunique()),
                "raw_point_count": int(len(raw)),
                "internal_point_count": int(len(internal)),
                "net_added_points": int(len(internal) - len(raw)),
                "total_anchor_slots": int(raw["Experimental_index"].nunique() * len(ANCHOR_SPECS)),
                "anchor_inserted_count": int(per_anchor["anchor_inserted"].sum()),
                "anchor_collision_count": int((~per_anchor["anchor_inserted"]).sum()),
                "raw_duplicate_drop_count": int(per_curve["raw_duplicate_drop_count"].sum()),
                "curves_matching_reconstruction": int(per_curve["curve_matches_reconstruction"].sum()),
                "all_curves_match_reconstruction": bool(per_curve["curve_matches_reconstruction"].all()),
                "outer_merge_left_only_rows": int((comparison_merge["_merge"] == "left_only").sum()),
                "outer_merge_right_only_rows": int((comparison_merge["_merge"] == "right_only").sum()),
            }
        ]
    )


def write_summary(
    per_curve: pd.DataFrame,
    per_anchor: pd.DataFrame,
    transformation_summary: pd.DataFrame,
    out_path: Path,
) -> None:
    ts = transformation_summary.iloc[0]
    anchor_stats = (
        per_anchor.groupby("anchor_time", as_index=False)
        .agg(
            inserted_anchor_count=("anchor_inserted", "sum"),
            raw_round_collision_count=("raw_has_round_collision", "sum"),
            raw_exact_anchor_count=("raw_has_exact_anchor_time", "sum"),
        )
        .sort_values("anchor_time")
    )

    lines = [
        "# internal181_vs_raw181 audit",
        "",
        "This audit reconstructs the current `data/curves_long.csv` internal181 corpus",
        "from the raw Bannigan/Lai 181-source spreadsheet snapshot.",
        "",
        "## Recovered transformation",
        "",
        "1. Round raw `Time` and `Release` to 2 decimals using Excel-style half-up rounding.",
        "2. If multiple raw rows collapse onto the same rounded time, keep the first occurrence.",
        "3. Inject exact anchor points at `0.25`, `0.50`, and `1.00` day using rounded",
        "   `T=0.25`, `T=0.5`, and `T=1.0` values, but only when that rounded time is absent.",
        "",
        "## Overall counts",
        "",
        f"- raw curves: `{int(ts['n_curves'])}`",
        f"- raw points: `{int(ts['raw_point_count'])}`",
        f"- internal points: `{int(ts['internal_point_count'])}`",
        f"- net added points: `{int(ts['net_added_points'])}`",
        f"- total anchor slots: `{int(ts['total_anchor_slots'])}`",
        f"- inserted anchor points: `{int(ts['anchor_inserted_count'])}`",
        f"- skipped anchors due to rounded-time collisions: `{int(ts['anchor_collision_count'])}`",
        f"- raw duplicate points removed after rounding: `{int(ts['raw_duplicate_drop_count'])}`",
        f"- curves exactly matched by the recovered transformation: `{int(ts['curves_matching_reconstruction'])}` / `{int(ts['n_curves'])}`",
        "",
        "## Anchor-level breakdown",
        "",
    ]

    for _, row in anchor_stats.iterrows():
        lines.extend(
            [
                f"- `{row['anchor_time']:.2f}` day:",
                f"  inserted `{int(row['inserted_anchor_count'])}` times, "
                f"rounded-time collisions `{int(row['raw_round_collision_count'])}`, "
                f"raw exact anchor already present `{int(row['raw_exact_anchor_count'])}`",
            ]
        )

    delta_counts = per_curve["n_internal_points"].sub(per_curve["n_raw_points"]).value_counts().sort_index()
    lines.extend(
        [
            "",
            "## Per-curve net point deltas",
            "",
        ]
    )
    for delta, count in delta_counts.items():
        lines.append(f"- delta `{int(delta)}`: `{int(count)}` curves")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    internal = load_internal_curves(args.repo_root)
    raw = load_raw_curves(args.repo_root)
    reconstructed, per_curve, per_anchor = build_reconstructed_internal(raw)
    per_curve, comparison_merge = compare_internal_to_reconstruction(
        internal=internal,
        reconstructed=reconstructed,
        per_curve=per_curve,
    )
    transformation_summary = build_transformation_summary(
        raw=raw,
        internal=internal,
        per_curve=per_curve,
        per_anchor=per_anchor,
        comparison_merge=comparison_merge,
    )

    reconstructed.to_csv(args.outdir / "reconstructed_internal181.csv", index=False)
    per_curve.to_csv(args.outdir / "per_curve_summary.csv", index=False)
    per_anchor.to_csv(args.outdir / "per_anchor_audit.csv", index=False)
    transformation_summary.to_csv(args.outdir / "transformation_summary.csv", index=False)
    write_summary(
        per_curve=per_curve,
        per_anchor=per_anchor,
        transformation_summary=transformation_summary,
        out_path=args.outdir / "summary.md",
    )

    print(f"[internal181-vs-raw181] wrote outputs to {args.outdir}")
    print(
        "[internal181-vs-raw181] "
        f"raw_points={len(raw)} internal_points={len(internal)} "
        f"net_added={len(internal) - len(raw)} "
        f"all_curves_match={bool(transformation_summary.iloc[0]['all_curves_match_reconstruction'])}"
    )


if __name__ == "__main__":
    main()
