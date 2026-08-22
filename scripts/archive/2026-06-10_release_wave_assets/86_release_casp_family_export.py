"""
86 - Export low-rank CASP/FIB families into the shared posterior-family schema.

Purpose:
    Move the shared posterior-family object beyond point-estimate exports by
    reusing the existing FIB-CASP and Ensemble-CASP constructions at runtime.
    This script emits the actual low-rank `(mu, U, sigma)` family for each test
    curve in a chosen benchmark cell.

Important honesty rule:
    This exports the raw family construction from FIB or ensemble disagreement.
    It does NOT claim post-hoc conformal calibration or solved cross-mechanism
    uncertainty. It is a structural bridge from existing UQ machinery into the
    shared object schema.

Produces:
    outputs/86_release_casp_family_export/<mode>/<dataset>/<scheme>/
        *.json
        summary.csv
        summary.txt
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from casp import ensemble_posterior, fib_posterior, per_tree_predictions
from release_posterior_family import ReleasePosteriorFamily


def _load_script(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_bench62 = _load_script(Path(__file__).resolve().parent / "62_fib_casp_benchmark.py", "_bench62_86")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("fib", "ensemble"), required=True)
    ap.add_argument("--dataset", choices=("cross321", "internal181", "liposome"), required=True)
    ap.add_argument("--scheme", choices=("random_5fold", "group_by_drug", "group_by_polymer"), required=True)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--alpha-fib", type=float, default=1e-2)
    ap.add_argument("--sigma0-frac", type=float, default=0.05)
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outroot", type=Path, default=Path("outputs/86_release_casp_family_export"))
    return ap.parse_args()


def _load_bundle(dataset: str):
    loaders = {
        "cross321": _bench62.load_cross321,
        "internal181": _bench62.load_internal181,
        "liposome": _bench62.load_liposome,
    }
    if dataset not in loaders:
        raise ValueError(dataset)
    return loaders[dataset]()


def _standardize_train_test(bundle, tr: np.ndarray, te: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_form = bundle.n_form
    x_tr = bundle.X[tr].copy()
    x_te = bundle.X[te].copy()
    mu_f = x_tr[:, :n_form].mean(axis=0, keepdims=True)
    sd_f = x_tr[:, :n_form].std(axis=0, keepdims=True).clip(min=1e-6)
    x_tr[:, :n_form] = (x_tr[:, :n_form] - mu_f) / sd_f
    x_te[:, :n_form] = (x_te[:, :n_form] - mu_f) / sd_f
    return x_tr, x_te


def _curve_id(dataset: str, fid: int) -> str:
    if dataset == "liposome":
        return f"liposome_{int(fid):03d}"
    return f"{dataset}_{int(fid)}"


def _assay_context(bundle) -> dict[str, object]:
    if bundle.name == "liposome":
        return {
            "time_unit": "hours",
            "t_max": float(bundle.t_max),
            "release_object": "cumulative_fraction",
            "canonical_group_names": {"group_by_drug": "API_name", "group_by_polymer": "release_method"},
        }
    return {
        "time_unit": "days",
        "t_max": float(bundle.t_max),
        "release_object": "cumulative_fraction",
        "canonical_group_names": {"group_by_drug": "drug", "group_by_polymer": "polymer"},
    }


def _mechanism_id(bundle) -> str:
    sim_name = type(bundle.simulator).__name__
    if sim_name == "PLGABiphasic":
        return "plga_biphasic"
    if sim_name == "WeibullSimulator":
        return "liposome_weibull" if bundle.name == "liposome" else "weibull"
    return str(bundle.name)


def _family_from_fib(bundle, theta_hat: np.ndarray, t_obs: np.ndarray, rank: int, alpha_fib: float, sigma0_frac: float) -> dict[str, object]:
    sim = bundle.simulator
    prior = sim.prior().base_dist
    prior_low = prior.low.float()
    prior_high = prior.high.float()
    theta_t = torch.tensor(theta_hat, dtype=torch.float32)
    t_t = torch.tensor(t_obs, dtype=torch.float32)
    q_fib = fib_posterior(
        simulator=sim,
        theta_hat=theta_t,
        t=t_t,
        sigma_obs=bundle.sigma_obs,
        rank=rank,
        alpha=alpha_fib,
        sigma_0_frac=sigma0_frac,
        prior_low=prior_low,
        prior_high=prior_high,
    )
    return {
        "z_center": q_fib["mu"].cpu().numpy().astype(float),
        "z_basis": q_fib["U"].T.cpu().numpy().astype(float),
        "z_scale": q_fib["sigma"].cpu().numpy().astype(float),
        "extra": {
            "inactive_rank": int(q_fib["U_inactive"].shape[1]),
            "sigma_inactive_scalar": float(q_fib["sigma_inactive_scalar"].cpu().item()),
            "eigvals_full": q_fib["eigvals_full"].cpu().numpy().astype(float).tolist(),
        },
    }


def _family_from_ensemble(bundle, theta_trees: np.ndarray, rank: int) -> dict[str, object]:
    prior = bundle.simulator.prior().base_dist
    tgt = ensemble_posterior(
        theta_per_tree=theta_trees,
        rank=rank,
        prior_low=prior.low.numpy(),
        prior_high=prior.high.numpy(),
    )
    return {
        "z_center": tgt.mu.astype(float),
        "z_basis": tgt.U.T.astype(float),
        "z_scale": tgt.sigma_active.astype(float),
        "extra": {
            "n_trees_used": int(tgt.n_trees_used),
            "spectrum": tgt.spectrum.astype(float).tolist(),
        },
    }


def main() -> None:
    args = parse_args()
    if args.mode == "ensemble" and args.dataset == "liposome":
        raise ValueError("ensemble mode is currently scoped to PLGA zero-shot cells only")

    bundle = _load_bundle(args.dataset)
    if args.mode == "ensemble":
        bundle.X = bundle.X[:, :bundle.n_form].copy()

    splits = _bench62.make_splits(
        args.scheme,
        len(bundle.X),
        bundle.groups_drug,
        bundle.groups_polymer,
        n_folds=args.n_folds,
        seed=args.seed,
    )

    outdir = args.outroot / args.mode / args.dataset / args.scheme
    outdir.mkdir(parents=True, exist_ok=True)
    assay_context = _assay_context(bundle)
    summary_rows: list[dict[str, object]] = []
    summary_lines = [f"=== 86 -- {args.mode} family export ===", ""]

    for fold_idx, (tr, te) in enumerate(splits):
        x_tr, x_te = _standardize_train_test(bundle, tr, te)
        rf = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=None,
            n_jobs=-1,
            random_state=args.seed + fold_idx,
        )
        rf.fit(x_tr, bundle.theta_oracle[tr])

        if args.mode == "fib":
            theta_pred_raw = rf.predict(x_te).astype(np.float32)
            prior = bundle.simulator.prior().base_dist
            lo_np = prior.low.numpy()
            hi_np = prior.high.numpy()
            eps_box = 1e-4 * (hi_np - lo_np)
            theta_pred = np.clip(theta_pred_raw, lo_np + eps_box, hi_np - eps_box)
        else:
            tree_preds = per_tree_predictions(rf, x_te)
            prior = bundle.simulator.prior().base_dist
            lo_np = prior.low.numpy()
            hi_np = prior.high.numpy()
            eps_box = 1e-4 * (hi_np - lo_np)
            tree_preds = np.clip(tree_preds, lo_np + eps_box, hi_np - eps_box)

        for local_i, idx in enumerate(te):
            curve = bundle.curves[idx]
            t_obs = curve.t_obs.astype(np.float64)
            q_obs = curve.q_obs.astype(np.float64)
            mask = (t_obs > 0.0) & (t_obs <= bundle.t_max)
            if mask.sum() < 2:
                continue
            t_eval = t_obs[mask]

            if args.mode == "fib":
                fam = _family_from_fib(
                    bundle=bundle,
                    theta_hat=theta_pred[local_i],
                    t_obs=t_eval,
                    rank=args.rank,
                    alpha_fib=args.alpha_fib,
                    sigma0_frac=args.sigma0_frac,
                )
                metadata_extra = {
                    "family_constructor": "fib_posterior",
                    "sigma_obs": float(bundle.sigma_obs),
                    "alpha_fib": float(args.alpha_fib),
                    "sigma0_frac": float(args.sigma0_frac),
                }
            else:
                fam = _family_from_ensemble(
                    bundle=bundle,
                    theta_trees=tree_preds[local_i],
                    rank=args.rank,
                )
                metadata_extra = {
                    "family_constructor": "ensemble_posterior",
                    "n_estimators": int(args.n_estimators),
                }

            curve_id = _curve_id(args.dataset, int(bundle.fids[idx]))
            family = ReleasePosteriorFamily.low_rank_family(
                mechanism_id=_mechanism_id(bundle),
                decoder_handle=f"{type(bundle.simulator).__name__}.simulate_numpy",
                z_center=fam["z_center"],
                z_basis=fam["z_basis"],
                z_scale=fam["z_scale"],
                assay_context=assay_context,
                observation_context={
                    "fold": int(fold_idx),
                    "scheme": args.scheme,
                    "family_kind": args.mode,
                    "observed_timepoints_used": int(mask.sum()),
                },
                metadata={
                    "curve_id": curve_id,
                    "fid": int(bundle.fids[idx]),
                    "dataset": bundle.name,
                    "source_script": "86_release_casp_family_export.py",
                    "latent_coordinates": "theta_like_kinetic_state",
                    **metadata_extra,
                    **fam["extra"],
                },
            )
            family.to_json(outdir / f"{curve_id}.json")
            item = family.summary_dict()
            item["curve_id"] = curve_id
            item["fid"] = int(bundle.fids[idx])
            item["fold"] = int(fold_idx)
            summary_rows.append(item)
            summary_lines.extend(
                [
                    f"{curve_id}:",
                    f"  mechanism_id  : {item['mechanism_id']}",
                    f"  decoder_handle: {item['decoder_handle']}",
                    f"  latent_dim    : {item['latent_dim']}",
                    f"  rank          : {item['rank']}",
                    "",
                ]
            )

    pd.DataFrame(summary_rows).to_csv(outdir / "summary.csv", index=False)
    (outdir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
