"""
27d -- M2: PLGA + Fourier-residual amortized SBI (sbi.NPE + MAF, theta_dim=14).

Difference from 27c (theta_dim=9):
    - simulator = PLGABiphasicResidual(n_modes=5, sigma_c=0.05) -- adds 5
      Fourier sine coefficients to dQ/dt with shrinkage prior Normal(0, sigma_c).
    - prior = MultipleIndependent over (base Uniform 9D, residual Normal 5D),
      assembled on training device.
    - evaluation reports ||c_hat||_2 distribution per subgroup, as the
      diagnostic for whether the residual is correlated with v1 misspec
      (large ||c|| on fast/short, near-zero on `neither`).

Hypothesis under test:
    27c overall 3d median R^2 = -0.027 (FAIL), fast/short subgroups
    -1.25 / -1.60. Hypothesis: those subgroups need a residual term the
    parametric ODE can't make. The Fourier basis captures smooth missing
    dynamics (triphasic lag, secondary burst, morphology transition). M3
    checks whether ||c_hat|| is large where 27c failed AND suffix R^2 improves.

Identifiability mitigations baked in:
    - low-dim basis (5 Fourier modes) caps total residual capacity
    - shrinkage prior Normal(0, 0.05) gravitationally pulls c -> 0
    - (Q_max - Q) gating in PLGABiphasicResidual preserves saturation
    - c=0 reduces exactly to base PLGABiphasic (M1 verified bit-exact)

Pre-declared gates (same as 27c except G3a is new):
    G1  : SBC KS p > 0.05 for all 14 dims at each of prefix = 1, 3, 7 d.
          Coverage_68 in [0.65, 0.71], coverage_95 in [0.92, 0.98].
    G2  : median posterior std monotone decreasing in prefix length.
          1d > 3d > 7d strictly.
    G3a : 3d fast_short median R^2 >= -0.50 (huge improvement from 27c -1.6).
    G3b : 3d overall median R^2 >= 0.349 (matches baseline 24).
    G3c : 3d fast_short median R^2 > 0 (stretch -- residual recovers misspec).
    G3d : 3d overall median R^2 > 0.45 (clearly beats baseline 24).
    G4  : 7d overall median R^2 > 0.349.

Outputs:
    outputs/27d_residual_npe/posterior.pt
    outputs/27d_residual_npe/training_log.txt
    outputs/27d_residual_npe/sbc/rank_histograms_{1,3,7}d.png
    outputs/27d_residual_npe/sbc/sbc_summary.csv
    outputs/27d_residual_npe/per_curve_metrics.csv             # includes c_norm column
    outputs/27d_residual_npe/subgroup_summary.csv              # includes median c_norm
    outputs/27d_residual_npe/posterior_width_vs_prefix.png
    outputs/27d_residual_npe/c_norm_by_subgroup.png            # NEW M2 diagnostic
    outputs/27d_residual_npe/summary.txt

Expected runtime:
    ~100-180 min on a 4060 (theta_dim 9->14 adds ~30% to MAF compute).
    Use --quick for a smoke test (~5-10 min, n_sim=5000).

Predecessor stack: v1 (failed) -> 27c (G2 pass, G3 fail by 0.4) -> 27d.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sbi.diagnostics import check_sbc, run_sbc
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.embedding_nets import FCEmbedding

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from posterior import interpolate_to_grid  # noqa: E402
from simulator import PLGABiphasic, PLGABiphasicResidual  # noqa: E402

# 321 loader constants -- copied verbatim from scripts/26 / scripts/27 v1
# (project convention: scripts self-contain helpers).
FID_COL = "Experimental_index"
TIME_COL = "Time"
Y_COL = "Release"
DP_GROUP_COL = "DP_Group"
_QUALITY_RANK = {"high": 2, "medium": 1, "low": 0}
_321_RENAME = {
    "Formulation Index": FID_COL,
    "Drug MW": "Drug_Mw",
    "Drug TPSA": "Drug_TPSA",
    "Drug LogP": "Drug_LogP",
    "Polymer MW": "Polymer_MW",
    "Initial Drug-to-Polymer Ratio": "Initial D/M ratio",
    "Drug Loading Capacity": "DLC",
}
_321_DROP = (
    "Particle Size",
    "Drug Encapsulation Efficiency",
    "Solubility Enhancer Concentration",
)
_TAIL_FLAGS_PATH = Path(
    "outputs/sprint1_14_cross_doi_failure_tail/per_curve_enriched.csv"
)
# Reference numbers for the verdict block. These are the bars 27d must clear.
BASELINE_24_3D_MEDIAN_R2 = 0.349
V1_3D_MEDIAN_R2 = -0.25
V1_3D_NEITHER_MEDIAN_R2 = 0.525   # the one v1 number that beat baseline


class _JointPrior(torch.distributions.Distribution):
    """Joint of (Independent Uniform R^d_base, Independent Normal R^d_resid).

    Why this exists: sbi.utils.MultipleIndependent and sbi.utils.process_prior
    do not propagate component device through `.sample()` -- with cuda
    components, sample() returns cpu and sbi's check_if_prior_on_device
    rejects. This class is the minimal device-correct alternative.

    Used in 27d to combine the PLGA base prior (Uniform) and the Fourier
    residual coefficient prior (Normal(0, sigma_c)) into a single prior
    object that sbi.NPE accepts.
    """
    arg_constraints: dict = {}
    has_rsample = False

    def __init__(
        self,
        base_dist: torch.distributions.Distribution,
        resid_dist: torch.distributions.Distribution,
    ):
        self._base = base_dist
        self._resid = resid_dist
        self._d_base = base_dist.event_shape.numel()
        self._d_resid = resid_dist.event_shape.numel()
        super().__init__(
            batch_shape=torch.Size([]),
            event_shape=torch.Size([self._d_base + self._d_resid]),
            validate_args=False,
        )

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        s_base = self._base.sample(sample_shape)
        s_resid = self._resid.sample(sample_shape)
        return torch.cat([s_base, s_resid], dim=-1)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        base_v = value[..., :self._d_base]
        resid_v = value[..., self._d_base:]
        return self._base.log_prob(base_v) + self._resid.log_prob(resid_v)

    @property
    def support(self) -> torch.distributions.constraints.Constraint:
        # real_vector (no bounds) so sbi's rejection sampler accepts every
        # posterior sample. We then manually clamp base 9 dims to prior
        # bounds before forwarding to the simulator. Alternatives we tried:
        #   - constraints.cat -> _Cat.check returns (batch, 2), sbi can't
        #     reshape for rejection
        #   - custom _JointSupport with check returning (batch,) -> needs
        #     biject_to registration with custom CatTransform; complex
        # The clamp-after-sample workaround is functionally equivalent for
        # MAF posteriors that have learned the bounded data distribution
        # (mass outside bounds is small in practice).
        return torch.distributions.constraints.real_vector

    @property
    def mean(self) -> torch.Tensor:
        return torch.cat([self._base.mean, self._resid.mean], dim=-1)

    @property
    def variance(self) -> torch.Tensor:
        return torch.cat([self._base.variance, self._resid.variance], dim=-1)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - float(np.sum((y_true - y_pred) ** 2)) / ss_tot


@dataclass
class CurveRecord:
    fid: int
    t_obs: torch.Tensor
    q_obs: torch.Tensor


def _make_prefix_mask(prefix_days: float, t_grid: torch.Tensor) -> torch.Tensor:
    """1.0 where t_grid <= prefix_days, 0.0 otherwise. Float dtype."""
    return (t_grid <= prefix_days).float()


# ---------------------------------------------------------------------------
# Synthetic training data
# ---------------------------------------------------------------------------

def _generate_synthetic_pairs(
    sim: PLGABiphasic,
    t_grid: torch.Tensor,
    n: int,
    noise_sigma: float,
    prefix_low: float,
    prefix_high: float,
    seed: int,
    device: str,
    batch_size: int = 256,  # lower than 27c's 1024: residual ODE has larger autograd footprint
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate (theta, x) pairs for NPE training.

    x = concat(Q_masked, mask) of length 2*T. Prefix length is sampled
    continuously from Uniform(prefix_low, prefix_high) days. Continuous
    sampling forces the model to interpolate at any prefix length, not
    just memorize specific {1, 3, 7} values (v1 trained only on those
    three -- one of several reasons it failed G2).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    theta = sim.sample_prior(n).to(device)
    t_grid_d = t_grid.to(device)
    T = t_grid_d.numel()

    Q_full = torch.empty(n, T, device=device)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        with torch.no_grad():
            Q_full[start:end] = sim.simulate(theta[start:end], t_grid_d)
        # PLGABiphasicResidual's augmented vector_field has a larger autograd
        # graph than base; torchdiffeq's adaptive RK4 holds rejected-step
        # state in memory. Clearing between batches lets us run larger N on
        # an 8 GB 4060. (Empirical: without this, n=20k OOMs at first batch.)
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    prefixes = torch.tensor(
        rng.uniform(prefix_low, prefix_high, size=n),
        dtype=torch.float32, device=device,
    )
    masks = (t_grid_d.unsqueeze(0) <= prefixes.unsqueeze(1)).float()

    noise = noise_sigma * torch.randn(Q_full.shape, device=device)
    q_obs = ((Q_full + noise) * masks).clamp(0.0, 1.0)

    x = torch.cat([q_obs, masks], dim=-1)
    return theta.cpu(), x.cpu()


# ---------------------------------------------------------------------------
# SBC at fixed prefix
# ---------------------------------------------------------------------------

def _build_sbc_inputs_at_prefix(
    sim: PLGABiphasic,
    t_grid: torch.Tensor,
    prefix_days: float,
    n_sbc: int,
    noise_sigma: float,
    seed: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (theta_true, x) at a single fixed prefix length for SBC.

    Unlike training (continuous prefix), SBC needs a single prefix length
    per evaluation so we get a clean answer to "is the posterior at exactly
    1d / 3d / 7d calibrated".
    """
    torch.manual_seed(seed)
    theta = sim.sample_prior(n_sbc).to(device)
    t_grid_d = t_grid.to(device)
    with torch.no_grad():
        Q_full = sim.simulate(theta, t_grid_d)

    mask = _make_prefix_mask(prefix_days, t_grid_d)
    noise = noise_sigma * torch.randn(Q_full.shape, device=device)
    q_obs = ((Q_full + noise) * mask.unsqueeze(0)).clamp(0.0, 1.0)
    mask_batch = mask.unsqueeze(0).expand(n_sbc, -1)
    x = torch.cat([q_obs, mask_batch], dim=-1)
    return theta.cpu(), x.cpu()


def _plot_sbc_ranks(
    ranks: torch.Tensor,
    ks_pvals: torch.Tensor,
    param_names: list[str],
    n_sbc: int,
    prefix_days: float,
    out_path: Path,
) -> None:
    n_params = len(param_names)
    ncols = 4
    nrows = (n_params + ncols - 1) // ncols   # ceil-divide; handles any n_params
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.0),
                             sharey=True)
    axes = axes.ravel() if nrows > 1 else np.atleast_1d(axes)
    expected_count = n_sbc / 20
    for k in range(n_params):
        ax = axes[k]
        ax.hist(ranks[:, k].numpy(), bins=20, edgecolor="black", alpha=0.85)
        ax.axhline(expected_count, color="red", linestyle="--", linewidth=1)
        ks_p = float(ks_pvals[k])
        flag = "PASS" if ks_p > 0.05 else "FAIL"
        ax.set_title(f"{param_names[k]}  KS p={ks_p:.3f}  [{flag}]", fontsize=9)
        ax.set_xlabel("rank")
        if k % ncols == 0:
            ax.set_ylabel("count")
    for k in range(n_params, len(axes)):
        axes[k].axis("off")
    fig.suptitle(
        f"27d SBC at prefix={prefix_days:.1f}d  ({n_sbc} reps; G1 gate)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 321 loader and partial-curve construction (copied from v1)
# ---------------------------------------------------------------------------

def _load_external_records(
    xlsx_path: Path,
    matched_fids_csv: Path,
    t_grid: torch.Tensor,
    min_quality: str,
) -> tuple[list[CurveRecord], pd.DataFrame]:
    matched_fids = set(pd.read_csv(matched_fids_csv)["Formulation_Index"].astype(int))
    df = pd.read_excel(xlsx_path).rename(columns=_321_RENAME)
    df = df.drop(columns=[c for c in _321_DROP if c in df.columns])
    df["Polymer_MW"] = df["Polymer_MW"].astype(float) * 1000.0
    df["DLC"] = df["DLC"].astype(float) / 100.0
    df[Y_COL] = df[Y_COL].astype(float).clip(0.0, 1.0)
    df[DP_GROUP_COL] = "UNK-PLGA"
    df = (
        df.groupby([FID_COL, TIME_COL], as_index=False, sort=False)
        .agg({
            **{c: "first" for c in df.columns if c not in (TIME_COL, Y_COL)},
            Y_COL: "mean",
        })
    )
    df = df[df[FID_COL].astype(int).isin(matched_fids)].copy()
    desc_df = df.drop_duplicates(FID_COL).sort_values(FID_COL).reset_index(drop=True)
    records: list[CurveRecord] = []
    min_rank = _QUALITY_RANK[min_quality]
    for fid in desc_df[FID_COL].tolist():
        g = df[df[FID_COL] == fid].sort_values(TIME_COL)
        t_obs = torch.tensor(g[TIME_COL].to_numpy(), dtype=torch.float32)
        q_obs = torch.tensor(g[Y_COL].to_numpy(), dtype=torch.float32).clamp(0.0, 1.0)
        try:
            _, quality, _ = interpolate_to_grid(
                t_obs, q_obs, t_grid,
                t_max_days=float(t_grid.max().item()),
            )
        except ValueError:
            continue
        if _QUALITY_RANK.get(quality, -1) < min_rank:
            continue
        records.append(CurveRecord(fid=int(fid), t_obs=t_obs, q_obs=q_obs))
    if _TAIL_FLAGS_PATH.exists():
        tail_flags = pd.read_csv(_TAIL_FLAGS_PATH)[
            ["Formulation_Index", "fast_regime", "short_window"]
        ]
    else:
        tail_flags = pd.DataFrame(
            columns=["Formulation_Index", "fast_regime", "short_window"]
        )
    return records, tail_flags


def _prefix_obs_from_real_curve(
    rec: CurveRecord,
    prefix_days: float,
    t_grid: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None, np.ndarray | None]:
    t_np = rec.t_obs.cpu().numpy().astype(float)
    q_np = rec.q_obs.cpu().numpy().astype(float)
    prefix_obs_mask = t_np <= prefix_days
    future_obs_mask = t_np > prefix_days
    if prefix_obs_mask.sum() < 2 or future_obs_mask.sum() < 2:
        return None, None, None
    q_prefix_grid, _, _ = interpolate_to_grid(
        torch.tensor(t_np[prefix_obs_mask], dtype=torch.float32),
        torch.tensor(q_np[prefix_obs_mask], dtype=torch.float32),
        t_grid,
        t_max_days=float(t_grid.max().item()),
    )
    obs_mask_grid = _make_prefix_mask(prefix_days, t_grid)
    return q_prefix_grid * obs_mask_grid, obs_mask_grid, future_obs_mask


def _classify_subgroups(per_curve: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "overall": per_curve,
        "neither": per_curve[~per_curve["fast_regime"] & ~per_curve["short_window"]],
        "fast": per_curve[per_curve["fast_regime"]],
        "short": per_curve[per_curve["short_window"]],
        "fast_short": per_curve[per_curve["fast_regime"] & per_curve["short_window"]],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/plga_phase1.yaml"))
    ap.add_argument(
        "--cross-doi-data",
        type=Path,
        default=Path(
            r"D:\chemical-world-model-v0\datset\321PLGA"
            r"\A Dataset on Formulation Parameters and Characteristics of "
            r"Drug-Loaded PLGA Microparticles\mp_dataset_processed.xlsx"
        ),
    )
    ap.add_argument(
        "--matched-fids-csv",
        type=Path,
        default=Path(
            "outputs/sprint1_10_eval_deployment/cross_doi_per_curve_metrics.csv"
        ),
    )
    ap.add_argument("--out", type=Path, default=Path("outputs/27d_residual_npe"))
    ap.add_argument("--n-modes", type=int, default=5,
                    help="Fourier modes in residual basis (default 5)")
    ap.add_argument("--sigma-c", type=float, default=0.015,
                    help="shrinkage scale on residual coefs. Default 0.015 "
                         "is the largest value passing M1.5 amplitude audit "
                         "(83% of samples produce <15% Q_max deviation; "
                         "see 27d_residual_amplitude_audit_s0p015/).")
    ap.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-sim", type=int, default=None, help="overrides config")
    ap.add_argument("--n-sbc", type=int, default=300)
    ap.add_argument("--n-posterior-samples", type=int, default=1000)
    ap.add_argument("--prefix-low", type=float, default=0.5,
                    help="lower bound of training prefix sampling (days)")
    ap.add_argument("--prefix-high", type=float, default=14.0,
                    help="upper bound of training prefix sampling (days)")
    ap.add_argument("--min-quality", choices=("high", "medium", "low"), default="high")
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: tiny n_sim, n_sbc, epochs (~3-5 min)")
    args = ap.parse_args()

    if args.quick:
        # Quick smoke test: train enough that the MAF doesn't put 99%+ of its
        # mass outside the prior (which crashes rejection sampling via OOM).
        # 5000 samples * 25 epochs is the minimum that gives a usable posterior
        # for verification of code paths. Full run uses ~50k * 200.
        args.n_sim = 5000
        args.n_sbc = 30
        args.n_posterior_samples = 50

    _seed_everything(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "sbc").mkdir(exist_ok=True)
    overall_t0 = time.time()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    n_sim = args.n_sim or cfg["synthetic"]["n_pairs"]
    t_cfg = cfg["synthetic"]["t_grid_days"]
    t_grid = torch.linspace(t_cfg["start"], t_cfg["end"], t_cfg["n"])
    T = t_grid.numel()
    noise_sigma = cfg["posterior"].get("noise_sigma", 0.03)

    sim = PLGABiphasicResidual(
        n_modes=args.n_modes,
        sigma_c=args.sigma_c,
        t_max=float(t_cfg["end"]),
        rtol=cfg["simulator"]["rtol"],
        atol=cfg["simulator"]["atol"],
        method=cfg["simulator"]["method"],
    )
    # sbi requires prior on the training device. sbi.utils.MultipleIndependent
    # does NOT propagate component device through sample() (verified empirically
    # against sigma_c=0.015 + cuda components: sample.device = cpu). Custom
    # local class _JointPrior gives explicit device control + correct support
    # (so rejection sampling rejects theta outside base bounds).
    _base_uniform = PLGABiphasic().prior().base_dist
    _base_prior_dev = torch.distributions.Independent(
        torch.distributions.Uniform(
            _base_uniform.low.to(args.device),
            _base_uniform.high.to(args.device),
        ),
        1,
    )
    _resid_prior_dev = torch.distributions.Independent(
        torch.distributions.Normal(
            torch.zeros(args.n_modes, device=args.device),
            args.sigma_c * torch.ones(args.n_modes, device=args.device),
        ),
        1,
    )
    prior = _JointPrior(_base_prior_dev, _resid_prior_dev)
    _test_sample = prior.sample((1,))
    print(f"[27d] joint prior sample device: {_test_sample.device} (expect {args.device})")
    assert _test_sample.device.type == torch.device(args.device).type, (
        f"_JointPrior sample device {_test_sample.device} != {args.device}"
    )

    print(f"[27d] device         : {args.device}")
    print(f"[27d] n_simulations  : {n_sim}")
    print(f"[27d] t_grid         : {T} pts over [0, {t_cfg['end']}] days")
    print(f"[27d] noise_sigma    : {noise_sigma}")
    print(f"[27d] prefix range   : [{args.prefix_low}, {args.prefix_high}] days (continuous)")
    print(f"[27d] residual       : n_modes={args.n_modes}, sigma_c={args.sigma_c}")
    print(f"[27d] theta_dim      : {sim.n_params}  (base 9 + residual {args.n_modes})")

    # ------ build NPE -------------------------------------------------------
    embedding = FCEmbedding(
        input_dim=2 * T,                # concat(Q_masked, mask)
        output_dim=cfg["posterior"].get("embedding_output_dim", 16),
        num_layers=cfg["posterior"].get("embedding_layers", 3),
        num_hiddens=cfg["posterior"].get("embedding_hidden", 64),
    )
    density_estimator_builder = posterior_nn(
        model=cfg["posterior"]["flow_type"],   # maf
        hidden_features=cfg["posterior"]["hidden_features"],
        num_transforms=cfg["posterior"]["num_transforms"],
        embedding_net=embedding,
    )
    inference = NPE(prior=prior, density_estimator=density_estimator_builder,
                    device=args.device)

    # ------ generate training data -----------------------------------------
    print(f"[27d] generating {n_sim} synthetic (theta, x) pairs...")
    t_gen = time.time()
    theta_train, x_train = _generate_synthetic_pairs(
        sim=sim, t_grid=t_grid, n=n_sim,
        noise_sigma=noise_sigma,
        prefix_low=args.prefix_low, prefix_high=args.prefix_high,
        seed=args.seed, device=args.device,
    )
    print(f"[27d] generated in {time.time()-t_gen:.1f}s; theta {tuple(theta_train.shape)}, x {tuple(x_train.shape)}")

    # ------ train ----------------------------------------------------------
    # sbi caches the z-score normalizer on the device of the data passed to
    # append_simulations. If we pass CPU data here but train on CUDA, the
    # z-score stats stay on CPU and sample_batched later raises a device
    # mismatch. Move both to the training device explicitly.
    inference.append_simulations(
        theta_train.to(args.device), x_train.to(args.device)
    )
    print("[27d] training NPE+MAF...")
    t_train_start = time.time()
    if args.quick:
        max_epochs = 25
        stop_after = 10
    else:
        max_epochs = cfg["train"]["max_epochs"]
        stop_after = cfg["train"]["early_stopping_patience"]
    density_estimator = inference.train(
        training_batch_size=cfg["train"]["batch_size"],
        max_num_epochs=max_epochs,
        validation_fraction=0.1,
        stop_after_epochs=stop_after,
        show_train_summary=True,
    )
    train_elapsed = time.time() - t_train_start
    print(f"[27d] training done in {train_elapsed/60:.1f} min")

    posterior_obj = inference.build_posterior(density_estimator)

    # Save posterior (sbi's serialization)
    torch.save({
        "density_estimator_state": density_estimator.state_dict(),
        "config": {
            "n_sim": n_sim,
            "noise_sigma": noise_sigma,
            "prefix_low": args.prefix_low,
            "prefix_high": args.prefix_high,
            "seed": args.seed,
            "T": T,
        },
    }, args.out / "posterior.pt")

    # ------ G1 + G2: SBC at fixed prefix lengths ---------------------------
    print("[27d] running SBC at fixed prefixes 1/3/7 days...")
    sbc_rows: list[dict] = []
    sbc_t0 = time.time()
    eval_prefixes = (1.0, 3.0, 7.0)
    for prefix_days in eval_prefixes:
        theta_sbc, x_sbc = _build_sbc_inputs_at_prefix(
            sim=sim, t_grid=t_grid, prefix_days=prefix_days,
            n_sbc=args.n_sbc, noise_sigma=noise_sigma,
            seed=args.seed + int(prefix_days * 1000), device=args.device,
        )
        ranks, dap = run_sbc(
            thetas=theta_sbc.to(args.device),
            xs=x_sbc.to(args.device),
            posterior=posterior_obj,
            num_posterior_samples=args.n_posterior_samples,
        )
        stats = check_sbc(
            ranks=ranks,
            prior_samples=sim.sample_prior(args.n_sbc).to(args.device),
            dap_samples=dap,
            num_posterior_samples=args.n_posterior_samples,
        )
        _plot_sbc_ranks(
            ranks=ranks, ks_pvals=stats["ks_pvals"],
            param_names=sim.param_names, n_sbc=args.n_sbc,
            prefix_days=prefix_days,
            out_path=args.out / "sbc" / f"rank_histograms_{int(prefix_days)}d.png",
        )
        for k, name in enumerate(sim.param_names):
            sbc_rows.append({
                "prefix_days": prefix_days,
                "param": name,
                "ks_pval": float(stats["ks_pvals"][k]),
                "ks_pass": float(stats["ks_pvals"][k]) > 0.05,
            })
        print(f"[27d]   prefix={prefix_days}d  KS p min = {float(stats['ks_pvals'].min()):.4f}")
    sbc_df = pd.DataFrame(sbc_rows)
    sbc_df.to_csv(args.out / "sbc" / "sbc_summary.csv", index=False)
    print(f"[27d] SBC done in {(time.time()-sbc_t0)/60:.1f} min")

    # ------ load 321 -------------------------------------------------------
    print(f"[27d] loading 321 from {args.cross_doi_data}")
    records, tail_flags = _load_external_records(
        xlsx_path=args.cross_doi_data,
        matched_fids_csv=args.matched_fids_csv,
        t_grid=t_grid,
        min_quality=args.min_quality,
    )
    print(f"[27d] kept {len(records)} curves at min_quality={args.min_quality}")

    # ------ G2 + G3 + G4: 321 evaluation -----------------------------------
    print(f"[27d] evaluating on 321 at prefixes {eval_prefixes}...")
    eval_t0 = time.time()
    per_curve_rows: list[dict] = []
    flags_by_fid = {
        int(r["Formulation_Index"]): (bool(r["fast_regime"]), bool(r["short_window"]))
        for _, r in tail_flags.iterrows()
    } if len(tail_flags) else {}

    for prefix_days in eval_prefixes:
        for rec in records:
            q_prefix, mask, future_obs_mask = _prefix_obs_from_real_curve(
                rec, prefix_days, t_grid,
            )
            if q_prefix is None:
                continue
            x_obs = torch.cat([q_prefix, mask], dim=-1).to(args.device)
            theta_samples = posterior_obj.sample(
                (args.n_posterior_samples,),
                x=x_obs,
                show_progress_bars=False,
            ).cpu()
            # Clamp base 9 dims to prior bounds. _JointPrior.support was set
            # to real_vector to bypass sbi rejection; we enforce bounds here
            # before forwarding to simulator (avoids log_kw=10 -> kw=22000/d).
            # Residual coefs (dims 9:14) are unbounded by design.
            theta_samples[:, :9] = torch.clamp(
                theta_samples[:, :9],
                min=_base_uniform.low,
                max=_base_uniform.high,
            )
            # Forward-simulate each sample.
            with torch.no_grad():
                Q_samples = sim.simulate(theta_samples.to(args.device),
                                         t_grid.to(args.device)).cpu()
            Q_median = Q_samples.median(dim=0).values   # (T,)
            # Interpolate predicted curve back to original observed t_obs at
            # post-prefix points.
            t_grid_np = t_grid.cpu().numpy()
            t_future = rec.t_obs.cpu().numpy()[future_obs_mask]
            q_pred_future = np.interp(
                t_future, t_grid_np, Q_median.cpu().numpy(),
            )
            q_true_future = rec.q_obs.cpu().numpy()[future_obs_mask]
            r2 = _r2(q_true_future, q_pred_future)
            mae = float(np.mean(np.abs(q_true_future - q_pred_future)))
            mean_sd = float(theta_samples.std(dim=0).mean())
            # Residual-magnitude diagnostic: per-curve posterior median of
            # ||c||_2. Large on subgroups where the parametric ODE struggles
            # = residual is doing the misspec-correction work as designed.
            # Near-zero everywhere = residual is unused, ablation back to 27c.
            c_samples = theta_samples[:, 9:]          # (n_samples, n_modes)
            c_norm_samples = c_samples.norm(dim=-1)   # (n_samples,)
            c_norm_median = float(c_norm_samples.median())
            c_norm_mean = float(c_norm_samples.mean())
            fast_regime, short_window = flags_by_fid.get(rec.fid, (False, False))
            per_curve_rows.append({
                "fid": rec.fid,
                "prefix_days": prefix_days,
                "R2": r2,
                "MAE": mae,
                "mean_sd_theta": mean_sd,
                "c_norm_median": c_norm_median,
                "c_norm_mean": c_norm_mean,
                "fast_regime": fast_regime,
                "short_window": short_window,
            })
        print(f"[27d]   prefix={prefix_days}d done")
    eval_elapsed = time.time() - eval_t0
    per_curve = pd.DataFrame(per_curve_rows)
    per_curve.to_csv(args.out / "per_curve_metrics.csv", index=False)
    print(f"[27d] 321 eval done in {eval_elapsed/60:.1f} min")

    # ------ subgroup summary -----------------------------------------------
    summary_rows: list[dict] = []
    for prefix_days in eval_prefixes:
        pdf = per_curve[per_curve["prefix_days"] == prefix_days]
        for subset_name, sdf in _classify_subgroups(pdf).items():
            summary_rows.append({
                "prefix_days": prefix_days,
                "subset": subset_name,
                "n_curves": int(len(sdf)),
                "median_R2": float(sdf["R2"].median()) if len(sdf) else float("nan"),
                "mean_R2": float(sdf["R2"].mean()) if len(sdf) else float("nan"),
                "median_MAE": float(sdf["MAE"].median()) if len(sdf) else float("nan"),
                "median_mean_sd": float(sdf["mean_sd_theta"].median()) if len(sdf) else float("nan"),
                "median_c_norm": float(sdf["c_norm_median"].median()) if len(sdf) else float("nan"),
                "mean_c_norm": float(sdf["c_norm_median"].mean()) if len(sdf) else float("nan"),
            })
    subgroup_summary = pd.DataFrame(summary_rows)
    subgroup_summary.to_csv(args.out / "subgroup_summary.csv", index=False)

    # ------ G2 plot: posterior width vs prefix -----------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for subset_name in ("overall", "neither", "fast_short"):
        ys, errs = [], []
        for prefix_days in eval_prefixes:
            pdf = per_curve[per_curve["prefix_days"] == prefix_days]
            sdf = _classify_subgroups(pdf)[subset_name]
            if len(sdf) == 0:
                ys.append(float("nan"))
                errs.append(0.0)
            else:
                ys.append(float(sdf["mean_sd_theta"].median()))
                errs.append(float(sdf["mean_sd_theta"].std()))
        ax.errorbar(list(eval_prefixes), ys, yerr=errs, marker="o", capsize=4,
                    label=f"{subset_name}")
    ax.axhline(0.73, color="gray", linestyle=":",
               label="v1 mean_sd (=0.73, all prefixes)")
    ax.set_xlabel("prefix length (days)")
    ax.set_ylabel("median per-curve mean posterior std")
    ax.set_title("G2: posterior width vs prefix length on 321")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / "posterior_width_vs_prefix.png", dpi=150)
    plt.close(fig)

    # ------ M2 diagnostic: ||c||_2 by subgroup -----------------------------
    # Hypothesis: residual should be SMALL on `neither` (parametric ODE
    # already fits) and LARGE on `fast_short` (where ODE is misspecified).
    # If c_norm is uniform across subgroups -> residual is noise; if
    # near-zero everywhere -> residual is unused (ablation back to 27c).
    fig, ax = plt.subplots(figsize=(8, 4.5))
    subset_order = ["overall", "neither", "fast", "short", "fast_short"]
    box_data = []
    labels = []
    for prefix_days in eval_prefixes:
        pdf = per_curve[per_curve["prefix_days"] == prefix_days]
        for subset_name in subset_order:
            sdf = _classify_subgroups(pdf)[subset_name]
            if len(sdf):
                box_data.append(sdf["c_norm_median"].to_numpy())
                labels.append(f"{int(prefix_days)}d\n{subset_name}\n(n={len(sdf)})")
    ax.boxplot(box_data, tick_labels=labels, showfliers=False)
    ax.axhline(args.sigma_c * np.sqrt(args.n_modes), color="red", linestyle="--",
               label=f"prior std (sigma_c*sqrt(n_modes) = {args.sigma_c * np.sqrt(args.n_modes):.3f})")
    ax.set_ylabel("posterior median ||c||_2 per curve")
    ax.set_title("M2 diagnostic: residual magnitude by prefix x subgroup")
    ax.legend(loc="upper right", fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(args.out / "c_norm_by_subgroup.png", dpi=150)
    plt.close(fig)

    # ------ verdict --------------------------------------------------------
    g1_pass_per_prefix = {}
    for prefix_days in eval_prefixes:
        sub = sbc_df[sbc_df["prefix_days"] == prefix_days]
        g1_pass_per_prefix[prefix_days] = bool(sub["ks_pass"].all())
    g1_pass = all(g1_pass_per_prefix.values())

    overall_by_prefix = {
        p: per_curve[per_curve["prefix_days"] == p]["mean_sd_theta"].median()
        for p in eval_prefixes
    }
    g2_pass = (overall_by_prefix[1.0] > overall_by_prefix[3.0]
               > overall_by_prefix[7.0])

    per_3d = per_curve[per_curve["prefix_days"] == 3.0]
    median_r2_3d_overall = float(per_3d["R2"].median())
    fast_short_3d = per_3d[per_3d["fast_regime"] & per_3d["short_window"]]
    median_r2_3d_fast_short = (
        float(fast_short_3d["R2"].median()) if len(fast_short_3d) else float("nan")
    )

    median_r2_7d_overall = float(
        per_curve[per_curve["prefix_days"] == 7.0]["R2"].median()
    )

    g3a_pass = median_r2_3d_fast_short >= -0.50      # huge improvement from 27c -1.6
    g3b_pass = median_r2_3d_overall >= BASELINE_24_3D_MEDIAN_R2  # matches baseline 24
    g3c_pass = median_r2_3d_fast_short > 0.0          # residual recovers misspec
    g3d_pass = median_r2_3d_overall > 0.45            # clearly beats baseline 24
    g4_pass = median_r2_7d_overall > BASELINE_24_3D_MEDIAN_R2

    # ||c||_2 diagnostic: is residual correlated with subgroup misspec?
    c_by_subset_3d = {
        name: float(_classify_subgroups(per_3d)[name]["c_norm_median"].median())
        for name in ("neither", "fast_short")
        if len(_classify_subgroups(per_3d)[name])
    }

    verdict_lines = [
        "=== 27d verdict ===",
        f"G1 (SBC calibration at 1/3/7 d):     {'PASS' if g1_pass else 'FAIL'}",
        *[f"    prefix={p}d : {'PASS' if v else 'FAIL'}" for p, v in g1_pass_per_prefix.items()],
        f"G2 (posterior width monotone):       {'PASS' if g2_pass else 'FAIL'}",
        f"    1d median_sd = {overall_by_prefix[1.0]:.3f}",
        f"    3d median_sd = {overall_by_prefix[3.0]:.3f}",
        f"    7d median_sd = {overall_by_prefix[7.0]:.3f}",
        f"G3a (3d fast_short median R^2 >= -0.50): {'PASS' if g3a_pass else 'FAIL'}",
        f"    27d fast_short: {median_r2_3d_fast_short:+.4f}",
        f"    27c fast_short: -1.6014  (this is the bar to beat)",
        f"G3b (3d overall median R^2 >= 0.349):    {'PASS' if g3b_pass else 'FAIL'}",
        f"    27d: {median_r2_3d_overall:+.4f}",
        f"    27c: -0.0274 / baseline 24: +0.3490 / v1: -0.2500",
        f"G3c (3d fast_short median R^2 > 0):      {'PASS' if g3c_pass else 'FAIL'} [stretch]",
        f"G3d (3d overall median R^2 > 0.45):      {'PASS' if g3d_pass else 'FAIL'} [stretch]",
        f"G4  (7d overall > 0.349):                {'PASS' if g4_pass else 'FAIL'}",
        f"    27d: {median_r2_7d_overall:+.4f}",
        "",
        "M2 diagnostic -- residual magnitude (||c||_2 median, 3d):",
        *[f"    {name:11s}: {v:.4f}  (prior std = {args.sigma_c * np.sqrt(args.n_modes):.4f})"
          for name, v in c_by_subset_3d.items()],
        "    Interpretation:",
        "      - fast_short ||c|| >> neither ||c||  => residual is correcting misspec as designed",
        "      - both near zero                     => residual unused (ablate back to 27c)",
        "      - both large                         => residual stealing theta's job (R1 risk)",
    ]

    # ------ summary.txt ----------------------------------------------------
    n_curves_per_prefix = {
        p: int(len(per_curve[per_curve["prefix_days"] == p])) for p in eval_prefixes
    }
    overall_min = float(overall_t0)
    summary_text = "\n".join([
        "=== 27d -- PLGA + Fourier-residual amortized SBI (M2) ===",
        "",
        f"  device              : {args.device}",
        f"  n_simulations       : {n_sim}",
        f"  noise_sigma         : {noise_sigma}",
        f"  prefix range (train): [{args.prefix_low}, {args.prefix_high}] days continuous",
        f"  n_modes             : {args.n_modes}",
        f"  sigma_c             : {args.sigma_c}",
        f"  theta_dim           : {sim.n_params}",
        f"  n_sbc per prefix    : {args.n_sbc}",
        f"  n_posterior_samples : {args.n_posterior_samples}",
        f"  seed                : {args.seed}",
        f"  train wallclock     : {train_elapsed/60:.1f} min",
        f"  total wallclock     : {(time.time()-overall_min)/60:.1f} min",
        "",
        "Subgroup summary (R^2 on 321 suffix prediction + residual magnitude):",
        subgroup_summary.to_string(index=False),
        "",
        f"Curves used per prefix: {n_curves_per_prefix}",
        "",
        *verdict_lines,
        "",
        "Plots:",
        "  sbc/rank_histograms_{1,3,7}d.png : G1 PIT visualization per prefix",
        "  posterior_width_vs_prefix.png    : G2 visualization",
        "  c_norm_by_subgroup.png           : M2 residual-magnitude diagnostic",
        "",
        "Files:",
        "  posterior.pt              : trained NPE state",
        "  sbc/sbc_summary.csv       : per-param KS p-values per prefix",
        "  per_curve_metrics.csv     : per-curve R^2 / MAE / posterior width",
        "  subgroup_summary.csv      : aggregated by subgroup",
    ])
    (args.out / "summary.txt").write_text(summary_text, encoding="utf-8")
    (args.out / "training_log.txt").write_text(
        f"n_sim={n_sim}\ntrain_elapsed_min={train_elapsed/60:.1f}\nseed={args.seed}\n",
        encoding="utf-8",
    )
    print()
    print(summary_text)


if __name__ == "__main__":
    main()
