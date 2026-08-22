"""81 - Release-corpus CNP/LNN few-shot transfer probe.

Purpose:
    Use the 751-curve clean cumulative release corpus as a few-shot functional
    prediction benchmark:

        metadata + early context points -> future release function

    This is a probe, not a foundation-model or physical-state claim. Hidden
    states are learned release representations.

Consumes:
    outputs/148_release_main_cumulative_v1/curves_long.csv
    outputs/148_release_main_cumulative_v1/formulations.csv
    optionally outputs/158_freeze_theta_targets/theta_table_v1.parquet
    optionally outputs/161_hierarchical_early_obs_adapter/per_curve_metrics.csv

Produces:
    outputs/81_release_corpus_cnp_lnn_probe/summary_by_system.csv
    outputs/81_release_corpus_cnp_lnn_probe/summary_by_budget.csv
    outputs/81_release_corpus_cnp_lnn_probe/summary_by_time_budget.csv
    outputs/81_release_corpus_cnp_lnn_probe/per_curve_metrics.csv
    outputs/81_release_corpus_cnp_lnn_probe/budget_curve.csv
    outputs/81_release_corpus_cnp_lnn_probe/pairwise_bootstrap.csv
    outputs/81_release_corpus_cnp_lnn_probe/baseline_delta_summary.csv
    outputs/81_release_corpus_cnp_lnn_probe/predictions_long.csv
    outputs/81_release_corpus_cnp_lnn_probe/split_manifest.csv
    outputs/81_release_corpus_cnp_lnn_probe/ablation_summary.csv
    outputs/81_release_corpus_cnp_lnn_probe/lock_metadata.json
    outputs/81_release_corpus_cnp_lnn_probe/summary.md

Smoke:
    .\\.venv\\Scripts\\python.exe scripts\\81_release_corpus_cnp_lnn_probe.py `
        --max-curves-per-system 3 --epochs 5 --ensemble-size 1
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shape_baseline_utils import FAMILY_FUNCS, SAFE_NUMERIC_COLS, safe_r2  # noqa: E402


DEFAULT_POOL_DIR = Path("outputs/148_release_main_cumulative_v1")
DEFAULT_THETA_DIR = Path("outputs/158_freeze_theta_targets")
DEFAULT_REF161_DIR = Path("outputs/161_hierarchical_early_obs_adapter")
DEFAULT_OUT = Path("outputs/81_release_corpus_cnp_lnn_probe")
BUDGETS = (0, 1, 2, 3, 5)
LARGE_SYSTEMS = {"PLGA", "liposome"}


@dataclass(frozen=True)
class CurveRecord:
    curve_id: str
    system_id: str
    source_dataset: str
    polymer_family: str
    payload_name: str
    numeric: dict[str, float]
    categorical: dict[str, str]
    time_days: np.ndarray
    release_fraction: np.ndarray


@dataclass
class Preprocessor:
    numeric_cols: list[str]
    categorical_cols: list[str]
    mean: np.ndarray
    std: np.ndarray
    medians: dict[str, float]
    vocabs: dict[str, dict[str, int]]

    def transform_numeric(self, curves: list[CurveRecord]) -> np.ndarray:
        rows = []
        for curve in curves:
            vals = []
            for col in self.numeric_cols:
                value = float(curve.numeric.get(col, self.medians[col]))
                vals.append(value if np.isfinite(value) else self.medians[col])
            rows.append(vals)
        arr = np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, len(self.numeric_cols)), dtype=np.float32)
        return (arr - self.mean) / self.std

    def transform_categorical(self, curves: list[CurveRecord]) -> np.ndarray:
        rows = []
        for curve in curves:
            vals = []
            for col in self.categorical_cols:
                vocab = self.vocabs[col]
                vals.append(vocab.get(curve.categorical.get(col, "__MISSING__"), vocab["__UNK__"]))
            rows.append(vals)
        return np.asarray(rows, dtype=np.int64) if rows else np.zeros((0, len(self.categorical_cols)), dtype=np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Release-corpus CNP/LNN few-shot transfer probe.")
    parser.add_argument("--pool-dir", type=Path, default=DEFAULT_POOL_DIR)
    parser.add_argument("--theta-dir", type=Path, default=DEFAULT_THETA_DIR)
    parser.add_argument("--ref161-dir", type=Path, default=DEFAULT_REF161_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--steps-per-epoch", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--metadata-mode", choices=["safe", "all", "none"], default="safe")
    parser.add_argument("--decoder-mode", choices=["lnn", "mlp"], default="lnn")
    parser.add_argument("--time-budgets-days", default="0.25,1,3,7,14")
    parser.add_argument("--max-curves-per-system", type=int, default=0)
    parser.add_argument("--within-holdout-frac", type=float, default=0.2)
    parser.add_argument(
        "--only-heldout-systems",
        default="",
        help="Comma-separated LOSO heldout systems to run. Default runs every system.",
    )
    parser.add_argument(
        "--skip-within-large",
        action="store_true",
        help="Skip the PLGA/liposome within-large-system heldout splits.",
    )
    parser.add_argument(
        "--skip-loso-system",
        action="store_true",
        help="Skip LOSO-system splits and run only the requested within-large-system splits.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_git_hash() -> str:
    try:
        return subprocess.run(
            ["git", "-c", "safe.directory=D:/release-foundation", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def stable_int_hash(text: str) -> int:
    out = 0
    for idx, ch in enumerate(text):
        out += (idx + 1) * ord(ch)
    return out


def parse_time_budgets(text: str) -> tuple[float, ...]:
    if not str(text).strip():
        return ()
    out = []
    for part in str(text).split(","):
        value = float(part.strip())
        if value < 0:
            raise ValueError(f"time budget must be non-negative, got {value}")
        out.append(value)
    return tuple(sorted(set(out)))


def parse_system_filter(text: str, available_systems: list[str]) -> list[str]:
    if not str(text).strip():
        return list(available_systems)
    available = set(available_systems)
    requested = [part.strip() for part in str(text).split(",") if part.strip()]
    unknown = sorted(set(requested) - available)
    if unknown:
        raise ValueError(f"Unknown heldout system(s): {unknown}. Available systems: {available_systems}")
    return [system for system in available_systems if system in set(requested)]


def map_system_id(source_dataset: str, polymer_family: str) -> str:
    if source_dataset in {"internal181", "cross321"}:
        return "PLGA"
    return str(polymer_family) if str(polymer_family).strip() else str(source_dataset)


def clean_category(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "__MISSING__"
    text = str(value).strip()
    return text if text else "__MISSING__"


def categorical_columns_for_mode(metadata_mode: str) -> list[str]:
    if metadata_mode == "all":
        return ["system_id", "source_dataset", "polymer_family", "payload_name"]
    if metadata_mode == "safe":
        return ["polymer_family", "payload_name"]
    if metadata_mode == "none":
        return []
    raise ValueError(f"Unknown metadata_mode={metadata_mode}")


def load_curve_records(pool_dir: Path, max_curves_per_system: int, seed: int) -> list[CurveRecord]:
    curves = pd.read_csv(pool_dir / "curves_long.csv")
    forms = pd.read_csv(pool_dir / "formulations.csv")
    required_curves = {"unified_curve_id", "time_days", "release_fraction", "source_dataset"}
    required_forms = {"unified_curve_id", "source_dataset", "polymer_family", "payload_name"}
    missing_curves = required_curves - set(curves.columns)
    missing_forms = required_forms - set(forms.columns)
    if missing_curves or missing_forms:
        raise ValueError(f"Missing required columns: curves={sorted(missing_curves)}, forms={sorted(missing_forms)}")

    curves = curves.copy()
    curves["time_days"] = pd.to_numeric(curves["time_days"], errors="coerce")
    curves["release_fraction"] = pd.to_numeric(curves["release_fraction"], errors="coerce")
    curves = curves.dropna(subset=["unified_curve_id", "time_days", "release_fraction"]).copy()
    curves = curves.sort_values(["unified_curve_id", "time_days"]).reset_index(drop=True)

    forms = forms.drop_duplicates("unified_curve_id").copy()
    forms["system_id"] = [
        map_system_id(str(ds), str(poly))
        for ds, poly in zip(forms["source_dataset"], forms["polymer_family"])
    ]
    form_map = forms.set_index("unified_curve_id")
    numeric_cols = [col for col in SAFE_NUMERIC_COLS if col in forms.columns]
    categorical_cols = ["system_id", "source_dataset", "polymer_family", "payload_name"]

    records: list[CurveRecord] = []
    for curve_id, sub in curves.groupby("unified_curve_id", sort=True):
        if curve_id not in form_map.index:
            continue
        t = sub["time_days"].to_numpy(dtype=float)
        y_raw = sub["release_fraction"].to_numpy(dtype=float)
        mask = np.isfinite(t) & np.isfinite(y_raw)
        t = t[mask]
        y_raw = y_raw[mask]
        if len(t) < 3:
            continue
        order = np.argsort(t)
        t = t[order]
        y_raw = y_raw[order]
        if len(np.unique(t)) < len(t):
            unique = pd.DataFrame({"time_days": t, "release_fraction": y_raw}).groupby("time_days", as_index=False).mean()
            t = unique["time_days"].to_numpy(dtype=float)
            y_raw = unique["release_fraction"].to_numpy(dtype=float)
        if len(t) < 3:
            continue
        # Preserve raw file values in source files; only clipped model input is kept in memory.
        y = np.clip(y_raw, 0.0, 1.2)
        row = form_map.loc[curve_id]
        numeric = {
            col: float(pd.to_numeric(row.get(col, np.nan), errors="coerce"))
            for col in numeric_cols
        }
        categorical = {col: clean_category(row.get(col, "__MISSING__")) for col in categorical_cols}
        records.append(
            CurveRecord(
                curve_id=str(curve_id),
                system_id=str(row["system_id"]),
                source_dataset=str(row["source_dataset"]),
                polymer_family=str(row["polymer_family"]),
                payload_name=str(row["payload_name"]),
                numeric=numeric,
                categorical=categorical,
                time_days=t,
                release_fraction=y,
            )
        )

    if max_curves_per_system > 0:
        rng = np.random.default_rng(seed)
        capped: list[CurveRecord] = []
        by_system: dict[str, list[CurveRecord]] = {}
        for record in records:
            by_system.setdefault(record.system_id, []).append(record)
        for system, system_records in sorted(by_system.items()):
            if len(system_records) <= max_curves_per_system:
                capped.extend(system_records)
            else:
                idx = np.sort(rng.choice(len(system_records), size=max_curves_per_system, replace=False))
                capped.extend([system_records[int(i)] for i in idx])
        records = capped

    if not records:
        raise ValueError(f"No valid curves loaded from {pool_dir}")
    return records


def fit_preprocessor(train_curves: list[CurveRecord], metadata_mode: str) -> Preprocessor:
    numeric_cols = [col for col in SAFE_NUMERIC_COLS if any(col in c.numeric for c in train_curves)]
    categorical_cols = categorical_columns_for_mode(metadata_mode)

    numeric_rows = []
    for curve in train_curves:
        numeric_rows.append([curve.numeric.get(col, np.nan) for col in numeric_cols])
    numeric_arr = np.asarray(numeric_rows, dtype=float) if numeric_rows else np.zeros((0, len(numeric_cols)))
    if numeric_arr.size:
        med_vals = []
        for j in range(numeric_arr.shape[1]):
            col = numeric_arr[:, j]
            finite = col[np.isfinite(col)]
            med_vals.append(float(np.median(finite)) if len(finite) else 0.0)
        med = np.asarray(med_vals, dtype=float)
    else:
        med = np.zeros(len(numeric_cols), dtype=float)
    med = np.nan_to_num(med, nan=0.0, posinf=0.0, neginf=0.0)
    filled = np.where(np.isfinite(numeric_arr), numeric_arr, med[None, :]) if numeric_arr.size else numeric_arr
    mean = filled.mean(axis=0).astype(np.float32) if filled.size else np.zeros(len(numeric_cols), dtype=np.float32)
    std = filled.std(axis=0).astype(np.float32) if filled.size else np.ones(len(numeric_cols), dtype=np.float32)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    medians = {col: float(med[i]) for i, col in enumerate(numeric_cols)}

    vocabs: dict[str, dict[str, int]] = {}
    for col in categorical_cols:
        values = sorted({curve.categorical.get(col, "__MISSING__") for curve in train_curves})
        vocab = {"__UNK__": 0, "__MISSING__": 1}
        for value in values:
            if value not in vocab:
                vocab[value] = len(vocab)
        vocabs[col] = vocab

    return Preprocessor(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        mean=mean,
        std=std,
        medians=medians,
        vocabs=vocabs,
    )


class LiquidCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.candidate = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.tau = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Softplus(),
        )

    def forward(self, h: torch.Tensor, x: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        hx = torch.cat([h, x], dim=-1)
        cand = self.candidate(hx)
        tau = self.tau(hx) + 1e-3
        alpha = 1.0 - torch.exp(-torch.clamp(dt, min=0.0) / tau)
        return h + alpha * (cand - h)


class ReleaseCnpLnn(nn.Module):
    def __init__(
        self,
        numeric_dim: int,
        categorical_cardinalities: list[int],
        hidden_dim: int,
        t_scale: float,
        decoder_mode: str,
    ) -> None:
        super().__init__()
        self.t_scale = float(max(t_scale, 1e-6))
        self.decoder_mode = decoder_mode
        emb_dim = min(16, max(4, hidden_dim // 8))
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, emb_dim) for cardinality in categorical_cardinalities]
        )
        meta_in = numeric_dim + emb_dim * len(categorical_cardinalities)
        self.meta_encoder = nn.Sequential(
            nn.Linear(meta_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.context_point = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.context_summary = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )
        self.cell = LiquidCell(hidden_dim, hidden_dim)
        self.rate_head = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.qmax_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def encode_meta(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        pieces = [x_num]
        for i, emb in enumerate(self.embeddings):
            pieces.append(emb(x_cat[:, i]))
        return self.meta_encoder(torch.cat(pieces, dim=-1))

    def encode_context(self, times: torch.Tensor, y_context: torch.Tensor, context_mask: torch.Tensor) -> torch.Tensor:
        t_scaled = torch.log1p(torch.clamp(times, min=0.0)) / np.log1p(self.t_scale)
        point_in = torch.stack([t_scaled, y_context], dim=-1)
        encoded = self.context_point(point_in) * context_mask[..., None]
        denom = context_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = encoded.sum(dim=1) / denom

        counts = context_mask.sum(dim=1).long()
        batch_idx = torch.arange(times.shape[0], device=times.device)
        last_idx = torch.clamp(counts - 1, min=0)
        prev_idx = torch.clamp(counts - 2, min=0)
        has_one = (counts > 0).to(times.dtype).unsqueeze(1)
        has_two = (counts > 1).to(times.dtype).unsqueeze(1)
        last_t = t_scaled[batch_idx, last_idx].unsqueeze(1) * has_one
        last_y = y_context[batch_idx, last_idx].unsqueeze(1) * has_one
        prev_t = t_scaled[batch_idx, prev_idx].unsqueeze(1) * has_two
        prev_y = y_context[batch_idx, prev_idx].unsqueeze(1) * has_two
        slope = ((last_y - prev_y) / (last_t - prev_t).abs().clamp_min(1e-3)) * has_two
        n_frac = (counts.to(times.dtype).unsqueeze(1) / max(1, times.shape[1])).clamp(0.0, 1.0)
        summary = torch.cat([last_t, last_y, slope.clamp(-10.0, 10.0), n_frac], dim=1)
        return pooled + self.context_summary(summary)

    def forward(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
        times: torch.Tensor,
        y_context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        meta = self.encode_meta(x_num, x_cat)
        context = self.encode_context(times, y_context, context_mask)
        z = self.fuse(torch.cat([meta, context], dim=-1))
        qmax = 0.25 + 0.95 * torch.sigmoid(self.qmax_head(z))

        h = z
        prev_t = torch.zeros(times.shape[0], 1, device=times.device, dtype=times.dtype)
        cumulative = torch.zeros(times.shape[0], 1, device=times.device, dtype=times.dtype)
        preds = []
        t_scaled_all = torch.log1p(torch.clamp(times, min=0.0)) / np.log1p(self.t_scale)
        for j in range(times.shape[1]):
            t_j = times[:, j:j + 1]
            t_scaled = t_scaled_all[:, j:j + 1]
            dt = (t_j - prev_t).clamp_min(0.0) / self.t_scale
            if self.decoder_mode == "lnn":
                h = self.cell(h, z, dt)
            elif self.decoder_mode == "mlp":
                h = z
            else:
                raise ValueError(f"Unknown decoder_mode={self.decoder_mode}")
            rate = F.softplus(self.rate_head(torch.cat([h, t_scaled], dim=-1))) + 1e-5
            cumulative = cumulative + rate * dt.clamp_min(1e-4)
            preds.append(qmax * (1.0 - torch.exp(-cumulative)))
            prev_t = t_j
        raw = torch.cat(preds, dim=1).clamp(0.0, 1.2)

        counts = context_mask.sum(dim=1).long()
        has_context = (counts > 0).to(times.dtype).unsqueeze(1)
        batch_idx = torch.arange(times.shape[0], device=times.device)
        last_idx = torch.clamp(counts - 1, min=0)
        last_y = y_context[batch_idx, last_idx].unsqueeze(1) * has_context
        raw_at_last = raw[batch_idx, last_idx].unsqueeze(1) * has_context
        residual_fraction = ((raw - raw_at_last) / (qmax - raw_at_last).clamp_min(1e-4)).clamp(0.0, 1.0)
        anchored = last_y + (qmax - last_y).clamp_min(0.0) * residual_fraction
        return (has_context * anchored + (1.0 - has_context) * raw).clamp(0.0, 1.2)


def sample_batch(
    curves_by_system: dict[str, list[int]],
    train_curves: list[CurveRecord],
    x_num: np.ndarray,
    x_cat: np.ndarray,
    batch_size: int,
    budgets: tuple[int, ...],
    rng: np.random.Generator,
) -> dict[str, torch.Tensor]:
    systems = sorted(curves_by_system)
    chosen_indices = []
    for _ in range(batch_size):
        system = systems[int(rng.integers(0, len(systems)))]
        candidates = curves_by_system[system]
        chosen_indices.append(candidates[int(rng.integers(0, len(candidates)))])
    selected = [train_curves[i] for i in chosen_indices]
    max_len = max(len(curve.time_days) for curve in selected)

    times = np.zeros((batch_size, max_len), dtype=np.float32)
    y = np.zeros((batch_size, max_len), dtype=np.float32)
    context_mask = np.zeros((batch_size, max_len), dtype=np.float32)
    target_mask = np.zeros((batch_size, max_len), dtype=np.float32)
    for row, curve in enumerate(selected):
        n = len(curve.time_days)
        k = int(budgets[int(rng.integers(0, len(budgets)))])
        k = min(k, n - 1)
        times[row, :n] = curve.time_days.astype(np.float32)
        y[row, :n] = curve.release_fraction.astype(np.float32)
        if k > 0:
            context_mask[row, :k] = 1.0
        target_mask[row, k:n] = 1.0

    return {
        "x_num": torch.tensor(x_num[chosen_indices], dtype=torch.float32),
        "x_cat": torch.tensor(x_cat[chosen_indices], dtype=torch.long),
        "times": torch.tensor(times, dtype=torch.float32),
        "y": torch.tensor(y, dtype=torch.float32),
        "context_mask": torch.tensor(context_mask, dtype=torch.float32),
        "target_mask": torch.tensor(target_mask, dtype=torch.float32),
    }


def train_model(
    train_curves: list[CurveRecord],
    preprocessor: Preprocessor,
    *,
    seed: int,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    hidden_dim: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    t_scale: float,
    decoder_mode: str,
) -> tuple[ReleaseCnpLnn, dict[str, float]]:
    seed_all(seed)
    x_num = preprocessor.transform_numeric(train_curves).astype(np.float32)
    x_cat = preprocessor.transform_categorical(train_curves)
    curves_by_system: dict[str, list[int]] = {}
    for idx, curve in enumerate(train_curves):
        curves_by_system.setdefault(curve.system_id, []).append(idx)

    model = ReleaseCnpLnn(
        numeric_dim=x_num.shape[1],
        categorical_cardinalities=[len(preprocessor.vocabs[col]) for col in preprocessor.categorical_cols],
        hidden_dim=hidden_dim,
        t_scale=t_scale,
        decoder_mode=decoder_mode,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    for _epoch in range(epochs):
        model.train()
        for _step in range(steps_per_epoch):
            batch = sample_batch(curves_by_system, train_curves, x_num, x_cat, batch_size, BUDGETS, rng)
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = model(batch["x_num"], batch["x_cat"], batch["times"], batch["y"] * batch["context_mask"], batch["context_mask"])
            resid = (pred - batch["y"]) * batch["target_mask"]
            denom = batch["target_mask"].sum().clamp_min(1.0)
            loss = (resid.square().sum() / denom)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_val = float(loss.detach().cpu())
            if loss_val < best_loss:
                best_loss = loss_val
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    meta = {
        "best_train_mse": best_loss,
        "n_params": float(sum(p.numel() for p in model.parameters())),
    }
    return model, meta


def predict_curve(
    model: ReleaseCnpLnn,
    curve: CurveRecord,
    x_num: np.ndarray,
    x_cat: np.ndarray,
    budget: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(curve.time_days)
    k = min(int(budget), n - 1)
    times = torch.tensor(curve.time_days[None, :], dtype=torch.float32, device=device)
    y = torch.tensor(curve.release_fraction[None, :], dtype=torch.float32, device=device)
    context_mask = torch.zeros((1, n), dtype=torch.float32, device=device)
    if k > 0:
        context_mask[:, :k] = 1.0
    with torch.no_grad():
        pred = model(
            torch.tensor(x_num[None, :], dtype=torch.float32, device=device),
            torch.tensor(x_cat[None, :], dtype=torch.long, device=device),
            times,
            y * context_mask,
            context_mask,
        ).cpu().numpy()[0]
    target_mask = np.zeros(n, dtype=bool)
    target_mask[k:n] = True
    return curve.time_days[target_mask], curve.release_fraction[target_mask], pred[target_mask]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2))) if len(y_true) else np.nan


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred))) if len(y_true) else np.nan


def bounded_r2(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, bool]:
    value = safe_r2(y_true, y_pred)
    if np.isfinite(value):
        return float(value), True
    return 0.0, False


def crps_ensemble(samples: np.ndarray, y: np.ndarray) -> float:
    samples = np.asarray(samples, dtype=float)
    y = np.asarray(y, dtype=float)
    if samples.ndim != 2 or samples.shape[0] == 0:
        return np.nan
    term1 = np.mean(np.abs(samples - y[None, :]))
    diffs = np.abs(samples[:, None, :] - samples[None, :, :])
    term2 = 0.5 * np.mean(diffs)
    return float(term1 - term2)


def interp_mean_prediction(train_curves: list[CurveRecord], target_times: np.ndarray, same_system: str | None = None) -> np.ndarray | None:
    source = [c for c in train_curves if same_system is None or c.system_id == same_system]
    if not source:
        return None
    preds = [
        np.interp(target_times, c.time_days, c.release_fraction, left=c.release_fraction[0], right=c.release_fraction[-1])
        for c in source
    ]
    return np.mean(np.vstack(preds), axis=0)


def interp_group_mean_prediction(train_curves: list[CurveRecord], target_times: np.ndarray, field: str, value: str) -> np.ndarray | None:
    source = [c for c in train_curves if str(getattr(c, field)) == str(value)]
    if not source:
        return None
    preds = [
        np.interp(target_times, c.time_days, c.release_fraction, left=c.release_fraction[0], right=c.release_fraction[-1])
        for c in source
    ]
    return np.mean(np.vstack(preds), axis=0)


def anchored_mean_prediction(
    train_curves: list[CurveRecord],
    target_times: np.ndarray,
    context_times: np.ndarray,
    context_y: np.ndarray,
    *,
    same_system: str | None = None,
    field: str | None = None,
    value: str | None = None,
) -> np.ndarray | None:
    if len(context_times) == 0:
        return None
    source = train_curves
    if same_system is not None:
        source = [c for c in source if c.system_id == same_system]
    if field is not None:
        source = [c for c in source if str(getattr(c, field)) == str(value)]
    if not source:
        return None
    last_t = float(context_times[-1])
    last_y = float(context_y[-1])
    query = np.concatenate([[last_t], target_times])
    prior = np.mean(
        np.vstack(
            [
                np.interp(query, c.time_days, c.release_fraction, left=c.release_fraction[0], right=c.release_fraction[-1])
                for c in source
            ]
        ),
        axis=0,
    )
    pred = last_y + (prior[1:] - prior[0])
    pred = np.maximum.accumulate(np.maximum(pred, last_y))
    return np.clip(pred, 0.0, 1.2)


def carry_forward_prediction(target_times: np.ndarray, context_y: np.ndarray) -> np.ndarray | None:
    if len(target_times) == 0 or len(context_y) == 0:
        return None
    return np.full(len(target_times), float(context_y[-1]), dtype=float)


def shape_increment_anchored_prediction(
    fn: Any,
    params: np.ndarray,
    target_times: np.ndarray,
    context_times: np.ndarray,
    context_y: np.ndarray,
) -> np.ndarray | None:
    if len(target_times) == 0 or len(context_times) == 0:
        return None
    query = np.concatenate([context_times, target_times])
    prior = np.clip(fn(query, params), 0.0, 1.2)
    prior_context = prior[: len(context_times)]
    prior_target = prior[len(context_times) :]
    last_y = float(context_y[-1])
    last_prior = float(prior_context[-1])

    scale = 1.0
    if len(context_times) >= 2:
        dp = prior_context - float(prior_context[0])
        dy = context_y - float(context_y[0])
        denom = float(np.dot(dp, dp))
        if denom > 1e-12:
            scale = float(np.dot(dp, dy) / denom)
            scale = float(np.clip(scale, 0.0, 5.0))

    pred = last_y + scale * (prior_target - last_prior)
    pred = np.maximum.accumulate(np.maximum(pred, last_y))
    return np.clip(pred, 0.0, 1.2)


def budget_specs_for_curve(curve: CurveRecord, time_budgets: tuple[float, ...]) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    n = len(curve.time_days)
    for budget in BUDGETS:
        k = min(int(budget), n - 1)
        specs.append(
            {
                "budget_kind": "point",
                "budget_value": float(budget),
                "budget_label": f"k={budget}",
                "context_count": k,
            }
        )
    for horizon in time_budgets:
        k = int(np.searchsorted(curve.time_days, horizon, side="right"))
        k = min(k, n - 1)
        specs.append(
            {
                "budget_kind": "time_days",
                "budget_value": float(horizon),
                "budget_label": f"t<={horizon:g}d",
                "context_count": k,
            }
        )
    return specs


def theta_from_family_fits(fits: pd.DataFrame) -> pd.DataFrame | None:
    required = {"unified_curve_id", "family", "param_1", "param_2", "param_3"}
    if not required.issubset(fits.columns):
        return None
    out = pd.DataFrame(
        {
            "unified_curve_id": fits["unified_curve_id"].astype(str),
            "theta_family": fits["family"].astype(str),
            "theta_param_1": pd.to_numeric(fits["param_1"], errors="coerce"),
            "theta_param_2": pd.to_numeric(fits["param_2"], errors="coerce"),
            "theta_param_3": pd.to_numeric(fits["param_3"], errors="coerce"),
        }
    )
    if "param_4" in fits.columns:
        out["theta_param_4"] = pd.to_numeric(fits["param_4"], errors="coerce")
    return out


def load_theta_table(theta_dir: Path) -> pd.DataFrame | None:
    path = theta_dir / "theta_table_v1.parquet"
    if not path.exists():
        path = theta_dir / "all_family_fits.csv"
        if not path.exists():
            return None
    try:
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        return theta_from_family_fits(pd.read_csv(path))
    except Exception:
        fallback = theta_dir / "all_family_fits.csv"
        if path == fallback or not fallback.exists():
            return None
        try:
            return theta_from_family_fits(pd.read_csv(fallback))
        except Exception:
            return None


def median_shape_params(theta: pd.DataFrame, train_ids: set[str]) -> dict[str, np.ndarray]:
    if theta is None or theta.empty:
        return {}
    sub = theta[theta["unified_curve_id"].astype(str).isin(train_ids)].copy()
    out: dict[str, np.ndarray] = {}
    for family, fam_df in sub.groupby("theta_family"):
        param_cols = [c for c in ["theta_param_1", "theta_param_2", "theta_param_3", "theta_param_4"] if c in fam_df.columns]
        vals = fam_df[param_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        vals = vals[:, ~np.all(~np.isfinite(vals), axis=0)]
        if vals.size:
            out[str(family)] = np.nanmedian(vals, axis=0)
    return out


def reference_161_map(ref161_dir: Path) -> dict[tuple[str, int], float]:
    path = ref161_dir / "per_curve_metrics.csv"
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    required = {"unified_curve_id", "n_obs", "curve_rmse"}
    if not required.issubset(df.columns):
        return {}
    out: dict[tuple[str, int], float] = {}
    for _, row in df.iterrows():
        out[(str(row["unified_curve_id"]), int(row["n_obs"]))] = float(row["curve_rmse"])
    return out


def evaluate_split(
    split_label: str,
    train_curves: list[CurveRecord],
    test_curves: list[CurveRecord],
    args: argparse.Namespace,
    device: torch.device,
    theta_table: pd.DataFrame | None,
    ref161: dict[tuple[str, int], float],
    split_kind: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    if not train_curves or not test_curves:
        return [], [], [], [], {"heldout_system": split_label, "split_kind": split_kind, "skipped": True}

    pre = fit_preprocessor(train_curves, args.metadata_mode)
    train_ids = {c.curve_id for c in train_curves}
    shape_priors = median_shape_params(theta_table, train_ids)
    t_scale = max(max(c.time_days) for c in train_curves + test_curves)
    fold_seed_base = args.seed + stable_int_hash(f"{split_kind}:{split_label}") % 100_000

    models: list[ReleaseCnpLnn] = []
    train_meta: list[dict[str, float]] = []
    for member in range(args.ensemble_size):
        model, meta = train_model(
            train_curves,
            pre,
            seed=fold_seed_base + member * 1009,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
            t_scale=t_scale,
            decoder_mode=args.decoder_mode,
        )
        model.eval()
        models.append(model)
        train_meta.append(meta)

    test_num = pre.transform_numeric(test_curves)
    test_cat = pre.transform_categorical(test_curves)
    per_curve_rows: list[dict[str, object]] = []
    baseline_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    model_name = f"CNP-{args.decoder_mode.upper()}-ensemble"
    time_budgets = parse_time_budgets(args.time_budgets_days)

    for role, role_curves in [("train", train_curves), ("test", test_curves)]:
        for curve in role_curves:
            manifest_rows.append(
                {
                    "split_kind": split_kind,
                    "heldout_system": split_label,
                    "role": role,
                    "unified_curve_id": curve.curve_id,
                    "system_id": curve.system_id,
                    "source_dataset": curve.source_dataset,
                    "polymer_family": curve.polymer_family,
                    "payload_name": curve.payload_name,
                    "n_timepoints": int(len(curve.time_days)),
                    "min_time_days": float(np.min(curve.time_days)),
                    "max_time_days": float(np.max(curve.time_days)),
                }
            )

    for i, curve in enumerate(test_curves):
        for spec in budget_specs_for_curve(curve, time_budgets):
            budget = int(spec["context_count"])
            budget_kind = str(spec["budget_kind"])
            budget_value = float(spec["budget_value"])
            budget_label = str(spec["budget_label"])
            target_t, y_true, _ = predict_curve(models[0], curve, test_num[i], test_cat[i], budget, device)
            if len(y_true) == 0:
                continue
            k_context = min(int(budget), len(curve.time_days) - 1)
            context_t = curve.time_days[:k_context]
            context_y = curve.release_fraction[:k_context]
            member_preds = []
            for model in models:
                _, _, pred = predict_curve(model, curve, test_num[i], test_cat[i], budget, device)
                member_preds.append(pred)
            samples = np.vstack(member_preds)
            mean_pred = samples.mean(axis=0)
            if (
                not np.isfinite(samples).all()
                or np.any(np.diff(mean_pred) < -1e-6)
                or np.min(samples) < -1e-6
                or np.max(samples) > 1.200001
            ):
                raise RuntimeError(f"Invalid {model_name} prediction for {curve.curve_id} at n_obs={budget}")
            q05 = np.quantile(samples, 0.05, axis=0)
            q25 = np.quantile(samples, 0.25, axis=0)
            q75 = np.quantile(samples, 0.75, axis=0)
            q95 = np.quantile(samples, 0.95, axis=0)
            r2_value, r2_defined = bounded_r2(y_true, mean_pred)
            per_curve_rows.append(
                {
                    "split_kind": split_kind,
                    "heldout_system": split_label,
                    "unified_curve_id": curve.curve_id,
                    "source_dataset": curve.source_dataset,
                    "budget_kind": budget_kind,
                    "budget_value": budget_value,
                    "budget_label": budget_label,
                    "n_obs": int(budget),
                    "method": model_name,
                    "decoder_mode": args.decoder_mode,
                    "metadata_mode": args.metadata_mode,
                    "n_target": int(len(y_true)),
                    "rmse": rmse(y_true, mean_pred),
                    "mae": mae(y_true, mean_pred),
                    "r2": r2_value,
                    "r2_defined": bool(r2_defined),
                    "cov90": float(np.mean((y_true >= q05) & (y_true <= q95))),
                    "cov50": float(np.mean((y_true >= q25) & (y_true <= q75))),
                    "width90": float(np.mean(q95 - q05)),
                    "width50": float(np.mean(q75 - q25)),
                    "crps": crps_ensemble(samples, y_true),
                }
            )
            for point_idx, (time_days, y_point, pred_mean, pred_p05, pred_p50, pred_p95) in enumerate(
                zip(target_t, y_true, mean_pred, q05, np.quantile(samples, 0.50, axis=0), q95)
            ):
                prediction_rows.append(
                    {
                        "split_kind": split_kind,
                        "heldout_system": split_label,
                        "unified_curve_id": curve.curve_id,
                        "source_dataset": curve.source_dataset,
                        "system_id": curve.system_id,
                        "budget_kind": budget_kind,
                        "budget_value": budget_value,
                        "budget_label": budget_label,
                        "n_obs": int(budget),
                        "target_point_idx": int(point_idx),
                        "time_days": float(time_days),
                        "y_true": float(y_point),
                        "y_pred_mean": float(pred_mean),
                        "y_pred_p05": float(pred_p05),
                        "y_pred_p50": float(pred_p50),
                        "y_pred_p95": float(pred_p95),
                        "model_name": model_name,
                        "decoder_mode": args.decoder_mode,
                        "metadata_mode": args.metadata_mode,
                        "ensemble_size": int(args.ensemble_size),
                    }
                )

            global_pred = interp_mean_prediction(train_curves, target_t)
            if global_pred is not None:
                baseline_rows.append(
                    {
                        "split_kind": split_kind,
                        "heldout_system": split_label,
                        "unified_curve_id": curve.curve_id,
                        "budget_kind": budget_kind,
                        "budget_value": budget_value,
                        "budget_label": budget_label,
                        "n_obs": int(budget),
                        "baseline": "GlobalMean",
                        "rmse": rmse(y_true, global_pred),
                    }
                )
            carry_pred = carry_forward_prediction(target_t, context_y)
            if carry_pred is not None:
                baseline_rows.append(
                    {
                        "split_kind": split_kind,
                        "heldout_system": split_label,
                        "unified_curve_id": curve.curve_id,
                        "budget_kind": budget_kind,
                        "budget_value": budget_value,
                        "budget_label": budget_label,
                        "n_obs": int(budget),
                        "baseline": "LastValueCarryForward",
                        "rmse": rmse(y_true, carry_pred),
                    }
                )
            anchored_global = anchored_mean_prediction(train_curves, target_t, context_t, context_y)
            if anchored_global is not None:
                baseline_rows.append(
                    {
                        "split_kind": split_kind,
                        "heldout_system": split_label,
                        "unified_curve_id": curve.curve_id,
                        "budget_kind": budget_kind,
                        "budget_value": budget_value,
                        "budget_label": budget_label,
                        "n_obs": int(budget),
                        "baseline": "GlobalMeanAnchored",
                        "rmse": rmse(y_true, anchored_global),
                    }
                )
            for baseline_name, field, value in [
                ("PolymerFamilyMean", "polymer_family", curve.polymer_family),
                ("PayloadNameMean", "payload_name", curve.payload_name),
            ]:
                group_pred = interp_group_mean_prediction(train_curves, target_t, field, value)
                if group_pred is not None:
                    baseline_rows.append(
                        {
                            "split_kind": split_kind,
                            "heldout_system": split_label,
                            "unified_curve_id": curve.curve_id,
                            "budget_kind": budget_kind,
                            "budget_value": budget_value,
                            "budget_label": budget_label,
                            "n_obs": int(budget),
                            "baseline": baseline_name,
                            "rmse": rmse(y_true, group_pred),
                        }
                    )
                anchored_group = anchored_mean_prediction(train_curves, target_t, context_t, context_y, field=field, value=value)
                if anchored_group is not None:
                    baseline_rows.append(
                        {
                            "split_kind": split_kind,
                            "heldout_system": split_label,
                            "unified_curve_id": curve.curve_id,
                            "budget_kind": budget_kind,
                            "budget_value": budget_value,
                            "budget_label": budget_label,
                            "n_obs": int(budget),
                            "baseline": f"{baseline_name}Anchored",
                            "rmse": rmse(y_true, anchored_group),
                        }
                    )
            local_pred = interp_mean_prediction(train_curves, target_t, same_system=curve.system_id)
            if local_pred is not None:
                baseline_rows.append(
                    {
                        "split_kind": split_kind,
                        "heldout_system": split_label,
                        "unified_curve_id": curve.curve_id,
                        "budget_kind": budget_kind,
                        "budget_value": budget_value,
                        "budget_label": budget_label,
                        "n_obs": int(budget),
                        "baseline": "SystemLocalMean",
                        "rmse": rmse(y_true, local_pred),
                    }
                )
            anchored_local = anchored_mean_prediction(train_curves, target_t, context_t, context_y, same_system=curve.system_id)
            if anchored_local is not None:
                baseline_rows.append(
                    {
                        "split_kind": split_kind,
                        "heldout_system": split_label,
                        "unified_curve_id": curve.curve_id,
                        "budget_kind": budget_kind,
                        "budget_value": budget_value,
                        "budget_label": budget_label,
                        "n_obs": int(budget),
                        "baseline": "SystemLocalMeanAnchored",
                        "rmse": rmse(y_true, anchored_local),
                    }
                )
            for family, params in shape_priors.items():
                fn = FAMILY_FUNCS.get(family)
                if fn is None:
                    continue
                shape_pred = np.clip(fn(target_t, params), 0.0, 1.2)
                baseline_rows.append(
                    {
                        "split_kind": split_kind,
                        "heldout_system": split_label,
                        "unified_curve_id": curve.curve_id,
                        "budget_kind": budget_kind,
                        "budget_value": budget_value,
                        "budget_label": budget_label,
                        "n_obs": int(budget),
                        "baseline": f"ShapePrior:{family}",
                        "rmse": rmse(y_true, shape_pred),
                    }
                )
                anchored_shape_pred = shape_increment_anchored_prediction(fn, params, target_t, context_t, context_y)
                if anchored_shape_pred is not None:
                    baseline_rows.append(
                        {
                            "split_kind": split_kind,
                            "heldout_system": split_label,
                            "unified_curve_id": curve.curve_id,
                            "budget_kind": budget_kind,
                            "budget_value": budget_value,
                            "budget_label": budget_label,
                            "n_obs": int(budget),
                            "baseline": f"ShapePriorIncrementAnchored:{family}",
                            "rmse": rmse(y_true, anchored_shape_pred),
                        }
                    )
            ref_key = (curve.curve_id, int(budget))
            if budget_kind == "point" and ref_key in ref161:
                baseline_rows.append(
                    {
                        "split_kind": split_kind,
                        "heldout_system": split_label,
                        "unified_curve_id": curve.curve_id,
                        "budget_kind": budget_kind,
                        "budget_value": budget_value,
                        "budget_label": budget_label,
                        "n_obs": int(budget),
                        "baseline": "Reference161ThetaAdapter",
                        "rmse": ref161[ref_key],
                    }
                )

    meta = {
        "heldout_system": split_label,
        "split_kind": split_kind,
        "n_train": len(train_curves),
        "n_test": len(test_curves),
        "n_params_mean": float(np.mean([m["n_params"] for m in train_meta])),
        "best_train_mse_mean": float(np.mean([m["best_train_mse"] for m in train_meta])),
        "skipped": False,
    }
    return per_curve_rows, baseline_rows, prediction_rows, manifest_rows, meta


def evaluate_loso_fold(
    heldout_system: str,
    all_curves: list[CurveRecord],
    args: argparse.Namespace,
    device: torch.device,
    theta_table: pd.DataFrame | None,
    ref161: dict[tuple[str, int], float],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    train_curves = [c for c in all_curves if c.system_id != heldout_system]
    test_curves = [c for c in all_curves if c.system_id == heldout_system]
    if any(c.system_id == heldout_system for c in train_curves):
        raise RuntimeError(f"LOSO leakage detected for heldout system {heldout_system}")
    return evaluate_split(
        heldout_system,
        train_curves,
        test_curves,
        args,
        device,
        theta_table,
        ref161,
        split_kind="loso-system",
    )


def pooled_r2_from_rows(df: pd.DataFrame) -> float:
    # Per-curve tables do not store all target points, so report median per-curve R2
    # as the robust curve-level summary in this probe.
    vals = pd.to_numeric(df["r2"], errors="coerce").dropna().to_numpy(dtype=float)
    return float(np.median(vals)) if len(vals) else np.nan


def summarize(per_curve: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_by_system = (
        per_curve.groupby(["split_kind", "heldout_system", "budget_kind", "budget_label", "budget_value"], dropna=False)
        .agg(
            n_curves=("unified_curve_id", "nunique"),
            median_n_obs=("n_obs", "median"),
            median_rmse=("rmse", "median"),
            mean_rmse=("rmse", "mean"),
            median_mae=("mae", "median"),
            median_curve_r2=("r2", "median"),
            mean_cov90=("cov90", "mean"),
            mean_cov50=("cov50", "mean"),
            mean_width90=("width90", "mean"),
            mean_width50=("width50", "mean"),
            mean_crps=("crps", "mean"),
        )
        .reset_index()
        .sort_values(["split_kind", "heldout_system", "budget_kind", "budget_value"])
    )
    summary_by_budget = (
        per_curve.groupby(["split_kind", "budget_kind", "budget_label", "budget_value"], dropna=False)
        .agg(
            n_curves=("unified_curve_id", "nunique"),
            median_n_obs=("n_obs", "median"),
            median_rmse=("rmse", "median"),
            mean_rmse=("rmse", "mean"),
            median_mae=("mae", "median"),
            median_curve_r2=("r2", "median"),
            mean_cov90=("cov90", "mean"),
            mean_cov50=("cov50", "mean"),
            mean_width90=("width90", "mean"),
            mean_width50=("width50", "mean"),
            mean_crps=("crps", "mean"),
        )
        .reset_index()
        .sort_values(["split_kind", "budget_kind", "budget_value"])
    )

    budget_rows = []
    for (split_kind, system, budget_kind), sub in summary_by_system.groupby(["split_kind", "heldout_system", "budget_kind"]):
        sub = sub.sort_values("budget_value")
        base = sub.iloc[0]
        base_width = float(base["mean_width90"])
        base_rmse = float(base["median_rmse"])
        for _, row in sub.iterrows():
            width_ratio = float(row["mean_width90"] / base_width) if base_width > 1e-12 else 1.0
            rmse_ratio = float(row["median_rmse"] / base_rmse) if base_rmse > 1e-12 else 1.0
            budget_rows.append(
                {
                    "split_kind": split_kind,
                    "heldout_system": system,
                    "budget_kind": budget_kind,
                    "budget_label": row["budget_label"],
                    "budget_value": float(row["budget_value"]),
                    "median_n_obs": float(row["median_n_obs"]),
                    "median_rmse": float(row["median_rmse"]),
                    "mean_width90": float(row["mean_width90"]),
                    "width90_ratio_to_first_budget": width_ratio,
                    "rmse_ratio_to_first_budget": rmse_ratio,
                }
            )
    budget_curve = pd.DataFrame(budget_rows)
    return summary_by_system, summary_by_budget, budget_curve


def bootstrap_pairwise(per_curve: pd.DataFrame, baselines: pd.DataFrame, n_bootstrap: int, seed: int) -> pd.DataFrame:
    rows = []
    if baselines.empty:
        return pd.DataFrame(rows)
    rng = np.random.default_rng(seed)
    group_cols = ["split_kind", "heldout_system", "budget_kind", "budget_label", "budget_value"]
    for (split_kind, system, budget_kind, budget_label, budget_value), sub in per_curve.groupby(group_cols):
        lnn = sub[["unified_curve_id", "rmse"]].set_index("unified_curve_id")
        base_sub = baselines[
            (baselines["split_kind"] == split_kind)
            & (baselines["heldout_system"] == system)
            & (baselines["budget_kind"] == budget_kind)
            & (baselines["budget_label"] == budget_label)
        ]
        for baseline, bdf in base_sub.groupby("baseline"):
            joined = lnn.join(bdf[["unified_curve_id", "rmse"]].set_index("unified_curve_id"), lsuffix="_lnn", rsuffix="_baseline", how="inner")
            if len(joined) < 2:
                continue
            deltas = joined["rmse_lnn"].to_numpy(dtype=float) - joined["rmse_baseline"].to_numpy(dtype=float)
            boot = []
            for _ in range(n_bootstrap):
                idx = rng.integers(0, len(deltas), size=len(deltas))
                boot.append(float(np.mean(deltas[idx])))
            ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
            rows.append(
                {
                    "split_kind": split_kind,
                    "heldout_system": system,
                    "budget_kind": budget_kind,
                    "budget_label": budget_label,
                    "budget_value": float(budget_value),
                    "median_n_obs": float(sub["n_obs"].median()),
                    "baseline": baseline,
                    "n_pairs": int(len(deltas)),
                    "delta_rmse_lnn_minus_baseline": float(np.mean(deltas)),
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "lnn_better": bool(ci_high < 0.0),
                    "baseline_better": bool(ci_low > 0.0),
                }
            )
    return pd.DataFrame(rows)


def summarize_baseline_delta(per_curve: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    if baselines.empty:
        return pd.DataFrame()
    key_cols = ["split_kind", "heldout_system", "unified_curve_id", "budget_kind", "budget_label", "budget_value"]
    model = per_curve[key_cols + ["rmse"]].rename(columns={"rmse": "rmse_model"})
    base = baselines[key_cols + ["baseline", "rmse"]].rename(columns={"rmse": "rmse_baseline"})
    joined = model.merge(base, on=key_cols, how="inner")
    if joined.empty:
        return pd.DataFrame()
    joined["delta"] = joined["rmse_model"] - joined["rmse_baseline"]
    return (
        joined.groupby(["split_kind", "budget_kind", "budget_label", "budget_value", "baseline"], dropna=False)
        .agg(
            n_pairs=("delta", "size"),
            median_model=("rmse_model", "median"),
            median_baseline=("rmse_baseline", "median"),
            median_delta_model_minus_baseline=("delta", "median"),
            mean_delta_model_minus_baseline=("delta", "mean"),
        )
        .reset_index()
        .sort_values(["split_kind", "budget_kind", "budget_value", "median_delta_model_minus_baseline"])
    )


def write_summary(out: Path, summary_by_system: pd.DataFrame, summary_by_budget: pd.DataFrame, fold_meta: list[dict[str, object]]) -> None:
    lines = [
        "# Release-Corpus CNP/LNN Probe",
        "",
        "Probe-only result. Hidden state is a learned release representation, not a physical state claim.",
        "Intervals are raw ensemble quantiles, not calibrated posterior uncertainty.",
        "",
        "## Aggregate by observation budget",
        "",
        summary_by_budget.to_markdown(index=False),
        "",
        "## Fold metadata",
        "",
    ]
    for meta in fold_meta:
        if meta.get("skipped"):
            lines.append(f"- `{meta['split_kind']}::{meta['heldout_system']}`: skipped")
        else:
            lines.append(
                f"- `{meta['split_kind']}::{meta['heldout_system']}`: train={meta['n_train']}, test={meta['n_test']}, "
                f"params~{meta['n_params_mean']:.0f}, train_mse={meta['best_train_mse_mean']:.4g}"
            )
    lines.extend(["", "## By heldout system", "", summary_by_system.to_markdown(index=False), ""])
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    curves = load_curve_records(args.pool_dir, args.max_curves_per_system, args.seed)
    systems = sorted({c.system_id for c in curves})
    heldout_systems = parse_system_filter(args.only_heldout_systems, systems)
    theta = load_theta_table(args.theta_dir)
    ref161 = reference_161_map(args.ref161_dir)

    all_per_curve: list[dict[str, object]] = []
    all_baselines: list[dict[str, object]] = []
    all_predictions: list[dict[str, object]] = []
    all_manifest: list[dict[str, object]] = []
    fold_meta: list[dict[str, object]] = []
    for system in ([] if args.skip_loso_system else heldout_systems):
        per_curve_rows, baseline_rows, prediction_rows, manifest_rows, meta = evaluate_loso_fold(system, curves, args, device, theta, ref161)
        all_per_curve.extend(per_curve_rows)
        all_baselines.extend(baseline_rows)
        all_predictions.extend(prediction_rows)
        all_manifest.extend(manifest_rows)
        fold_meta.append(meta)
        print(f"[81] heldout={system} rows={len(per_curve_rows)} baselines={len(baseline_rows)}")

    within_systems = [] if args.skip_within_large else sorted(set(LARGE_SYSTEMS) & set(heldout_systems))
    for system in within_systems:
        system_curves = [c for c in curves if c.system_id == system]
        if len(system_curves) < 3:
            continue
        rng = np.random.default_rng(args.seed + stable_int_hash(f"within:{system}"))
        n_holdout = int(round(len(system_curves) * float(args.within_holdout_frac)))
        n_holdout = min(max(1, n_holdout), len(system_curves) - 1)
        heldout_idx = set(int(i) for i in rng.choice(len(system_curves), size=n_holdout, replace=False))
        heldout_ids = {system_curves[i].curve_id for i in heldout_idx}
        train_curves = [c for c in curves if c.curve_id not in heldout_ids]
        test_curves = [c for c in curves if c.curve_id in heldout_ids]
        split_label = f"{system}:within"
        per_curve_rows, baseline_rows, prediction_rows, manifest_rows, meta = evaluate_split(
            split_label,
            train_curves,
            test_curves,
            args,
            device,
            theta,
            ref161,
            split_kind="within-large-system",
        )
        all_per_curve.extend(per_curve_rows)
        all_baselines.extend(baseline_rows)
        all_predictions.extend(prediction_rows)
        all_manifest.extend(manifest_rows)
        fold_meta.append(meta)
        print(f"[81] heldout={split_label} rows={len(per_curve_rows)} baselines={len(baseline_rows)}")

    per_curve = pd.DataFrame(all_per_curve)
    baselines = pd.DataFrame(all_baselines)
    predictions = pd.DataFrame(all_predictions)
    split_manifest = pd.DataFrame(all_manifest)
    if per_curve.empty:
        raise RuntimeError("No per-curve metrics produced")
    per_curve.to_csv(args.out / "per_curve_metrics.csv", index=False)
    baselines.to_csv(args.out / "baseline_per_curve_metrics.csv", index=False)
    predictions.to_csv(args.out / "predictions_long.csv", index=False)
    split_manifest.to_csv(args.out / "split_manifest.csv", index=False)
    if per_curve.replace([np.inf, -np.inf], np.nan)[["rmse", "mae", "cov90", "width90", "crps"]].isna().any().any():
        raise RuntimeError("NaN/inf detected in primary metric columns")

    summary_by_system, summary_by_budget, budget_curve = summarize(per_curve)
    summary_by_time_budget = summary_by_budget[summary_by_budget["budget_kind"] == "time_days"].copy()
    pairwise = bootstrap_pairwise(per_curve, baselines, args.n_bootstrap, args.seed)
    baseline_delta = summarize_baseline_delta(per_curve, baselines)
    ablation_summary = summary_by_budget.copy()
    ablation_summary.insert(0, "decoder_mode", args.decoder_mode)
    ablation_summary.insert(1, "metadata_mode", args.metadata_mode)

    summary_by_system.to_csv(args.out / "summary_by_system.csv", index=False)
    summary_by_budget.to_csv(args.out / "summary_by_budget.csv", index=False)
    summary_by_time_budget.to_csv(args.out / "summary_by_time_budget.csv", index=False)
    budget_curve.to_csv(args.out / "budget_curve.csv", index=False)
    pairwise.to_csv(args.out / "pairwise_bootstrap.csv", index=False)
    baseline_delta.to_csv(args.out / "baseline_delta_summary.csv", index=False)
    ablation_summary.to_csv(args.out / "ablation_summary.csv", index=False)

    metadata = {
        "script": "scripts/81_release_corpus_cnp_lnn_probe.py",
        "git_hash": get_git_hash(),
        "pool_dir": str(args.pool_dir),
        "theta_dir": str(args.theta_dir),
        "ref161_dir": str(args.ref161_dir),
        "out": str(args.out),
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "steps_per_epoch": int(args.steps_per_epoch),
        "batch_size": int(args.batch_size),
        "hidden_dim": int(args.hidden_dim),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "ensemble_size": int(args.ensemble_size),
        "metadata_mode": str(args.metadata_mode),
        "decoder_mode": str(args.decoder_mode),
        "max_curves_per_system": int(args.max_curves_per_system),
        "within_holdout_frac": float(args.within_holdout_frac),
        "only_heldout_systems": str(args.only_heldout_systems),
        "skip_within_large": bool(args.skip_within_large),
        "skip_loso_system": bool(args.skip_loso_system),
        "device": str(device),
        "budgets": list(BUDGETS),
        "time_budgets_days": list(parse_time_budgets(args.time_budgets_days)),
        "systems": systems,
        "evaluated_loso_systems": heldout_systems,
        "evaluated_within_systems": within_systems,
        "n_curves": int(len(curves)),
        "note": "Probe only. Do not claim foundation model, mechanism learned, physical hidden state, or calibrated uncertainty.",
    }
    (args.out / "lock_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_summary(args.out, summary_by_system, summary_by_budget, fold_meta)

    print("[81] wrote", args.out)
    print(summary_by_budget.to_string(index=False))


if __name__ == "__main__":
    main()
