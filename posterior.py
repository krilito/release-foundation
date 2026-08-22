"""
Neural posterior estimation for amortized SBI on drug release.

Phase 1 ships `CurvePosterior` — MVP-B stage 1, `q_φ(θ | Q)`. Given a
release curve on the canonical time grid, return a posterior over kinetic
parameters. Trained on synthetic `(θ, Q)` pairs produced by a
`ReleaseSimulator`.

Phase 1 day 3 adds `DescriptorPosterior` — MVP-B stage 2, `r_ψ(θ | x)`.
Trained by knowledge distillation from `q_φ`: for each real curve i,
draw K θ samples from `q_φ(θ | Q_real_i)` and treat the `(θ_k, x_i)`
pairs as supervised training data for r_ψ. This minimizes
`KL(q_φ(θ|Q) || r_ψ(θ|x))` and is NOT the SRDS cascade approach
(which fits a single θ̂ per curve and regresses x → θ̂). See ADR-018.

The "amortized" property: once trained, posterior inference for any new
observation is a forward pass, not optimization. This is the architectural
difference from the SRDS cascade. See ADR-009 in DECISIONS.md.
"""
from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import torch
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.embedding_nets import FCEmbedding
from torch import Tensor
from torch.distributions import Distribution, Independent, Uniform

from encoder import FormulationFeaturizer, MLPFormulationEncoder
from simulator import ReleaseSimulator


class CurvePosterior:
    """Amortized neural posterior `q_φ(θ | Q)` for a `ReleaseSimulator`.

    Workflow:
        sim = PLGABiphasic()
        t_grid = torch.linspace(0., 90., 64)
        cp = CurvePosterior(sim, t_grid)
        cp.train(n_simulations=50_000)
        samples = cp.sample(Q_observed, n_samples=1000)  # θ samples
    """

    def __init__(
        self,
        simulator: ReleaseSimulator,
        t_grid: Tensor,
        flow: str = "maf",
        hidden_features: int = 64,
        num_transforms: int = 5,
        embedding_output_dim: int = 16,
        embedding_hidden: int = 64,
        embedding_layers: int = 3,
        noise_sigma: float = 0.03,
        device: str = "cpu",
    ) -> None:
        if t_grid.ndim != 1:
            raise ValueError(f"t_grid must be 1-D, got {tuple(t_grid.shape)}")
        if noise_sigma < 0:
            raise ValueError(f"noise_sigma must be >= 0, got {noise_sigma}")
        self.simulator = simulator
        self.t_grid = t_grid
        self.device = device
        self.flow = flow
        self.hidden_features = hidden_features
        self.num_transforms = num_transforms
        self.embedding_output_dim = embedding_output_dim
        self.embedding_hidden = embedding_hidden
        self.embedding_layers = embedding_layers
        self.noise_sigma = noise_sigma

        # The observation Q is a 64-D curve with strong inter-dimensional
        # correlation. Without an embedding net, the flow has to discover
        # the low-effective-dimensional structure on its own and tends to
        # produce dense-but-mis-located posteriors. See ADR-010.
        self._embedding_net = FCEmbedding(
            input_dim=len(t_grid),
            output_dim=embedding_output_dim,
            num_layers=embedding_layers,
            num_hiddens=embedding_hidden,
        )
        self._density_estimator_builder = posterior_nn(
            model=flow,
            hidden_features=hidden_features,
            num_transforms=num_transforms,
            embedding_net=self._embedding_net,
        )
        self._inference: NPE | None = None
        self._estimator = None
        self._posterior = None
        self._training_log: dict | None = None

    # ------------------------------------------------------------------
    # Observation-noise model
    # ------------------------------------------------------------------
    def add_obs_noise(self, Q: Tensor, generator: torch.Generator | None = None) -> Tensor:
        """Add Gaussian observation noise to a clean simulator output.

        The deterministic `simulator.simulate()` produces curves with a
        delta-function posterior over directly observable parameters
        (q_burst, Q_max). MAF / NSF flows cannot represent deltas, so
        SBC catastrophically fails on noiseless data. Adding observation
        noise of physically realistic magnitude (~2-5 % of Q range)
        regularizes the posterior to a width the flow can capture.

        See ADR-014 for the diagnostic chain that established this.

        Args:
            Q: simulated curves, shape (B, T), values in [0, 1].
            generator: optional torch generator for reproducibility.

        Returns:
            Q_obs: noisy observations, shape (B, T), clipped to [0, 1].
        """
        if self.noise_sigma <= 0:
            return Q
        if generator is None:
            noise = torch.randn_like(Q) * self.noise_sigma
        else:
            noise = torch.randn(Q.shape, generator=generator, device=Q.device) * self.noise_sigma
        return (Q + noise).clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Synthetic-data generation
    # ------------------------------------------------------------------
    def generate_synthetic_pairs(
        self,
        n: int,
        seed: int = 0,
        batch_size: int = 1024,
        add_noise: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Sample `(θ, Q)` pairs from prior + simulator (+ obs noise).

        Batched through the simulator to amortize torchdiffeq overhead.
        Returns CPU tensors regardless of device (training moves them).

        `add_noise=True` (default) injects observation noise via
        `add_obs_noise`. SBC and training must both call with the same
        setting; setting `False` is for diagnostic purposes only.
        """
        torch.manual_seed(seed)
        theta = self.simulator.sample_prior(n)
        Q_chunks = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            with torch.no_grad():
                Q_chunks.append(self.simulator.simulate(theta[start:end], self.t_grid))
        Q = torch.cat(Q_chunks, dim=0)
        if add_noise:
            Q = self.add_obs_noise(Q)
        return theta, Q

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        n_simulations: int = 50_000,
        training_batch_size: int = 256,
        max_num_epochs: int = 200,
        seed: int = 0,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        verbose: bool = True,
    ) -> dict:
        """Generate synthetic data and train the conditional flow."""
        if verbose:
            print(f"[posterior] generating {n_simulations} synthetic (θ, Q) pairs...")
        theta, Q = self.generate_synthetic_pairs(n_simulations, seed=seed)

        if verbose:
            print(f"[posterior] training NPE on (θ ∈ {tuple(theta.shape)}, Q ∈ {tuple(Q.shape)})")
        current_batch_size = training_batch_size
        while True:
            self._inference = NPE(
                prior=self._prior_on_device(),
                density_estimator=self._density_estimator_builder,
                device=self.device,
            )
            # Keep the full synthetic training set on CPU. On laptop-class
            # WDDM GPUs, eagerly moving the entire dataset to CUDA can push
            # peak memory over the edge before sbi's mini-batch loop even
            # starts. Let sbi own the host->device transfer path during
            # training instead of duplicating the dataset on GPU up front.
            self._inference.append_simulations(theta, Q)
            try:
                self._estimator = self._inference.train(
                    training_batch_size=current_batch_size,
                    max_num_epochs=max_num_epochs,
                    validation_fraction=validation_fraction,
                    stop_after_epochs=stop_after_epochs,
                    show_train_summary=verbose,
                )
                break
            except RuntimeError as exc:
                msg = str(exc).lower()
                is_oom = "out of memory" in msg or "bad allocation" in msg
                if not (is_oom and str(self.device).startswith("cuda") and current_batch_size > 32):
                    raise
                next_batch_size = max(32, current_batch_size // 2)
                if next_batch_size == current_batch_size:
                    raise
                if verbose:
                    print(
                        "[posterior] CUDA OOM during NPE.train(); "
                        f"retrying with smaller batch size {current_batch_size} -> {next_batch_size}"
                    )
                self._inference = None
                self._estimator = None
                self._posterior = None
                gc.collect()
                torch.cuda.empty_cache()
                current_batch_size = next_batch_size
        self._estimator.to(self.device)
        self._posterior = self._inference.build_posterior(self._estimator)
        # sbi 0.26 stores per-epoch validation log probs but no explicit
        # 'best' key. Derive best from max of recorded series, robust to
        # future renames.
        summary = getattr(self._inference, "_summary", {})

        def last(*candidates: str):
            for key in candidates:
                values = summary.get(key)
                if values:
                    try:
                        return float(values[-1])
                    except (TypeError, ValueError):
                        return values[-1]
            return None

        def best(*candidates: str):
            for key in candidates:
                values = summary.get(key)
                if values:
                    try:
                        return float(max(values))
                    except (TypeError, ValueError):
                        return None
            return None

        self._training_log = {
            "n_simulations": n_simulations,
            "batch_size": current_batch_size,
            "max_epochs": max_num_epochs,
            "seed": seed,
            "epochs_trained": last("epochs_trained", "epochs"),
            "best_val_log_prob": best("validation_log_probs", "best_validation_log_prob"),
            "final_train_loss": last("training_log_probs", "training_loss", "train_loss"),
            "final_val_loss": last("validation_log_probs", "validation_loss", "val_loss"),
        }
        return self._training_log

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def sample(
        self,
        Q_obs: Tensor,
        n_samples: int = 1000,
        show_progress_bars: bool = False,
    ) -> Tensor:
        """Posterior samples `θ ~ q_φ(θ | Q_obs)`.

        Q_obs must be on the canonical `t_grid`. Real curves should be
        interpolated upstream (see `data.interpolate_to_grid`).
        """
        self._require_trained()
        if Q_obs.ndim == 1:
            Q_obs = Q_obs.unsqueeze(0)  # sbi expects a leading sample dim
        if Q_obs.shape[-1] != len(self.t_grid):
            raise ValueError(
                f"Q_obs last dim {Q_obs.shape[-1]} != t_grid len {len(self.t_grid)}. "
                f"Interpolate the real curve to the canonical grid first."
            )
        Q_obs = Q_obs.to(self._estimator_device())
        return self._posterior.sample(
            (n_samples,), x=Q_obs.squeeze(0), show_progress_bars=show_progress_bars
        )

    def log_prob(self, theta: Tensor, Q_obs: Tensor) -> Tensor:
        """log q_φ(θ | Q_obs)."""
        self._require_trained()
        if Q_obs.ndim == 1:
            Q_obs = Q_obs.unsqueeze(0)
        device = self._estimator_device()
        theta = theta.to(device)
        Q_obs = Q_obs.to(device)
        return self._posterior.log_prob(theta, x=Q_obs.squeeze(0))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        self._require_trained()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "estimator_state_dict": self._estimator.state_dict(),
                "t_grid": self.t_grid,
                "flow": self.flow,
                "hidden_features": self.hidden_features,
                "num_transforms": self.num_transforms,
                "embedding_output_dim": self.embedding_output_dim,
                "embedding_hidden": self.embedding_hidden,
                "embedding_layers": self.embedding_layers,
                "noise_sigma": self.noise_sigma,
                "training_log": self._training_log,
                "simulator_class": type(self.simulator).__name__,
                "n_params": self.simulator.n_params,
                "param_names": self.simulator.param_names,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        simulator: ReleaseSimulator,
        device: str = "cpu",
    ) -> CurvePosterior:
        """Load a trained CurvePosterior. The simulator must match the one
        used for training (same class, same n_params)."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        if ckpt["simulator_class"] != type(simulator).__name__:
            raise ValueError(
                f"Checkpoint was trained with {ckpt['simulator_class']}, "
                f"got {type(simulator).__name__}"
            )
        cp = cls(
            simulator=simulator,
            t_grid=ckpt["t_grid"],
            flow=ckpt["flow"],
            hidden_features=ckpt["hidden_features"],
            num_transforms=ckpt["num_transforms"],
            embedding_output_dim=ckpt.get("embedding_output_dim", 16),
            embedding_hidden=ckpt.get("embedding_hidden", 64),
            embedding_layers=ckpt.get("embedding_layers", 3),
            noise_sigma=ckpt.get("noise_sigma", 0.03),
            device=device,
        )
        # Rebuild the inference object so we can restore the estimator weights.
        cp._inference = NPE(
            prior=cp._prior_on_device(),
            density_estimator=cp._density_estimator_builder,
            device=device,
        )
        # We need at least one (theta, x) pair to instantiate the estimator
        # architecture inside NPE before loading weights.
        dummy_theta, dummy_Q = cp.generate_synthetic_pairs(n=4, seed=0, batch_size=4)
        dummy_theta = dummy_theta.to(device)
        dummy_Q = dummy_Q.to(device)
        cp._inference.append_simulations(dummy_theta, dummy_Q)
        cp._estimator = cp._inference._build_neural_net(dummy_theta, dummy_Q)
        cp._estimator.load_state_dict(ckpt["estimator_state_dict"])
        cp._estimator.to(device)
        cp._posterior = cp._inference.build_posterior(cp._estimator)
        cp._training_log = ckpt.get("training_log")
        return cp

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _require_trained(self) -> None:
        if self._posterior is None:
            raise RuntimeError(
                "CurvePosterior has not been trained. Call .train() first, "
                "or .load() from a saved checkpoint."
            )

    def _estimator_device(self) -> torch.device:
        if self._estimator is None:
            return torch.device(self.device)
        try:
            return next(self._estimator.parameters()).device
        except StopIteration:
            pass
        try:
            return next(self._estimator.buffers()).device
        except StopIteration:
            return torch.device(self.device)

    def _prior_on_device(self) -> Distribution:
        """Return simulator prior on the training device.

        `sbi` checks that prior parameters live on the same device as the
        density estimator. Our simulator keeps the canonical prior CPU-side
        for general use, so the posterior wrapper mirrors it onto CUDA only
        for training/loading.
        """
        prior = self.simulator.prior()
        if self.device == "cpu":
            return prior
        base = getattr(prior, "base_dist", None)
        if isinstance(prior, Independent) and isinstance(base, Uniform):
            return Independent(
                Uniform(base.low.to(self.device), base.high.to(self.device)),
                prior.reinterpreted_batch_ndims,
            )
        raise TypeError(f"Unsupported prior type for device transfer: {type(prior)!r}")

    @property
    def is_trained(self) -> bool:
        return self._posterior is not None


# ======================================================================
# Stage 2: r_ψ(θ | x)
# ======================================================================
class DescriptorPosterior:
    """Amortized neural posterior `r_ψ(θ | x)` over formulation descriptors.

    Workflow:
        feat = FormulationFeaturizer.fit(df_unique)
        enc = MLPFormulationEncoder(input_dim=feat.input_dim, embed_dim=64)
        rp = DescriptorPosterior(sim, feat, enc)
        targets = rp.build_teacher_targets(q_phi, Q_grid, K=64)
        rp.train(x_features, targets)
        samples = rp.sample(x_features_new, n_samples=1000)

    Trained via knowledge distillation from `CurvePosterior` q_φ; see
    ADR-018 for the choice of KL training target over a cascade point
    estimate.

    sbi's `NPE.train` does not expose `weight_decay`; regularization in
    v0 relies on encoder-level dropout (default 0.1) plus early stopping
    via `stop_after_epochs`. If cross-DOI generalization shows
    overfitting we can switch to a custom training loop.
    """

    def __init__(
        self,
        simulator: ReleaseSimulator,
        featurizer: FormulationFeaturizer,
        encoder: MLPFormulationEncoder,
        flow: str = "maf",
        hidden_features: int = 64,
        num_transforms: int = 5,
        device: str = "cpu",
    ) -> None:
        if encoder.input_dim != featurizer.input_dim:
            raise ValueError(
                f"encoder.input_dim={encoder.input_dim} != "
                f"featurizer.input_dim={featurizer.input_dim}"
            )
        self.simulator = simulator
        self.featurizer = featurizer
        self.encoder = encoder
        self.flow = flow
        self.hidden_features = hidden_features
        self.num_transforms = num_transforms
        self.device = device

        # The encoder itself is the embedding net. sbi's posterior_nn
        # threads it into the flow's conditioner, so encoder weights are
        # trained jointly with the flow under the NPE-C log-likelihood.
        self._density_estimator_builder = posterior_nn(
            model=flow,
            hidden_features=hidden_features,
            num_transforms=num_transforms,
            embedding_net=encoder,
        )
        self._inference: NPE | None = None
        self._estimator = None
        self._posterior = None
        self._training_log: dict | None = None

    # ------------------------------------------------------------------
    # Teacher target generation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def build_teacher_targets(
        self,
        q_phi: CurvePosterior,
        Q_grid: Tensor,
        K: int = 64,
        seed: int = 0,
        show_progress_bars: bool = False,
    ) -> Tensor:
        """Sample K θ samples from `q_φ(θ | Q_grid_i)` per curve.

        Args:
            q_phi: trained Stage-1 CurvePosterior, used as a frozen
                teacher.
            Q_grid: real curves projected onto `q_phi.t_grid`, shape
                `(N, T)`.
            K: samples per curve. KL is approximated as a Monte-Carlo
                expectation, so larger K means tighter approximation
                but more compute.

        Returns:
            theta_targets: shape `(N, K, n_params)`, on CPU.
        """
        if Q_grid.ndim != 2:
            raise ValueError(f"Q_grid must be 2-D, got shape {tuple(Q_grid.shape)}")
        if not q_phi.is_trained:
            raise RuntimeError("q_phi is not trained")
        torch.manual_seed(seed)
        out = []
        for i in range(Q_grid.shape[0]):
            theta_k = q_phi.sample(
                Q_grid[i], n_samples=K, show_progress_bars=show_progress_bars,
            )
            out.append(theta_k.cpu())
        return torch.stack(out, dim=0)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        x: Tensor,
        theta_targets: Tensor,
        training_batch_size: int = 256,
        max_num_epochs: int = 200,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        seed: int = 0,
        verbose: bool = True,
    ) -> dict:
        """Train r_ψ on stacked `(x_repeated, theta_k)` pairs.

        Args:
            x: per-curve features from `featurizer.transform`, shape
                `(N, input_dim)`.
            theta_targets: samples from `q_φ` per curve, shape
                `(N, K, n_params)`, as produced by
                `build_teacher_targets`.
        """
        if x.ndim != 2:
            raise ValueError(f"x must be 2-D, got shape {tuple(x.shape)}")
        if theta_targets.ndim != 3:
            raise ValueError(
                f"theta_targets must be (N, K, n_params), got {tuple(theta_targets.shape)}"
            )
        N, K, P = theta_targets.shape
        if x.shape[0] != N:
            raise ValueError(
                f"x has {x.shape[0]} rows but theta_targets has {N} curves"
            )
        if self.simulator.n_params != P:
            raise ValueError(
                f"theta_targets last dim {P} != simulator.n_params {self.simulator.n_params}"
            )

        torch.manual_seed(seed)
        # Flatten (N, K, P) -> (N*K, P), and broadcast x to match.
        x_repeated = x.unsqueeze(1).expand(N, K, -1).reshape(N * K, -1)
        theta_flat = theta_targets.reshape(N * K, P)

        if verbose:
            print(
                f"[descriptor-posterior] training on N={N} curves × K={K} = "
                f"{N * K} (θ, x) pairs"
            )

        theta_flat = theta_flat.to(self.device)
        x_repeated = x_repeated.to(self.device)

        self._inference = NPE(
            prior=self._prior_on_device(),
            density_estimator=self._density_estimator_builder,
            device=self.device,
        )
        self._inference.append_simulations(theta_flat, x_repeated)
        self._estimator = self._inference.train(
            training_batch_size=training_batch_size,
            max_num_epochs=max_num_epochs,
            learning_rate=learning_rate,
            validation_fraction=validation_fraction,
            stop_after_epochs=stop_after_epochs,
            show_train_summary=verbose,
        )
        self._estimator.to(self.device)
        self._posterior = self._inference.build_posterior(self._estimator)

        summary = getattr(self._inference, "_summary", {})

        def last(*candidates: str):
            for key in candidates:
                values = summary.get(key)
                if values:
                    try:
                        return float(values[-1])
                    except (TypeError, ValueError):
                        return values[-1]
            return None

        def best(*candidates: str):
            for key in candidates:
                values = summary.get(key)
                if values:
                    try:
                        return float(max(values))
                    except (TypeError, ValueError):
                        return None
            return None

        self._training_log = {
            "n_curves": N,
            "K": K,
            "n_pairs": N * K,
            "batch_size": training_batch_size,
            "max_epochs": max_num_epochs,
            "seed": seed,
            "epochs_trained": last("epochs_trained", "epochs"),
            "best_val_log_prob": best("validation_log_probs", "best_validation_log_prob"),
            "final_train_loss": last("training_log_probs", "training_loss", "train_loss"),
            "final_val_loss": last("validation_log_probs", "validation_loss", "val_loss"),
        }
        return self._training_log

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def sample(
        self,
        x: Tensor,
        n_samples: int = 1000,
        show_progress_bars: bool = False,
    ) -> Tensor:
        """Posterior samples `θ ~ r_ψ(θ | x)` for one or more formulations."""
        self._require_trained()
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.shape[-1] != self.featurizer.input_dim:
            raise ValueError(
                f"x last dim {x.shape[-1]} != featurizer.input_dim "
                f"{self.featurizer.input_dim}"
            )
        x = x.to(self._estimator_device())
        return self._posterior.sample(
            (n_samples,), x=x.squeeze(0), show_progress_bars=show_progress_bars,
        )

    def log_prob(self, theta: Tensor, x: Tensor) -> Tensor:
        """log r_ψ(θ | x)."""
        self._require_trained()
        if x.ndim == 1:
            x = x.unsqueeze(0)
        device = self._estimator_device()
        theta = theta.to(device)
        x = x.to(device)
        return self._posterior.log_prob(theta, x=x.squeeze(0))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        self._require_trained()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "estimator_state_dict": self._estimator.state_dict(),
                "flow": self.flow,
                "hidden_features": self.hidden_features,
                "num_transforms": self.num_transforms,
                "encoder_state_dict": self.encoder.state_dict(),
                "encoder_class": type(self.encoder).__name__,
                "encoder_input_dim": self.encoder.input_dim,
                "encoder_embed_dim": self.encoder.embed_dim,
                "featurizer": {
                    "continuous_cols": self.featurizer.continuous_cols,
                    "polymer_families": self.featurizer.polymer_families,
                    "mean": self.featurizer.mean,
                    "std": self.featurizer.std,
                },
                "training_log": self._training_log,
                "simulator_class": type(self.simulator).__name__,
                "n_params": self.simulator.n_params,
                "param_names": self.simulator.param_names,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        simulator: ReleaseSimulator,
        device: str = "cpu",
    ) -> DescriptorPosterior:
        """Load a trained DescriptorPosterior. simulator must match."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        if ckpt["simulator_class"] != type(simulator).__name__:
            raise ValueError(
                f"Checkpoint was trained with {ckpt['simulator_class']}, "
                f"got {type(simulator).__name__}"
            )
        feat_ckpt = ckpt["featurizer"]
        featurizer = FormulationFeaturizer(
            continuous_cols=tuple(feat_ckpt["continuous_cols"]),
            polymer_families=tuple(feat_ckpt["polymer_families"]),
            mean=np.asarray(feat_ckpt["mean"], dtype=np.float32),
            std=np.asarray(feat_ckpt["std"], dtype=np.float32),
        )
        if ckpt["encoder_class"] != "MLPFormulationEncoder":
            raise NotImplementedError(
                f"Loading encoder class {ckpt['encoder_class']!r} not supported"
            )
        encoder = MLPFormulationEncoder(
            input_dim=ckpt["encoder_input_dim"],
            embed_dim=ckpt["encoder_embed_dim"],
        )
        encoder.load_state_dict(ckpt["encoder_state_dict"])

        rp = cls(
            simulator=simulator,
            featurizer=featurizer,
            encoder=encoder,
            flow=ckpt["flow"],
            hidden_features=ckpt["hidden_features"],
            num_transforms=ckpt["num_transforms"],
            device=device,
        )
        rp._inference = NPE(
            prior=rp._prior_on_device(),
            density_estimator=rp._density_estimator_builder,
            device=device,
        )
        # Dummy batch to instantiate the flow before loading weights.
        dummy_theta = simulator.sample_prior(4).to(device)
        dummy_x = torch.zeros((4, featurizer.input_dim), device=device)
        rp._inference.append_simulations(dummy_theta, dummy_x)
        rp._estimator = rp._inference._build_neural_net(dummy_theta, dummy_x)
        rp._estimator.load_state_dict(ckpt["estimator_state_dict"])
        rp._estimator.to(device)
        rp._posterior = rp._inference.build_posterior(rp._estimator)
        rp._training_log = ckpt.get("training_log")
        return rp

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _require_trained(self) -> None:
        if self._posterior is None:
            raise RuntimeError(
                "DescriptorPosterior has not been trained. Call .train() "
                "first, or .load() from a saved checkpoint."
            )

    def _estimator_device(self) -> torch.device:
        if self._estimator is None:
            return torch.device(self.device)
        try:
            return next(self._estimator.parameters()).device
        except StopIteration:
            pass
        try:
            return next(self._estimator.buffers()).device
        except StopIteration:
            return torch.device(self.device)

    def _prior_on_device(self) -> Distribution:
        prior = self.simulator.prior()
        if self.device == "cpu":
            return prior
        base = getattr(prior, "base_dist", None)
        if isinstance(prior, Independent) and isinstance(base, Uniform):
            return Independent(
                Uniform(base.low.to(self.device), base.high.to(self.device)),
                prior.reinterpreted_batch_ndims,
            )
        raise TypeError(f"Unsupported prior type for device transfer: {type(prior)!r}")

    @property
    def is_trained(self) -> bool:
        return self._posterior is not None


# ======================================================================
# Stage 2 helpers: real-curve interpolation + quality flag
# ======================================================================
def interpolate_to_grid(
    t_obs: Tensor,
    Q_obs: Tensor,
    t_grid: Tensor,
    t_max_days: float = 90.0,
) -> tuple[Tensor, str, dict]:
    """Project an irregularly-sampled (t_obs, Q_obs) curve onto t_grid.

    Observed points with t > t_max_days are discarded (q_φ was trained on
    [0, 90] only and extrapolation is unsafe). For grid points within
    [0, t_obs.max()] linear interpolation is used; for grid points
    beyond t_obs.max() the last observed value is held (last-value
    extension). The quality flag warns when the held tail is large
    relative to the observed support.

    Quality bands (per ADR-018):
        high   : tmax >= 60 OR lastQ >= 0.8
        low    : tmax <  90 AND lastQ <  0.7
        medium : everything else

    Returns:
        Q_grid: shape `(T,)`, float32, clamped to `[0, 1]`.
        quality: 'high' | 'medium' | 'low'.
        meta: diagnostic dict with `tmax`, `lastQ`, `n_obs_kept`.
    """
    if t_obs.ndim != 1 or Q_obs.ndim != 1:
        raise ValueError("t_obs and Q_obs must be 1-D")
    if t_obs.shape != Q_obs.shape:
        raise ValueError(f"shape mismatch t_obs {t_obs.shape} vs Q_obs {Q_obs.shape}")

    mask = (t_obs <= t_max_days) & (t_obs >= 0.0)
    t_keep = t_obs[mask]
    Q_keep = Q_obs[mask]
    if t_keep.numel() < 2:
        raise ValueError(
            f"Fewer than 2 observed points within [0, {t_max_days}] days; "
            f"got {int(t_keep.numel())}."
        )

    # numpy.interp clamps to xp endpoints when grid points fall outside
    # [xp.min(), xp.max()], which gives the last-value extension we want
    # for the tail. We also do it for the head in case t_obs[0] > 0.
    Q_grid_np = np.interp(
        t_grid.cpu().numpy(),
        t_keep.cpu().numpy(),
        Q_keep.cpu().numpy(),
    )
    Q_grid = torch.tensor(Q_grid_np, dtype=torch.float32).clamp(0.0, 1.0)

    tmax = float(t_keep.max())
    last_Q = float(Q_keep[-1] if t_keep[-1] == tmax else Q_keep[t_keep.argmax()])
    high = (tmax >= 60.0) or (last_Q >= 0.8)
    low = (tmax < 90.0) and (last_Q < 0.7)
    quality = "high" if high and not low else ("low" if low else "medium")

    meta = {
        "tmax": tmax,
        "last_Q": last_Q,
        "n_obs_kept": int(t_keep.numel()),
    }
    return Q_grid, quality, meta
