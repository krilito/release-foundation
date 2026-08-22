"""SimulatorDecoder — wraps any ReleaseSimulator into a CASP-friendly call.

Why a thin wrapper. CASP's encoder/ELBO must be simulator-agnostic. The
decoder is the only place that touches the simulator API. If a new
mechanism (Weibull, LNP, hydrogel) needs different t-grid handling or
clamping, the change lives here, not in the encoder.

Also: theta from the encoder can be sampled (shape (S, B, P)) or a single
batch ((B, P)). The decoder flattens, calls simulate, and reshapes back.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from simulator import ReleaseSimulator


class SimulatorDecoder(nn.Module):
    def __init__(self, simulator: ReleaseSimulator) -> None:
        super().__init__()
        self.simulator = simulator
        self.n_params = simulator.n_params

    def forward(self, theta: Tensor, t: Tensor) -> Tensor:
        """Simulate Q(t) for each theta.

        theta shape:  (B, P) or (S, B, P)
        t shape:      (T,)  with t[0] > 0 OK (we prepend 0 internally)
        return shape: (B, T) or (S, B, T) — Q evaluated at the input t.

        Why we prepend 0: torchdiffeq.odeint uses t[0] as the integration
        START time. The simulator's initial state (h=0, m=1, Q=0) is
        physical at t=0, not at t=1. Calling simulate with t=[1,3,...]
        without a leading 0 would misplace the initial condition at t=1
        and silently drop the t=0..t=t[0] dynamics — including the burst
        release, which usually completes in <1 day. This was a bug in
        the first FIB v0 run (R^2 dropped from 0.957 to 0.50 just from
        this offset; cumulative bias propagates through the whole curve).
        """
        t = t.to(theta.device)
        if t[0].item() != 0.0:
            t_full = torch.cat([torch.zeros(1, device=t.device, dtype=t.dtype), t])
            slice_start = 1
        else:
            t_full = t
            slice_start = 0

        if theta.dim() == 2:
            Q = self.simulator.simulate(theta, t_full)              # (B, T+1)
            return Q[..., slice_start:].clamp(0.0, 1.0)
        if theta.dim() == 3:
            S, B, P = theta.shape
            flat = theta.reshape(S * B, P)
            Q = self.simulator.simulate(flat, t_full)               # (S*B, T+1)
            Q = Q[..., slice_start:]
            return Q.reshape(S, B, -1).clamp(0.0, 1.0)
        raise ValueError(f"theta dim must be 2 or 3, got {theta.dim()}")
