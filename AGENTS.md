# AGENTS.md

Instructions for AI assistants (Claude Code, Cursor, Copilot, Aider, ...)
working in this repository. Humans should also read this — it doubles as the
project's coding contract.

**General behavioral guidelines live in [CLAUDE.md](CLAUDE.md)** (the four
Karpathy principles: think before coding, simplicity first, surgical changes,
goal-driven execution). Read that first. This file adds project-specific
hard constraints on top.

## North star

This codebase has a long-horizon ambition toward cross-mechanism drug-release
modeling, starting from a PLGA-only amortized simulation-based inference (SBI)
baseline. Do not describe the current paper as a "foundation model" unless the
data scale and joint multi-mechanism training criteria are actually met. The
architecture must remain extensible to new release mechanisms (LNP, liposome,
microsphere, hydrogel) without rewrites.

**Read [ARCHITECTURE.md](ARCHITECTURE.md) before making non-trivial changes.**

## Current research framing

As of 2026-06-12, do **not** optimize the paper story around "RMSE beats all
black boxes" or "we propose a stronger predictor." The locked interpretation is
an information-budget / release-state inference result:

```text
drug release prediction is release-state inference under limited
pre-experimental descriptors and costly observations;
static descriptors provide a prior over latent release state;
measured early observations expose missing kinetic state;
after early Q is known, point prediction can become close to information-saturated.
```

Read `docs/project_archaeology_inventory_2026-06-12.md` before starting any
new modeling work, and read
`docs/plga_information_budget_claim_table_2026-06-11.md` before touching PLGA.

For release-state forensics work and scripts `120+`, read
`docs/release_state_inference_formalism_2026-06-13.md` first and use
`release_state.py` as the interface contract. Residuals are valid only inside
one declared `ReleaseStateSpace`; data-driven residual symptoms must be decoded
into time-domain perturbations before receiving kinetic labels. Script `120`
is residual mapping only. MSVS / candidate measurement scoring starts after
the residual map exists.

For screening-platform work, read
`docs/release_screening_platform_design_contract_2026-06-14.md`,
`docs/release_screening_platform_mathematical_contract_2026-06-14.md`, and
`docs/release_screening_platform_formal_groundwork_2026-06-14.md` before
opening any new route or utility experiment. The active gate from Experiment
123 is:

```text
X_i^base and optional state proxies must stay separated;
R_i^model and R_i^audit must stay separated;
future-window scoring must exclude observed early points;
route and neural screening-platform modeling remain blocked until factual
preparation-route fields are standardized cleanly enough to instantiate R_i^model.
```

Prioritize experiments that measure observation-budget value, uncertainty
contraction, stopping rules, and mechanism-vs-shape-prior controls. Treat
`best-route` gains as exploratory oracle evidence until re-tested with one
frozen route, equal early timepoints, equal future grid, equal HPO budget, and
paired CIs.

The next cycle is consolidation only unless explicitly superseded: unified
tables, unified split taxonomy, claim-strength ladder, and manuscript synthesis.
Do not open new SRDS, PySR-main, RSSM/world-model, LNN/CNP, KAN, PINN, RL,
foundation-model, MoE, or generic neural architecture sprints unless a
preregistered diagnostic first states which gap is being closed:
`Q(t)->z`, `X->z`, `X+Y_k->z`, or `z/Q_hat(t)->design`.

## Hard constraints

1. **Mechanism modules are subclasses of `ReleaseSimulator`.**
   Never hard-code PLGA-specific logic into training loops, encoders, or
   posterior networks. New mechanisms drop in as siblings of `PLGABiphasic`,
   they do not require rewrites elsewhere.

2. **Descriptor encoders are subclasses of `FormulationEncoder`.**
   The MLP encoder is Phase 1 only. Phase 2 swaps in a transformer over
   formulation tokens. Anything that consumes encoder output depends only on
   the abstract interface (`embed_dim` contract).

3. **All randomness is seeded.**
   Every script accepts `--seed` and seeds torch, numpy, python `random`,
   and any sbi/scipy code paths. Default seed: `0`.

4. **No symbolic regression in main training loops.**
   PySR-style search has been retired (see ADR-003). Symbolic distillation,
   if needed for interpretation, runs offline on trained models in
   `scripts/post_*.py`. Not in `train.py`. Not in `posterior.py`.

5. **Real data lives in `data/` and is gitignored.**
   Synthetic data is generated on the fly from the simulator. Do not commit
   simulated trajectories — they regenerate from seeds.

6. **One source of truth per concept.**
   The ODE form lives in `simulator.py` and nowhere else. Hyperparameters
   live in `configs/*.yaml` and nowhere else. If a number appears twice,
   one of them is wrong.

## Style (Karpathy-flavored minimalism)

- **Flat repo.** One concept per file at the root. Avoid nested packages.
- **Comments explain WHY** (physics choice, identifiability concern,
  sparse-data caveat), not WHAT (which the code already says).
- **Scripts in `scripts/` are numbered** (`01_`, `02_`, ...) and documented
  at the top: what they consume, what they produce, expected runtime.
- **Type hints on public function signatures.** `from __future__ import
  annotations` at the top of every `.py` file.
- **Pure functions over classes** unless statefulness is essential. ABCs
  are reserved for the two extension points (Simulator, Encoder).

## Hard YES list (process rules)

1. **Evidence before configuration changes.** Before proposing any change
   to the prior, the t_grid, the ODE form, or the simulator's parameter
   bounds, the assistant MUST first produce a diagnostic that supports
   the change. A reasoned argument without a measurement is not enough.
   This rule exists because ADR-011 (t_grid 90 → 180) was reasoned to but
   measured against, and turned out to hurt; ADR-012 reverted it.

   Exceptions: trivial fixes (typos, type errors, obvious bugs) and
   changes the user has explicitly asked for.

2. **Check `git status` before editing files the user might be editing.**
   When the assistant is about to edit a file, run `git status --short`
   first. If the target file shows `M` (modified, uncommitted), the user
   has work in progress there. Surface this in the response and ask
   before overwriting. When the assistant commits with `git add -A`, any
   uncommitted user work in the same file is silently swept into the
   assistant's commit — see the post-mortem on ADR-013 / ADR-014, where
   the user's local device-alignment patch on `scripts/08_pipeline_sanity.py`
   was absorbed into the assistant's noise-injection commit, causing
   ambiguity about who owned which line.

   This applies to file-write operations as well as `git add -A`; prefer
   adding specific paths over `-A` when uncommitted user changes exist.

## Hard NO list

- ❌ Editing ARCHITECTURE.md to match drifting code. Architecture leads;
  code follows. If you must deviate, append a new entry to DECISIONS.md
  **first**, then change code.
- ❌ Committing `outputs/`, `wandb/`, `data/`, `*.pt`, `*.pkl`, `__pycache__/`.
- ❌ `pip install` directly. Use `uv add`. The lockfile is canonical.
- ❌ Silently swallowing exceptions in the simulator or ODE solver.
  Numerical failures must surface so the user can diagnose stiffness or
  parameter-range issues.
- ❌ Adding a dependency to fix a one-off bug. Every entry in
  `pyproject.toml` is justified in DECISIONS.md.
- ❌ Re-introducing the legacy SRDS modules (KAN teacher, PySR scaffold
  search, LightGBM mapper). They are deliberately not here.
- ❌ Treating LNN/CNP/RSSM or any new neural architecture as the next natural
  step without a release-state diagnostic. Historical runs demote these routes
  to ablations or negative controls, not active claims.
- ❌ Writing the manuscript as "we developed a better release-curve predictor."
  The current claim is information budget, release-state inference, and
  experiment-aware design utility.

## Active Observer Research (Sprint 3+)

Scripts `68_*` implement a Particle Active Kinetic Observer for PLGA release.
Full constraints and research log in `research/active_observer/CLAUDE.md`.

Key invariant: `simulate_plga_ode` MUST use `PLGABiphasic.simulate_numpy()`.
No surrogate, no neural ODE, no custom solver.

## Conversational style with the user

The user is the lab's sole computational lead, self-taught Python, fast
learner, ambitious. Their PhD advisor is the corresponding author and
fully supports this direction.

- Be direct. Push back on bad ideas.
- Explain physics when introducing new ODE terms, priors, or constraints.
- Do not sandbag — they want expert-level critique, not encouragement.
- When uncertain, say so and propose a diagnostic, not a guess.
- Keep responses focused. Code blocks over prose when code is the answer.
