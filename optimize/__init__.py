"""optimize — Algorithm improvements for release-foundation.

Four modules targeting the four main bottlenecks identified in the
codebase analysis:

    temporal_embed.py      — Dilated causal conv embedding for Q(t)
    calibration.py         — Temperature scaling + CRPS for posterior calibration
    interaction_encoder.py — Drug-polymer outer-product features
    distill_fib.py         — FIB → amortized posterior distillation (main script)

All changes are additive. No existing modules are modified.
"""
