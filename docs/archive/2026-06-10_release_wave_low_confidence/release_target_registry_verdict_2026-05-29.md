# Release Target Registry Verdict

Date: `2026-05-29`

Purpose:

```text
State the strongest current executable version of the target-layer claim:
the unified stack now has a mechanism-aware target registry artifact.
```

Primary outputs:

- [release_target_registry.json](D:/release-foundation/outputs/106_release_target_registry/release_target_registry.json)
- [registry_summary.csv](D:/release-foundation/outputs/106_release_target_registry/registry_summary.csv)
- [summary.txt](D:/release-foundation/outputs/106_release_target_registry/summary.txt)

Code:

- [release_target_registry.py](D:/release-foundation/release_target_registry.py)
- [106_release_target_registry_export.py](D:/release-foundation/scripts/106_release_target_registry_export.py)

## 1. What now exists

The target layer is no longer only a memo or csv table.

It now exists as a JSON registry artifact that records, per mechanism:

- shared target ids
- recommendation status
- default targets when allowed
- caveats
- evidence fields

It now covers both:

- `timing_threshold`
- `shape_descriptor`

## 2. Why this matters

This is the executable form of the more honest bold claim:

> unified release intelligence should share target classes and reporting
> contracts, while allowing mechanism-aware defaults and exclusions.

That is a stronger systems claim than:

> we have one universal timing target

and it is much more defensible from current evidence.

## 3. Bottom line

The unified stack now has:

- a posterior-family object
- a target-layer contract
- a target registry artifact

And that registry now spans:

- timing anchors
- shape targets
- mechanism-aware default sets

That means the "shared release-intelligence stack" claim is no longer just
about benchmark philosophy. It now has real object-level interfaces at both
the latent-family layer and the reporting-target layer.
