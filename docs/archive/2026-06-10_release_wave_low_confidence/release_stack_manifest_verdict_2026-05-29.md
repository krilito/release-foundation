# Release Stack Manifest Verdict

Date: `2026-05-29`

Purpose:

```text
State the strongest current object-level version of the unified release
claim: the stack now has a combined manifest binding family objects and
target-layer defaults.
```

Primary outputs:

- [release_stack_manifest.json](D:/release-foundation/outputs/108_release_stack_manifest/release_stack_manifest.json)
- [stack_manifest_summary.csv](D:/release-foundation/outputs/108_release_stack_manifest/stack_manifest_summary.csv)
- [summary.txt](D:/release-foundation/outputs/108_release_stack_manifest/summary.txt)

Code:

- [108_release_stack_manifest.py](D:/release-foundation/scripts/108_release_stack_manifest.py)

## 1. What now exists

The project no longer just has:

- a posterior-family object
- a target registry

It now also has a combined manifest that binds them per mechanism.

That means each mechanism entry now carries, in one place:

- family example / decoder handle
- latent dimension / rank
- timing defaults
- shape defaults
- target-layer status

## 2. Why this matters

This is the strongest current executable version of the bold claim:

> drug release can be unified at the stack-contract level even when decoding
> remains mechanism-specific.

## 3. Bottom line

The unified release-intelligence story is no longer just:

- a benchmark idea
- a posterior-family schema
- a target registry

It is now also:

> a mechanism-indexed stack manifest that binds latent-family objects and
> reporting-target defaults into one executable contract.
