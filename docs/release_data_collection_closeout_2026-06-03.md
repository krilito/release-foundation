## Release Data Collection Closeout

Date: `2026-06-03`

Scope:

- close the current source-intake / corpus-standardization wave on
  `codex/curve-world-probes`
- freeze which assets belong in the clean cumulative pool, the caveat pool,
  the proxy-only pool, and provenance-only storage
- stop generic source hunting until the next model-side probes actually need it

Evidence base:

1. `outputs/127_release_data_asset_registry/asset_registry.csv`
2. `outputs/128_external_source_intake_registry/source_intake_registry.csv`
3. `outputs/147_layered_release_corpus_registry/layered_asset_registry.csv`
4. `outputs/148_release_main_cumulative_v1/manifest.json`
5. `outputs/149_release_caveat_augmented_cumulative_v1/manifest.json`

## Verdict

The collection phase is closed enough for the current branch.

We are no longer data-starved for the next curve-first probes. The right move
now is not "collect more random sources"; it is "run shape-family and
cross-system structure tests on the clean cumulative pool first."

## What We Have

### Clean cumulative pool

Primary modeling pool:

- `outputs/148_release_main_cumulative_v1/`
- `751` curves
- `12,804` release observations
- `751` formulation rows
- `8` component corpora

Component corpora:

1. `release_corpus_v1`
2. `external_liposome_ivr`
3. `external_bmp2_hydrogel_zenodo`
4. `external_degrapol_mendeley`
5. `external_alginate_mendeley`
6. `external_starch_mendeley`
7. `external_caseinate_gallic_mendeley`
8. `external_golfball_microspheres_mendeley`

This is the default pool for any Probe 2 / Probe 3 analysis.

### Caveat cumulative pool

Sensitivity-only augmentation:

- `outputs/149_release_caveat_augmented_cumulative_v1/`
- `784` curves
- `13,153` release observations
- adds `33` curves and `349` observations beyond the clean pool

Caveat assets:

1. `external_timp1_mendeley`
2. `external_nt3_hydrogel_mendeley`
3. `external_laponite_mendeley`
4. `external_plga_dox_mendeley`
5. `external_halloysite_origin_mendeley`

These are usable for robustness checks, but they should not silently define the
main claim-facing corpus.

### Proxy-only assets

Do not merge into cumulative-release modeling pools:

- `external_chitosan_zeolite_mendeley`
- `external_caseinate_guar_proxy_mendeley`
- `external_cure_proxy_mendeley`

Current total:

- `22` curves
- `191` observations

Reason:

- these are release-level proxies (absorbance / concentration / HPLC signal),
  not the same target as cumulative percent release

### Metadata-only and provenance-only assets

Keep, but do not count as extra curve supervision:

- `external_nanomed_registry`: `271` metadata entries, `0` local curves
- `bannigan_lai_181_raw`: provenance copy for transformation audit
- `external_laponite_nh2_mendeley`: superseded partial extraction
- `external_laponite_opju_registry`: provenance registry retained after numeric recovery

## What Is Closed

Closed as standardized and usable:

- PLGA base corpus
- liposome IVR corpus
- BMP-2 hydrogel corpus
- DegraPol corpus
- alginate corpus
- starch corpus
- caseinate-gallic corpus
- golfball microsphere corpus
- layered registry
- materialized clean/caveat merged pools

Closed as intentionally non-mainline:

- proxy corpora
- caveat corpora
- provenance registries

Closed as blocked / not worth forcing right now:

1. `methylcellulose_hydrogel_zenodo_17360516`
   provider-side `403` payloads instead of CSV
2. `plos_figshare_28819167`
   local file is zero bytes
3. `phage_hydrogel_zenodo_17252016`
   downloaded, but not a clean release-curve corpus
4. `hs5bgtstk3_mendeley`
   document-only handoff, no numeric release tables found
5. `external_nanomed_registry`
   metadata exists, raw release curves are still absent locally

## Operational Rule Going Forward

Use these pools explicitly:

1. main experiments: `outputs/148_release_main_cumulative_v1/`
2. sensitivity reruns: `outputs/149_release_caveat_augmented_cumulative_v1/`
3. proxy analyses: separate route only, never mixed into cumulative targets
4. metadata analyses: descriptor-side only, never counted as curve data

## Next Step

Do not spend the next cycle on more intake unless a missing mechanism becomes a
specific blocker.

The next honest step is:

1. run shape-family sufficiency on `release_main_cumulative_v1`
2. if needed, rerun the same audit on `release_caveat_augmented_cumulative_v1`
3. only then decide whether a larger curve-world route is justified

Short version:

- data collection is no longer the bottleneck
- target mismatch is the real bottleneck
- the branch should now move from intake to structure-testing
