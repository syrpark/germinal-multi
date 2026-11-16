# Stage 5 — Final filter

The acceptance decision. Every AbMPNN candidate from stage 4 is cofolded
against every antigen state at a larger seed budget, relaxed as an ensemble,
scored, and tested against the stringent threshold set.

**Code:** `germinal/filters/filter_utils.py` — same
`fold_designs_for_filter` / `multi_run_filters` pair as stage 3, with
`is_final=True`.
**Runs in:** phase D, once per batch of candidates.
**Config:** `configs/filter/final/*.yaml`, selected with
`filter/final=<name>`.
**Writes:** passing designs to `accepted/`, failing ones to `filter_failed/`.

## What `is_final=True` changes

| | Initial | Final |
|---|---------|-------|
| fold seeds | `num_fold_seeds_initial` (3) | `num_fold_seeds_final` (5) |
| samples requested | selected | **all** (`want_all_samples`) |
| FastRelax | single | parallel ensemble |
| ProteinMPNN scoring | no | yes |
| thresholds | lenient | stringent |

The final stage asks the backend for every diffusion sample rather than a
selected one, because the ensemble is what the confidence aggregation is
computed over.

## Aggregation

- **Across targets:** worst, as in stage 3 — minimum for higher-is-better,
  maximum for lower-is-better.
- **Across seeds:** confidence metrics come from the **worst** seed. Seeds
  are independent attempts at the complex, so the worst is the conservative
  estimate.
- **Across diffusion samples within a seed:** the **best**. Samples differ
  only by sampling noise.

This two-axis split is `resolve_selection`; see
[folding-backends.md](folding-backends.md#two-selection-axes-not-one).

- **Across the FastRelax ensemble:** controlled by `relax_score_mode`
  (`n_relax: 5` runs).

  | Value | Meaning |
  |-------|---------|
  | `"best"` (default) | every metric from the lowest-energy run — one real conformation |
  | `"average"` | mean each metric across runs — corresponds to no single structure, kept only for comparison with older runs |

  `score_interface_ensemble` returns the best relaxed PDB, which becomes
  `external_relaxed_pdb` for the rest of the stage. Energy is `binder_score`
  (total Rosetta energy, more negative is better).

## Thresholds

The shipped `scfv_multi` final set:

| Metric | Operator | Value |
|--------|----------|-------|
| `clashes` | `<=` | 0.2 |
| `sc_rmsd` | `<=` | 6 |
| `binder_near_hotspot` | `==` | true |
| `percent_interface_cdr` | `>` | 0.5 |
| `interface_shape_comp` | `>=` | 0.55 |
| `interface_hbonds` | `>=` | 5 |
| `surface_hydrophobicity` | `<` | 0.45 |
| `interface_hydrophobicity` | `>` | 35 |
| `plddt` | `>=` | 0.80 |
| `iptm` | `>=` | 0.73 |
| `ptm` | `>=` | 0.80 |
| `ipsae` | `>=` | 0.55 |

All AND. Keys are stage-prefixed in the YAML and stripped before evaluation.

## Outcomes

| Result | Destination | `designs.csv` row |
|--------|-------------|-------------------|
| pass | `accepted/` | yes |
| fail | `filter_failed/` | yes |

Both are written by `save_trajectory`, which copies the structure flat into
the destination's `structures/`. Every design also gets a row in
`all_trajectories.csv` regardless of outcome, and each failed threshold is
charged in `failure_counts.csv`.

Typical acceptance across the whole pipeline is on the order of a few per
cent.

## Notes

- `redesign_candidates/` is **not** where failed finals go. It is a structure
  store for predictions, and its `designs.csv` is deliberately kept empty.
  This changed; tools written against the older meaning read an empty CSV and
  silently pool nothing. See [output-layout.md](output-layout.md).
- Candidates are folded in batches rather than one per candidate, so a
  design with five variants pays one container start instead of five. The
  batch size comes from `folding.resolve_batch_size(run_settings, "filter")`,
  and each batch is scored before the next is folded, so an interrupted run
  loses one batch rather than all of them.
