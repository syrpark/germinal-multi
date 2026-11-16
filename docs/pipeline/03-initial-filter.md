# Stage 3 — Initial filter

The first external structure prediction. Every design that survived
hallucination and the early check is cofolded against every antigen state,
relaxed, scored, and tested against a lenient threshold set. Designs that
pass go on to redesign; designs that fail are dead ends.

**Code:** `germinal/filters/filter_utils.py` —
`fold_designs_for_filter` (batched fold), `multi_run_filters` (scoring).
**Runs in:** phase B, once per batch.
**Config:** `configs/filter/initial/*.yaml`, selected with
`filter/initial=<name>`.
**Writes:** `redesign_candidates/structures/{design}/` — predictions and the
relaxed structures derived from them. Failures are saved to `trajectories/`.

## One fold per batch, not per design

`fold_designs_for_filter` folds the whole batch in a single call: D designs ×
T targets jobs, one container start. See
[folding-backends.md](folding-backends.md#batching) for why this matters.

The batched result is handed to `multi_run_filters` as `prediction_results`.
If it were not, `multi_run_filters` would fold again — silently correct, and
silently paying the container start it just avoided. Pinned by
`tests/test_batched_filter_stages.py`.

Results are matched to targets **positionally**, so a result list of the
wrong length raises `ValueError` naming both counts rather than scoring a
design against the wrong antigen.

A `None` in a result slot is guarded before unpacking and raises naming the
target: one bad design must not discard a whole batch, but it must not be
scored as if it had folded either.

## Budget

| Key | Value in the shipped multi configs |
|-----|------------------------------------|
| `num_fold_seeds_initial` | 3 |
| `num_samples` (folding config) | 5 |
| `num_cycles` (folding config) | 10 |

`num_samples` and `num_cycles` are passed explicitly to the backend. AF3's own
defaults are 5 and 10, so omitting them was not neutral — it meant the
sampling budget was chosen by the container rather than by config, and
silently differed from the Protenix stage, which was being told 1.

## What is measured

Per target, on the relaxed structure:

- **Confidence:** `plddt`, `ptm`, `iptm`, `pae`, `ipsae`
- **Interface:** `interface_shape_comp`, `interface_hbonds`, `interface_dG`,
  `interface_dSASA`
- **Hotspot/CDR:** `binder_near_hotspot`, `percent_interface_cdr`
- **Hydrophobicity:** `surface_hydrophobicity`, `interface_hydrophobicity`
- **Geometry:** `clashes`, `sc_rmsd` (against the AF2 hallucination PDB)

Sequence-level metrics are computed once rather than per target: framework
mutation count (`get_framework_mutations`) and IgLM log-likelihood
(`get_iglm_ll`).

## Aggregation across targets

Acceptance is **per-target AND-filtering**: a design must pass every
threshold against every antigen state. There is no cross-state balance gate.

The design-level value written to the CSVs is the **worst** across targets —
minimum for higher-is-better metrics, maximum for lower-is-better.

## Thresholds

Keys are stage-prefixed in the YAML (`initial.clashes`) and the prefix is
stripped by `strip_stage_prefix` before evaluation. The shipped
`scfv_multi` initial set:

| Metric | Operator | Value |
|--------|----------|-------|
| `clashes` | `<` | 1 |
| `binder_near_hotspot` | `==` | true |
| `percent_interface_cdr` | `>` | 0.5 |
| `interface_shape_comp` | `>=` | 0.50 |

Deliberately lenient — this stage is a triage, not the acceptance decision.

## Notes

- Predictions live under `redesign_candidates/`, not `trajectories/`.
  `trajectories/structures/` holds only the AF2 hallucination PDBs, which are
  the `sc_rmsd` reference. See [output-layout.md](output-layout.md).
- On the `logits` arms this stage is the throughput bottleneck; on the
  `germinal` arms hallucination is.
