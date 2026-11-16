# Stage 4 — Redesign and recycling

Everything that passed the initial filter gets its sequence re-proposed by a
multi-state ProteinMPNN (AbMPNN), producing candidate variants for the final
filter. Runs with `recycling: true` additionally loop
predict → redesign → predict, keeping a candidate only while it improves.

**Code:** `germinal/design/redesign.py` (single/multi-state redesign),
`germinal/design/recycle.py` (the loop), `external/proteinmpnn/`
(Kuhlman-Lab multi-state ProteinMPNN).
**Runs in:** phase C.
**Writes:** `redesign_candidates/structures/{design}/` — recycle predictions,
nested per design.

## How `num_seqs` reaches ProteinMPNN

```
config run/*.yaml : num_seqs: 16
  → run_settings["num_seqs"]
  → redesign.py      num_seqs = run_settings.get("num_seqs", 40)
  → CLI             --num_seq_per_target 16
```

`num_seqs` is how many sequences ProteinMPNN **samples**. It is not how many
become candidates.

## How the sampled set is cut down

```python
if max_sequences is None:
    max_sequences = run_settings.get("max_mpnn_sequences", 4)
result_seqs = novel_seqs[:max_sequences]
```

`novel_seqs` are the sampled sequences that differ from the input, in
ProteinMPNN's own output order. The first `max_sequences` are returned.

Two callers, two different caps:

| Caller | Cap | Why |
|--------|-----|-----|
| non-recycling redesign | `max_mpnn_sequences` (config) | every returned variant becomes a real final-filter candidate |
| recycle loop | `max_sequences=1` at the call site | the loop needs one sequence to carry forward, not a set |

`max_mpnn_sequences` is deliberately **not** read at the recycle call site.
The loop states its own cap explicitly so the config key governs only the
non-recycling path. The effective value is recorded in `final_config.yaml`.

## `mpnn_fix_interface`

Holds interface residues fixed during redesign, so ProteinMPNN re-proposes
only the rest.

```python
fix_interface = run_settings.get("mpnn_fix_interface", False)
if fix_interface and atom_distance_cutoff > 0.0:
    ...
```

Both conditions are required. `atom_distance_cutoff` gates the second half,
and it is `0.0` until a recycle improves — so `mpnn_fix_interface: true`
alone switches nothing on at the first pass. This was a real bug: the cutoff
was previously consulted in a way that made the flag a no-op.

### `interface_ratchet`

When `interface_ratchet: true`, the cutoff is raised to
`fix_interface_cutoff` once a recycle improves — "not yet" is `0.0`, so the
interface is progressively frozen as the design gets better rather than
fixed from the start against an interface that has not formed.

```yaml
mpnn_fix_interface: true
interface_ratchet: true
fix_interface_cutoff: 3.0   # only consulted when interface_ratchet is true
```

## The recycling loop

Only runs when the run config sets `recycling: true`. The runner prints its
state at startup: `Hallucination: full | recycling: False`.

```
for recycle in 1..num_recycles:
    redesign (max_sequences=1)
    fold the new sequence           num_fold_seeds_recycle seeds
    keep the best backbone so far, and continue
```

`recycle_ipsae_threshold` does **not** stop the loop. It is applied afterwards,
deciding which of the scored candidates reach the final filters; a design with
none above it is charged `recycling_ipsae_gate`. The loop itself truncates only
on a redesign failure, an incomplete prediction, or a scoring failure.

| Key | Shipped value | Meaning |
|-----|---------------|---------|
| `num_recycles` | 5 | recycle iterations |
| `num_recycles_design` | 3 | recycles inside the design model |
| `num_fold_seeds_recycle` | 1 | folding seeds per recycle fold |
| `recycle_ipsae_threshold` | 0.4 | ipSAE gate to continue |

The recycle stage folds with `recycle_folder`, which defaults to `af3` as the
filter stage does. The two are independent, so the recycle loop can be switched
to `protenix` on its own — see [folding-backends.md](folding-backends.md).

Each design's recycle predictions go in their own subdirectory
`redesign_candidates/structures/{design}/`, so discarding one design's output
is a directory removal rather than a glob over filenames. **Every recycle is
currently retained**; there is no deletion pass, and a glob over
`_recycle*.pdb` would be the wrong way to add one -- those filenames are
referenced by CSV rows.

## Failure accounting

A design whose redesign fails is recorded against the **initial** stage, not
the final one — the final filter never ran for it, so charging it there would
inflate final-stage failure counts with designs that never reached them.

## Notes

- The multi-state path uses Kuhlman-Lab ProteinMPNN with AbMPNN weights at
  `external/proteinmpnn/run/model_weights/abmpnn.pt`.
- Constraint syntax: `;` separates constraint groups, `,` separates states
  within a group, so `,` cannot appear inside a residue spec. Use ranges
  (`B30-B33`), not comma-separated positions.
