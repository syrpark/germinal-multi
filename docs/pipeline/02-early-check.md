# Stage 2 — Early check

A free pre-screen between hallucination and the initial filter. It applies the
same clash / hotspot / CDR-fraction gates the initial filter applies, but
reads them off the **AF2 hallucination PDBs** with PyRosetta and a KD-tree
instead of a cofold. A design that cannot pass these geometric gates on its
own hallucinated structure will not pass them after an AF3 fold either, so
rejecting here saves a prediction.

**Code:** `germinal/filters/early_check.py`
**Runs in:** phase A, immediately after each trajectory.
**Writes:** nothing. Rejections are recorded in `failure_counts.csv` and the
design is saved to `trajectories/`.

It lives in `germinal/filters/` rather than `germinal/utils/` because it calls
`filter_utils.compute_hotspot_proximity`, and `filter_utils` imports `utils` —
putting it in `utils` would close an import cycle.

## What it checks, per target

1. **Clashes** — `calculate_clash_score(..., only_ca=True, num_targets=1)` on
   the hallucinated PDB. Any CA clash sets `any_clashes`.
2. **Hotspot proximity** — `compute_hotspot_proximity`, giving `near_hotspot`,
   `cdr3_contacts`, `cdr_contacts`.
3. **CDR fraction at the interface** — against `min_cdr_fraction`.

Each target is checked with its own hotspot list
(`target_settings["target_hotspots"][t_idx]`), so a dual-target design must
satisfy both epitopes.

## A crash is not a failed design

The hotspot block catches exceptions, but deliberately does **not** record
`near_hotspot=False`:

```python
except Exception as e:
    print(f"  Early check: hotspot proximity CRASHED for target {t_idx} ...")
    crashed = True
    target_info["near_hotspot"] = None
```

`False` would be indistinguishable from a binder that genuinely misses its
epitope. Recording `None` and setting `crashed` keeps "the measurement did not
happen" separate from "the design is bad", and leaves `all_hotspot_pass`
untouched so a tool failure cannot silently reject a design. The crash is
printed with its exception type and the offending PDB path.

## Config keys

| Key | Meaning |
|-----|---------|
| `clash_threshold` | distance below which two CAs count as clashing |
| `hotspot_distance_threshold` | binder-to-hotspot distance cutoff |
| `residue_contact_distance` | residue-residue contact cutoff |
| `min_cdr_hotspot_contacts` | contacts required to count as near the hotspot |

`min_cdr_fraction` is passed by the caller, not read from config here.

## Notes

- CDR3 positions arrive 0-indexed from `run_settings` and are converted with
  `+ 1` for PDB numbering at `early_check.py`.
- Rejections here are charged to their own counters, so an arm whose designs
  die at this stage is distinguishable from one failing the initial filter.
