# Stage 1 — Hallucination

Generates a binder sequence by gradient optimisation against a frozen AF2
(ColabDesign) model, then gates it on the confidence AF2 reports for its own
prediction. No external folding model is involved; this stage is cheap
relative to stages 3 and 5 but is the throughput bottleneck on the `germinal`
arms.

**Code:** `germinal/design/design.py`. The full and logits-only schedules are
one implementation, selected by `STAGES_FULL` vs `STAGES_LOGITS`.
**Runs in:** phase A, one trajectory at a time.
**Writes:** `trajectories/structures/{design}.pdb` and
`{design}_target{N}.pdb` — the AF2 hallucination PDBs. Nothing else writes
there; they are the `sc_rmsd` reference for every later stage.

## Stages within the trajectory

`stages` selects which optimisation phases run: `logits`, `soft`, `greedy`.
A run that lists only `logits` is the "logits-only" path; one that includes
`soft`/`greedy` is the "full" path. The runner prints which at startup:

```
Hallucination: full | recycling: False
```

## The three confidence gates

All three read the **same** three thresholds, assembled once at
`design.py`:

```python
save_filters = {
    "plddt":  run_settings.get("plddt_threshold", 0.84),
    "i_ptm":  run_settings.get("i_ptm_threshold", 0.65),
    "i_pae":  run_settings.get("i_pae_threshold", 0.3),
}
```

| Gate | Where | What it decides |
|------|-------|-----------------|
| `logits` | `design.py` | whether to proceed to the softmax stage |
| `softmax` | `design.py` | whether to proceed to greedy |
| `final` | `design.py` | whether the trajectory is kept at all |

The `logits` gate is skipped entirely when neither `soft` nor `greedy` is in
`stages` (`run_later_stages`, `design.py`) — there is nothing for it to
decide, so a logits-only run falls straight through to the final gate.

### Gate directions

These are not uniform, and the asymmetry has caused confusion:

```
pass  ⇔  plddt >  T_plddt          higher is better
     and  iptm  >= T_iptm           higher is better
     and  ipae  <  T_ipae           LOWER is better
```

So **lowering** `i_pae_threshold` makes the gate **stricter**, while lowering
`i_ptm_threshold` makes it looser. The `final` gate is written as the negated
disjunction (`final_ipae >= save_filters["i_pae"]` fails), which is the same
rule.

Because the gates are conjunctions, loosening two thresholds at once is
superlinear in designs recovered: a design blocked by both is recovered only
when both move.

## Failure accounting

The `final` gate is a disjunction, so more than one metric can be to blame. It
charges **each** metric that fell short rather than attributing everything to
the first (`design.py`), giving
`hallucination_final_plddt`, `hallucination_final_i_ptm`,
`hallucination_final_i_pae` in `failure_counts.csv`.

A clashing structure (`ca_clashes > 0`) terminates as `Clashing` before the
confidence check and is charged to `hallucination_clashes`.

## Config keys

| Key | Meaning |
|-----|---------|
| `plddt_threshold` | pLDDT gate, pass if strictly greater |
| `i_ptm_threshold` | iPTM gate, pass if greater or equal |
| `i_pae_threshold` | interface PAE gate, pass if strictly less |
| `stages` | which optimisation phases run |
| `softmax_steps` | 0 disables the softmax stage even when listed |
| `max_hallucination_passing` | passers that close a phase-A batch |
| `max_hallucination_attempts` | attempts that close a phase-A batch |

`max_hallucination_*` bound the batch, not the run; `max_trajectories` bounds
the run.

## Notes

- The hallucination PDBs are written **before** the terminate flag is set, so
  their presence does not mean the trajectory finished. Resume must not treat
  `trajectories/structures/` as proof — see
  [output-layout.md](output-layout.md).
- On the `germinal` (full) arms this stage dominates wall-clock; on the
  `logits` arms the initial filter does.
