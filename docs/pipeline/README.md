# Pipeline documentation

One file per stage. Each describes what the stage does, the config keys that
steer it, where it writes, and the failure modes that have actually bitten.

Entry point: `run_germinal_multi.py`. Hydra composes the config from
`configs/config.yaml` plus the `run=`, `target=`, `filter/initial=` and
`filter/final=` groups given on the command line, with the folding backend
selected per stage as `folding@recycle_folder=` / `folding@filter_folder=`.

## Stages

| # | Stage | Doc | Code |
|---|-------|-----|------|
| 1 | Hallucination | [01-hallucination.md](01-hallucination.md) | `germinal/design/design.py` |
| 2 | Early check | [02-early-check.md](02-early-check.md) | `germinal/filters/early_check.py` |
| 3 | Initial filter | [03-initial-filter.md](03-initial-filter.md) | `germinal/filters/filter_utils.py` |
| 4 | Redesign and recycling | [04-redesign-and-recycling.md](04-redesign-and-recycling.md) | `germinal/design/redesign.py`, `germinal/design/recycle.py` |
| 5 | Final filter | [05-final-filter.md](05-final-filter.md) | `germinal/filters/filter_utils.py` |

## Cross-cutting

- [folding-backends.md](folding-backends.md) — how a stage gets a structure
  predictor, and the rule that keeps af3 and protenix from drifting apart.
- [msa-infrastructure.md](msa-infrastructure.md) — MSA generation, CDR
  masking, and the two caches.
- [output-layout.md](output-layout.md) — what lands in which directory, and
  how a restarted run takes over work a killed one left behind.

## Phases vs stages

The five stages above are the conceptual pipeline. The runner executes them
in four **phases**, because folding is batched: one container start costs
roughly as much as five predictions, so designs are folded in groups rather
than one at a time.

```
PHASE A   hallucinate a batch          stages 1 and 2, per trajectory
            closes on max_hallucination_passing passers
            or max_hallucination_attempts attempts, whichever first
PHASE B   ONE fold of every survivor   stage 3
PHASE C   redesign / recycling         stage 4, per surviving design
PHASE D   fold a chunk of candidates,  stage 5
          score it, repeat
          batch cleanup
```

So a trajectory does not run end to end before the next one starts, and a
kill mid-batch catches several designs at once. This is what resume has to
cope with (see [output-layout.md](output-layout.md)): a per-design working
directory exists from phase B, long before any CSV row is written in phase
D, so directory existence is not proof that a design finished, and the
in-flight records under `inflight/` are what make the gap recoverable.

The four phase boundaries live in `run_germinal_multi.py`. Adopted designs
join at A (to refold) or at D (candidates
already in hand); phase D folds in groups of
`folding.resolve_batch_size(run_settings, "filter")` candidates and scores
each group before folding the next, so a kill costs one group.

## Conventions

- CDR positions are **0-indexed** in `run_settings`, **1-indexed** in PDB
  numbering. Conversion sites add 1 explicitly.
- Design names: `{target}_{type}_s{seed}`; AbMPNN variants append
  `_abmpnn_{N}`; recycle predictions append `_recycle{N}`.
- Thresholds use AND logic throughout. Every one must pass.
