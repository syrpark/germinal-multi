# Output layout

What lands where, what "this seed is already done" means, and how a
restarted run takes over work a killed one left behind.

**Code:** `germinal/utils/io.py` — `RunLayout` (the layout),
`IO.save_trajectory` (routing), `IO.check_existing_seed` (spent seeds),
`IO.find_adoptable` (resume).

## Directory structure

```
results/{experiment_name}/
├── trajectories/
│   ├── structures/          AF2 hallucination PDBs ONLY (the sc_rmsd reference)
│   └── designs.csv
├── redesign_candidates/
│   ├── structures/
│   │   ├── {design}/        filter-model + recycling-model predictions
│   │   └── _batch_fold/     batched filter-stage predictions
│   └── designs.csv          exists but stays EMPTY: structures only
├── filter_failed/
│   ├── structures/
│   └── designs.csv
├── accepted/
│   ├── structures/
│   └── designs.csv
├── inflight/                one {design}.json per unfinished design
├── msa_cache/               target + binder-framework MSAs, run-level
├── all_trajectories.csv     master record of every design
├── failure_counts.csv       which filters caused rejections (stage-prefixed)
└── final_config.yaml        full Hydra config snapshot
```

Each backend stages its own inputs **inside** whichever directory it was
given as `output_dir` — `af3_inputs/` for AF3, `protenix_inputs/` for
Protenix, each with `msas/`. So these appear per design and under
`_batch_fold/`, not as a shared sibling. They are copies out of
`msa_cache/`, which is where the search cost is actually paid.

`_batch_fold/` persists between batches; each batch clears its own input
JSONs and any stale per-job output directory, not the whole tree.

## Record directories vs the structure store

Three directories are **records** — a design lands there because a decision
was made about it, and it gets a `designs.csv` row:

| Directory | Meaning |
|-----------|---------|
| `trajectories/` | failed before the final filter: hallucination, early check, initial filter, redesign, or the recycling gate |
| `filter_failed/` | reached the final filter and failed it |
| `accepted/` | passed everything |

`redesign_candidates/` is **not** a record. It is a structure store holding
every structure *predicted* by the filter model or the recycling model. Its
`designs.csv` is deliberately kept empty, and
`tests/test_design_classification.py` pins it at 0 rows as a regression guard
against a misrouting bug.

### `candidate_source`

Each row names which sequence it describes, because the hallucinated sequence
is scored alongside its redesigns rather than only being a starting point:

| `candidate_source` | Row | Name |
|---|---|---|
| `codesign` | the hallucinated sequence itself, unredesigned | `<design>_abmpnn_0` |
| `abmpnn` | a multi-state AbMPNN redesign of it | `<design>_abmpnn_1` .. `_abmpnn_N` |

The `_abmpnn_0` slot exists only where the co-designed candidate is added
(`build_candidates(..., include_codesign=True)`, the non-recycling path). The
recycling loop scores the hallucinated sequence within the loop instead, so it
passes `include_codesign=False` and its rows start at `_abmpnn_1`.

### This changed, and it breaks older tooling

`redesign_candidates/` previously meant "reached the final filter and failed
it". Runs produced before the change have three design directories and no
`filter_failed/`. Any tool that hardcodes `redesign_candidates` as the
failed-finals leg will, on a current run, read an empty CSV, find nothing,
and report a clean result built from the accepted designs alone — no error,
just a silently smaller pool.

Resolve it from what is on disk instead: the presence of `filter_failed/`
identifies a current-layout run.

## Resume

Two separate questions, answered by two separate mechanisms:

1. **Is this seed spent?** `check_existing_seed`, below. Only meaningful under
   `pregenerate_seeds`; a run drawing random seeds never asks about one it has
   already used.
2. **Is there unfinished work to take over?** `find_adoptable`, over the
   records in `inflight/`. This is what makes a restart cheap, and it is the
   one that matters in production, where seeds are random.

### Taking over unfinished work

A design gets a record in `inflight/{design}.json` once it passes
hallucination and the early check, superseded at each later boundary
(`hallucinated` → `filtered` → `candidates_ready`) and deleted once every one
of its candidates has a row. An empty `inflight/` means the last run ended
clean; anything in it is work a restart can finish.

The record is **not** a row in `all_trajectories.csv`, and must not become
one: a row there would make `check_existing_seed` report the design finished
and would count against `max_hallucinated_trajectories`, so a crash would
retire the seed with no result rather than leaving it recoverable.

Adoption keys on the record, never on stray structures. A hallucination PDB
on its own proves nothing, because the confidence gates that judged it
(pLDDT, iPTM, i_pae) live only in memory — output from a design that *failed*
them looks exactly like output from one that passed. The record is written
after those gates, so it is the only evidence the work was worth keeping. A
record is adopted when all of:

- no row carries its exact `design_name` (an exact match, not the delimited
  `_s{seed}` search);
- for a `candidates_ready` record, at least one listed candidate has no row.
  **Not optional**: a design that finishes scoring never gets a row of its
  own, only candidate rows, so asking "does the parent have a row" would
  mistake every finished design for unfinished work;
- `target_fp` matches — a design built against different antigen states is not
  the same object;
- `filter_fp` matches, but **only** for a `candidates_ready` record. That is
  the one phase which inherits a verdict instead of recomputing it: it skips
  phases A-C, so its initial-filter result and its recycle-built candidates
  both come from the old config. `hallucinated` and `filtered` records are
  re-folded and re-filtered under the current config on adoption, so gating
  them on the filter set only stranded work that would have been re-evaluated
  correctly. For the same reason `filter_fp` does not digest the **final**
  filters at all: Phase D applies those to every adopted design, so moving a
  final threshold invalidates nothing;
- every structure the record points at is still on disk.

A record failing any of these is reported and **left in place**. Deciding that
unfinished work is rubbish is a human's call.

An adopted design occupies a fold slot, so it counts against
`max_hallucination_passing`, but costs no attempt and no trajectory — those
were charged to the run that hallucinated it. Its rows carry `resumed`,
because `design_time` then covers only the second process's share.

A design part-way through scoring its candidates keeps both its record and its
structures at batch cleanup, which is what lets the next run finish the set.

### A run that ends is not a run that finished

**Expect leftovers, and expect them to be invisible.** A run killed by a
wall-clock limit almost always parks designs mid-pipeline, most often at
`hallucinated`: they passed hallucination but never filled a fold batch, so
they were never folded, never scored, and — because a record is deliberately
not a row — appear in no CSV at all. A log ending in `0 designs accepted` can
sit on top of real, already-paid-for work.

The end-of-run summary therefore reports `inflight/` explicitly
(`_unfinished_work_note`), in the log and in the saved run summary: how many
designs, at which phase, what each is still owed, and that **relaunching the
same `experiment_name` adopts them** rather than redoing them. The clean case
says so too, since "no warning" cannot be told from "never checked".

One cause is worth recognising on sight. A batch folds only once
`max_hallucination_passing` designs pass hallucination, or
`max_hallucination_attempts` are spent on that batch — whichever comes first.
A target whose hallucination gates reject most designs can therefore spend an
entire run accumulating passers and fold **none** of them: the passer count
never fills, and the attempt cap arrives after the wall. The symptom is a run
whose rows are all `hallucination_failed` with zero folds in the log and a
handful of records parked in `inflight/`. The fix is to lower
`max_hallucination_passing` for that target, which trades a little batching
efficiency for actually folding what the run produced.

### Is this seed spent?

A seed is done if either:

1. a row in `all_trajectories.csv` carries it (including a
   `..._abmpnn_N` candidate row — phase D writes candidate rows, not a row
   for the parent, and the seed lives in the candidate name); or
2. a saved structure in a **terminal** directory carries it. The terminal
   directories are `accepted/` and `filter_failed/` only.

Seed matching is **delimited, not substring**. `seed in fname` reported 8876
as used because `..._s488766_...` contains `8876`; every false positive burns
a trajectory slot.

### Why working directories are not proof

`redesign_candidates/structures/` is a **working** directory and must never
be treated as terminal. Its per-design subdirectory is created in phase B to
receive predictions, and its name carries `_s{seed}` — but the row may not be
written until phase D. A crash in between would leave a seed that resume
skips with no result for it. Re-running a design costs GPU time; silently
dropping one costs the design.

The same applies to `trajectories/structures/`: the hallucination PDBs are
written before the terminate flag is set, so they exist for designs that
never finished.

None of this changes now that `inflight/` exists — the two coexist. A working
directory or a stray PDB still proves nothing on its own; what makes the work
recoverable is the record naming it, and a structure with no record is still
ignored.

Pinned by `tests/test_resume_under_batching.py`, including that neither the
`redesign_candidates` working directory nor a recycle prediction inside it
marks a seed done.

## Saving

`save_trajectory` routes on `trajectory.save_location`, one of
`trajectories`, `redesign_candidates`, `filter_failed`, `accepted`. It copies
(`shutil.copy`) the structure into the destination's `structures/` — flat,
regardless of where the prediction was nested.

Writes are durable and thread-safe: CSV writes go through a lock file
(`{filename}.lock`) and a temp-file rename, and a corrupt master CSV is
quarantined rather than overwritten.

## Retention

Every recycle prediction is currently **kept**. Nesting them per design makes
a future cleanup a safe `rmtree` of one directory rather than a glob over
filenames, but no deletion pass exists. A glob over `_recycle*.pdb` would be
the wrong way to add one: those filenames are referenced by CSV rows.
