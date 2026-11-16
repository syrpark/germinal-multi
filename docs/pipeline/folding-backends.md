# Folding backends

Two structure predictors are available, AF3 and Protenix. Which one a stage
uses is a **config choice**; the runner never branches on a backend name.

**Code:** `germinal/filters/folding.py` (router),
`germinal/filters/af3.py`, `germinal/filters/protenix.py` (backends).

## The rule

> One implementation per job, routed to models through config. Never a fork
> per backend.

Adding a backend must not mean touching the runner, and adding a capability
to one backend must not leave the other behind. Read every backend setting
through `folding.resolve_folder(run_settings, stage)` rather than by name.

The failure mode this guards against is silent: a cache defined in one
backend and wired into the other only, so the unwired path keeps paying for
work the cache exists to eliminate, with nothing in the logs to show it. Both
backends call the shared `af3.resolve_binder_msa` for exactly that reason.

## Selecting a backend

`configs/config.yaml` picks one per stage via Hydra config groups:

```yaml
defaults:
  - folding@recycle_folder: af3
  - folding@filter_folder: af3
```

Both stages default to `af3`, which is what the shipped configs use. Override
per run config with `- override folding@recycle_folder: protenix`. The two
stages are independent: the filter stage and the recycling loop *can* run
different models.

## Router API

| Function | Returns |
|----------|---------|
| `resolve_folder(run_settings, stage)` | the stage's folding config dict |
| `resolve_backend_name(run_settings, stage)` | `"af3"` or `"protenix"` |
| `resolve_msa_policy(run_settings, stage)` | `(msa_method, msa_scope)` |
| `resolve_selection(run_settings, stage)` | `(sample_select_mode, seed_select_mode)` |
| `build_context(...)` | a `FoldContext` for the backend |
| `register_backend(name, fold)` | add a backend |
| `available_backends()` | registered names |
| `validate_backends(run_settings)` | startup check across all stages |

`stage` is `"filter"` or `"recycle"`, mapped to config keys by `STAGE_KEYS`.

## No legacy fallbacks, on purpose

`resolve_folder` raises rather than defaulting when a stage config is missing
or malformed. The old flat keys (`recycle_structure_model`,
`protenix_samples`, `af3_structure_select_mode`) silently disagreed across
backends — AF3 was never told a sample count and used its own default of 5
while Protenix was told 1. A config written against the old scheme must fail
loudly rather than resolve to something nobody chose.

`resolve_selection` raises the same way on the removed flat `select_mode`.

## Two selection axes, not one

```yaml
sample_select_mode: best    # diffusion samples within a seed
seed_select_mode:   worst   # independent attempts at the complex
```

Diffusion samples within a seed differ only by sampling noise, so taking the
best is right. Seeds are independent attempts, so taking the worst is the
conservative estimate. A single mode over the flat pool conflated them and
charged every design for its unluckiest diffusion draw.

## Batching

`folding.fold_designs_against_targets` folds a whole batch in one call. AF3
runs the batch in a single container; Protenix loops internally. The caller
does not need to know which.

This matters because the AF3 container start costs far more than a single
prediction. Folding D designs in one call instead of D pays that start-up once
rather than D times, and lets every design in the batch share one target MSA.

Results come back positionally, so a short result list is rejected rather
than silently scoring a design against the wrong antigen.

## Failure scope

If every job in a batch fails, that is normally fatal. But when all the failed
jobs are legs of **one** design sharing a single binder sequence, one flaky
MSA search would otherwise kill an entire multi-hour run. `folding.py` raises
the underlying error in that case instead of `AllJobsFailed`:

```python
if jobs and len(errors) == len(jobs):
    if len({job.binder_sequence for job in jobs}) <= 1:
        raise errors[0]
    raise AllJobsFailed(...)
```

Without this, one flaky MSA search ends a multi-hour run.
