# MSA infrastructure

How each chain gets an alignment, how binder CDRs are masked, and the two
caches that stop the same search being paid for repeatedly.

**Code:** `germinal/filters/af3.py` (generation, masking, both caches,
`resolve_binder_msa`), `germinal/filters/protenix.py` (same functions, called
not copied), `germinal/filters/folding.py` (policy resolution).

## Two axes, not one

`msa_method` — how an alignment is built. Both options are colabfold; they
differ only in where the databases live.

| Value | Meaning |
|-------|---------|
| `colabfold_remote` | MMseqs2 remote API |
| `colabfold_local` | local databases at `msa_db_dir` |

`msa_scope` — who gets one.

| Value | Meaning |
|-------|---------|
| `none` | nobody; every chain folds single-sequence |
| `target_only` | antigens do; the binder gets a one-sequence stub |
| `binder_masked` | the binder does too, CDR columns gapped in every homolog |

There is deliberately **no unmasked binder option**. The CDRs were designed,
so homolog CDRs are germline loops the design is not, and keeping them pulls
the prediction back toward germline.

These were previously one enum, which conflated the axes and left "local
databases, targets only" inexpressible. `_LEGACY_MSA_MODES` documents the old
mapping; `resolve_msa_policy` returns the pair.

### Current defaults

| Stage | Backend | `msa_scope` |
|-------|---------|-------------|
| filter | af3 | `target_only` |
| recycle | af3 | `target_only` |

Both stages default to `af3`. The row below applies when a stage is switched
to `protenix`, whose own config sets `msa_scope: "binder_masked"`:

| Stage | Backend | `msa_scope` |
|-------|---------|-------------|
| either | protenix | `binder_masked` |

Protenix needs the binder alignment far more than AF3 does: on one accepted
design AF3 scored ipSAE 0.67 from the stub alone while Protenix scored 0.000,
and 0.80 once given the alignment.

## CDR masking

`mask_binder_cdrs(msa_path, cdr_positions, design_name, binder_seq)` gaps the
CDR columns in every homolog row and leaves the query row intact. It returns
the number of homologs masked.

It **fails loudly** rather than skipping. It raises `CdrMaskError` on a
missing path, empty `cdr_positions`, a query row that does not match
`binder_seq`, or positions out of range — and warns when it masks 0 homologs,
since a mask that silently does nothing is indistinguishable from no mask at
all.

A3M lowercase characters are insertions and occupy no query column; the
column arithmetic accounts for this.

CDR positions come from the config (`cdr_positions`, 0-indexed), never from a
hardcoded layout.

## The two caches

Both live under `root/msa_cache/`, set in `config.py` as
`run_settings["msa_cache_dir"]`. It is derived, not a YAML key. It is **not**
AF3-specific despite having once been called `af3_msa_cache`: both backends
read and write it.

### Target cache — keyed by sequence

`_target_msa_cache_path` → `target_{sha256(sequence)[:16]}.a3m`

The antigen sequence is identical across every design in a run, so this
collapses one search per antigen state for the whole run. `output_dir` is
per-design on the filter path and per-process on the recycle path, so the
local `msas/` check alone almost never hits — without this cache every design
pays a fresh search per antigen.

### Framework cache — keyed by framework

`_framework_msa_cache_path` → `binder_fw_{framework_key}.a3m`, where

```python
def binder_framework_key(binder_seq, cdr_positions) -> str:
    cdrs = {int(p) for p in (cdr_positions or [])}
    framework = "".join(ch for i, ch in enumerate(binder_seq) if i not in cdrs)
    payload = f"{len(binder_seq)}:{sorted(cdrs)}:{framework}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
```

Once the CDR columns are gapped, the homolog block is identical across every
binder sharing a framework — which is typically every candidate of every
trajectory in a run.

## Two keys, deliberately

`resolve_binder_msa` uses both:

- the **file** is per binder sequence (`binder_{sha256(seq)[:16]}.a3m`),
  because its query row must be that exact binder;
- the **cache** is per framework, because the homolog block is shared.

On a cache hit the query row is swapped for the current binder. Entries are
published to the cache **only after masking**, so every reuse inherits gapped
CDRs.

### One implementation

`resolve_binder_msa` is shared by both backends. Keying the binder MSA on a
name that embeds the target leg defeats it: both legs of one design then
search separately and no design reuses another's framework. The signature in
the logs is one binder search per leg with zero reuse, alongside correctly
cached targets.

Pinned by `tests/test_binder_framework_msa_cache.py` and
`tests/test_binder_msa_cdr_mask.py`, which assert both that each backend
reaches masking and that the shared resolver itself masks before returning —
otherwise delegating is how masking disappears from both backends at once.

## Timeouts and retries

`call_generate_colabfold_msa_with_timeout(..., attempts=2)` retries once. A
SIGKILLed worker leaves a partial `{name}_all/` directory behind, which
`_clear_partial_msa_output` removes before each retry; without that the retry
inherits the corpse and fails the same way.

A single 600 s timeout once failed both legs of a design and, through
`AllJobsFailed`, aborted an otherwise healthy 9-hour run. See
[folding-backends.md](folding-backends.md#failure-scope).

## Where files land

MSA inputs are staged inside whichever directory the backend is given as
`output_dir` — `af3_inputs/` for AF3, `protenix_inputs/` for Protenix, each
with an `msas/` subdirectory. So they appear inside each per-design directory
and inside `_batch_fold/`, not as one shared sibling. Nothing expensive is
duplicated: these are copies out of `root/msa_cache/`, which is where the
search cost is actually paid.
