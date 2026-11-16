"""Backend-agnostic structure folding.

The recycle loop and the filter stages pick their backend independently, via
``recycle_folder`` and ``filter_folder``. Everything model-specific lives
behind this module, so no caller branches on the model.

To add a backend, write ``fold(jobs, ctx) -> list`` and call
``register_backend(name, fold)``; a model that folds one complex per call can
wrap ``fold_one`` in :func:`sequential` instead.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from germinal.filters import af3

from germinal.utils.log import get_logger

log = get_logger(__name__)


#: Stage name -> the run-settings key holding that stage's folding config,
#: each one instance of the `configs/folding` group.
STAGE_KEYS = {
    "recycle": "recycle_folder",
    "filter": "filter_folder",
}

#: The only values either selection axis accepts.
SELECT_MODES = frozenset({"best", "worst"})

#: Replaced by sample_select_mode + seed_select_mode. A config still carrying
#: it is rejected rather than silently reduced to one axis.
LEGACY_SELECT_KEY = "select_mode"

#: (pdb_path, scores, ipsae), plus every sample when ctx.want_all_samples.
FoldResult = Tuple


class AllJobsFailed(RuntimeError):
    """Every job in a folding batch raised.

    One job failing is a design outcome; every job failing is an environment
    fault. Raised by the backend so the cause survives, but the caller decides
    whether to abort or record it and move on.
    """


@dataclass(frozen=True)
class FoldJob:
    """One complex to fold: a binder against a single target."""

    name: str
    binder_sequence: str
    target_sequence: str
    target_index: int


@dataclass(frozen=True)
class FoldContext:
    """Everything a backend needs that is the same for every job in a call.

    The four selection numbers -- num_samples, seeds, sample_select_mode,
    seed_select_mode -- are spelled out here rather than left in `folder`,
    and resolved in this module only, never inside a backend.
    """

    target_chain: str
    binder_chain: str
    output_dir: str
    run_settings: Dict[str, Any]

    #: Diffusion samples per seed, identical for af3 and protenix. Each
    #: backend has its own default if not told, so always pass it.
    num_samples: int
    #: One prediction set per seed; len(seeds) is "how many seeds".
    seeds: List[int]
    #: Within one seed, across its diffusion samples. "best": a seed's samples
    #: differ only by sampling noise.
    sample_select_mode: str
    #: Across seeds, over each seed's chosen sample. "worst": seeds are
    #: independent attempts and a design is only as good as its weakest.
    seed_select_mode: str

    #: How alignments are built: "colabfold_remote" or "colabfold_local".
    msa_method: str
    #: Who gets one: "none", "target_only" or "binder_masked".
    msa_scope: str

    #: Pairformer cycles. AF3 --num_recycles, Protenix --cycle.
    num_cycles: int
    #: This stage's resolved folding config, for backend-specific settings
    #: only (conda_env, model_name, num_steps).
    folder: Dict[str, Any]
    #: Return every sample alongside the chosen one, for the relax ensemble.
    #: Which stage this context serves, so a backend can batch where that
    #: pays and stay sequential where it does not. Unset means sequential.
    stage: str = ""
    want_all_samples: bool = False
    #: AF3 only: structure from the best-ranked prediction, scores and ipSAE
    #: from the selected one.
    decoupled: bool = False


#: ``fold(jobs, ctx) -> list``: one entry per job, in order, FoldResult or
#: None. The entry points, not the backend, decide whether a None is fatal.
Backend = Callable[[Sequence[FoldJob], FoldContext], List[Optional[FoldResult]]]

_BACKENDS: Dict[str, Backend] = {}


def register_backend(name: str, fold: Backend) -> None:
    """Make a folding backend selectable by ``name`` in a run config."""
    _BACKENDS[name.lower()] = fold


def available_backends() -> List[str]:
    return sorted(_BACKENDS)


def sequential(fold_one: Callable[[FoldJob, FoldContext], FoldResult]) -> Backend:
    """Build a backend from a folder that handles one complex per call.

    Supplies the loop, the ordering guarantee and the per-job error handling.
    A job that raises becomes None in its slot; every job raising becomes
    :class:`AllJobsFailed`.
    """

    def fold(jobs: Sequence[FoldJob], ctx: FoldContext) -> List[Optional[FoldResult]]:
        results: List[Optional[FoldResult]] = []
        errors: List[Exception] = []
        for job in jobs:
            try:
                results.append(fold_one(job, ctx))
            except Exception as e:
                log.info(f"  Folding failed for {job.name}: {type(e).__name__}: {e}")
                errors.append(e)
                results.append(None)

        if jobs and len(errors) == len(jobs):
            first = errors[0]
            # All-failed is systemic only when the jobs could fail
            # independently; legs of one design share an MSA, so they cannot.
            if len({job.binder_sequence for job in jobs}) <= 1:
                log.info(
                    f"  All {len(jobs)} folding job(s) failed, but they are "
                    f"legs of ONE design sharing a single binder sequence, so "
                    f"this is one failure and not {len(jobs)} independent "
                    f"ones. Propagating the cause as a per-design fault."
                )
                raise first

            raise AllJobsFailed(
                f"all {len(jobs)} folding jobs failed across "
                f"{len({job.binder_sequence for job in jobs})} distinct binder "
                f"sequences; first failure was "
                f"{type(first).__name__}: {first}"
            ) from first

        return results

    return fold


# --- Backend selection ---


def resolve_folder(run_settings: Dict[str, Any], stage: str) -> Dict[str, Any]:
    """The folding-backend config selected for ``stage``.

    Read every backend setting through this, never by name.

    Raises:
        ValueError: on an unknown stage, or a missing/malformed stage config.
    """
    if stage not in STAGE_KEYS:
        raise ValueError(
            f"Unknown folding stage {stage!r}; expected one of {sorted(STAGE_KEYS)}"
        )
    key = STAGE_KEYS[stage]
    folder = run_settings.get(key)
    # No legacy fallback: a config written against the old flat keys must
    # fail here rather than resolve to something nobody chose.
    if not isinstance(folder, dict):
        raise ValueError(
            f"run_settings[{key!r}] is {type(folder).__name__}, expected the "
            f"folding config for the {stage!r} stage. Select one in the "
            f"defaults list, e.g. `- folding@{key}: af3`."
        )
    return folder


#: Designs per backend call when a caller folds in batches and scores between
#: them, bounding what a kill mid-fold discards. 0 means one call for all.
DEFAULT_BATCH_SIZE = 10


def resolve_batch_size(run_settings: Dict[str, Any], stage: str) -> int:
    """Designs per backend call for ``stage``.

    Backend-dependent: it trades a container start against how much folding a
    kill throws away, and the backends start at different costs. Absent from a
    config it defaults rather than raising, since getting it wrong costs time
    and never correctness.

    Raises:
        ValueError: on a negative or non-integer value.
    """
    folder = resolve_folder(run_settings, stage)
    if "batch_size" not in folder:
        return DEFAULT_BATCH_SIZE
    value = folder["batch_size"]
    try:
        size = int(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{STAGE_KEYS[stage]}.batch_size is {value!r}; expected an "
            f"integer number of designs per backend call."
        )
    if size < 0:
        raise ValueError(
            f"{STAGE_KEYS[stage]}.batch_size is {size}; expected >= 0, where "
            f"0 folds everything in one call."
        )
    return size


#: How an alignment is built. Both are colabfold; they differ only in where
#: the databases live.
MSA_METHODS = frozenset({"colabfold_remote", "colabfold_local"})

#: Who gets an alignment: none, target_only (binder gets a one-sequence
#: stub), or binder_masked (binder too, CDR columns gapped in every homolog).
MSA_SCOPES = frozenset({"none", "target_only", "binder_masked"})

#: Legacy single-enum setting -> (method, scope), which it conflated.
_LEGACY_MSA_MODES = {
    "none": ("colabfold_remote", "none"),
    "local": ("colabfold_local", "binder_masked"),
    "colabfold": ("colabfold_remote", "binder_masked"),
    "target": ("colabfold_remote", "target_only"),
}


def resolve_msa_policy(
    run_settings: Dict[str, Any], stage: str
) -> Tuple[str, str]:
    """``(msa_method, msa_scope)``: how alignments are built, and for whom.

    Resolved per stage, because af3 tolerates target_only and protenix does
    not. A config carrying the old run-level `msa_mode` is mapped, not
    rejected, since every shipped run config still uses it.

    Raises:
        ValueError: on an unknown method, scope, or legacy mode.
    """
    folder = resolve_folder(run_settings, stage)
    method = folder.get("msa_method", run_settings.get("msa_method"))
    scope = folder.get("msa_scope", run_settings.get("msa_scope"))
    if method is None or scope is None:
        legacy = run_settings.get("msa_mode")
        if legacy is None:
            raise ValueError(
                f"No MSA policy: set msa_method ({sorted(MSA_METHODS)}) and "
                f"msa_scope ({sorted(MSA_SCOPES)})."
            )
        if legacy not in _LEGACY_MSA_MODES:
            raise ValueError(
                f"msa_mode={legacy!r} is not a known mode; expected one of "
                f"{sorted(_LEGACY_MSA_MODES)}. Prefer msa_method + msa_scope."
            )
        mapped_method, mapped_scope = _LEGACY_MSA_MODES[legacy]
        method = mapped_method if method is None else method
        scope = mapped_scope if scope is None else scope

    if method not in MSA_METHODS:
        raise ValueError(
            f"msa_method={method!r} is not a known method; expected one of "
            f"{sorted(MSA_METHODS)}."
        )
    if scope not in MSA_SCOPES:
        raise ValueError(
            f"msa_scope={scope!r} is not a known scope; expected one of "
            f"{sorted(MSA_SCOPES)}."
        )
    return str(method), str(scope)


def resolve_selection(run_settings: Dict[str, Any], stage: str) -> Tuple[str, str]:
    """``(sample_select_mode, seed_select_mode)`` for ``stage``.

    Separate decisions: samples within a seed differ only by noise ("best"),
    while seeds are independent attempts at the complex ("worst").

    Raises:
        ValueError: on a missing or non-{"best","worst"} value, or on a config
            still carrying the removed flat ``select_mode``.
    """
    folder = resolve_folder(run_settings, stage)
    key = STAGE_KEYS[stage]
    if LEGACY_SELECT_KEY in folder:
        raise ValueError(
            f"{key}.{LEGACY_SELECT_KEY} has been replaced by "
            f"{key}.sample_select_mode and {key}.seed_select_mode. The old key "
            f"reduced samples and seeds together; set the two axes explicitly "
            f"(see configs/folding/af3.yaml)."
        )
    modes = []
    for axis in ("sample_select_mode", "seed_select_mode"):
        mode = folder.get(axis)
        if mode not in SELECT_MODES:
            raise ValueError(
                f"{key}.{axis}={mode!r} is not a selection mode; expected one "
                f"of {sorted(SELECT_MODES)}."
            )
        modes.append(str(mode))
    return modes[0], modes[1]


def build_context(
    run_settings: Dict[str, Any],
    stage: str,
    *,
    target_chain: str,
    binder_chain: str,
    output_dir,
    seeds: List[int],
    want_all_samples: bool = False,
    sample_select_mode: Optional[str] = None,
    seed_select_mode: Optional[str] = None,
) -> "FoldContext":
    """The one place a stage's config becomes a FoldContext.

    Every entry point goes through here, so num_samples, num_cycles and the
    two selection axes are read from the config once and reach both backends
    the same way. An override is for a deliberate caller, not a default.
    """
    folder = resolve_folder(run_settings, stage)
    cfg_sample, cfg_seed = resolve_selection(run_settings, stage)
    msa_method, msa_scope = resolve_msa_policy(run_settings, stage)
    return FoldContext(
        stage=stage,
        target_chain=target_chain,
        binder_chain=binder_chain,
        output_dir=output_dir,
        run_settings=run_settings,
        num_samples=folder["num_samples"],
        seeds=seeds,
        sample_select_mode=sample_select_mode or cfg_sample,
        seed_select_mode=seed_select_mode or cfg_seed,
        msa_method=msa_method,
        msa_scope=msa_scope,
        num_cycles=folder["num_cycles"],
        folder=folder,
        want_all_samples=want_all_samples,
        decoupled=bool(folder.get("decoupled", False)),
    )


def resolve_backend_name(run_settings: Dict[str, Any], stage: str) -> str:
    """Name of the folding backend configured for ``stage``.

    Raises:
        ValueError: on an unknown stage or an unknown backend name.
    """
    folder = resolve_folder(run_settings, stage)
    name = str(folder.get("backend", "")).lower()
    if name not in _BACKENDS:
        raise ValueError(
            f"{STAGE_KEYS[stage]}.backend={name!r} is not a known folding "
            f"backend; expected one of {available_backends()}"
        )
    return name


def validate_backends(run_settings: Dict[str, Any]) -> None:
    """Check both stages at startup, before any GPU time is spent.

    A bad backend name would otherwise surface minutes into the first
    trajectory, or only at its final filter.

    Raises:
        ValueError: on an unknown backend, or an AF3-only select mode paired
            with a non-AF3 filter backend.
    """
    for stage in STAGE_KEYS:
        resolve_backend_name(run_settings, stage)

    for stage in STAGE_KEYS:
        resolve_msa_policy(run_settings, stage)

    # Both stages must fold with MSAs: without them Protenix is faster but
    # far less confident, so the wrong setting reads as weak binding.
    for stage in STAGE_KEYS:
        if not resolve_folder(run_settings, stage).get("use_msa", False):
            raise ValueError(
                f"{STAGE_KEYS[stage]}.use_msa is false. Both folding stages "
                f"must use MSAs; folding a target without one degrades the "
                f"structure and the design is then rejected on metrics that "
                f"say nothing about the design."
            )

    for stage in ("filter", "recycle"):
        backend = resolve_backend_name(run_settings, stage)
        resolve_selection(run_settings, stage)
        if resolve_folder(run_settings, stage).get("decoupled", False) and backend != "af3":
            raise ValueError(
                f"{STAGE_KEYS[stage]}.decoupled is true, but "
                f"{STAGE_KEYS[stage]}.backend={backend!r}. Only the af3 "
                f"extractor can take the structure and the scores from "
                f"different predictions."
            )


# --- AF3 backend: batched ---


def fold_af3(
    jobs: Sequence[FoldJob], ctx: FoldContext
) -> List[Optional[FoldResult]]:
    """Fold every job in one Singularity call.

    AF3 pays its container start-up and model load once for the whole list,
    which is the reason this backend does not go through `sequential`.
    """
    input_jsons = [
        af3.create_input_dict(
            job.binder_sequence,
            job.target_sequence,
            ctx.binder_chain,
            ctx.target_chain,
            job.name,
            ctx.seeds,
        )
        for job in jobs
    ]

    try:
        succeeded = set(
            af3._run_af3_batch_inference(
                input_jsons=input_jsons,
                output_dir=str(ctx.output_dir),
                binder_chain=ctx.binder_chain,
                msa_method=ctx.msa_method,
                msa_scope=ctx.msa_scope,
                run_settings=ctx.run_settings,
                target_ids=[job.target_index for job in jobs],
                folder=ctx.folder,
                num_samples=ctx.num_samples,
                num_cycles=ctx.num_cycles,
            )
        )
    except FileNotFoundError as e:
        # Produced-nothing is all-None, not a raise: the Backend contract puts
        # that decision in the entry point, which escalates it once for every
        # backend. Raising here instead reached no such decision -- it crashed
        # the run even for a lone design, where _run treats it as a per-design
        # fault. Per-job misses are already Nones below; this is the all-miss
        # case, which _run_af3_batch_inference signals by raising.
        log.info(f"  AF3 batch produced no output at all: {e}")
        succeeded = set()

    results: List[Optional[FoldResult]] = []
    for job in jobs:
        if job.name not in succeeded:
            results.append(None)
            continue
        if ctx.want_all_samples:
            # Two extractions: the chosen sample's confidences, then every
            # sample for the relax ensemble. cleanup only on the second.
            chosen = af3.extract_structure_and_scores(
                str(ctx.output_dir), job.name,
                sample_select_mode=ctx.sample_select_mode,
                seed_select_mode=ctx.seed_select_mode,
                decoupled=ctx.decoupled, cleanup=False,
            )
            all_samples = af3.extract_structure_and_scores(
                str(ctx.output_dir), job.name, want_all=True, cleanup=True,
            )
            results.append((*chosen, all_samples))
        else:
            results.append(
                af3.extract_structure_and_scores(
                    str(ctx.output_dir), job.name,
                    sample_select_mode=ctx.sample_select_mode,
                    seed_select_mode=ctx.seed_select_mode,
                    decoupled=ctx.decoupled, cleanup=True,
                )
            )
    return results


# --- Protenix backend: one complex per call ---


def fold_one_protenix(job: FoldJob, ctx: FoldContext) -> FoldResult:
    """Fold one complex with Protenix.

    Imported here, not at module scope: Protenix is optional, lives in its own
    conda environment, and an af3-only install must not pay for it or fail on
    its absence.
    """
    from germinal.filters import protenix

    # The same four as the af3 backend gets, spelled the same way.
    common = dict(
        binder_seq=job.binder_sequence,
        target_seq=job.target_sequence,
        target_chains=ctx.target_chain,
        output_dir=str(ctx.output_dir),
        design_name=job.name,
        run_settings=ctx.run_settings,
        binder_chain=ctx.binder_chain,
        msa_method=ctx.msa_method,
        msa_scope=ctx.msa_scope,
        cleanup=True,
        folder=ctx.folder,
        num_samples=ctx.num_samples,
        num_cycles=ctx.num_cycles,
        seed=ctx.seeds,
        sample_select_mode=ctx.sample_select_mode,
        seed_select_mode=ctx.seed_select_mode,
    )

    if not ctx.want_all_samples:
        return protenix.run_protenix(**common)

    # One inference, every sample. The chosen one cannot be read off the ends
    # of `all_samples`, which is ranked over the flat pool.
    chosen, all_samples = protenix.run_protenix(want_all=True, **common)
    return (*chosen, all_samples)


register_backend("af3", fold_af3)
def fold_protenix(jobs, ctx: FoldContext):
    """Protenix, batched at the filter stages and sequential at the recycle one.

    The filter stages fold many designs at once, so one invocation over all of
    them loads the model once instead of per complex. The recycle loop folds
    one candidate at a time, so batching there would add staging for nothing.
    """
    if ctx.stage != "filter" or len(jobs) <= 1:
        return sequential(fold_one_protenix)(jobs, ctx)

    from germinal.filters import protenix

    target_chains = (ctx.target_chain.split(",")
                     if isinstance(ctx.target_chain, str) else ctx.target_chain)
    results = protenix.run_protenix_batch(
        [(j.name, j.binder_sequence, [j.target_sequence]) for j in jobs],
        output_dir=str(ctx.output_dir),
        seed=ctx.seeds,
        run_settings=ctx.run_settings,
        binder_chain=ctx.binder_chain,
        target_chains=target_chains,
        msa_method=ctx.msa_method,
        msa_scope=ctx.msa_scope,
        want_all=ctx.want_all_samples,
        cleanup=True,
        num_samples=ctx.num_samples,
        num_cycles=ctx.num_cycles,
        sample_select_mode=ctx.sample_select_mode,
        seed_select_mode=ctx.seed_select_mode,
        folder=ctx.folder,
    )
    # run_protenix returns (chosen, all_samples) under want_all; the batch
    # returns whatever extract gave it, so flatten it the same way.
    if ctx.want_all_samples:
        results = [None if r is None else (*r[0], r[1]) for r in results]
    return results


register_backend("protenix", fold_protenix)


# --- Public entry points ---


def _draw_seeds(n: int) -> List[int]:
    return [int(x) for x in np.random.randint(0, 999999, size=n)]


#: Consecutive batches that failed entirely but were excused as one design's
#: legs (see _run). One such batch is a per-design fault; a run of them is the
#: backend being broken, which the carve-out would otherwise hide one design
#: at a time. Reset by any job that produces output.
_SINGLE_DESIGN_FAIL_STREAK = 0


def _run(stage: str, jobs: Sequence[FoldJob], ctx: FoldContext):
    global _SINGLE_DESIGN_FAIL_STREAK
    backend_name = resolve_backend_name(ctx.run_settings, stage)
    backend = _BACKENDS[backend_name]
    try:
        results = backend(jobs, ctx)
    except AllJobsFailed as e:
        # Name the stage and backend: the message ends up in the run artifacts,
        # where "all jobs failed" alone does not say which fold was running.
        raise AllJobsFailed(f"{stage} stage ({backend_name}): {e}") from e

    # Before the all-None check below: all() over an empty list is True, so a
    # backend returning nothing would report as a systemic fold failure and
    # hide the positional mismatch that actually went wrong.
    if len(results) != len(jobs):
        raise RuntimeError(
            f"The {stage} folding backend returned {len(results)} results for "
            f"{len(jobs)} jobs; results are matched to jobs by position."
        )

    # A batch backend reports failure as all-None rather than by raising, so
    # the escalation has to happen here -- the backends disagreed otherwise
    # (af3 raised, protenix returned Nones) for one and the same condition.
    if any(r is not None for r in results):
        _SINGLE_DESIGN_FAIL_STREAK = 0
    if jobs and all(r is None for r in results):
        # Same carve-out sequential() makes: legs of ONE design share a binder
        # sequence, so they cannot fail independently and all of them failing
        # is one per-design fault, not an environment one. Batches of a single
        # design are routine whenever few hallucinations pass.
        distinct = len({job.binder_sequence for job in jobs})
        if distinct <= 1:
            _SINGLE_DESIGN_FAIL_STREAK += 1
            limit = int(ctx.run_settings.get("max_consecutive_tool_failures", 10))
            if limit and _SINGLE_DESIGN_FAIL_STREAK >= limit:
                raise AllJobsFailed(
                    f"{stage} stage ({backend_name}): {_SINGLE_DESIGN_FAIL_STREAK} "
                    f"batches in a row produced no output. Each was excused as "
                    f"one design's legs failing together, but a run of them is "
                    f"the backend failing, not {_SINGLE_DESIGN_FAIL_STREAK} bad "
                    f"designs -- check the backend's output above. Designs "
                    f"already past hallucination are journaled and a rerun will "
                    f"adopt them. Set max_consecutive_tool_failures to 0 to "
                    f"disable this check."
                )
            # WARNING, not INFO: at log_level WARNING, documented as "just the
            # problems", an INFO line here means a backend that never works
            # prints nothing explaining why.
            log.warning(
                f"  All {len(jobs)} {stage} folding job(s) ({backend_name}) "
                f"produced no output, but they are legs of ONE design sharing "
                f"a single binder sequence, so this is one failure and not "
                f"{len(jobs)} independent ones. Treating it as a per-design "
                f"fault: the design is dropped and the run continues "
                f"({_SINGLE_DESIGN_FAIL_STREAK}/{limit} in a row)."
            )
        else:
            raise AllJobsFailed(
                f"{stage} stage ({backend_name}): every one of the "
                f"{len(jobs)} folding jobs produced no output, across "
                f"{distinct} distinct binder sequences. Independent designs "
                f"cannot all fail at once for design reasons, so this is an "
                f"environment or config fault -- check the backend's output "
                f"above. Designs already past hallucination are journaled and "
                f"a rerun will adopt them."
            )

    return results


def fold_candidates(
    binder_sequences: List[str],
    target_sequences: List[str],
    target_chain: str,
    binder_chain: str,
    output_dir,
    design_name_prefix: str,
    run_settings: Dict[str, Any],
    sample_select_mode: Optional[str] = None,
    seed_select_mode: Optional[str] = None,
) -> List[Optional[FoldResult]]:
    """Fold every (binder sequence x target) pair, with the recycle backend.

    Either mode left None is resolved from ``recycle_folder``. Seeds come
    from ``num_fold_seeds_recycle``, not ``num_fold_seeds_initial``: the loop
    folds far more often, so it has its own smaller budget.

    Returns:
        One (pdb, scores, ipsae) per pair, ordered
        [seq0_t0, seq0_t1, seq1_t0, ...], or None in a slot that produced no
        output -- one bad candidate must not discard the rest of the cycle, so
        callers skip None slots.
    """
    jobs = [
        FoldJob(
            name=f"{design_name_prefix}_seq{seq_idx}_target{t_idx}",
            binder_sequence=binder_seq,
            target_sequence=target_seq,
            target_index=t_idx,
        )
        for seq_idx, binder_seq in enumerate(binder_sequences)
        for t_idx, target_seq in enumerate(target_sequences)
    ]
    ctx = build_context(
        run_settings, "recycle",
        target_chain=target_chain,
        binder_chain=binder_chain,
        output_dir=output_dir,
        seeds=_draw_seeds(run_settings["num_fold_seeds_recycle"]),
        sample_select_mode=sample_select_mode,
        seed_select_mode=seed_select_mode,
    )
    return _run("recycle", jobs, ctx)


def fold_designs_against_targets(
    designs: Sequence[Tuple[str, str]],
    target_sequences: List[str],
    target_names: List[str],
    target_chain: str,
    binder_chain: str,
    output_dir,
    run_settings: Dict[str, Any],
    sample_select_mode: Optional[str] = None,
    seed_select_mode: Optional[str] = None,
    is_final: bool = False,
) -> Dict[str, List[Optional[FoldResult]]]:
    """Fold MANY designs against every target in one call.

    Unlike ``fold_against_targets``, which batches one design's targets,
    this batches across designs as well.

    Args:
        designs: (design_name, binder_sequence) pairs. Names must be unique.

    Returns:
        design_name -> one result per target, in target order. A design with a
        missing target gets None in that slot; the caller decides whether that
        is fatal, because one bad design must not discard the batch.
    """
    names = [n for n, _ in designs]
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(
            f"fold_designs_against_targets needs unique design names; got "
            f"duplicates {dupes}. Results are matched back by name."
        )
    jobs = [
        FoldJob(
            name=f"{design_name}_{target_name}",
            binder_sequence=binder_seq,
            target_sequence=target_seq,
            target_index=t_idx,
        )
        for design_name, binder_seq in designs
        for t_idx, (target_seq, target_name) in enumerate(
            zip(target_sequences, target_names)
        )
    ]
    ctx = build_context(
        run_settings, "filter",
        target_chain=target_chain,
        binder_chain=binder_chain,
        output_dir=output_dir,
        seeds=_draw_seeds(
            run_settings["num_fold_seeds_final"]
            if is_final
            else run_settings["num_fold_seeds_initial"]
        ),
        sample_select_mode=sample_select_mode,
        seed_select_mode=seed_select_mode,
        want_all_samples=is_final,
    )
    results = _run("filter", jobs, ctx)

    per_target = len(target_names)
    return {
        design_name: results[i * per_target : (i + 1) * per_target]
        for i, (design_name, _) in enumerate(designs)
    }


def fold_against_targets(
    binder_sequence: str,
    target_sequences: List[str],
    target_names: List[str],
    target_chain: str,
    binder_chain: str,
    output_dir,
    design_name: str,
    run_settings: Dict[str, Any],
    sample_select_mode: Optional[str] = None,
    seed_select_mode: Optional[str] = None,
    is_final: bool = False,
) -> List[FoldResult]:
    """Fold one binder against every target, with the filter backend.

    Seeds come from ``num_fold_seeds_final`` when ``is_final``, else
    ``num_fold_seeds_initial``. Every target is needed to score the design,
    so unlike the recycle path a missing job raises.

    Returns:
        One tuple per target: (pdb, scores, ipsae), plus a fourth element
        holding every sample when ``is_final``.
    """
    jobs = [
        FoldJob(
            name=f"{design_name}_{target_name}",
            binder_sequence=binder_sequence,
            target_sequence=target_seq,
            target_index=t_idx,
        )
        for t_idx, (target_seq, target_name) in enumerate(
            zip(target_sequences, target_names)
        )
    ]
    ctx = build_context(
        run_settings, "filter",
        target_chain=target_chain,
        binder_chain=binder_chain,
        output_dir=output_dir,
        seeds=_draw_seeds(
            run_settings["num_fold_seeds_final"]
            if is_final
            else run_settings["num_fold_seeds_initial"]
        ),
        sample_select_mode=sample_select_mode,
        seed_select_mode=seed_select_mode,
        want_all_samples=is_final,
    )

    results = _run("filter", jobs, ctx)
    missing = [job.name for job, r in zip(jobs, results) if r is None]
    if missing:
        raise FileNotFoundError(
            f"{resolve_backend_name(run_settings, 'filter')} produced no output "
            f"for {len(missing)} of {len(jobs)} target(s) of design "
            f"{design_name}: {missing}. Every target is required to score a "
            f"multi-target design."
        )
    return results
