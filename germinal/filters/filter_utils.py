"""Filter utilities for structure prediction and interface quality assessment.

This module contains functions for running filters and computing metrics for a single design trajectory.
"""

import os
import tempfile
import time
from typing import Any, Dict, Optional, Tuple, Sequence, Union, List
import numpy as np
from iglm import IgLM
from germinal.utils import utils
from germinal.filters import folding, pyrosetta_utils
from germinal.utils.io import IO, Trajectory

from germinal.utils.log import get_logger

log = get_logger(__name__)


# None-safe aggregation. An unmeasured metric is None, never a sentinel:
# these helpers drop Nones and return None only when nothing was measured.
def _present(values):
    return [v for v in values if v is not None]


def _fmt(value, decimals=3):
    """Format a metric for logging, tolerating the None an early-rejected
    target reports for every metric it skipped."""
    if value is None:
        return "None"
    try:
        return f"{value:.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def _agg_min(values):
    vals = _present(values)
    return np.min(vals) if vals else None


def _agg_max(values):
    vals = _present(values)
    return np.max(vals) if vals else None


def _agg_mean(values):
    vals = _present(values)
    return np.mean(vals) if vals else None


def _store_per_leg(out: dict, metric_name: str, values) -> None:
    """Per-leg columns keyed by leg position, not target name.

    Target-named columns made every arm's CSV a different schema, so a
    concatenation across arms was block-diagonal. The index is resolved back to
    a name through the target_leg<i> columns.
    """
    for i, value in enumerate(values):
        out[f"{metric_name}.leg{i}"] = value


#: Stages that may own a metric. Columns read "<stage>.<agg>_<metric>" or

#: Aggregation prefixes a recorded aggregate column can carry.
AGG_PREFIXES = ("min", "max", "mean")


def strip_stage_prefix(filter_set: dict, stage: str) -> dict:
    """Return `filter_set` keyed by bare metric name, validating the stage.

    Filter configs name their stage ("final.ipsae") so a config key reads the
    same as the column it produces. Everything inside this module works on bare
    names, so the prefix is removed once, here, rather than threaded through
    ten call sites.

    Validating rather than just stripping catches a file placed in the wrong
    directory, or copied between stages without re-keying -- which would
    otherwise apply the final thresholds at the initial stage and silently
    change what the run measures.
    """
    out = {}
    for key, spec in filter_set.items():
        prefix, _, bare = key.partition(".")
        if not bare:
            raise ValueError(
                f"Filter key {key!r} has no stage prefix. Expected "
                f"'{stage}.{key}' -- keys name their stage so that a config "
                f"key and the CSV column it produces read the same."
            )
        if prefix != stage:
            raise ValueError(
                f"Filter key {key!r} is for the {prefix!r} stage but this is "
                f"the {stage!r} filter. A config keyed for the wrong stage "
                f"would apply the wrong thresholds without failing."
            )
        if bare.split("_", 1)[0] in AGG_PREFIXES:
            raise ValueError(
                f"Filter key {key!r} carries an aggregation prefix. Thresholds "
                f"are applied per target leg, never to a cross-leg aggregate, "
                f"so '{stage}.{bare.split('_', 1)[1]}' is the correct form."
            )
        out[bare] = spec
    return out


def resolve_target_sequences(target_settings: dict) -> List[str]:
    """Target chain sequences, one per antigen state.

    Identical for every design in a run, so a batched fold resolves it once
    instead of re-parsing every target PDB per design.
    """
    target_chain = target_settings["target_chain"]
    return [
        utils.get_sequence_from_pdb(path)[target_chain]
        for path in target_settings["target_pdb_path"]
    ]


def fold_designs_for_filter(
    designs: Sequence[Tuple[str, str]],
    run_settings: dict,
    target_settings: dict,
    io: IO,
    is_final: bool = False,
) -> Dict[str, List]:
    """Fold MANY designs against every target in ONE backend call.

    The batched counterpart to the fold inside multi_run_filters: it pays the
    container start once and lets every design share one target MSA.

    Args:
        designs: (design_name, binder_sequence) pairs. Names must be unique.

    Returns:
        {design_name: [result per target]}, in target order. A target that
        produced no output is None in its slot; the caller decides, because one
        bad design must not discard a whole batch.
    """
    if not designs:
        return {}

    batch_dir = io.layout.redesign_candidates / "structures" / "_batch_fold"
    os.makedirs(batch_dir, exist_ok=True)

    backend_name = folding.resolve_backend_name(run_settings, "filter")
    stage = "final" if is_final else "initial"
    t_start = time.time()
    results = folding.fold_designs_against_targets(
        designs=designs,
        target_sequences=resolve_target_sequences(target_settings),
        target_names=target_settings["target_name"],
        target_chain=target_settings["target_chain"],
        binder_chain=target_settings["binder_chain"],
        output_dir=batch_dir,
        run_settings=run_settings,
        is_final=is_final,
    )
    log.info(
        f"Batched {backend_name} {stage} prediction: {len(designs)} design(s) x "
        f"{len(target_settings['target_name'])} target(s) took (seconds): "
        f"{time.time() - t_start:.2f}"
    )
    return results


def _as_float(value) -> float:
    """A confidence metric as a float, whatever shape the backend reported.

    af3 reports numpy scalars; protenix reports 1-element lists for ptm, iptm
    and aggregate_score. Both have to reach the CSV as a number, and
    float([0.8]) raises.
    """
    if isinstance(value, (list, tuple, np.ndarray)):
        return float(np.mean(value))
    return float(value)


def multi_run_filters(
    trajectory: Trajectory,
    run_settings: dict,
    target_settings: dict,
    filter_set: dict,
    io: IO,
    trajectory_sequence: str,
    rmsd_reference_pdbs: List[str],
    is_final: bool = False,
    prediction_results: Optional[List] = None,
) -> Tuple[dict, dict, bool, str]:
    """Run filters and compute metrics for multi-target design trajectory.

    Per target: fold, relax, score the interface, then aggregate across
    targets (worst leg) and evaluate against the filter set.

    Args:
        trajectory: Trajectory metadata for the current design.
        run_settings: Configuration for the run (model choice, CDRs, etc.).
        target_settings: Target/binder chains, hotspots, target length.
        filter_set: Mapping of metric name to threshold spec with an operator.
        io: IO/layout helper providing directory structure.
        trajectory_sequence: Final amino acid sequence of the binder.
        rmsd_reference_pdbs: Reference PDBs for sc_rmsd alignment, one per
            target: the hallucination output at the initial stage, and the
            candidate's own designed_on at the final stage.
        is_final: If True, use num_fold_seeds_final folding seeds, parallel
            relax and ensemble scoring; otherwise num_fold_seeds_initial.
        prediction_results: This design's folds, one per target, already done
            by fold_designs_for_filter as part of a batch. None folds here for
            this design alone -- correct, but it pays a container start per
            design, which is what the batched path exists to avoid.

    Returns:
        Tuple containing:
            - filter_metrics: Dict of aggregated metrics for the trajectory
            - filter_results: Dict mapping '<metric>_filter' to pass/fail booleans
            - accepted: True if all filters in the set passed, else False
            - external_relaxed_pdbs: List of relaxed complex PDB paths, one
              per antigen state (io.save_trajectory iterates them)
    """
    assert len(rmsd_reference_pdbs) >= 2, "multi_run_filters expects at least 2 reference PDB files"

    # Checked first: results are matched to targets by POSITION, so a short
    # list would score the design against the wrong antigen.
    if prediction_results is not None and (
        len(prediction_results) != len(target_settings["target_name"])
    ):
        raise ValueError(
            f"{trajectory.design_name}: got {len(prediction_results)} fold "
            f"result(s) for {len(target_settings['target_name'])} targets. "
            f"Results are positional, so a mismatched list would score the "
            f"design against the wrong antigen."
        )

    stage = "final" if is_final else "initial"
    filter_set = strip_stage_prefix(filter_set, stage)

    # Extract target-specific settings
    target_names = target_settings["target_name"]
    target_hotspots_list = target_settings["target_hotspots"]
    target_pdb_paths = target_settings["target_pdb_path"]


    # Filter-model predictions live under redesign_candidates/, beside the
    # recycling model's.
    structures_directory = io.layout.redesign_candidates / "structures"
    design_structures_dir = structures_directory / trajectory.design_name
    os.makedirs(design_structures_dir, exist_ok=True)
    target_chain = target_settings["target_chain"]
    binder_chain = target_settings["binder_chain"]

    # Get target sequences for each target
    target_sequences = resolve_target_sequences(target_settings)

    # Calculate CDR3 positions
    # 1-indexed for PDB numbering; see utils.cdr3_positions for the nb/scfv split
    cdr3 = np.array(utils.cdr3_positions(run_settings)) + 1

    # ========================== Calculate number of framework mutations ==========================
    # Computed once since this is a sequence-level metric
    n_framework_mutations, framework_mutations = get_framework_mutations(
        trajectory_sequence,
        run_settings["starting_binder_seq"],
        run_settings["cdr_positions"],
    )
    log.info(f"Framework mutations: {framework_mutations}")

    # ========================== Get Log-likelihood from IgLM ==========================
    # Computed once since this is a sequence-level metric
    iglm_ll = get_iglm_ll(
        sequence=trajectory_sequence,
        species_token=run_settings["iglm_species"],
        vh_first=run_settings["vh_first"],
        vh_len=run_settings["vh_len"],
        vl_len=run_settings["vl_len"],
    )

    # ========================== Batched Structure Prediction ==========================
    # All targets in one call where the backend supports it; which backend
    # runs is resolved inside folding.py.
    if prediction_results is None:
        backend_name = folding.resolve_backend_name(run_settings, "filter")
        t_start = time.time()
        prediction_results = _batch_af3_multi_targets(
            trajectory_sequence=trajectory_sequence,
            target_sequences=target_sequences,
            target_chain=target_chain,
            binder_chain=binder_chain,
            structures_directory=design_structures_dir,
            design_name=trajectory.design_name,
            run_settings=run_settings,
            target_names=target_names,
            is_final=is_final,
        )
        log.info(
            f"Batched {backend_name} prediction for {len(target_names)} targets "
            f"took (seconds): {time.time() - t_start:.2f}"
        )

    # Run filtering for each target
    all_target_metrics = []
    all_target_results = []
    all_target_accepted = []
    external_relaxed_pdbs = []
    early_rejection_meta_cache = None  # first early-reject context, for design-level logs

    for target_idx, (rmsd_reference_pdb, target_sequence, target_hotspots, target_name, target_pdb_path) in enumerate(
        zip(rmsd_reference_pdbs, target_sequences, target_hotspots_list, target_names, target_pdb_paths)
    ):
        log.info(f"\n{'='*60}")
        log.info(f"Processing target {target_idx + 1}/{len(target_names)}: {target_name}")
        log.info(f"{'='*60}\n")

        # Create target-specific settings for this iteration
        target_settings_single = target_settings.copy()
        target_settings_single["target_hotspots"] = target_hotspots

        # Create run_settings copy with the correct PDB for this target
        run_settings_single = run_settings.copy()
        run_settings_single["starting_pdb_complex"] = target_pdb_path

        # ========================== Structure Prediction Result ==========================
        # A batched backend reports a per-job failure as None in its slot,
        # not by raising, so the length check above cannot catch it.
        prediction_result = prediction_results[target_idx]
        if prediction_result is None:
            # Tool failure, not a weak design: drop this design but let the
            # rest of the batch score. Metrics stay absent, so it is unmeasured.
            log.info(
                f"  WARNING: {trajectory.design_name}: no fold result for target "
                f"{target_idx} ({target_name}); the backend returned None for "
                f"this job. Skipping this design -- it is NOT a filter failure. "
                f"See the folding errors above."
            )
            skip_metrics = {
                "failure_stage": "missing_fold",
                "missing_fold_target": target_name,
            }
            skip_results = {f"{name}_filter": False for name in filter_set}
            return skip_metrics, skip_results, False, [None] * len(target_names)

        if is_final:
            external_pdb, confidence_source_metrics, confidence_source_ipsae, all_structures = prediction_result
        else:
            external_pdb, confidence_source_metrics, confidence_source_ipsae = prediction_result
            all_structures = None

        # ===================== Early Confidence Rejection =====================
        # Decision-neutral: a design rejected here would fail the same filter
        # after relax and interface scoring anyway.
        confidence_filter_map = {
            "plddt": "plddt",
            "ptm": "ptm",
            "iptm": "iptm",
            "pae": "pae",
        }
        early_reject = False
        for filter_name, metric_key in confidence_filter_map.items():
            if filter_name in filter_set and metric_key in confidence_source_metrics:
                metric_value = confidence_source_metrics[metric_key]
                if isinstance(metric_value, (list, np.ndarray)):
                    metric_value = float(np.mean(metric_value))
                _, check = evaluate_filters(
                    {filter_name: filter_set[filter_name]},
                    {filter_name: metric_value},
                )
                if not check.get(f"{filter_name}_filter", True):
                    log.info(
                        f"  Early reject on {filter_name}: {metric_value} "
                        f"(threshold: {filter_set[filter_name]['operator']} "
                        f"{filter_set[filter_name]['value']})"
                    )
                    early_reject = True

        if "ipsae" in filter_set and confidence_source_ipsae is None:
            # ipsae is None, which fails every filter anyway, so reject before
            # paying for relax. It stays None and buckets as unmeasured.
            log.info(
                "  Early reject: ipSAE could not be computed (tool failure, not "
                "a weak interface); skipping relax+scoring for this target."
            )
            early_reject = True
        elif "ipsae" in filter_set:
            ipsae_scalar = (
                confidence_source_ipsae["ipsae"]
                if isinstance(confidence_source_ipsae, dict)
                else confidence_source_ipsae
            )
            _, check = evaluate_filters(
                {"ipsae": filter_set["ipsae"]}, {"ipsae": ipsae_scalar}
            )
            if not check.get("ipsae_filter", True):
                log.info(
                    f"  Early reject on ipsae: {ipsae_scalar} "
                    f"(threshold: {filter_set['ipsae']['operator']} "
                    f"{filter_set['ipsae']['value']})"
                )
                early_reject = True

        if early_reject:
            log.info(f"\nTarget {target_name} - EARLY REJECTED (skipping relax+scoring)")
            (
                early_metrics,
                early_results,
                early_rejection_meta,
            ) = build_early_reject_metrics(
                target_idx=target_idx,
                confidence_source_metrics=confidence_source_metrics,
                confidence_source_ipsae=confidence_source_ipsae,
                target_name=target_name,
                filter_set=filter_set,
                lm_ll=iglm_ll,
                n_framework_mutations=n_framework_mutations,
                framework_mutations=framework_mutations,
            )
            all_target_metrics.append(early_metrics)
            all_target_results.append(early_results)
            all_target_accepted.append(False)
            external_relaxed_pdbs.append(external_pdb)
            if early_rejection_meta_cache is None:
                early_rejection_meta_cache = early_rejection_meta
            continue

        # ========================== Run Structure Scoring ==========================
        t_start = time.time()
        (target_filter_metrics, target_filter_results, target_accepted,
         external_relaxed_pdb) = run_structure_scoring(
            external_pdb=external_pdb,
            confidence_source_metrics=confidence_source_metrics,
            confidence_source_ipsae=confidence_source_ipsae,
            all_structures=all_structures,
            design_structures_dir=design_structures_dir,
            design_name=f"{trajectory.design_name}_{target_name}",
            run_settings=run_settings_single,
            target_settings=target_settings_single,
            filter_set=filter_set,
            binder_chain=binder_chain,
            target_chain=target_chain,
            cdr3=cdr3,
            trajectory_sequence=trajectory_sequence,
            rmsd_reference_pdb=rmsd_reference_pdb,
            is_final=is_final,
            n_framework_mutations=n_framework_mutations,
            framework_mutations=framework_mutations,
            iglm_ll=iglm_ll,
        )
        external_relaxed_pdbs.append(external_relaxed_pdb)
        t_prediction = time.time() - t_start
        log.info(f"Structure scoring took (seconds): {t_prediction:.2f}")

        log.info(f"\nTarget {target_name} - {'ACCEPTED' if target_accepted else 'REJECTED'}")
        if not target_accepted:
            for filter_key, passed in target_filter_results.items():
                if not passed:
                    metric_name = filter_key.replace("_filter", "")
                    metric_val = target_filter_metrics.get(metric_name, "N/A")
                    threshold = filter_set[metric_name]["value"]
                    operator = filter_set[metric_name]["operator"]
                    if isinstance(metric_val, float):
                        log.info(f"  FAILED {metric_name}: {metric_val:.4f} (threshold: {operator} {threshold})")
                    else:
                        log.info(f"  FAILED {metric_name}: {metric_val} (threshold: {operator} {threshold})")

        all_target_metrics.append(target_filter_metrics)
        all_target_results.append(target_filter_results)
        all_target_accepted.append(target_accepted)
    
    # ========================== Aggregate Metrics Across Targets ==========================
    log.info(f"\n{'='*60}")
    log.info("Aggregating metrics across targets...")
    log.info(f"{'='*60}\n")
    
    aggregated_metrics = {}

    # Resolves the positional .leg<i> columns back to a target.
    for i, target_name in enumerate(target_names):
        aggregated_metrics[f"target_leg{i}"] = target_name

    # Union of keys: an early-rejected target and a fully scored one carry
    # different key sets, and a missing key must aggregate as unmeasured.
    metric_names = []
    for metrics in all_target_metrics:
        for key in metrics:
            if key not in metric_names:
                metric_names.append(key)

    for metric_name in metric_names:
        values = [metrics.get(metric_name) for metrics in all_target_metrics]
        
        # Confidence metrics - take minimum (both must be confident)
        if metric_name in ["plddt", "ptm", "iptm", 
                          "aggregate_score",
                          "pdockq_pDockQ2",
                          "lis_lis", "lis_lia", "ipsae"]:
            aggregated_metrics[f"min_{metric_name}"] = _agg_min(values)
            # store per-target values for tracking (if needed)
            _store_per_leg(aggregated_metrics, metric_name, values)
        
        # PAE metrics - take maximum
        elif metric_name in ["pae"]:
            aggregated_metrics[f"max_{metric_name}"] = _agg_max(values)
            _store_per_leg(aggregated_metrics, metric_name, values)
        
        # Sequence-level metrics - same for both targets
        elif metric_name in ["iglm_ll", "n_framework_mutations", "framework_mutations"]:
            aggregated_metrics[metric_name] = values[0]

        # MPNN score is a negative log-likelihood, so the worst-across-targets
        # value is the maximum.
        elif metric_name == "mpnn_score_mean":
            aggregated_metrics[f"max_{metric_name}"] = _agg_max(values)
            _store_per_leg(aggregated_metrics, metric_name, values)

        # A spread, not a quality: the mean is the meaningful summary.
        elif metric_name == "mpnn_score_std":
            aggregated_metrics[f"mean_{metric_name}"] = _agg_mean(values)
            _store_per_leg(aggregated_metrics, metric_name, values)
        
        # Higher is worse, so the maximum is the worst-across-targets value.
        # binder_score and surface_hydrophobicity belong here for the same reason.
        elif metric_name in ["clashes", "clashes_unrelaxed", "num_clashes_trajectory", 
                            "num_clashes_relaxed", "hydrophobic_patches_binder", 
                            "sap_score", "cdr_sap",
                            "binder_score", "surface_hydrophobicity",
                            "interface_dG_SASA_ratio"]:
            aggregated_metrics[f"max_{metric_name}"] = _agg_max(values)
            _store_per_leg(aggregated_metrics, metric_name, values)
        
        # Interface quality - take minimum (higher is better for every metric
        # here, so the minimum is the worst-across-targets value)
        elif metric_name in ["interface_shape_comp", "interface_packstat",
                            "interface_dSASA", "interface_fraction",
                            "interface_nres", "interface_hbonds", "interface_hbond_percentage",
                            "percent_interface_cdr",
                            "interface_hydrophobicity"]:
            aggregated_metrics[f"min_{metric_name}"] = _agg_min(values)
            _store_per_leg(aggregated_metrics, metric_name, values)
        
        # Binding energy - take max 
        elif metric_name == "interface_dG":
            aggregated_metrics[f"max_{metric_name}"] = _agg_max(values)  # Least negative
            aggregated_metrics[f"min_{metric_name}"] = _agg_min(values)  # Most negative
            _store_per_leg(aggregated_metrics, metric_name, values)
        
        # Unsaturated H-bonds - take max
        elif metric_name in ["interface_delta_unsat_hbonds", "interface_delta_unsat_hbonds_percentage"]:
            aggregated_metrics[f"max_{metric_name}"] = _agg_max(values)
            _store_per_leg(aggregated_metrics, metric_name, values)
        
        # Hotspot contacts - take min
        elif metric_name in ["binder_near_hotspot", "cdr3_hotspot_contacts", "cdr_hotspot_contacts"]:
            # Min only: the legs are deliberately the same epitope, so summing
            # their contacts would double-count one interface.
            aggregated_metrics[f"min_{metric_name}"] = _agg_min(values)
            _store_per_leg(aggregated_metrics, metric_name, values)
        
        # Secondary structure - report per-target
        elif metric_name in ["alpha_interface", "beta_interface", "loops_interface",
                            "alpha_all", "beta_all", "loops_all"]:
            aggregated_metrics[f"mean_{metric_name}"] = _agg_mean(values)
            _store_per_leg(aggregated_metrics, metric_name, values)
        
        # RMSD - take maximum
        elif metric_name == "sc_rmsd":
            aggregated_metrics[f"max_{metric_name}"] = _agg_max(values)
            _store_per_leg(aggregated_metrics, metric_name, values)

        # Detailed data - store per-target only
        elif metric_name in ["interface_AA", "interface_residues", "ss_content"]:
            _store_per_leg(aggregated_metrics, metric_name, values)
        
        # Default - minimum (conservative approach)
        else:
            aggregated_metrics[f"min_{metric_name}"] = _agg_min(values)
            log.warning(f"Unknown metric '{metric_name}', using minimum aggregation")
            _store_per_leg(aggregated_metrics, metric_name, values)
    
    # ========================== Aggregate Filter Results ==========================
    # For filter results, a design passes only if it passes ALL targets (AND logic)
    aggregated_results = {}
    for result_name in all_target_results[0].keys():
        individual_results = [results[result_name] for results in all_target_results]
        aggregated_results[result_name] = all(individual_results)
        
        # Log which targets failed which filters
        if not aggregated_results[result_name]:
            failed_targets = [
                target_names[i] for i, passed in enumerate(individual_results) if not passed
            ]
            log.info(f"Filter '{result_name}' failed for targets: {failed_targets}")
    
    # Acceptance is per-target AND-filtering: a design must pass every filter
    # against every antigen state.
    overall_accepted = all(all_target_accepted)

    # ========================== Print Summary ==========================
    log.info(f"\n{'='*60}")
    log.info(f"OVERALL RESULT: {'ACCEPTED' if overall_accepted else 'REJECTED'}")
    log.info(f"{'='*60}\n")
    
    # Prefixes must match the direction each metric aggregates in: min for
    # higher-is-better, max for lower-is-better.
    log.info("Key aggregated metrics:")
    log.info(f"  pLDDT (min): {_fmt(aggregated_metrics.get('min_plddt'), 3)}")
    log.info(f"  iPTM (min): {_fmt(aggregated_metrics.get('min_iptm'), 3)}")
    log.info(f"  Clashes (max): {aggregated_metrics.get('max_clashes')}")
    log.info(f"  CDR hotspot contacts (min): {aggregated_metrics.get('min_cdr_hotspot_contacts')}")
    log.info(f"  Interface dG (worst): {_fmt(aggregated_metrics.get('max_interface_dG'), 2)}")
    
    if early_rejection_meta_cache is not None:
        aggregated_metrics.update(early_rejection_meta_cache)

    # Return every state's relaxed PDB; the caller saves one per target
    return aggregated_metrics, aggregated_results, overall_accepted, external_relaxed_pdbs

def build_early_reject_metrics(
    confidence_source_metrics: dict,
    confidence_source_ipsae,
    target_name: str,
    target_idx: int,
    filter_set: dict,
    lm_ll,
    n_framework_mutations,
    framework_mutations,
) -> Tuple[Dict[str, Any], Dict[str, bool], Dict[str, Any]]:
    """
    Build the metrics dict for a target rejected by the confidence early check.

    The real AF3 confidence values are kept; everything that would have come
    from relax and interface scoring is None, not a sentinel. Keys match
    build_filter_metrics() so aggregation treats both kinds of leg alike.

    Returns:
        (metrics, filter_results, rejection_meta). rejection_meta is separate so
        it can be attached once at design level rather than aggregated per metric.
    """
    ipsae_scalar = (
        confidence_source_ipsae["ipsae"]
        if isinstance(confidence_source_ipsae, dict)
        else confidence_source_ipsae
    )

    def _scalar(v):
        if isinstance(v, (list, np.ndarray)):
            return float(np.mean(v))
        return v

    # Measured: the AF3 confidence values that caused the rejection.
    metrics = {
        "plddt": _scalar(confidence_source_metrics.get("plddt")),
        "ptm": _scalar(confidence_source_metrics.get("ptm")),
        "iptm": _scalar(confidence_source_metrics.get("iptm")),
        "pae": _scalar(confidence_source_metrics.get("pae")),
        "aggregate_score": _scalar(
            confidence_source_metrics.get("aggregate_score")
        ),
        "ipsae": ipsae_scalar,
        # Sequence-level metrics are computed before prediction, so they are real.
        "n_framework_mutations": n_framework_mutations,
        "framework_mutations": framework_mutations,
        "iglm_ll": lm_ll,
    }

    # Never measured: everything downstream of relax and interface scoring.
    not_measured = [
        "binder_near_hotspot",
        "clashes_unrelaxed",
        "clashes",
        "binder_score",
        "surface_hydrophobicity",
        "interface_shape_comp",
        "interface_packstat",
        "interface_dG",
        "interface_dSASA",
        "interface_dG_SASA_ratio",
        "interface_fraction",
        "interface_hydrophobicity",
        "interface_nres",
        "interface_hbonds",
        "interface_hbond_percentage",
        "interface_delta_unsat_hbonds",
        "interface_delta_unsat_hbonds_percentage",
        "hydrophobic_patches_binder",
        "sap_score",
        "cdr_sap",
        "cdr3_hotspot_contacts",
        "cdr_hotspot_contacts",
        "pdockq_pDockQ2",
        "lis_lis",
        "lis_lia",
        "percent_interface_cdr",
        "alpha_interface",
        "beta_interface",
        "loops_interface",
        "alpha_all",
        "beta_all",
        "loops_all",
        "sc_rmsd",
        "interface_AA",
        "interface_residues",
        "ss_content",
        "mpnn_score_mean",
        "mpnn_score_std",
    ]
    for key in not_measured:
        metrics[key] = None

    # The target is rejected, but not by every filter: evaluate the metrics
    # that exist, so one rejection does not read as a dozen failures.
    results = {}
    for metric_name, filter_config in filter_set.items():
        if metrics.get(metric_name) is None:
            results[f"{metric_name}_filter"] = False
            continue
        _, check = evaluate_filters(
            {metric_name: filter_config}, {metric_name: metrics[metric_name]}
        )
        results[f"{metric_name}_filter"] = check[f"{metric_name}_filter"]

    rejection_meta = {
        "failure_stage": "early_confidence_filter",
        "failed_target": target_name,
        "failed_leg": target_idx,
        "early_filter_ipsae": ipsae_scalar,
        "early_filter_plddt": confidence_source_metrics.get("plddt"),
        "early_filter_iptm": confidence_source_metrics.get("iptm"),
        "early_filter_ptm": confidence_source_metrics.get("ptm"),
        "early_filter_pae": confidence_source_metrics.get("pae"),
        "failure_reason": (
            f"early_filter: ipsae={ipsae_scalar:.3f} on {target_name}"
            if isinstance(ipsae_scalar, (int, float))
            else f"early_filter: confidence threshold failed on {target_name}"
        ),
    }
    return metrics, results, rejection_meta


def select_failed_filters(filter_results: dict, filter_metrics: dict) -> Dict[str, int]:
    """Pick the failure counters to charge for a rejected design.

    A filter whose metric was never measured cannot have fired, so those
    collapse into one counter naming the stage; filters that fired on a real
    measurement are charged individually, and 0 or 0.0 is a measurement.

    Per-leg values are checked before the aggregate, which on a mixed run
    carries the measured leg's value. NaN counts as unmeasured as well as
    None, since the per-leg columns come back as NaN.

    Args:
        filter_results: Mapping of '<metric>_filter' to pass/fail booleans.
        filter_metrics: Metrics for the same design, aggregated and, on a
            multi-target run, per target as '<metric>_<target_name>', plus
            'failed_target' naming the leg that was rejected. A None or NaN
            value means the metric was never computed.

    Returns:
        Counter name -> increment, ready for `IO.update_failures`.
    """

    def _is_measured(value: Any) -> bool:
        if value is None:
            return False
        try:
            return not np.isnan(value)
        except (TypeError, ValueError):
            # Booleans, strings and the like are measurements.
            return True

    failed_target = filter_metrics.get("failed_target")

    def _metric_value(metric_name: str) -> Any:
        """The value that decides whether this filter could have fired.

        Neither "<metric>.<target>" nor "<agg>_<metric>" is reachable under
        the bare filter name. The per-leg value wins; otherwise the aggregate
        is tried under each
        aggregation prefix, then the bare name for binder-level metrics
        (iglm_ll, framework mutations) that are not aggregated at all.
        """
        failed_leg = filter_metrics.get("failed_leg")
        if failed_leg is not None:
            per_leg = f"{metric_name}.leg{failed_leg}"
            if per_leg in filter_metrics:
                return filter_metrics[per_leg]
        # Target-named fallback: reads CSVs written before the leg<i> rename.
        if failed_target:
            per_leg = f"{metric_name}.{failed_target}"
            if per_leg in filter_metrics:
                return filter_metrics[per_leg]
        for agg in AGG_PREFIXES:
            key = f"{agg}_{metric_name}"
            if key in filter_metrics:
                return filter_metrics[key]
        return filter_metrics.get(metric_name)

    failed = {}
    unmeasured = False

    for filter_key, passed in filter_results.items():
        if passed:
            continue
        metric_name = (
            filter_key[: -len("_filter")]
            if filter_key.endswith("_filter")
            else filter_key
        )
        if not _is_measured(_metric_value(metric_name)):
            unmeasured = True
            continue
        failed[filter_key] = 1

    if unmeasured:
        stage = filter_metrics.get("failure_stage")
        # Leading underscore marks a short-circuit bucket, not a filter, so it
        # is never summed alongside real filter counts.
        bucket = {
            "early_confidence_filter": "_short_circuit_early_confidence",
            "missing_fold": "_short_circuit_missing_fold",
        }.get(stage, "_unmeasured_metric")
        failed[bucket] = 1

    return failed


def build_filter_metrics(
    confidence_metrics: dict,
    interface_metrics: dict,
    hydrophobic_patches_binder,
    sap_score,
    cdr_sap,
    cdr3_hotspot_contacts,
    cdr_hotspot_contacts,
    pdockq_metrics: dict,
    lis_metrics: dict,
    percent_interface_cdr,
    ss_content: dict,
    binder_rmsd,
    n_framework_mutations,
    framework_mutations,
    num_clashes_trajectory,
    num_clashes_relaxed,
    binder_near_hotspot,
    iglm_ll,
    mpnn_score_mean=None,
    mpnn_score_std=None,
) -> Dict[str, Any]:
    """
    Aggregate all metrics into comprehensive evaluation dict (floats rounded to 4 decimals).

    Returns:
        Dict[str, Any]: Confidence, interface, structural, biological, and sequence metrics
    """
    metrics = {
        # confidence
        "plddt": confidence_metrics["plddt"],
        "ptm": confidence_metrics["ptm"],
        "iptm": confidence_metrics["i_ptm"],
        "pae": confidence_metrics["pae"],
        "aggregate_score": confidence_metrics["aggregate_score"],
        "ipsae":confidence_metrics["ipsae"],
        # structure + interface
        "binder_near_hotspot": binder_near_hotspot,
        "clashes_unrelaxed": num_clashes_trajectory,
        "clashes": num_clashes_relaxed,  # relaxed clashes
        "binder_score": interface_metrics["interface_scores"]["binder_score"],
        "surface_hydrophobicity": interface_metrics["interface_scores"][
            "surface_hydrophobicity"
        ],
        "interface_shape_comp": interface_metrics["interface_scores"]["interface_sc"],
        "interface_packstat": interface_metrics["interface_scores"][
            "interface_packstat"
        ],
        "interface_dG": interface_metrics["interface_scores"]["interface_dG"],
        "interface_dSASA": interface_metrics["interface_scores"]["interface_dSASA"],
        "interface_dG_SASA_ratio": interface_metrics["interface_scores"][
            "interface_dG_SASA_ratio"
        ],
        "interface_fraction": interface_metrics["interface_scores"][
            "interface_fraction"
        ],
        "interface_hydrophobicity": interface_metrics["interface_scores"][
            "interface_hydrophobicity"
        ],
        "interface_nres": interface_metrics["interface_scores"]["interface_nres"],
        "interface_hbonds": interface_metrics["interface_scores"][
            "interface_interface_hbonds"
        ],
        "interface_hbond_percentage": interface_metrics["interface_scores"][
            "interface_hbond_percentage"
        ],
        "interface_delta_unsat_hbonds": interface_metrics["interface_scores"][
            "interface_delta_unsat_hbonds"
        ],
        "interface_delta_unsat_hbonds_percentage": interface_metrics[
            "interface_scores"
        ]["interface_delta_unsat_hbonds_percentage"],
        "hydrophobic_patches_binder": len(hydrophobic_patches_binder),
        "sap_score": sap_score,
        "cdr_sap": cdr_sap,
        "cdr3_hotspot_contacts": cdr3_hotspot_contacts,
        "cdr_hotspot_contacts": cdr_hotspot_contacts,
        # derived confidence
        "pdockq_pDockQ2": pdockq_metrics["pDockQ2"],
        "lis_lis": lis_metrics["lis"],
        "lis_lia": lis_metrics["lia"],
        # secondary structure + framework metrics
        "percent_interface_cdr": percent_interface_cdr,
        "alpha_interface": ss_content["alpha_i"],
        "beta_interface": ss_content["beta_i"],
        "loops_interface": ss_content["loops_i"],
        "alpha_all": ss_content["alpha_"],
        "beta_all": ss_content["beta_"],
        "loops_all": ss_content["loops_"],
        "sc_rmsd": binder_rmsd,
        "n_framework_mutations": n_framework_mutations,
        # large logs
        "framework_mutations": framework_mutations,
        "interface_AA": interface_metrics["interface_AA"],
        "interface_residues": interface_metrics["interface_residues"],
        "ss_content": ss_content,
        # iglm
        "iglm_ll": iglm_ll,
        # sequence scoring (informational, not used as filters)
        "mpnn_score_mean": mpnn_score_mean,
        "mpnn_score_std": mpnn_score_std,
    }

    # round floats to 4 decimals for compactness
    metrics = {
        k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()
    }
    return metrics


def evaluate_filters(
    filter_set: dict, filter_metrics: dict
) -> Tuple[bool, Dict[str, bool]]:
    """
    Evaluate metrics against quality filters (operators: <, <=, >, >=, ==, =).

    Args:
        filter_set: {metric_name: {"value": threshold, "operator": op}}
        filter_metrics: Calculated metrics dict

    Returns:
        Tuple[bool, Dict[str, bool]]: (all_passed, individual_results)
    """
    filter_results = {}

    missing = [name for name in filter_set if name not in filter_metrics]
    if missing:
        # A filter naming a metric that does not exist is a config error: it
        # would otherwise reject every design and look like weak binding.
        raise KeyError(
            f"Filter config names metric(s) that the pipeline does not "
            f"produce: {sorted(missing)}. Available: "
            f"{sorted(filter_metrics)}. Every filter key must match a produced "
            f"metric exactly -- an unmatched key would otherwise fail every "
            f"design silently."
        )

    for filter_name, filter_config in filter_set.items():

        metric_value = filter_metrics[filter_name]
        threshold = filter_config["value"]
        operator = filter_config["operator"]

        # None means the metric could not be computed. Comparing it would raise
        # TypeError, so treat it as a failed filter instead.
        if metric_value is None:
            log.info(
                f"Warning: metric '{filter_name}' is None (could not be computed), "
                f"treating as failed filter"
            )
            filter_results[f"{filter_name}_filter"] = False
            continue

        # Evaluate based on operator
        if operator == "<":
            passed = metric_value < threshold
        elif operator == "<=":
            passed = metric_value <= threshold
        elif operator == ">":
            passed = metric_value > threshold
        elif operator == ">=":
            passed = metric_value >= threshold
        elif operator == "==":
            passed = metric_value == threshold
        elif operator == "=":
            passed = metric_value == threshold
        else:
            log.warning(f"Unknown operator '{operator}' for filter '{filter_name}'")
            passed = False

        filter_results[f"{filter_name}_filter"] = passed

    # All filters must pass
    all_passed = all(filter_results.values())

    return all_passed, filter_results


def get_framework_mutations(
    trajectory_sequence: str,
    framework_sequence: str,
    cdr_positions: Sequence[int],
) -> Tuple[int, List[str]]:
    """
    Identify mutations outside CDR regions (format: 'A123B').

    Args:
        trajectory_sequence: Final designed sequence
        framework_sequence: Original reference sequence
        cdr_positions: CDR positions (0-indexed)

    Returns:
        Tuple[int, List[str]]: (count, mutation_list)
    """
    framework_mutations = []
    for i, (seq, framework) in enumerate(zip(trajectory_sequence, framework_sequence)):
        if seq != framework and i not in cdr_positions:
            framework_mutations.append(f"{framework}{i + 1}{seq}")
    return len(framework_mutations), framework_mutations


def is_binder_near_hotspot(
    target_contacts: Sequence[int],
    target_hotspots: Sequence[int],
    binder_contacts: Sequence[int],
    cdr_positions: Sequence[int],
    cdr3: Sequence[int],
    min_hotspot_contacts: int = 3,
) -> Tuple[bool, int, int]:
    """Check whether the binder interface is near target hotspot residues.

    Args:
        target_contacts: Target residue indices at the interface (1-indexed).
        target_hotspots: Target hotspot residue indices (1-indexed).
        binder_contacts: Binder residue indices at the interface (1-indexed).
        cdr_positions: Binder CDR residue indices (1-indexed).
        cdr3: Binder CDR3 residue indices (1-indexed).
        min_hotspot_contacts: CDR-hotspot contacts needed to accept.

    Returns:
        (accepted, num_cdr3_hotspot_contacts, num_cdr_hotspot_contacts)
    """
    cdr_hotspot_contacts = []
    cdr3_hotspot_contacts = []
    for i, tc in enumerate(target_contacts):
        if tc in target_hotspots:
            if binder_contacts[i] in cdr_positions:
                cdr_hotspot_contacts.append(binder_contacts[i])
            if binder_contacts[i] in cdr3:
                cdr3_hotspot_contacts.append(binder_contacts[i])
    cdr_hotspot_contacts = set(cdr_hotspot_contacts)
    cdr3_hotspot_contacts = set(cdr3_hotspot_contacts)
    accepted = len(cdr_hotspot_contacts) >= min_hotspot_contacts

    return accepted, len(cdr3_hotspot_contacts), len(cdr_hotspot_contacts)

def run_structure_prediction_batch(
    binder_sequences: List[str],
    target_sequences: List[str],
    target_chain: str,
    binder_chain: str,
    structures_directory: str,
    design_name_prefix: str,
    run_settings: dict,
) -> List[Tuple[str, dict, Any]]:
    """
    Predict every (binder sequence x target) pair in a single batched AF3 call.

    Results are ordered [seq0_t0, seq0_t1, seq1_t0, seq1_t1, ...]. A pair whose
    AF3 output is missing yields None in its slot rather than aborting the batch,
    so one bad candidate does not discard the rest. Callers must skip None slots.

    Args:
        binder_sequences: Candidate binder sequences.
        target_sequences: Target protein sequences.
        target_chain: Target chain ID.
        binder_chain: Binder chain ID.
        structures_directory: Output directory for AF3 results.
        design_name_prefix: Prefix for naming each job.
        run_settings: Config holding the AF3 paths and settings.
    Returns:
        One (pdb_path, metrics, ipsae) tuple per pair, or None where prediction
        produced no output.
    """
    # Thin wrapper: the backend choice and seed count both live in folding.py.
    return folding.fold_candidates(
        binder_sequences=binder_sequences,
        target_sequences=target_sequences,
        target_chain=target_chain,
        binder_chain=binder_chain,
        output_dir=structures_directory,
        design_name_prefix=design_name_prefix,
        run_settings=run_settings,
    )


def _batch_af3_multi_targets(
    trajectory_sequence: str,
    target_sequences: List[str],
    target_chain: str,
    binder_chain: str,
    structures_directory,
    design_name: str,
    run_settings: dict,
    target_names: List[str],
    is_final: bool = False,
) -> List[tuple]:
    """Predict one binder against every target in a single batched AF3 call.

    Saves one container start-up per extra target. Every target is needed to
    score the design, so a missing output raises rather than returning a partial
    result.

    Args:
        trajectory_sequence: Designed antibody sequence.
        target_sequences: Target protein sequences.
        target_chain: Target chain ID.
        binder_chain: Binder chain ID.
        structures_directory: Output directory for AF3 results.
        design_name: Base design name; the target name is appended.
        run_settings: Config holding the AF3 paths and settings.
        target_names: Target name strings, used for job naming.
        is_final: If True, run num_fold_seeds_final seeds and extract every sample.

    Returns:
        One result tuple per target, each shaped:
        (pdb, metrics, ipsae) normally, plus all_structures when is_final.
    """
    # Thin wrapper over the folding router. Seed counts and backend choice
    # (af3 / protenix, set independently from the recycle stage) live there.
    return folding.fold_against_targets(
        binder_sequence=trajectory_sequence,
        target_sequences=target_sequences,
        target_names=target_names,
        target_chain=target_chain,
        binder_chain=binder_chain,
        output_dir=structures_directory,
        design_name=design_name,
        run_settings=run_settings,
        is_final=is_final,
    )


def run_structure_scoring(
    external_pdb: str,
    confidence_source_metrics: dict,
    confidence_source_ipsae: Union[dict, float, None],
    all_structures: list,
    design_structures_dir,
    design_name: str,
    run_settings: dict,
    target_settings: dict,
    filter_set: dict,
    binder_chain: str,
    target_chain: str,
    cdr3: np.ndarray,
    trajectory_sequence: str,
    rmsd_reference_pdb: str,
    is_final: bool,
    n_framework_mutations: int,
    framework_mutations: list,
    iglm_ll: float,
) -> Tuple[dict, dict, bool, str]:
    """Run FastRelax, structure scoring, and filter evaluation after structure prediction.

    Pipeline:
    1) Relax the predicted structure (parallel ensemble if is_final)
    2) Compute clashes, secondary structure content, and interface metrics
    3) Compute confidence metrics (pDockQ2, LIS/LIA)
    4) Compute hydrophobic patch and hotspot proximity metrics
    5) If is_final: score with ProteinMPNN
    6) Aggregate all metrics and evaluate against the provided filter set

    Args:
        external_pdb: Path to predicted complex PDB.
        confidence_source_metrics: Metrics dict from structure prediction.
        confidence_source_ipsae: ipSAE results from structure prediction. A dict
            with "ipsae", "pdockq2" and "lis" from AF3, or a bare float.
        all_structures: List of (pdb, metrics, ipsae) for all seeds (is_final only).
        design_structures_dir: Directory for output structures.
        design_name: Design identifier (may include target suffix).
        run_settings: Run configuration dict.
        target_settings: Target-specific settings dict.
        filter_set: Filter thresholds for evaluation.
        binder_chain: Binder chain ID.
        target_chain: Target chain ID.
        cdr3: CDR3 positions (1-indexed).
        trajectory_sequence: Designed antibody sequence.
        rmsd_reference_pdb: Reference PDB for RMSD alignment (the AF2
            hallucination output; see multi_run_filters).
        is_final: If True, use ensemble relax and scoring.
        n_framework_mutations: Number of framework mutations (precomputed).
        framework_mutations: List of framework mutation strings (precomputed).
        iglm_ll: IgLM log-likelihood (precomputed).

    Returns:
        Tuple of (filter_metrics, filter_results, accepted, relaxed_pdb)
    """

    # ========================== FastRelax ==========================
    if is_final:
        relaxed_paths = pyrosetta_utils.pr_relax_parallel(
            external_pdb,
            str(design_structures_dir),
            design_name,
            run_settings["dalphaball_path"],
            n_relax=run_settings.get("n_relax", 5),
        )
        # "best" takes every metric from the single lowest-energy relax run;
        # averaging would describe no physical structure.
        relax_score_mode = run_settings.get("relax_score_mode", "best")
        (ensemble_interface_scores, best_interface_AA, best_interface_residues,
         best_relaxed_pdb) = pyrosetta_utils.score_interface_ensemble(
            relaxed_paths, binder_chain, target_chain,
            score_mode=relax_score_mode
        )
        external_relaxed_pdb = best_relaxed_pdb

        clash_threshold = run_settings["clash_threshold"]
        num_clashes_trajectory = utils.calculate_clash_score(
            external_pdb, only_ca=True, threshold=clash_threshold
        )
        # Mean across the whole relax ensemble, unlike the interface metrics:
        # clashing is a property of the ensemble, not of one relaxation.
        num_clashes_relaxed = np.mean([
            utils.calculate_clash_score(p, only_ca=True, threshold=clash_threshold)
            for p in relaxed_paths
        ])

        interface_metrics = {
            "interface_scores": ensemble_interface_scores,
            "interface_AA": best_interface_AA,
            "interface_residues": best_interface_residues,
        }
    else:
        external_relaxed_pdb = os.path.join(
            design_structures_dir, design_name + "_relaxed.pdb"
        )
        pyrosetta_utils.pr_relax(external_pdb, external_relaxed_pdb)

        clash_threshold = run_settings["clash_threshold"]
        num_clashes_trajectory = utils.calculate_clash_score(
            external_pdb, only_ca=True, threshold=clash_threshold
        )
        num_clashes_relaxed = utils.calculate_clash_score(
            external_relaxed_pdb, only_ca=True, threshold=clash_threshold
        )

        interface_metric_names = ["interface_scores", "interface_AA", "interface_residues"]
        interface_metrics = {
            k: v
            for k, v in zip(
                interface_metric_names,
                pyrosetta_utils.score_interface(
                    external_relaxed_pdb, target_settings["binder_chain"],
                    target_chain=target_chain,
                ),
            )
        }

    # ========================== Secondary structure content ==========================
    ss_content = utils.calc_ss_percentage(
        external_pdb, run_settings, binder_chain, return_dict=True
    )

    # ========================== Hotspot proximity ==========================
    one_indexed_cdr_positions = np.array(run_settings["cdr_positions"]) + 1

    binder_near_hotspot, cdr3_hotspot_contacts, cdr_hotspot_contacts = (
        compute_hotspot_proximity(
            external_relaxed_pdb=external_relaxed_pdb,
            target_settings=target_settings,
            binder_chain=binder_chain,
            target_chain=target_chain,
            one_indexed_cdr_positions=one_indexed_cdr_positions,
            cdr3=cdr3,
            distance_threshold=run_settings["hotspot_distance_threshold"],
            contact_distance=run_settings["residue_contact_distance"],
            min_hotspot_contacts=run_settings["min_cdr_hotspot_contacts"],
        )
    )

    # ========================== Interface CDR % ==========================
    percent_interface_cdr = utils.interface_cdrs(
        interface_metrics["interface_residues"],
        run_settings["cdr_positions"],
    )

    # ========================== pDockQ2, LIS/LIA ==========================
    pdockq_metrics, lis_metrics, pDockQ2_out = compute_pdockq_and_lis(
        ipsae=confidence_source_ipsae
    )

    # ========================== Aggregate Confidence Metrics ==========================
    # ipsae arrives as a dict from AF3; unwrap to the scalar the filters compare against.
    ipsae_scalar = (
        confidence_source_ipsae["ipsae"]
        if isinstance(confidence_source_ipsae, dict)
        else confidence_source_ipsae
    )
    confidence_metrics = {
        "plddt": _as_float(confidence_source_metrics["plddt"]),
        "ptm": _as_float(confidence_source_metrics["ptm"]),
        "i_ptm": _as_float(confidence_source_metrics["iptm"]),
        "pae": _as_float(confidence_source_metrics["pae"]),
        "aggregate_score": _as_float(confidence_source_metrics["aggregate_score"]),
        "ipsae": ipsae_scalar,
    }

    # ========================== SAP score ==========================
    sap_score, cdr_sap, _, hydrophobic_patches_binder = pyrosetta_utils.get_sap_score(
        external_relaxed_pdb,
        binder_chain=binder_chain,
        only_binder=True,
        limit_sasa=run_settings["sap_limit_sasa"],
        patch_radius=run_settings["sap_patch_radius"],
        avg_sasa_patch_thr=run_settings["sap_avg_sasa_patch_thr"],
        cdrs=run_settings["cdr_positions"],
    )

    # ========================== RMSD ==========================
    # unaligned_rmsd re-reads from disk, so persist the alignment to a temp
    # copy, never onto the hallucination artifact.
    aligned_reference = None
    try:
        fd, aligned_reference = tempfile.mkstemp(suffix="_aligned.pdb")
        os.close(fd)
        pyrosetta_utils.align_pdbs(
            external_pdb,
            rmsd_reference_pdb,
            target_chain,
            target_chain,
            output_pdb=aligned_reference,
        )
        binder_rmsd = pyrosetta_utils.unaligned_rmsd(
            external_pdb, aligned_reference, binder_chain, binder_chain
        )
    except Exception as e:
        # None, not a sentinel: a sentinel would be indistinguishable from a
        # real measurement and would skew the other target's aggregate.
        log.warning(
            f"Could not compute binder RMSD ({type(e).__name__}: {e})\n"
            f"  external: {external_pdb}\n"
            f"  reference:  {rmsd_reference_pdb}\n"
            f"  sc_rmsd recorded as None (unmeasured); the design will fail the "
            f"sc_rmsd filter."
        )
        binder_rmsd = None
    finally:
        if aligned_reference and os.path.exists(aligned_reference):
            os.remove(aligned_reference)

    # ========================== Sequence Scoring (final only) ==========================
    mpnn_score_mean, mpnn_score_std = None, None
    if is_final and all_structures:
        all_pdb_paths = [s[0] for s in all_structures]
        try:
            mpnn_score_mean, mpnn_score_std = score_sequence_mpnn(
                all_pdb_paths, trajectory_sequence, target_chain, binder_chain
            )
            log.info(f"ProteinMPNN score ({design_name}): {mpnn_score_mean:.4f} +/- {mpnn_score_std:.4f}")
        except Exception as e:
            # Do not swallow this. mpnn_score_mean would stay None, and
            # evaluate_filters turns None into a failed filter -- so a dead
            # checkpoint, a missing import or a GPU OOM would be recorded as a
            # design that merely scored badly. Tool failure, not a design
            # property.
            raise RuntimeError(
                f"ProteinMPNN scoring failed for {design_name} "
                f"({type(e).__name__}: {e}). This is a tool failure, not a "
                f"weak design."
            ) from e

    # ========================== Aggregate Filter Metrics ==========================
    target_filter_metrics = build_filter_metrics(
        confidence_metrics,
        interface_metrics,
        hydrophobic_patches_binder,
        sap_score,
        cdr_sap,
        cdr3_hotspot_contacts,
        cdr_hotspot_contacts,
        pdockq_metrics,
        lis_metrics,
        percent_interface_cdr,
        ss_content,
        binder_rmsd,
        n_framework_mutations,
        framework_mutations,
        num_clashes_trajectory,
        num_clashes_relaxed,
        binder_near_hotspot,
        iglm_ll,
        mpnn_score_mean=mpnn_score_mean,
        mpnn_score_std=mpnn_score_std,
    )

    # ========================== Evaluate Filter Set ==========================
    target_accepted, target_filter_results = evaluate_filters(filter_set, target_filter_metrics)

    return target_filter_metrics, target_filter_results, target_accepted, external_relaxed_pdb


def score_sequence_mpnn(pdb_paths, sequence, target_chain, binder_chain):
    """
    Score a candidate sequence with ProteinMPNN against multiple structures.

    Uses the colabdesign ProteinMPNN model to compute the negative log-likelihood
    of the candidate sequence given each structure backbone.

    Args:
        pdb_paths (list[str]): List of PDB file paths (e.g. one per final-filter folding seed).
        sequence (str): Candidate amino acid sequence to score.
        target_chain (str): Target chain ID.
        binder_chain (str): Binder chain ID.

    Returns:
        tuple: (mean_score, std_score) across all structures.
    """
    from colabdesign.mpnn import mk_mpnn_model

    scores = []
    for pdb_path in pdb_paths:
        try:
            mpnn_model = mk_mpnn_model(weights="original")
            mpnn_model.prep_inputs(
                pdb_filename=pdb_path,
                chain=f"{target_chain},{binder_chain}",
                fix_pos=f"{target_chain}",
            )
            result = mpnn_model.score(seq=sequence)
            scores.append(float(result["score"]))
        except Exception as e:
            log.warning(f"MPNN scoring failed for {pdb_path}: {e}")
            continue

    if not scores:
        raise RuntimeError("All ProteinMPNN scoring attempts failed")

    return float(np.mean(scores)), float(np.std(scores))


def compute_hotspot_proximity(
    external_relaxed_pdb: str,
    target_settings: dict,
    binder_chain: str,
    target_chain: str,
    one_indexed_cdr_positions: Sequence[int],
    cdr3: Sequence[int],
    distance_threshold: float = 5.3,
    contact_distance: float = 6.0,
    min_hotspot_contacts: int = 3,
) -> Tuple[bool, int, int]:
    """
    Compute CDR contacts with target hotspot residues (5.3Å threshold, ≥3 contacts required).

    Args:
        external_relaxed_pdb: Relaxed complex PDB path
        target_settings: Config with hotspot definitions
        binder_chain: Binder chain ID
        target_chain: Target chain ID
        one_indexed_cdr_positions: All CDR positions (1-indexed)
        cdr3: CDR3 positions (1-indexed)

    Returns:
        Tuple[bool, int, int]: (near_hotspot, cdr3_contacts, cdr_contacts)
    """
    # Default values when no hotspot specification is provided
    binder_near_hotspot, cdr3_hotspot_contacts, cdr_hotspot_contacts = True, 1, 1

    if len(target_settings["target_hotspots"]) > 0:
        target_hotspots = (
            np.array(
                utils.idx_from_ranges(
                    target_settings["target_hotspots"],
                    target_chain,
                )
            )
            + 1
        )

        hotspot_region = pyrosetta_utils.find_nearby_residues_from_pdb(
            external_relaxed_pdb,
            target_hotspots,
            distance_threshold=distance_threshold,
            chain=target_chain,
        )

        contacts = pyrosetta_utils.get_residue_contacts(
            external_relaxed_pdb, target_chain, binder_chain, contact_distance
        )
        contacts_per_chain = np.array(list(contacts.keys()))

        # No contact pair gives a shape-(0,) array, which would IndexError. A
        # dissociated prediction is routine: report "not near the hotspot".
        if contacts_per_chain.ndim != 2 or contacts_per_chain.shape[0] == 0:
            return False, 0, 0

        binder_near_hotspot, cdr3_hotspot_contacts, cdr_hotspot_contacts = (
            is_binder_near_hotspot(
                contacts_per_chain[:, 0],
                hotspot_region,
                contacts_per_chain[:, 1],
                one_indexed_cdr_positions,
                cdr3,
                min_hotspot_contacts=min_hotspot_contacts,
            )
        )

    return binder_near_hotspot, cdr3_hotspot_contacts, cdr_hotspot_contacts


def compute_pdockq_and_lis(ipsae: dict = None) -> Tuple[dict, dict, dict]:
    """
    Report docking quality metrics: pDockQ2, LIS, LIA (0-1 scale, higher=better).

    Both values come from the ipSAE CLI output; there is no local recomputation.
    The former pDockQ.pDockQ2() / calculate_lis() fallback read row 0 of
    `pmidockq`, which is the TARGET chain rather than the binder, so it filed
    the wrong chain's score under the binder's name. A wrong number is worse
    than no number: none of these are gated on by any filter config -- they are
    reported columns only -- so when ipSAE is unavailable they are recorded as
    None (unmeasured) instead of being guessed at.

    Args:
        ipsae: ipsae dict carrying pdockq2 and lis, or None/scalar when the
            ipSAE CLI failed (af3.extract_structure_and_scores returns None).

    Returns:
        Tuple[dict, dict, dict]: (pdockq_metrics, lis_metrics, pDockQ2_out).
        pDockQ2_out is always None now and has no consumers left; the third
        element is kept only so existing unpacking keeps working.
    """
    has_ipsae = isinstance(ipsae, dict)
    pdockq2 = ipsae.get("pdockq2") if has_ipsae else None
    lis = ipsae.get("lis") if has_ipsae else None

    if pdockq2 is None or lis is None:
        log.info(
            "Warning: ipSAE reported no pDockQ2/LIS; recording them as None "
            "(unmeasured). No filter gates on these metrics."
        )

    return (
        {"pDockQ2": float(pdockq2) if pdockq2 is not None else None},
        {"lis": float(lis) if lis is not None else None, "lia": None},
        None,
    )


def get_iglm_ll(
    sequence,
    chain_token="[HEAVY]",
    species_token="[CAMEL]",
    vh_first=True,
    vh_len=0,
    vl_len=0,
):
    """
    Calculate antibody sequence log-likelihood using IgLM language model.

    Attribution: Shuai, R. W., Ruffolo, J. A., & Gray, J. J. (2023). IgLM: Infilling
    language modeling for antibody sequence design. Cell Systems, 14(11), 979-989.
    License: JHU Academic Software License (non-commercial use). Commercial inquiries: awichma2@jhu.edu
    Source: https://github.com/Graylab/IgLM

    Args:
        sequence: Antibody amino acid sequence
        chain_token: "[HEAVY]" or "[LIGHT]"
        species_token: "[HUMAN]", "[CAMEL]", etc.
        vh_first: Heavy chain first in scFv sequence
        vh_len: Heavy chain length (0 for nanobodies)
        vl_len: Light chain length (0 for nanobodies)

    Returns:
        float: Log-likelihood score (higher = more natural)
    """

    # Initialize the model
    model = IgLM()

    # Compute the log likelihood, depending on nanobody or scfv
    if vl_len and vh_len:
        if vh_first:
            log_likelihood_h = model.log_likelihood(
                sequence[:vh_len], "[HEAVY]", species_token
            )
            log_likelihood_l = model.log_likelihood(
                sequence[-vl_len:], "[LIGHT]", species_token
            )
        else:
            log_likelihood_l = model.log_likelihood(
                sequence[:vl_len], "[LIGHT]", species_token
            )
            log_likelihood_h = model.log_likelihood(
                sequence[-vh_len:], "[HEAVY]", species_token
            )
        log_likelihood = log_likelihood_h + log_likelihood_l
    else:
        log_likelihood = model.log_likelihood(sequence, chain_token, species_token)

    return log_likelihood


