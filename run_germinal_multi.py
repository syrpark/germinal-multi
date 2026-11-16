"""Run Germinal multi-target antibody design.

Pipeline, both modes:
    hallucination -> early check -> initial filters
      -> recycling loop | AbMPNN redesign
      -> final filters per candidate -> cleanup

Two config keys route it:
    logits_only   logits-only hallucination, else the full soft+greedy schedule
    recycling     run the recycling loop instead of a single AbMPNN redesign

Multi-target only: every config supplies >= 2 antigen states.
"""

import os
import signal
import time

from omegaconf import DictConfig
import hydra
import pyrosetta as pr
import numpy as np

from germinal.design.design import germinal_design, STAGES_FULL, STAGES_LOGITS
from germinal.filters import filter_utils, folding
from germinal.filters.early_check import early_check
from germinal.design import redesign as redesign_mod
from germinal.design.redesign import redesign
from germinal.design.recycle import recycle
from germinal.utils import utils, config
from germinal.utils.io import Trajectory, config_fingerprints

from germinal.utils.log import configure as configure_logging, get_logger

# Not __name__: this file runs as a script, so __name__ is "__main__", which is
# not under the "germinal" logger that configure() attaches the handler to --
# every record would be dropped.
log = get_logger("germinal.runner")


def _setup(cfg):
    """Process config, validate it, start PyRosetta, return the run handles."""
    processed_cfg = config.process_config(cfg)
    prelim_run_settings, target_settings, initial_filters, final_filters = (
        processed_cfg.values()
    )
    # Before anything is created or any GPU time is spent; unset these surface
    # inside the container, well into the first trajectory.
    config.validate_af3_paths(prelim_run_settings)
    config.validate_redesign_submodule()
    config.validate_dalphaball(prelim_run_settings)
    config.validate_filter_metric_names(initial_filters, final_filters)
    io, run_settings = config.initialize_germinal_run(
        prelim_run_settings, target_settings
    )

    recycling = bool(run_settings.get("recycling", False))

    complex_paths = run_settings.get("complex_path", None)
    if complex_paths is not None:
        assert isinstance(complex_paths, list), (
            f"complex_path must be a list of one PDB per antigen state, got "
            f"{type(complex_paths).__name__}"
        )
        run_settings["starting_pdb_complex"] = complex_paths
        # Take the binder from the real complex, not the auto-generated one.
        binder_chain = target_settings["binder_chain"]
        binder_seqs = [
            utils.get_sequence_from_pdb(p)[binder_chain] for p in complex_paths
        ]
        assert all(s == binder_seqs[0] for s in binder_seqs), (
            "Binder sequences differ across complex_path PDBs:\n"
            + "\n".join(
                f"  {p}: {s[:40]}..." for p, s in zip(complex_paths, binder_seqs)
            )
        )
        run_settings["starting_binder_seq"] = binder_seqs[0]

    config.validate_binder_length(run_settings, target_settings)
    config.validate_num_targets(run_settings, target_settings)
    config.validate_target_numbering(target_settings)
    if recycling and run_settings.get("num_recycles", 0) <= 0:
        raise ValueError(
            "recycling is enabled but num_recycles is "
            f"{run_settings.get('num_recycles', 0)}. The loop would score "
            "nothing and every trajectory would be dropped at the ipSAE gate. "
            "Set num_recycles > 0 or recycling: false."
        )
    if not recycling:
        # Only meaningful without the loop: the ratchet raises the cutoff from
        # 0.0 later, so a recycling config legitimately starts at 0.0.
        config.validate_single_path_redesign_cutoff(run_settings)

    num_targets = target_settings["num_targets"]
    assert num_targets >= 2, (
        f"target_pdb_path has {num_targets} entry/entries; Germinal is "
        f"multi-target only and needs at least 2 antigen states, each with its "
        f"own target_name and target_hotspots."
    )

    io.save_run_config(
        run_settings, target_settings, initial_filters, final_filters
    )
    pr.init(
        f"-ignore_unrecognized_res -ignore_zero_occupancy -mute all "
        f"-holes:dalphaball {run_settings['dalphaball_path']} "
        f"-corrections::beta_nov16 true -relax:default_repeats 1"
    )
    log.info(
        f"============================\n"
        f"Experiment name: {run_settings['experiment_name']}\n"
        f"============================"
    )
    log.info(f"Processed config: {target_settings}\n{initial_filters}\n{final_filters}")
    log.info(
        f"Hallucination: {'logits-only' if run_settings.get('logits_only') else 'full'}"
        f" | recycling: {recycling}"
    )

    target_sequences = [
        utils.get_sequence_from_pdb(p)[target_settings["target_chain"]]
        for p in target_settings["target_pdb_path"]
    ]

    return (
        io,
        run_settings,
        target_settings,
        initial_filters,
        final_filters,
        target_sequences,
    )






def _remove_all(paths):
    for p in paths:
        if os.path.exists(p):
            os.remove(p)


#: Set by _request_stop; read at phase boundaries.
_STOP_REQUESTED = False


def _tool_failure(count, limit, reason):
    """Count one consecutive tool failure and abort if they do not stop.

    A failed external tool is recorded as a failed design, which is right for
    one design and wrong for a run: if AbMPNN cannot start at all, every
    design fails for the same reason and the run reports a 0% acceptance rate
    instead of an error. The caller resets the count as soon as any design
    produces candidates, so this only fires on an unbroken streak.

    Raises:
        RuntimeError: once `limit` designs in a row have failed this way.
    """
    count += 1
    if limit and count >= limit:
        raise RuntimeError(
            f"{count} designs in a row failed because an external tool "
            f"failed (most recently: {reason}). This is an environment or "
            f"configuration fault, not a property of the designs -- nothing "
            f"can be accepted until it is fixed. Check the ProteinMPNN "
            f"submodule, params/abmpnn.pt, GPU memory, and "
            f"`nvidia-smi --query-gpu=compute_mode --format=csv` (see "
            f"docs/af3-setup.md). Set max_consecutive_tool_failures to 0 to "
            f"disable this check."
        )
    log.warning(
        f"Tool failure {count}/{limit} in a row ({reason}); aborting the run "
        f"if this reaches {limit}."
    )
    return count


def _request_stop(signum, frame):
    """Ask the loop to stop at its next boundary.

    Sets a flag and nothing else, by necessity rather than style: it can fire
    while a frame holds a FileLock, and re-taking that lock here would
    deadlock unkillably. Printing (stdio lock) and raising (aborts a fold
    mid-write) are out for the same reason. Nothing needs flushing: the
    in-flight records were written at the boundaries they describe.
    """
    global _STOP_REQUESTED
    _STOP_REQUESTED = True




def _slot_from_record(record, candidates):
    """Rebuild a batch slot from an in-flight record.

    `resumed` marks the row, because design_time then covers only this
    process's share of the work.
    """
    trajectory = Trajectory.from_snapshot(record["trajectory"])
    trajectory.update_other_metrics({"resumed": True})
    return {
        "trajectory": trajectory,
        "design_name": record["design_name"],
        "seed": record.get("seed"),
        "sequence": record["sequence"],
        "hallucination_pdbs": list(record.get("hallucination_pdbs") or []),
        "target_len": record.get("target_len"),
        "elapsed": float(record.get("elapsed_s") or 0.0),
        "any_accepted": False,
        "candidates": list(candidates),
        "filter_metrics": {},
        "filter_results": {},
        "final_struct": record.get("final_struct"),
    }




def _journal(io, slot, phase, fingerprints):
    """Record how far this design got, so a later run can adopt it.

    Called at each phase boundary, not at shutdown: a killed process gets no
    chance to flush, so the record has to already be true.
    """
    target_fp, filter_fp = fingerprints
    io.record_inflight(slot["design_name"], {
        "phase": phase,
        "seed": slot.get("seed"),
        "sequence": slot["sequence"],
        "hallucination_pdbs": slot["hallucination_pdbs"],
        "target_len": slot.get("target_len"),
        "elapsed_s": slot["elapsed"],
        "target_fp": target_fp,
        "filter_fp": filter_fp,
        "trajectory": slot["trajectory"].snapshot(),
        "final_struct": slot.get("final_struct"),
        "candidates": slot.get("candidates") or [],
    })


#: What each in-flight phase is still owed, for the end-of-run report.
_PHASE_OWED = {
    "hallucinated": "folded and filtered",
    "filtered": "redesigned and scored",
    "candidates_ready": "scored",
}


def _unfinished_work_note(io) -> str:
    """What this run is leaving behind, and what finishes it.

    The counterpart to the adoption report at startup. A run that ends at a
    wall-clock limit routinely parks designs that passed hallucination but
    never filled a fold batch: real work, already paid for, invisible in every
    CSV because a record is deliberately not a row. Saying nothing here reads
    as "nothing left to do".
    """
    records = io.read_inflight()
    if not records:
        return "No unfinished work left behind; inflight/ is empty."

    by_phase = {}
    for record in records:
        by_phase.setdefault(record.get("phase", "<unknown>"), []).append(record)

    lines = [
        f"UNFINISHED WORK: {len(records)} design(s) left in inflight/, not "
        f"counted in any CSV above.",
    ]
    for phase, group in sorted(by_phase.items(), key=lambda kv: -len(kv[1])):
        owed = _PHASE_OWED.get(phase, "finished")
        lines.append(f"  {len(group):>4} at '{phase}' -- still to be {owed}: "
                     f"{', '.join(sorted(r['design_name'] for r in group)[:4])}"
                     + (" ..." if len(group) > 4 else ""))
    lines.append(
        "  These are NOT failures. Relaunching the same experiment_name "
        "adopts them and finishes where this run stopped; they cost no "
        "attempt and no trajectory. Nothing needs redoing by hand."
    )
    # The common cause, and the one knob that fixes it, named here because the
    # run that hit it is the run whose log someone reads.
    parked = len(by_phase.get("hallucinated", []))
    if parked:
        lines.append(
            f"  {parked} of them never reached a fold at all: a batch folds "
            f"once max_hallucination_passing designs pass hallucination (or "
            f"max_hallucination_attempts are spent). A target with a low pass "
            f"rate can park designs for a whole run without folding any -- "
            f"lower max_hallucination_passing if that is what happened here."
        )
    return "\n".join(lines)










@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Before anything logs: level and format come from config, so a run can be
    # turned down to warnings or given timestamps without touching code.
    configure_logging(
        level=cfg.get("log_level", "INFO"),
        timestamps=cfg.get("log_timestamps", False),
    )
    (
        io,
        run_settings,
        target_settings,
        initial_filters,
        final_filters,
        target_sequences,
    ) = _setup(cfg)

    # SIGUSR1 is what a scheduler sends ahead of the wall; SIGTERM is what it
    # sends at it. Both mean the same thing here: stop at the next boundary.
    signal.signal(signal.SIGUSR1, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    # Stamped into every in-flight record; a later run adopts only matching work.
    fingerprints = config_fingerprints(
        run_settings, target_settings, initial_filters
    )

    num_targets = target_settings["num_targets"]
    logits_only = bool(run_settings.get("logits_only", False))
    recycling = bool(run_settings.get("recycling", False))
    stages = STAGES_LOGITS if logits_only else STAGES_FULL
    cdr_threshold = run_settings["early_check_min_cdr_fraction"]
    ipsae_threshold = run_settings.get("recycle_ipsae_threshold", 0.5)
    # Filter-model and recycling-model predictions both land here; accepted/
    # and filter_failed/ get their own copies via save_trajectory.
    structures_directory = str(io.layout.redesign_candidates / "structures")

    if run_settings["pregenerate_seeds"]:
        np.random.seed(1)
        run_seeds = [
            np.random.randint(2**32 - 1)
            for _ in range(run_settings["max_trajectories"])
        ]
    else:
        init_seed = int(time.time_ns()) % (2**32 - 1)
        log.info(f"Initial seed: {init_seed}")
        np.random.seed(init_seed)


    start_time = time.time()
    # Three distinct populations: designs lost before filtering, designs a
    # filter rejected, and final-stage CANDIDATES rejected (not a design count).
    failed_design = 0
    num_accepted = 0
    num_failed = 0
    num_candidates_rejected = 0
    # `i` is undefined when max_trajectories is 0, so the summary cannot use it.
    n_attempted = 0
    # Consecutive designs lost to a failed external tool. Reset by any design
    # that produces candidates; see _tool_failure.
    consecutive_tool_failures = 0
    tool_failure_limit = int(run_settings.get("max_consecutive_tool_failures", 10))

    # Four phases: A hallucinate a batch, B fold the survivors in one call, C
    # redesign/recycle per design, D fold every candidate in one call.
    io.clear_batch_fold_dir()

    # Batch size is however many designs pass hallucination, bounded by
    # max_hallucination_passing and max_hallucination_attempts.

    def _design_time(slot):
        """Active seconds spent on this design, summed across phases.

        Wall clock since the design started would include every other design
        in the batch, because the phases interleave them.
        """
        return utils.get_clean_time(slot["elapsed"], 0.0)

    # Unfinished work from a killed run, taken over rather than redone. Done
    # once: draining inside the batch loop would re-adopt forever.
    inflight_records = io.read_inflight()
    to_refold, to_rescore = [], []
    for record in io.find_adoptable(*fingerprints):
        if record["phase"] == "candidates_ready":
            to_rescore.append(record)
        else:
            # A filtered design re-enters at the fold too: skipping ahead
            # would buy one shared fold for a second code path.
            to_refold.append(record)
    adopted_names = {
        r["design_name"] for r in (*to_refold, *to_rescore)
    }
    # Said either way: silence here cannot distinguish "nothing to adopt" from
    # "adoption did not run", and the difference costs a forensic comparison of
    # record timestamps against the job's start.
    if adopted_names:
        log.info(f"\nAdopting {len(to_refold)} design(s) to refold and "
              f"{len(to_rescore)} with candidates left to score.\n")
    elif inflight_records:
        log.info(f"\n{len(inflight_records)} in-flight record(s) found, none "
                 f"adoptable (reasons above); starting clean.\n")
    else:
        log.info("\nNo unfinished work to adopt; starting clean.\n")

    i = 0
    while i < run_settings["max_trajectories"]:
        terminate, reason = io.check_termination_conditions(
            run_settings, n_trajectories=i
        )
        if terminate:
            log.info(reason)
            break

        # ==================== PHASE A: hallucinate a batch ====================
        # `slots` holds the passers; its length is the batch size for B-D.
        # Adopted designs part-fill it: they occupy a fold slot but cost no
        # attempt and no trajectory, both charged to the run that made them.
        slots = [
            _slot_from_record(r, candidates=[])
            for r in utils.take(to_refold, run_settings["max_hallucination_passing"])
        ]
        rescore_batch = [
            _slot_from_record(r, candidates=r["missing_candidates"])
            for r in utils.take(to_rescore, run_settings["max_hallucination_passing"])
        ]
        batch_attempts = 0
        while i < run_settings["max_trajectories"]:
            if len(slots) >= run_settings["max_hallucination_passing"]:
                log.info(f"\nBatch full: {len(slots)} passer(s) from "
                      f"{batch_attempts} attempt(s). Folding.")
                break
            if batch_attempts >= run_settings["max_hallucination_attempts"]:
                log.info(f"\nBatch attempt limit: {batch_attempts} attempt(s) for "
                      f"{len(slots)} passer(s). Folding.")
                break

            terminate, reason = io.check_termination_conditions(
                run_settings, n_trajectories=i
            )
            if terminate:
                log.info(reason)
                break
            if _STOP_REQUESTED:
                break

            traj_index = i
            i += 1
            n_attempted = traj_index + 1
            trajectory_start_time = time.time()
            seed = (
                run_seeds[traj_index]
                if run_settings["pregenerate_seeds"]
                else int(np.random.randint(0, 999999, size=1, dtype=int)[0])
            )

            target_name = "_".join(target_settings["target_name"])
            design_name = f"{target_name}_{run_settings['type']}_s{seed}"

            if io.check_existing_seed(seed):
                log.info(f"Trajectory {traj_index} with seed {seed} already "
                      f"exists. Trying new seed.")
                continue

            # An adopted design has no row, so check_existing_seed cannot see
            # it; a collision would abort the batch on duplicate fold names.
            if design_name in adopted_names:
                log.info(f"Trajectory {traj_index} with seed {seed} is being "
                      f"adopted from an earlier run. Trying new seed.")
                continue

            # Counted here, not at `i`: a seed skipped above cost nothing, and
            # counting it would close batches early on resume.
            batch_attempts += 1

            trajectory = Trajectory(
                design_name,
                run_settings["experiment_name"],
                ",".join(map(str, run_settings["cdr_lengths"])),
                target_settings["target_hotspots"],
            )
            trajectory.set_save_location("trajectories")
            log.info(f"\nStarting trajectory {traj_index + 1}: {design_name}")


            # ==================== 1. Hallucination ====================
            design_output = germinal_design(
                design_name,
                run_settings,
                target_settings,
                io,
                seed,
                stages=stages,
                model_pdb_path=str(
                    io.layout.trajectories / f"structures/{design_name}.pdb"
                ),
            )

            hallucination_failures = getattr(
                design_output, "_hallucination_failures", None
            )
            if hallucination_failures:
                io.update_failures({
                    utils.gate_counter("hallucination", k[len("hallucination_"):]
                                  if k.startswith("hallucination_") else k): v
                    for k, v in hallucination_failures.items()
                })

            # Never rebound: the sc_rmsd reference must not move when a recycle
            # wins. Bound before the terminate check, which also needs it.
            hallucination_pdbs = io.per_target_pdbs(design_name, num_targets)

            trajectory_metrics_last = utils.copy_dict(design_output.aux["log"])
            if trajectory_metrics_last.get("terminate", "") != "":
                trajectory_time = utils.get_clean_time(time.time(), trajectory_start_time)
                log.info(f"Trajectory took: {trajectory_time}\n")
                trajectory.update_other_metrics({
                    "status": "hallucination_failed",
                    "failure_stage": "hallucination",
                    "terminate_reason": trajectory_metrics_last.get("terminate", ""),
                    "design_time": trajectory_time,
                })
                io.save_to_all_trajectories(trajectory)
                # design.py writes the PDBs before it sets `terminate`, and the
                # row's final_structure_path is empty, so nothing would free them.
                _remove_all(hallucination_pdbs)
                failed_design += 1
                continue

            # hallucination.* -- ColabDesign's own confidence, distinct from the
            # same quantities measured later by AF3 (final.min_iptm etc).
            trajectory.update_trajectory_metrics(
                utils.stage_keys(utils.copy_dict(design_output._tmp["best"]["log"]), "hallucination")
            )
            trajectory_sequence = design_output._tmp["best"]["seq"]
            target_len = design_output._target_len

            trajectory.update_other_metrics({
                "trajectory_pdb_af": hallucination_pdbs,
                "trajectory_sequence": trajectory_sequence,
                "target_len": target_len,
            })
            trajectory.set_save_location("trajectories")
            log.info(f"Hallucination took: "
                  f"{utils.get_clean_time(time.time(), trajectory_start_time)}\n")

            # ==================== 2. Early check ====================
            passed_early, early_info = early_check(
                hallucination_pdbs,
                run_settings,
                target_settings,
                min_cdr_fraction=cdr_threshold,
            )
            log.info(f"  Early check result: passed={passed_early}")
            for t_idx, t_info in enumerate(early_info.get("per_target", [])):
                log.info(f"    Target {t_idx}: {t_info}")
            if not passed_early:
                early_crashed = early_info.get("early_check_crashed", False)
                log.info(f"  Early check FAILED: "
                      f"{early_info.get('reject_reason', 'unknown')}.")
                trajectory.update_other_metrics({
                    "status": "early_check_failed",
                    "failure_stage": "early_check",
                    "reject_reason": early_info.get("reject_reason", "unknown"),
                    # Marks a row whose gate values are incomplete because a tool
                    # raised; exclude these from any rate computed off the CSVs.
                    "early_check_crashed": early_crashed,
                    "design_time": utils.get_clean_time(
                        time.time(), trajectory_start_time
                    ),
                })
                io.save_to_all_trajectories(trajectory)
                io.update_failures(
                    utils.gate_counter(
                        "early_check", "crashed" if early_crashed else "rejected"
                    )
                )
                # Delete only when the gate actually judged the design: a crash
                # is an environment fault and says nothing about design quality.
                if early_crashed:
                    log.info("  Keeping hallucination PDBs: early check CRASHED, so "
                          "the design was never judged. Re-score after fixing the "
                          "tool rather than re-hallucinating.")
                else:
                    _remove_all(hallucination_pdbs)
                failed_design += 1
                utils.clear_memory(clear_jax=False)
                continue

            slots.append({
                "trajectory": trajectory,
                "design_name": design_name,
                "seed": int(seed),
                "sequence": trajectory_sequence,
                "hallucination_pdbs": hallucination_pdbs,
                "target_len": target_len,
                "elapsed": time.time() - trajectory_start_time,
                "any_accepted": False,
                "candidates": [],
                "filter_metrics": {},
                "filter_results": {},
                "final_struct": None,
            })
            # First statement after the append: the unrecorded window is the gap.
            _journal(io, slots[-1], "hallucinated", fingerprints)

        # Stop before folding, not after: every slot here has a record, so the
        # next run adopts this batch instead of this one racing the wall.
        if _STOP_REQUESTED:
            log.info(f"\nStop requested. Leaving {len(slots) + len(rescore_batch)} "
                  f"design(s) for the next run to adopt.")
            break

        if not slots and not rescore_batch:
            continue
        utils.clear_memory(clear_jax=False)

        # ============ PHASE B: ONE fold of every survivor, then score ============
        batch_folds = filter_utils.fold_designs_for_filter(
            designs=[(s["design_name"], s["sequence"]) for s in slots],
            run_settings=run_settings,
            target_settings=target_settings,
            io=io,
            is_final=False,
        )

        passing = []
        for slot in slots:
            trajectory = slot["trajectory"]
            design_name = slot["design_name"]
            trajectory_sequence = slot["sequence"]
            hallucination_pdbs = slot["hallucination_pdbs"]
            phase_start = time.time()

            # ==================== 3. Initial filters ====================
            log.info("Running initial cofolding filters")
            filter_metrics, filter_results, pass_initial_filters, final_struct = (
                filter_utils.multi_run_filters(
                    trajectory,
                    run_settings,
                    target_settings,
                    initial_filters,
                    io,
                    trajectory_sequence,
                    hallucination_pdbs,
                    is_final=False,
                    prediction_results=batch_folds.get(design_name),
                )
            )
            utils.clear_memory(clear_jax=False)

            # Charged before the failure branch, which continues: otherwise a
            # rejected row excludes the phase it died in.
            slot["elapsed"] += time.time() - phase_start

            # Every design, not just failures: otherwise pooled initial.* is
            # conditioned on failure. Safe because final.* is a separate namespace.
            raw_filter_data = {**(filter_metrics or {}), **(filter_results or {})}
            complete_filter_data = utils.stage_keys(raw_filter_data, "initial")
            trajectory.update_filtering_metrics(complete_filter_data)

            if not pass_initial_filters:
                log.info("Trajectory not passing initial cofolding filters, skipping")
                trajectory.set_final_struct(final_struct)
                trajectory.update_other_metrics({
                    "status": "initial_filter_failed",
                    "design_time": _design_time(slot),
                    # Unprefixed: resolve_failure_stage reads a bare
                    # "failure_stage" key, which _stage_keys renames away.
                    "failure_stage": utils.resolve_failure_stage(
                        raw_filter_data, "initial_filter"
                    ),
                })
                trajectory.set_save_location("trajectories")
                io.save_trajectory(trajectory)
                # Stage-prefixed so failure_counts.csv can tell the two filter
                # stages apart; they previously shared one set of column names.
                failed_filters = utils.stage_keys(
                    filter_utils.select_failed_filters(filter_results, filter_metrics),
                    "initial",
                )
                if failed_filters:
                    io.update_failures(failed_filters)
                io.clear_inflight(design_name)
                _remove_all(hallucination_pdbs)
                num_failed += 1
                continue

            slot["filter_metrics"] = filter_metrics
            slot["filter_results"] = filter_results
            slot["final_struct"] = final_struct
            if pass_initial_filters:
                passing.append(slot)
                _journal(io, slot, "filtered", fingerprints)

        # ============ PHASE C: redesign / recycling for everything that passed ====
        with_candidates = []
        for slot in passing:
            # The longest stretch in the batch, so the check belongs here too:
            # a design still at `filtered` is adopted and re-enters at the fold.
            if _STOP_REQUESTED:
                log.info("\nStop requested. Leaving the remaining designs for the "
                      "next run to redesign.")
                break
            trajectory = slot["trajectory"]
            design_name = slot["design_name"]
            trajectory_sequence = slot["sequence"]
            hallucination_pdbs = slot["hallucination_pdbs"]
            filter_metrics = slot["filter_metrics"]
            filter_results = slot["filter_results"]
            final_struct = slot["final_struct"]
            phase_start = time.time()
            candidates = None

            # ==================== 4. Candidates ====================
            if recycling:
                scored_candidates, recycle_info = recycle(
                    run_settings,
                    target_settings,
                    design_name,
                    hallucination_pdbs,
                    trajectory_sequence,
                    target_sequences,
                    structures_directory,
                )
                trajectory.update_other_metrics({
                    "num_recycles_completed": recycle_info["recycles_completed"],
                    "recycle_truncation_reason": recycle_info["truncation_reason"],
                    "recycle_truncation_category": recycle_info["truncation_category"],
                })

                # Every candidate that cleared the epitope gate and the ipSAE
                # threshold goes forward, best first.
                eligible = [
                    c for c in scored_candidates
                    if c["hotspots_pass"]
                    and c["cdr_frac"] >= cdr_threshold
                    and c["min_ipsae"] >= ipsae_threshold
                ]
                eligible.sort(key=lambda c: c["min_ipsae"], reverse=True)
                trajectory.update_other_metrics({
                    "best_recycle_ipsae": (
                        max((c["min_ipsae"] for c in scored_candidates), default=None)
                    ),
                })

                if not eligible:
                    if recycle_info.get("tool_failure"):
                        consecutive_tool_failures = _tool_failure(
                            consecutive_tool_failures, tool_failure_limit,
                            recycle_info["truncation_reason"],
                        )
                    category = recycle_info["truncation_category"]
                    if not scored_candidates:
                        reason = recycle_info["truncation_reason"] or "no_recycle_scored"
                        category = category or "no_recycle_scored"
                        status = f"recycling_failed_{category}"
                        failure_stage = "recycling_truncated"
                        failure_counter = utils.gate_counter("recycling", category)
                        log.info(f"  Recycling produced no scored design ({reason}), "
                              f"skipping final filters.")
                    else:
                        status = "recycling_ipsae_failed"
                        failure_stage = "recycling_ipsae_gate"
                        failure_counter = utils.gate_counter("recycling", "ipsae_gate")
                        log.info(f"  No candidate cleared the epitope gate and ipSAE "
                              f"threshold ({ipsae_threshold}), skipping final filters.")
                    # The design cleared the initial filters, so keep what they
                    # measured and the structure they produced.
                    trajectory.update_filtering_metrics(
                        utils.stage_keys({**filter_metrics, **filter_results}, "initial")
                    )
                    # Before _design_time: this path continues, so the phase-end
                    # accounting below never runs.
                    slot["elapsed"] += time.time() - phase_start
                    trajectory.set_final_struct(final_struct)
                    trajectory.update_other_metrics({
                        "status": status,
                        "failure_stage": failure_stage,
                        "design_time": _design_time(slot),
                    })
                    trajectory.set_save_location("trajectories")
                    io.save_trajectory(trajectory)
                    io.update_failures(failure_counter)
                    io.clear_inflight(design_name)
                    _remove_all(hallucination_pdbs)
                    num_failed += 1
                    utils.clear_memory(clear_jax=False)
                    continue

                candidates = utils.build_final_candidates(
                    design_name=design_name,
                    trajectory_sequence=trajectory_sequence,
                    abmpnn_sequences=[
                        {
                            "seq": c["seq"],
                            "score": c["abmpnn_score"],
                            "seqid": c["abmpnn_seqid"],
                        }
                        for c in eligible
                    ],
                    include_codesign=False,
                )
                # The hallucinated sequence competes in the loop on its own merits,
                # so label it for what it is rather than as an AbMPNN proposal.
                for candidate, scored in zip(candidates, eligible):
                    if scored["is_baseline"]:
                        candidate["source"] = "codesign"
                    # sc_rmsd reference: this candidate's own recycle refold, so
                    # both sides come from the filter model.
                    candidate["designed_on"] = scored["pdbs"]
            else:
                log.info("\nStarting AbMPNN redesign...\n")
                abmpnn_sequences, abmpnn_success, redesign_reason = redesign(
                    trajectory_pdbs_af=hallucination_pdbs,
                    target_chain=target_settings["target_chain"],
                    binder_chain=target_settings["binder_chain"],
                    run_settings=run_settings,
                    atom_distance_cutoff=run_settings["atom_distance_cutoff"],
                )

                if not abmpnn_success:
                    if redesign_reason in redesign_mod.TOOL_FAILURE_REASONS:
                        log.warning(f"MPNN redesign FAILED ({redesign_reason}) -- tool "
                              f"failure, not a design property. Check the "
                              f"ProteinMPNN submodule, abmpnn.pt and GPU memory if "
                              f"it repeats on every trajectory.")
                        consecutive_tool_failures = _tool_failure(
                            consecutive_tool_failures, tool_failure_limit,
                            redesign_reason,
                        )
                    else:
                        log.info(f"MPNN redesign produced nothing ({redesign_reason}), "
                              f"skipping to next trajectory")
                    # "initial", not "final": redesign produced nothing to
                    # filter, so these are the initial filter's own metrics.
                    complete_filter_data = utils.stage_keys(
                        {**filter_metrics, **filter_results}, "initial"
                    )
                    trajectory.update_filtering_metrics(complete_filter_data)
                    # Before _design_time, same as the recycling path above:
                    # this one continues too.
                    slot["elapsed"] += time.time() - phase_start
                    # save_trajectory copies final_struct, which is unset on this path.
                    trajectory.set_final_struct(final_struct)
                    trajectory.update_other_metrics({
                        "status": "abmpnn_redesign_failed",
                        "design_time": _design_time(slot),
                        "failure_stage": "abmpnn_redesign",
                        "redesign_failure_reason": redesign_reason,
                    })
                    trajectory.set_save_location("trajectories")
                    io.save_trajectory(trajectory)
                    io.update_failures({utils.gate_counter("abmpnn_redesign", "failed"): 1})
                    io.clear_inflight(design_name)
                    _remove_all(hallucination_pdbs)
                    num_failed += 1
                    continue

                candidates = utils.build_final_candidates(
                    design_name=design_name,
                    trajectory_sequence=trajectory_sequence,
                    abmpnn_sequences=abmpnn_sequences,
                    include_codesign=True,
                )

            # The non-recycling path designs every candidate on the
            # hallucination output, so that is its designed-on backbone.
            for cand in candidates:
                cand.setdefault("designed_on", hallucination_pdbs)

            slot["elapsed"] += time.time() - phase_start
            if candidates:
                # The tools worked for this design, so the streak is broken.
                consecutive_tool_failures = 0
                slot["candidates"] = candidates
                with_candidates.append(slot)
                # Carries the candidate sequences: redesign would draw
                # different ones, so this is the only way to finish these.
                _journal(io, slot, "candidates_ready", fingerprints)

        # These already have their candidates, so they skip A-C. Added to
        # `slots` too, or cleanup would never clear their records.
        slots.extend(rescore_batch)
        with_candidates.extend(rescore_batch)

        # ============ PHASE D: fold a chunk of candidates, score it, repeat =====
        # Chunked so a kill costs one chunk of folding, not the whole batch.
        # A slot is never split: its candidates are scored together.
        for chunk in utils.chunks_by_candidate_count(
            with_candidates, folding.resolve_batch_size(run_settings, "filter")
        ):
            if _STOP_REQUESTED:
                log.info("\nStop requested. Leaving the remaining candidates for "
                      "the next run to score.")
                break
            final_designs = [
                (c["name"], c["seq"]) for s in chunk for c in s["candidates"]
            ]
            final_folds = filter_utils.fold_designs_for_filter(
                designs=final_designs,
                run_settings=run_settings,
                target_settings=target_settings,
                io=io,
                is_final=True,
            ) if final_designs else {}

            for slot in chunk:
                trajectory = slot["trajectory"]
                design_name = slot["design_name"]
                hallucination_pdbs = slot["hallucination_pdbs"]
                candidates = slot["candidates"]
                any_accepted = False
                phase_start = time.time()

                # ==================== 5. Final filters ====================
                any_accepted = False
                for candidate in candidates:
                    candidate_trajectory = trajectory.copy()
                    # Before multi_run_filters: it derives its output directory from
                    # design_name.
                    candidate_trajectory.rename(candidate["name"])

                    log.info(f"Running final filters on {candidate['name']} "
                          f"({candidate['source']})")

                    filter_metrics, filter_results, accepted, final_struct = (
                        filter_utils.multi_run_filters(
                            candidate_trajectory,
                            run_settings,
                            target_settings,
                            final_filters,
                            io,
                            candidate["seq"],
                            candidate["designed_on"],
                            is_final=True,
                            prediction_results=final_folds.get(candidate['name']),
                        )
                    )

                    raw_filter_data = {**filter_metrics, **filter_results}
                    complete_filter_data = utils.stage_keys(raw_filter_data, "final")
                    candidate_trajectory.update_filtering_metrics(complete_filter_data)
                    # Charged inside the candidate loop, since each candidate's row
                    # is written here. phase_start is reset to avoid double-counting.
                    slot["elapsed"] += time.time() - phase_start
                    phase_start = time.time()
                    candidate_trajectory.set_final_struct(final_struct)
                    candidate_trajectory.update_other_metrics({
                        "status": "accepted" if accepted else "final_filter_failed",
                        "trajectory_sequence": candidate["seq"],
                        "design_time": _design_time(slot),
                        "abmpnn_score": candidate["score"],
                        "abmpnn_seqid": candidate["seqid"],
                        "candidate_source": candidate["source"],
                        **({} if accepted else {
                            "failure_stage": utils.resolve_failure_stage(
                                raw_filter_data, "final_filter"
                            )
                        }),
                    })
                    candidate_trajectory.set_save_location(
                        "accepted" if accepted else "filter_failed"
                    )
                    io.save_trajectory(candidate_trajectory)
                    slot.setdefault("scored", []).append(candidate["name"])

                    if accepted:
                        log.info("========================================")
                        log.info(f"Design {candidate_trajectory.design_name} accepted!")
                        log.info("========================================")
                        num_accepted += 1
                        any_accepted = True
                    else:
                        failed_filters = utils.stage_keys(
                            filter_utils.select_failed_filters(
                                filter_results, filter_metrics
                            ),
                            "final",
                        )
                        if failed_filters:
                            io.update_failures(failed_filters)
                        num_candidates_rejected += 1
                    utils.clear_memory(clear_jax=False)

                slot["elapsed"] += time.time() - phase_start
                slot["any_accepted"] = any_accepted

        # ==================== Batch cleanup ====================
        # A stop request can break Phase C before a slot was redesigned. Such
        # a slot has neither candidates nor scored names, so the part-scored
        # guard below reads 0 != 0 and falls through to the delete -- throwing
        # away the hallucination and fold work for a design whose in-flight
        # record says it is adoptable. Leave everything in place instead.
        if _STOP_REQUESTED:
            log.warning(
                "Stop requested: keeping in-flight records and hallucination "
                "structures so the next run can adopt them."
            )
            utils.clear_memory(clear_jax=False)
            break
        # The hallucination PDBs are the sc_rmsd reference, so they survive A-D.
        for slot in slots:
            any_accepted = slot["any_accepted"]
            hallucination_pdbs = slot["hallucination_pdbs"]
            # A part-scored slot keeps its record AND the structures that
            # record points at, so the next run can finish the set.
            if len(slot.get("scored", [])) != len(slot["candidates"]):
                continue
            io.clear_inflight(slot["design_name"])
            if not any_accepted:
                _remove_all(hallucination_pdbs)
        io.clear_batch_fold_dir()
        utils.clear_memory(clear_jax=False)

    total_runtime = utils.get_clean_time(time.time(), start_time)
    run_summary = (
        f"Finished all designs after {n_attempted} attempted trajectories.\n"
        f"{failed_design} designs failed hallucination or the early check.\n"
        f"{num_failed} designs were rejected by a filter stage.\n"
        f"{num_accepted} designs passed all filters and were accepted.\n"
        f"({num_candidates_rejected} final-stage candidates rejected; a "
        f"design contributes several, so this is not a design count.)\n"
        f"Elapsed: {total_runtime}.\n"
        f"{_unfinished_work_note(io)}"
    )
    log.info(run_summary)
    io.save_run_summary(run_summary)


if __name__ == "__main__":
    main()
