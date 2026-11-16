"""The recycling loop: refold a design, redesign it, keep what improves.

Each cycle folds the current backbone, asks AbMPNN for variants, scores them
on epitope contact and ipSAE, and advances from the best one. Lives beside
redesign.py because the loop is mostly calls into it.
"""

import os
import time

import numpy as np

from germinal.filters import filter_utils, folding, pyrosetta_utils
from germinal.design import redesign as redesign_mod
from germinal.design.redesign import redesign
from germinal.utils import utils

from germinal.utils.log import get_logger

log = get_logger(__name__)


def _score_epitope(pdbs, run_settings, target_settings):
    """Hotspot-contact and CDR-fraction check on one candidate's structures.

    Returns (hotspots_pass, min_cdr_frac, scored_ok). scored_ok is False only
    when a scoring tool raised; 0.0 and False are legitimate results.
    """
    binder_chain = target_settings["binder_chain"]
    target_chain = target_settings["target_chain"]
    one_indexed_cdr_positions = np.array(run_settings["cdr_positions"]) + 1
    cdr3 = np.array(utils.cdr3_positions(run_settings)) + 1
    target_hotspots_list = target_settings["target_hotspots"]

    hotspots_pass = True
    min_cdr_frac = 1.0
    scored_ok = True

    for t_idx, pdb in enumerate(pdbs):
        target_settings_single = target_settings.copy()
        target_settings_single["target_hotspots"] = target_hotspots_list[t_idx]
        try:
            near_hotspot, _, _ = filter_utils.compute_hotspot_proximity(
                external_relaxed_pdb=pdb,
                target_settings=target_settings_single,
                binder_chain=binder_chain,
                target_chain=target_chain,
                one_indexed_cdr_positions=one_indexed_cdr_positions,
                cdr3=cdr3,
                distance_threshold=run_settings["hotspot_distance_threshold"],
                contact_distance=run_settings["residue_contact_distance"],
                min_hotspot_contacts=run_settings["min_cdr_hotspot_contacts"],
            )
        except Exception as e:
            log.info(f"    Target {t_idx} near_hotspot: UNSCORABLE "
                  f"({type(e).__name__}: {e}) [{pdb}]")
            scored_ok = False
            continue
        if not near_hotspot:
            hotspots_pass = False

        try:
            _, _, iface_res_str = pyrosetta_utils.score_interface(
                pdb, binder_chain=binder_chain, target_chain=target_chain
            )
            cdr_frac = utils.interface_cdrs(iface_res_str, run_settings["cdr_positions"])
        except Exception as e:
            log.info(f"    Target {t_idx} cdr_fraction: UNSCORABLE "
                  f"({type(e).__name__}: {e}) [{pdb}]")
            scored_ok = False
            continue
        min_cdr_frac = min(min_cdr_frac, cdr_frac)
        log.info(f"    Target {t_idx} near_hotspot={near_hotspot} "
              f"cdr_fraction={cdr_frac:.3f}")

    return hotspots_pass, min_cdr_frac, scored_ok

def recycle(
    run_settings,
    target_settings,
    design_name,
    trajectory_pdbs_af,
    trajectory_sequence,
    target_sequences,
    structures_directory,
):
    """Multi-state AbMPNN + refold loop.

    Returns (scored_candidates, info): every candidate folded, not just the
    winner. The caller selects which go to the final filters.
    """
    num_targets = target_settings["num_targets"]
    target_chain = target_settings["target_chain"]
    binder_chain = target_settings["binder_chain"]
    num_recycles = run_settings.get("num_recycles", 0)
    cdr_threshold = run_settings["early_check_min_cdr_fraction"]
    max_mpnn_score = run_settings.get("recycle_max_mpnn_score", 1.0)

    current_pdbs = list(trajectory_pdbs_af)
    current_sequence = trajectory_sequence
    scored_candidates = []
    info = {
        "recycles_completed": 0,
        "truncation_reason": "",
        "truncation_category": "",
        # True when the loop stopped because a TOOL failed rather than
        # because the design was not good enough. The caller counts these:
        # a broken install fails every design, and the run should say so
        # instead of recording thousands of rejections.
        "tool_failure": False,
    }

    # None, not 0.0: "nothing was ever scored" is a different state from "the
    # best scored 0.0", and the caller's gate distinguishes them.
    best_ipsae = None
    best_pdbs = list(current_pdbs)
    best_sequence = current_sequence

    interface_ratchet = run_settings.get("interface_ratchet", False)
    fix_interface_cutoff = run_settings.get("fix_interface_cutoff", 2.6)
    interface_locked = False
    best_avg_ipsae = 0.0

    tested_sequences = {current_sequence}

    log.info(f"\n--- Starting recycling pipeline ({num_recycles} iterations) ---")

    for recycle_idx in range(num_recycles):
        recycle_start = time.time()
        log.info(f"\n  Recycle {recycle_idx + 1}/{num_recycles}")

        atom_distance_cutoff = run_settings.get("atom_distance_cutoff", 0.0)
        if interface_locked:
            atom_distance_cutoff = fix_interface_cutoff
        log.info(f"  atom_distance_cutoff={atom_distance_cutoff}"
              f"{' (interface locked)' if interface_locked else ''}")

        abmpnn_sequences, abmpnn_success, redesign_reason = redesign(
            trajectory_pdbs_af=current_pdbs,
            target_chain=target_chain,
            binder_chain=binder_chain,
            run_settings=run_settings,
            atom_distance_cutoff=atom_distance_cutoff,
            exclude_sequences=tested_sequences,
            # The loop consumes one candidate per cycle, so ask for one.
            max_sequences=1,
        )

        if not abmpnn_success or len(abmpnn_sequences) == 0:
            info["truncation_reason"] = (
                f"redesign_{redesign_reason}_recycle_{recycle_idx + 1}"
            )
            info["truncation_category"] = "redesign"
            info["tool_failure"] = redesign_reason in redesign_mod.TOOL_FAILURE_REASONS
            if info["tool_failure"]:
                # WARNING, not INFO: this is a broken tool, and it repeats for
                # every design. At log_level WARNING -- documented as "just the
                # problems" -- an INFO line here means a run where nothing can
                # ever succeed prints nothing explaining why.
                log.warning(f"  Multi-state redesign FAILED at recycle "
                      f"{recycle_idx + 1} ({redesign_reason}) -- tool failure, "
                      f"not a design property. If it repeats, check the "
                      f"ProteinMPNN submodule, abmpnn.pt, GPU memory, and "
                      f"`nvidia-smi --query-gpu=compute_mode --format=csv`: on "
                      f"an Exclusive_Process cluster this process holds the "
                      f"only context the card allows and the redesign "
                      f"subprocess cannot start (see docs/af3-setup.md).")
            else:
                log.info(f"  Multi-state redesign produced nothing at recycle "
                      f"{recycle_idx + 1} ({redesign_reason}), stopping.")
            break

        mpnn_sequence = abmpnn_sequences[0]["seq"]
        abmpnn_score = abmpnn_sequences[0]["score"]
        abmpnn_seqid = abmpnn_sequences[0]["seqid"]
        log.info(f"  AbMPNN score: {abmpnn_score:.3f}, seqid: {abmpnn_seqid:.3f}")

        if abmpnn_score > max_mpnn_score:
            log.info(f"  AbMPNN score {abmpnn_score:.3f} > {max_mpnn_score}, "
                  f"stopping (AbMPNN is deterministic, retrying won't help).")
            info["truncation_reason"] = (
                f"abmpnn_score_{abmpnn_score:.3f}_over_{max_mpnn_score}"
                f"_recycle_{recycle_idx + 1}"
            )
            info["truncation_category"] = "abmpnn_score"
            break

        # Fold the incumbent once, in recycle 1: re-folding it every cycle
        # would let its ipSAE climb on seed noise alone.
        candidate_sequences = [mpnn_sequence]
        if recycle_idx == 0:
            candidate_sequences.append(current_sequence)
        tested_sequences.add(mpnn_sequence)

        # Nested per design so a cleanup is one rmtree, not a filename glob.
        recycle_dir = os.path.join(structures_directory, design_name)
        os.makedirs(recycle_dir, exist_ok=True)
        recycle_design_name = f"{design_name}_recycle{recycle_idx + 1}"
        log.info(f"  Predicting {len(candidate_sequences)} candidate(s) x "
              f"{num_targets} target(s) in one batched call")

        try:
            batch_results = filter_utils.run_structure_prediction_batch(
                binder_sequences=candidate_sequences,
                target_sequences=target_sequences,
                target_chain=target_chain,
                binder_chain=binder_chain,
                structures_directory=recycle_dir,
                design_name_prefix=recycle_design_name,
                run_settings=run_settings,
            )
        except folding.AllJobsFailed:
            # Raised only when EVERY job in the batch failed, which is
            # systemic -- bad environment or config, never N bad designs.
            raise
        except Exception as e:
            # A per-candidate fault that did not take the whole batch down is
            # a legitimate truncation: the trajectory simply stops recycling.
            cause = str(e).replace("\n", " ").strip()[:200]
            log.info(f"  Batched structure prediction failed: {cause}")
            info["truncation_reason"] = (
                f"prediction_failed_recycle_{recycle_idx + 1}: "
                f"{type(e).__name__}: {cause}"
            )
            info["truncation_category"] = "prediction_failed"
            break

        # Results are ordered [cand0_t0, cand0_t1, cand1_t0, ...]
        cycle_candidates = []
        for seq_idx, cand_seq in enumerate(candidate_sequences):
            cand_pdbs, cand_ipsae = [], []
            for t_idx in range(num_targets):
                result = batch_results[seq_idx * num_targets + t_idx]
                if result is None:
                    break
                cand_pdb, _, ipsae_raw = result
                cand_pdbs.append(cand_pdb)
                cand_ipsae.append(utils.extract_ipsae(ipsae_raw)["ipsae"])
            if len(cand_pdbs) != num_targets:
                log.info(f"    Candidate {seq_idx} incomplete, skipping")
                continue
            is_baseline = cand_seq == current_sequence
            log.info(f"    Candidate {seq_idx} ({'baseline' if is_baseline else 'mpnn'}): "
                  f"ipSAE {[f'{v:.4f}' for v in cand_ipsae]}, "
                  f"min {min(cand_ipsae):.4f}")
            cycle_candidates.append({
                "seq": cand_seq,
                "pdbs": cand_pdbs,
                # The backbone AbMPNN designed on, copied because current_pdbs
                # is rebound later. "pdbs" above is this candidate's own refold.
                "designed_on": list(current_pdbs),
                "min_ipsae": min(cand_ipsae),
                "avg_ipsae": sum(cand_ipsae) / len(cand_ipsae),
                "is_baseline": is_baseline,
                # The baseline was never proposed by AbMPNN, so it has no score.
                # None rather than 0.0, which is the BEST possible MPNN NLL.
                "abmpnn_score": None if is_baseline else abmpnn_score,
                "abmpnn_seqid": None if is_baseline else abmpnn_seqid,
                "recycle": recycle_idx + 1,
            })

        if not cycle_candidates:
            log.info("  No candidate produced a complete set of structures, stopping.")
            info["truncation_reason"] = (
                f"prediction_incomplete_recycle_{recycle_idx + 1}"
            )
            info["truncation_category"] = "prediction_incomplete"
            break

        # Epitope gate on every candidate, not just the winner: all of them are
        # eligible for the final filters, so all of them must be checked.
        for cand in cycle_candidates:
            hotspots_pass, min_cdr_frac, scored_ok = _score_epitope(
                cand["pdbs"], run_settings, target_settings
            )
            cand["hotspots_pass"] = hotspots_pass
            cand["cdr_frac"] = min_cdr_frac
            cand["scored_ok"] = scored_ok

        scorable = [c for c in cycle_candidates if c["scored_ok"]]
        if not scorable:
            info["truncation_reason"] = f"scoring_failed_recycle_{recycle_idx + 1}"
            info["truncation_category"] = "scoring_failed"
            log.info("  No candidate could be scored; reverting to best backbone "
                  "and stopping recycling for this design.")
            current_pdbs = list(best_pdbs)
            current_sequence = best_sequence
            info["recycles_completed"] = recycle_idx + 1
            break

        scored_candidates.extend(scorable)


        # Advance the loop from the best candidate that cleared the epitope gate.
        passing = [
            c for c in scorable
            if c["hotspots_pass"] and c["cdr_frac"] >= cdr_threshold
        ]
        contender = max(passing, key=lambda c: c["min_ipsae"]) if passing else None

        if contender is not None and (
            best_ipsae is None or contender["min_ipsae"] > best_ipsae
        ):
            log.info(f"  New best recycle design! (min ipSAE "
                  f"{contender['min_ipsae']:.4f})")
            current_pdbs = list(contender["pdbs"])
            current_sequence = contender["seq"]
            best_ipsae = contender["min_ipsae"]
            best_pdbs = list(contender["pdbs"])
            best_sequence = contender["seq"]
            if interface_ratchet and contender["avg_ipsae"] > best_avg_ipsae:
                best_avg_ipsae = contender["avg_ipsae"]
                if not interface_locked:
                    interface_locked = True
                    log.info(f"  Interface LOCKED, atom_distance_cutoff="
                          f"{fix_interface_cutoff} for remaining cycles.")
        else:
            log.info("  Design not improved, reverting to best backbone.")
            current_pdbs = list(best_pdbs)
            current_sequence = best_sequence

        info["recycles_completed"] = recycle_idx + 1
        log.info(f"  Recycle {recycle_idx + 1} took: "
              f"{utils.get_clean_time(time.time(), recycle_start)}")
        utils.clear_memory(clear_jax=False)

    log.info(f"\n--- Recycling complete after {info['recycles_completed']} "
          f"recycle(s). ---\n")
    return scored_candidates, info
