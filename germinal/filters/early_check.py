"""Quick quality gate on the hallucinated AF2 structures.

Runs between hallucination and the initial filters, testing the same clash /
hotspot / CDR-fraction gates from AF2 output, which costs nothing next to a
cofold. It lives here rather than in utils to avoid an import cycle.
"""

from typing import Tuple, List

import numpy as np

from germinal.filters import filter_utils, pyrosetta_utils
from germinal.utils import utils

from germinal.utils.log import get_logger

log = get_logger(__name__)


def early_check(
    trajectory_pdbs_af: List[str],
    run_settings: dict,
    target_settings: dict,
    min_cdr_fraction: float,
) -> Tuple[bool, dict]:
    """Quick quality gate on the hallucinated AF2 structures (per target).

    Checks per target (using AF2 hallucinated PDBs only):
        1. Clashes (CA clashes on hallucinated PDB)
        2. Hotspot proximity (CDR contacts with target hotspots)
        3. CDR fraction at the binding interface

    Returns:
        (passed, info_dict)
    """
    target_chain = target_settings["target_chain"]
    binder_chain = target_settings["binder_chain"]
    one_indexed_cdr_positions = np.array(run_settings["cdr_positions"]) + 1

    target_hotspots_list = target_settings["target_hotspots"]

    # CDR3 positions (1-indexed for PDB numbering)
    cdr3 = np.array(utils.cdr3_positions(run_settings)) + 1

    info = {"per_target": []}
    all_hotspot_pass = True
    min_cdr_frac = 1.0
    any_clashes = False
    crashed = False

    # --- Per-target checks on hallucinated PDBs ---
    for t_idx, pdb_path in enumerate(trajectory_pdbs_af):
        target_info = {}

        # 1. Clashes
        from germinal.utils.utils import calculate_clash_score
        ca_clashes = calculate_clash_score(
            pdb_path,
            run_settings["clash_threshold"],
            only_ca=True,
            num_targets=1,
        )
        target_info["ca_clashes"] = ca_clashes
        if ca_clashes > 0:
            any_clashes = True

        # 2. Hotspot proximity
        target_settings_single = target_settings.copy()
        target_settings_single["target_hotspots"] = target_hotspots_list[t_idx]

        try:
            near_hotspot, cdr3_contacts, cdr_contacts = filter_utils.compute_hotspot_proximity(
                external_relaxed_pdb=pdb_path,
                target_settings=target_settings_single,
                binder_chain=binder_chain,
                target_chain=target_chain,
                one_indexed_cdr_positions=one_indexed_cdr_positions,
                cdr3=cdr3,
                distance_threshold=run_settings["hotspot_distance_threshold"],
                contact_distance=run_settings["residue_contact_distance"],
                min_hotspot_contacts=run_settings["min_cdr_hotspot_contacts"],
            )
            target_info["near_hotspot"] = near_hotspot
            target_info["cdr3_contacts"] = cdr3_contacts
            target_info["cdr_contacts"] = cdr_contacts
            if not near_hotspot:
                all_hotspot_pass = False
        except Exception as e:
            # A crash is a tool failure: near_hotspot=False would be
            # indistinguishable from a binder that genuinely misses.
            log.info(f"  Early check: hotspot proximity CRASHED for target {t_idx} "
                  f"({type(e).__name__}: {e}) [{pdb_path}]")
            crashed = True
            target_info["near_hotspot"] = None
            target_info["cdr3_contacts"] = None
            target_info["cdr_contacts"] = None

        # 3. CDR fraction
        try:
            _, _, interface_residues_str = pyrosetta_utils.score_interface(
                pdb_path, binder_chain=binder_chain, target_chain=target_chain
            )
            cdr_fraction = utils.interface_cdrs(
                interface_residues_str,
                run_settings["cdr_positions"],
            )
            target_info["cdr_fraction"] = round(cdr_fraction, 3)
            min_cdr_frac = min(min_cdr_frac, cdr_fraction)
        except Exception as e:
            # 0.0 is a legitimate CDR fraction, so do not write it on a
            # crash. Leave min_cdr_frac untouched.
            log.info(f"  Early check: interface scoring CRASHED for target {t_idx} "
                  f"({type(e).__name__}: {e}) [{pdb_path}]")
            crashed = True
            target_info["cdr_fraction"] = None

        info["per_target"].append(target_info)

    info["all_hotspot_pass"] = all_hotspot_pass
    info["min_cdr_fraction"] = round(min_cdr_frac, 3)
    info["any_clashes"] = any_clashes
    # Recorded on every design so the column is always present. True marks a
    # row whose gate values are incomplete.
    info["early_check_crashed"] = crashed

    if crashed or any_clashes or not all_hotspot_pass or min_cdr_frac < min_cdr_fraction:
        reason = []
        if crashed:
            reason.append("scoring_crashed")
        if any_clashes:
            reason.append("clashes")
        if not all_hotspot_pass:
            reason.append("hotspot_failed")
        if min_cdr_frac < min_cdr_fraction:
            reason.append(f"cdr_frac={min_cdr_frac:.3f}<{min_cdr_fraction}")
        info["reject_reason"] = ", ".join(reason)
        return False, info

    return True, info
