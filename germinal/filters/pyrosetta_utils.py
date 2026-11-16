"""
PyRosetta utility functions from BindCraft.

Attribution:
This module is adapted from the PyRosetta utilities in the BindCraft project:
https://github.com/martinpacesa/BindCraft

If you use these utilities, please cite:
Pacesa, M., Nickel, L., Schellhaas, C. et al. One-shot design of functional protein binders with BindCraft. Nature (2025). https://doi.org/10.1038/s41586-025-09429-6

"""

import os
import multiprocessing
import pyrosetta as pr
from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta.rosetta.core.select.residue_selector import ChainSelector
from pyrosetta.rosetta.protocols.simple_moves import AlignChainMover
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.core.simple_metrics.metrics import RMSDMetric
from pyrosetta.rosetta.core.select import get_residues_from_subset
from pyrosetta.rosetta.core.io import pose_from_pose
from pyrosetta import pose_from_pdb
from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects
from pyrosetta.rosetta.core.pose import Pose
from pyrosetta.rosetta.core.select.residue_selector import ResidueIndexSelector
from pyrosetta.rosetta.core.select.residue_selector import NeighborhoodResidueSelector
from germinal.utils.utils import hotspot_residues, clean_pdb
from collections import defaultdict
import numpy as np
from pyrosetta.rosetta.core.pack.guidance_scoreterms.sap import calculate_per_res_sap
from pyrosetta.rosetta.core.select.residue_selector import TrueResidueSelector

from germinal.utils.log import get_logger

log = get_logger(__name__)


# Rosetta interface scores
def score_interface(pdb_file, binder_chain="B", target_chain="A"):
    # load pose
    pose = pr.pose_from_pdb(pdb_file)

    # analyze interface statistics
    iam = InterfaceAnalyzerMover()
    # From the configured chains, not a hardcoded "A_B", so every metric in
    # this function scores the same interface.
    iam.set_interface(f"{target_chain}_{binder_chain}")
    scorefxn = pr.get_fa_scorefxn()
    iam.set_scorefunction(scorefxn)
    iam.set_compute_packstat(True)
    iam.set_compute_interface_energy(True)
    iam.set_calc_dSASA(True)
    iam.set_calc_hbond_sasaE(True)
    iam.set_compute_interface_sc(True)
    iam.set_pack_separated(True)
    iam.apply(pose)

    # Initialize dictionary with all amino acids
    interface_AA = {aa: 0 for aa in "ACDEFGHIKLMNPQRSTVWY"}

    # Initialize list to store PDB residue IDs at the interface
    interface_residues_set = hotspot_residues(
        pdb_file, binder_chain, target_chain=target_chain
    )
    interface_residues_pdb_ids = []

    # Iterate over the interface residues
    for pdb_res_num, aa_type in interface_residues_set.items():
        # Increase the count for this amino acid type
        interface_AA[aa_type] += 1

        # Append the binder_chain and the PDB residue number to the list
        interface_residues_pdb_ids.append(f"{binder_chain}{pdb_res_num}")

    # count interface residues
    interface_nres = len(interface_residues_pdb_ids)

    # Convert the list into a comma-separated string
    interface_residues_pdb_ids_str = ",".join(interface_residues_pdb_ids)

    # Calculate the percentage of hydrophobic residues at the interface of the binder
    hydrophobic_aa = set("ACFILMPVWY")
    hydrophobic_count = sum(interface_AA[aa] for aa in hydrophobic_aa)
    if interface_nres != 0:
        interface_hydrophobicity = (hydrophobic_count / interface_nres) * 100
    else:
        interface_hydrophobicity = 0

    # retrieve statistics
    interfacescore = iam.get_all_data()
    interface_sc = interfacescore.sc_value  # shape complementarity
    interface_interface_hbonds = (
        interfacescore.interface_hbonds
    )  # number of interface H-bonds
    interface_dG = iam.get_interface_dG()  # interface dG
    interface_dSASA = (
        iam.get_interface_delta_sasa()
    )  # interface dSASA (interface surface area)
    interface_packstat = iam.get_interface_packstat()  # interface pack stat score
    interface_dG_SASA_ratio = (
        interfacescore.dG_dSASA_ratio * 100
    )  # ratio of dG/dSASA (normalised energy for interface area size)
    buns_filter = XmlObjects.static_get_filter(
        '<BuriedUnsatHbonds report_all_heavy_atom_unsats="true" scorefxn="scorefxn" ignore_surface_res="false" use_ddG_style="true" dalphaball_sasa="1" probe_radius="1.1" burial_cutoff_apo="0.2" confidence="0" />'
    )
    interface_delta_unsat_hbonds = buns_filter.report_sm(pose)

    if interface_nres != 0:
        interface_hbond_percentage = (
            interface_interface_hbonds / interface_nres
        ) * 100  # Hbonds per interface size percentage
        interface_bunsch_percentage = (
            interface_delta_unsat_hbonds / interface_nres
        ) * 100  # Unsaturated H-bonds per percentage
    else:
        interface_hbond_percentage = None
        interface_bunsch_percentage = None

    # calculate binder energy score
    chain_design = ChainSelector(binder_chain)
    tem = pr.rosetta.core.simple_metrics.metrics.TotalEnergyMetric()
    tem.set_scorefunction(scorefxn)
    tem.set_residue_selector(chain_design)
    binder_score = tem.calculate(pose)

    # calculate binder SASA fraction
    bsasa = pr.rosetta.core.simple_metrics.metrics.SasaMetric()
    bsasa.set_residue_selector(chain_design)
    binder_sasa = bsasa.calculate(pose)

    if binder_sasa > 0:
        interface_binder_fraction = (interface_dSASA / binder_sasa) * 100
    else:
        interface_binder_fraction = 0

    # calculate surface hydrophobicity
    binder_pose = {
        pose.pdb_info().chain(pose.conformation().chain_begin(i)): p
        for i, p in zip(range(1, pose.num_chains() + 1), pose.split_by_chain())
    }[binder_chain]

    layer_sel = pr.rosetta.core.select.residue_selector.LayerSelector()
    layer_sel.set_layers(pick_core=False, pick_boundary=False, pick_surface=True)
    surface_res = layer_sel.apply(binder_pose)

    exp_apol_count = 0
    total_count = 0

    # count apolar and aromatic residues at the surface
    for i in range(1, len(surface_res) + 1):
        if surface_res[i] == True:
            res = binder_pose.residue(i)

            # count apolar and aromatic residues as hydrophobic
            if (
                res.is_apolar() == True
                or res.name() == "PHE"
                or res.name() == "TRP"
                or res.name() == "TYR"
            ):
                exp_apol_count += 1
            total_count += 1

    surface_hydrophobicity = exp_apol_count / total_count

    # output interface score array and amino acid counts at the interface
    interface_scores = {
        "binder_score": binder_score,
        "surface_hydrophobicity": surface_hydrophobicity,
        "interface_sc": interface_sc,
        "interface_packstat": interface_packstat,
        "interface_dG": interface_dG,
        "interface_dSASA": interface_dSASA,
        "interface_dG_SASA_ratio": interface_dG_SASA_ratio,
        "interface_fraction": interface_binder_fraction,
        "interface_hydrophobicity": interface_hydrophobicity,
        "interface_nres": interface_nres,
        "interface_interface_hbonds": interface_interface_hbonds,
        "interface_hbond_percentage": interface_hbond_percentage,
        "interface_delta_unsat_hbonds": interface_delta_unsat_hbonds,
        "interface_delta_unsat_hbonds_percentage": interface_bunsch_percentage,
    }

    # Deliberately NOT rounded: these are compared against two-decimal
    # thresholds, so rounding here would move the accept/reject boundary.

    return interface_scores, interface_AA, interface_residues_pdb_ids_str


# align pdbs to have same orientation
def align_pdbs(reference_pdb, align_pdb, reference_chain_id, align_chain_id, output_pdb=None):
    # initiate poses
    reference_pose = pr.pose_from_pdb(reference_pdb)
    align_pose = pr.pose_from_pdb(align_pdb)

    align = AlignChainMover()
    align.pose(reference_pose)

    # If the chain IDs contain commas, split them and only take the first value
    reference_chain_id = reference_chain_id.split(",")[0]
    align_chain_id = align_chain_id.split(",")[0]

    # Get the chain number corresponding to the chain ID in the poses
    reference_chain = pr.rosetta.core.pose.get_chain_id_from_chain(
        reference_chain_id, reference_pose
    )
    align_chain = pr.rosetta.core.pose.get_chain_id_from_chain(
        align_chain_id, align_pose
    )

    align.source_chain(align_chain)
    align.target_chain(reference_chain)
    align.apply(align_pose)

    # Defaults to overwriting align_pdb, but callers should pass output_pdb:
    # align_pdb is a persisted artifact and would be rewritten in place.
    out_pdb = output_pdb or align_pdb
    align_pose.dump_pdb(out_pdb)
    clean_pdb(out_pdb)
    return out_pdb


def get_sap_score(
    pdb,
    binder_chain=None,
    only_binder=False,
    hydrophobic_aa=None,
    patch_radius=8,
    limit_sasa=1,
    avg_sasa_patch_thr=0.75,
    cdrs=None,
):
    pose_ = pr.pose_from_pdb(
        pdb
    )  # Assuming 'pdb' is defined and contains your PDB file path

    selector = TrueResidueSelector()
    if binder_chain is not None and only_binder:
        # Derive the chain id -> pose index map from the pose itself: pose
        # order need not match alphabetical rank from A.
        chain_idx = {
            pose_.pdb_info().chain(pose_.conformation().chain_begin(i)): i
            for i in range(1, pose_.num_chains() + 1)
        }
        if binder_chain not in chain_idx:
            raise ValueError(
                f"binder_chain {binder_chain!r} is not present in {pdb}; "
                f"chains found: {sorted(chain_idx)}"
            )
        pose = Pose()
        pose = pose_.split_by_chain(chain_idx[binder_chain])
    else:
        pose = pose_
        if binder_chain is not None:
            selector = ChainSelector(binder_chain)

    # Create vectors to store the results
    num_residues = pose.total_residue()

    if hydrophobic_aa is None:
        # Top from hydrophobic scale (Black S.D., Mould D.R.)
        hydrophobic_aa = ["LEU", "ILE", "PHE", "TRP", "VAL", "MET", "TYR", "ALA"]


    sap_score = calculate_per_res_sap(
        pose=pose, score_sel=selector, sap_calculate_sel=selector, sasa_sel=selector
    )

    def avg_sap_hydrophobic_patch(sap_score, residues):
        avg_sap = 0
        for r in residues:
            avg_sap += sap_score[r[0]]  # * scale[r[1]]
        avg_sap = avg_sap / len(residues)
        return avg_sap

    def patch_exists(hydrophobic_patches, nearby_res):
        if len(hydrophobic_patches) < 1:
            return False
        else:
            nrb = set(nearby_res)
            for hp in hydrophobic_patches:
                prev = set(hp[1])
                if len(nrb - prev) <= len(nrb) - 2:
                    return True
            return False

    exposed_hydrophobic_aa = []
    hydrophobic_patches = []
    for i in range(1, num_residues + 1):
        aa_type = pose.residue(i).name3()  # Get three letter code
        if binder_chain is not None and pose.pdb_info().chain(i) != binder_chain:
            continue
        if aa_type in hydrophobic_aa and (sap_score[i]) >= limit_sasa:
            exposed_hydrophobic_aa.append((i, aa_type))
            nearby_res = get_nearby_residues(pose, i, distance=patch_radius)
            avg_sap_patch = avg_sap_hydrophobic_patch(sap_score, nearby_res)

            if avg_sap_patch >= avg_sasa_patch_thr:
                if not patch_exists(hydrophobic_patches, nearby_res):
                    hydrophobic_patches.append((avg_sap_patch, nearby_res))

    cdr_sap = np.array(sap_score)
    if not cdrs is None:
        # cdrs is binder-relative, so the pose must BE the binder: with
        # only_binder=False these indices would land on the target.
        if not (binder_chain is not None and only_binder):
            raise ValueError(
                "get_sap_score(cdrs=...) requires only_binder=True and a "
                "binder_chain: cdr positions are binder-relative, so scoring "
                "them against a whole-complex pose indexes the target."
            )
        cdr_sap = sum(cdr_sap[cdrs])
    else:
        cdr_sap = sum(cdr_sap)

    return sum(sap_score), cdr_sap, exposed_hydrophobic_aa, hydrophobic_patches


def get_nearby_residues(pose, target_residue_number, distance=8.0):
    """
    Get all residues within a specified distance of a target residue

    Args:
        pose: PyRosetta Pose object
        target_residue_number: int, the residue number you're interested in
        distance: float, the cutoff distance in Angstroms (default 8.0Å)

    Returns:
        list of residue numbers that are within the specified distance
    """
    # Create selector for the target residue
    target_selector = ResidueIndexSelector(target_residue_number)

    # Create neighborhood selector
    neighbor_selector = NeighborhoodResidueSelector()
    neighbor_selector.set_focus_selector(target_selector)
    neighbor_selector.set_distance(distance)

    # Apply selector to get nearby residues
    nearby_residues = []
    for i in range(1, pose.total_residue() + 1):
        if neighbor_selector.apply(pose)[i]:
            nearby_residues.append((i, pose.residue(i).name3()))

    return nearby_residues


# calculate the rmsd without alignment
def unaligned_rmsd(reference_pdb, align_pdb, reference_chain_id, align_chain_id):
    reference_pose = pr.pose_from_pdb(reference_pdb)
    align_pose = pr.pose_from_pdb(align_pdb)

    # Define chain selectors for the reference and align chains
    reference_chain_selector = ChainSelector(reference_chain_id)
    align_chain_selector = ChainSelector(align_chain_id)

    # Apply selectors to get residue subsets
    reference_chain_subset = reference_chain_selector.apply(reference_pose)
    align_chain_subset = align_chain_selector.apply(align_pose)

    # Convert subsets to residue index vectors
    reference_residue_indices = get_residues_from_subset(reference_chain_subset)
    align_residue_indices = get_residues_from_subset(align_chain_subset)

    # Create empty subposes
    reference_chain_pose = pr.Pose()
    align_chain_pose = pr.Pose()

    # Fill subposes
    pose_from_pose(reference_chain_pose, reference_pose, reference_residue_indices)
    pose_from_pose(align_chain_pose, align_pose, align_residue_indices)

    # Calculate RMSD using the RMSDMetric
    rmsd_metric = RMSDMetric()
    rmsd_metric.set_comparison_pose(reference_chain_pose)
    rmsd = rmsd_metric.calculate(align_chain_pose)

    # Not rounded: sc_rmsd is compared against two-decimal thresholds
    # (< 4.5 initial, <= 6 final), so rounding would shift the cutoff.
    return rmsd


# Relax designed structure
def pr_relax(pdb_file, relaxed_pdb_path):
    if not os.path.exists(relaxed_pdb_path):
        # Generate pose
        pose = pr.pose_from_pdb(pdb_file)
        start_pose = pose.clone()

        ### Generate movemaps
        mmf = MoveMap()
        mmf.set_chi(True)  # enable sidechain movement
        mmf.set_bb(
            True
        )  # enable backbone movement, can be disabled to increase speed by 30% but makes metrics look worse on average
        mmf.set_jump(False)  # disable whole chain movement

        # Run FastRelax
        fastrelax = FastRelax()
        scorefxn = pr.get_fa_scorefxn()
        fastrelax.set_scorefxn(scorefxn)
        fastrelax.set_movemap(mmf)  # set MoveMap
        fastrelax.max_iter(200)  # default iterations is 2500
        fastrelax.min_type("lbfgs_armijo_nonmonotone")
        fastrelax.constrain_relax_to_start_coords(True)
        fastrelax.apply(pose)

        # Align relaxed structure to original trajectory
        align = AlignChainMover()
        align.source_chain(0)
        align.target_chain(0)
        align.pose(start_pose)
        align.apply(pose)

        # Copy B factors from start_pose to pose
        for resid in range(1, pose.total_residue() + 1):
            if pose.residue(resid).is_protein():
                # Get the B factor of the first heavy atom in the residue
                bfactor = start_pose.pdb_info().bfactor(resid, 1)
                for atom_id in range(1, pose.residue(resid).natoms() + 1):
                    pose.pdb_info().bfactor(resid, atom_id, bfactor)

        # output relaxed and aligned PDB
        pose.dump_pdb(relaxed_pdb_path)
        clean_pdb(relaxed_pdb_path)


def _relax_worker(pdb_file, relaxed_pdb_path, seed, dalphaball_path):
    """
    Worker function for parallel Rosetta relax. Runs in a child process
    with its own PyRosetta initialization.

    Args:
        pdb_file (str): Input PDB file path.
        relaxed_pdb_path (str): Output relaxed PDB file path.
        seed (int): Random seed for Rosetta RNG.
        dalphaball_path (str): Path to DAlphaBall executable.
    """
    pr.init(
        f"-ignore_unrecognized_res -ignore_zero_occupancy -mute all "
        f"-holes:dalphaball {dalphaball_path} "
        f"-corrections::beta_nov16 true -relax:default_repeats 1 "
        f"-run:constant_seed -run:jran {seed}"
    )

    pose = pr.pose_from_pdb(pdb_file)
    start_pose = pose.clone()

    mmf = MoveMap()
    mmf.set_chi(True)
    mmf.set_bb(True)
    mmf.set_jump(False)

    fastrelax = FastRelax()
    scorefxn = pr.get_fa_scorefxn()
    fastrelax.set_scorefxn(scorefxn)
    fastrelax.set_movemap(mmf)
    fastrelax.max_iter(200)
    fastrelax.min_type("lbfgs_armijo_nonmonotone")
    fastrelax.constrain_relax_to_start_coords(True)
    fastrelax.apply(pose)

    align = AlignChainMover()
    align.source_chain(0)
    align.target_chain(0)
    align.pose(start_pose)
    align.apply(pose)

    for resid in range(1, pose.total_residue() + 1):
        if pose.residue(resid).is_protein():
            bfactor = start_pose.pdb_info().bfactor(resid, 1)
            for atom_id in range(1, pose.residue(resid).natoms() + 1):
                pose.pdb_info().bfactor(resid, atom_id, bfactor)

    # Temp file then rename, so the existence of relaxed_pdb_path means a
    # COMPLETE structure even if a worker is killed mid-write.
    tmp_path = f"{relaxed_pdb_path}.tmp.{os.getpid()}"
    try:
        pose.dump_pdb(tmp_path)
        clean_pdb(tmp_path)
        os.replace(tmp_path, relaxed_pdb_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def pr_relax_parallel(pdb_file, output_dir, design_name, dalphaball_path, n_relax=10):
    """
    Run parallel Rosetta FastRelax with different random seeds.

    Spawns n_relax child processes, each with its own PyRosetta initialization
    and a unique random seed.

    Args:
        pdb_file (str): Input PDB file path.
        output_dir (str): Directory for output PDB files.
        design_name (str): Design name for output file naming.
        dalphaball_path (str): Path to DAlphaBall executable.
        n_relax (int): Number of parallel relax runs. Defaults to 10.

    Returns:
        list[str]: Paths to relaxed PDB files.
    """
    ctx = multiprocessing.get_context("spawn")
    relaxed_paths = []
    jobs = []  # (process, seed, output path)
    seeds = np.random.randint(0, 999999, size=n_relax).tolist()

    for i, seed in enumerate(seeds):
        relaxed_pdb_path = os.path.join(output_dir, f"{design_name}_relaxed_{i}.pdb")
        relaxed_paths.append(relaxed_pdb_path)

        # Safe to reuse: _relax_worker renames into place, so this file is
        # either absent or complete.
        if os.path.exists(relaxed_pdb_path):
            continue

        p = ctx.Process(
            target=_relax_worker,
            args=(pdb_file, relaxed_pdb_path, seed, dalphaball_path),
        )
        jobs.append((p, seed, relaxed_pdb_path))

    for p, _, _ in jobs:
        p.start()
    for p, _, _ in jobs:
        p.join()

    # A dead worker is a tool failure, not a design property: dropping it
    # would silently shrink the ensemble that score_mode="best" draws from.
    failures = []
    for p, seed, path in jobs:
        if p.exitcode != 0:
            failures.append(f"seed {seed} -> exitcode {p.exitcode}")
        elif not os.path.exists(path):
            failures.append(f"seed {seed} -> exited 0 but wrote no {os.path.basename(path)}")

    if failures:
        raise RuntimeError(
            f"{len(failures)}/{len(jobs)} Rosetta relax workers failed for "
            f"{design_name}: " + "; ".join(failures) + ". The relax ensemble "
            "would be scored short, biasing every interface metric taken from "
            "it. Check memory limits and the DAlphaBall path."
        )

    return relaxed_paths


def score_interface_ensemble(
    relaxed_pdb_paths, binder_chain="B", target_chain="A", score_mode="best"
):
    """
    Score interface metrics across an ensemble of relaxed structures.

    Calls score_interface() for each relaxed structure and identifies the best
    structure by lowest binder_score (total Rosetta energy).

    Args:
        relaxed_pdb_paths (list[str]): Paths to relaxed PDB files.
        binder_chain (str): Binder chain ID.
        target_chain (str): Target chain ID.
        score_mode (str): How to aggregate interface metrics across relax runs.
            - "best" (default): Return all metrics from the single best
              structure (lowest binder_score) with no mixing between runs, so
              every reported metric describes one real conformation.
            - "average": Average numeric metrics across all runs. The resulting
              row corresponds to no physical structure and mixes metrics from
              different conformations; kept for comparison with older runs.

    Returns:
        tuple: (interface_scores, best_interface_AA, best_interface_residues, best_relaxed_pdb_path)
    """
    all_scores = []
    all_aa = []
    all_residues = []
    # Accumulated in lockstep with the scores: best_idx indexes the surviving
    # scores, so one failure would shift it off the matching path.
    all_paths = []
    n_failed = 0

    for pdb_path in relaxed_pdb_paths:
        try:
            scores, aa, residues = score_interface(pdb_path, binder_chain, target_chain)
            all_scores.append(scores)
            all_aa.append(aa)
            all_residues.append(residues)
            all_paths.append(pdb_path)
        except Exception as e:
            n_failed += 1
            log.warning(f"score_interface failed for {pdb_path}: {e}")

    if not all_scores:
        raise RuntimeError("All score_interface calls failed in ensemble scoring")
    if n_failed:
        log.info(
            f"Warning: {n_failed} of {len(relaxed_pdb_paths)} relax members "
            f"failed interface scoring; the ensemble reflects the remaining "
            f"{len(all_scores)}."
        )

    # Find best structure by lowest binder_score (total Rosetta energy)
    best_idx = np.argmin([s["binder_score"] for s in all_scores])
    best_interface_AA = all_aa[best_idx]
    best_interface_residues = all_residues[best_idx]
    best_relaxed_pdb_path = all_paths[best_idx]

    if score_mode == "best":
        # Return all metrics from the single best relax run — no mixing
        result_scores = all_scores[best_idx]
    else:
        # Average numeric metrics across all runs
        result_scores = {}
        for key in all_scores[0]:
            values = [s[key] for s in all_scores if s[key] is not None]
            if values and isinstance(values[0], (int, float)):
                result_scores[key] = np.mean(values)
            else:
                result_scores[key] = all_scores[0][key]

    return result_scores, best_interface_AA, best_interface_residues, best_relaxed_pdb_path


def get_cb_coordinates(residue):
    """Get CB coordinates (CA for GLY)"""
    if residue.name3() == "GLY":
        return np.array(residue.xyz("CA"))
    return np.array(residue.xyz("CB"))


def get_key_atoms(residue_name):
    """
    Get key atoms to check for each residue type
    Returns list of important atoms for VDW contacts
    """
    # Backbone atoms for all residues
    backbone = ["CA"]  # Only using CA and CB from backbone for efficiency

    # Side chain atoms by residue type
    side_chain = {
        # Hydrophobic residues
        "VAL": ["CG1", "CG2"],
        "ILE": ["CG1", "CG2", "CD1"],
        "LEU": ["CG", "CD1", "CD2"],
        "MET": ["CG", "SD", "CE"],
        "ALA": [],  # CB already in backbone
        "PRO": ["CG", "CD"],
        # Aromatic residues
        "PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
        "TYR": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
        "TRP": ["CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
        # Charged residues
        "ASP": ["CG", "OD1", "OD2"],
        "GLU": ["CG", "CD", "OE1", "OE2"],
        "LYS": ["CG", "CD", "CE", "NZ"],
        "ARG": ["CG", "CD", "NE", "CZ", "NH1", "NH2"],
        "HIS": ["CG", "ND1", "CD2", "CE1", "NE2"],
        # Polar residues
        "SER": ["OG"],
        "THR": ["OG1", "CG2"],
        "ASN": ["CG", "OD1", "ND2"],
        "GLN": ["CG", "CD", "OE1", "NE2"],
        # Special cases
        "GLY": [],  # No side chain
        "CYS": ["SG"],
    }

    return (
        backbone
        + side_chain.get(residue_name, [])
        + (["CB"] if residue_name != "GLY" else [])
    )


def get_residue_contacts(pdb_path, chain1="A", chain2="B", cutoff_distance=4.0):
    """
    Identify all residue pairs that are in contact across the interface

    Parameters:
    - pose: PyRosetta pose
    - chain1, chain2: Chain identifiers
    - cutoff_distance: Distance cutoff for contacts (Angstroms)

    Returns:
    - Dictionary of contact information
    """
    contacts = defaultdict(list)

    pose = pose_from_pdb(pdb_path)
    missing_atoms = 0

    # Map each chain id to where it starts in the pose, rather than assuming
    # the target occupies positions 1..len(chain1).
    chain_begin = {
        pose.pdb_info().chain(pose.conformation().chain_begin(i)): (
            pose.conformation().chain_begin(i)
        )
        for i in range(1, pose.num_chains() + 1)
    }
    for name, cid in (("chain1", chain1), ("chain2", chain2)):
        if cid not in chain_begin:
            raise ValueError(
                f"{name} {cid!r} is not present in {pdb_path}; "
                f"chains found: {sorted(chain_begin)}"
            )
    chain1_begin = chain_begin[chain1]
    chain2_begin = chain_begin[chain2]

    # Get residues from each chain
    chain1_residues = []
    chain2_residues = []

    # Pre-calculate CB coordinates for quick distance screening
    cb_coords = {}

    for i in range(1, pose.total_residue() + 1):
        chain = pose.pdb_info().chain(i)
        if chain == chain1:
            chain1_residues.append(i)
            cb_coords[i] = get_cb_coordinates(pose.residue(i))
        elif chain == chain2:
            chain2_residues.append(i)
            cb_coords[i] = get_cb_coordinates(pose.residue(i))

    # Create HBond set for hydrogen bond detection
    hbond_set = pose.get_hbonds()

    # Pre-calculate all hydrogen bonds
    hbonds = defaultdict(list)
    for hbond in hbond_set.hbonds():
        don_res = hbond.don_res()
        acc_res = hbond.acc_res()
        if pose.pdb_info().chain(don_res) != pose.pdb_info().chain(acc_res):
            hbonds[(don_res, acc_res)].append(hbond)

    # Extended cutoff for initial screening (to catch all possible interactions)
    extended_cutoff = max(cutoff_distance, 5.0) + 4.0  # Add buffer for side chains

    # Analyze each residue pair, but only if CB atoms are within extended cutoff
    for res1 in chain1_residues:
        res1_obj = pose.residue(res1)
        res1_name = res1_obj.name3()

        for res2 in chain2_residues:
            # Quick CB distance check first
            if np.linalg.norm(cb_coords[res1] - cb_coords[res2]) > extended_cutoff:
                continue

            res2_obj = pose.residue(res2)
            res2_name = res2_obj.name3()

            # Initialize contact info
            contact_types = set()
            min_distance = float("inf")

            # Representative atoms only: backbone plus one or two side chain.
            atoms_to_check1 = get_key_atoms(res1_name)
            atoms_to_check2 = get_key_atoms(res2_name)

            # Check distances between representative atoms
            for atom1 in atoms_to_check1:
                for atom2 in atoms_to_check2:
                    try:
                        distance = np.linalg.norm(
                            np.array(res1_obj.xyz(atom1))
                            - np.array(res2_obj.xyz(atom2))
                        )
                        if distance <= cutoff_distance:
                            min_distance = min(min_distance, distance)
                            contact_types.add("VDW Contact")
                    except KeyError:
                        # Residue type names an atom this pose lacks (a
                        # non-standard residue or a truncated side chain).
                        # Counted and reported once below rather than per atom,
                        # since silently skipping undercounts the interface.
                        missing_atoms += 1
                        continue

            # Check for hydrogen bonds (using pre-calculated hbonds)
            if (res1, res2) in hbonds or (res2, res1) in hbonds:
                contact_types.add("H-bond")
                for hbond in hbonds.get((res1, res2), []) + hbonds.get(
                    (res2, res1), []
                ):
                    don_res = pose.residue(hbond.don_res())
                    acc_res = pose.residue(hbond.acc_res())
                    don_coords = np.array(don_res.xyz(hbond.don_hatm()))
                    acc_coords = np.array(acc_res.xyz(hbond.acc_atm()))
                    hbond_distance = np.linalg.norm(don_coords - acc_coords)
                    min_distance = min(min_distance, hbond_distance)

            # Check for potential salt bridge
            if min_distance <= 4.0:  # Typical salt bridge distance
                if (
                    res1_name in ["ARG", "LYS", "HIS"] and res2_name in ["ASP", "GLU"]
                ) or (
                    res2_name in ["ARG", "LYS", "HIS"] and res1_name in ["ASP", "GLU"]
                ):
                    contact_types.add("Salt-bridge")

            # Check for potential hydrophobic interaction
            hydrophobic = [
                "ALA",
                "VAL",
                "LEU",
                "ILE",
                "MET",
                "PHE",
                "TRP",
                "PRO",
                "TYR",
            ]
            if min_distance <= 5.0:  # Typical hydrophobic interaction distance
                if res1_name in hydrophobic and res2_name in hydrophobic:
                    contact_types.add("Hydrophobic")

            # If any contacts were found, store the information
            if contact_types:
                # Chain-relative and 1-indexed, matching hotspot_region and
                # the 1-indexed CDR positions.
                key = (res1 - chain1_begin + 1, res2 - chain2_begin + 1)
                contacts[key] = {
                    "distance": min_distance,
                    "types": sorted(list(contact_types)),
                }

    if missing_atoms:
        log.info(
            f"Warning: {missing_atoms} atom lookup(s) missing from the pose in "
            f"{pdb_path}; those pairs contributed no VDW contact, so the "
            f"interface contact count is a lower bound."
        )
    return contacts


def find_nearby_residues_from_pdb(
    pdb_path: str,
    target_residues: list[int],
    distance_threshold: float = 6.0,
    chain: str = "A",
) -> list[int]:
    """
    Find residues near target residue(s) within a specified distance threshold
    for a specific chain.

    Parameters:
    -----------
    pdb_path : str
        Path to the PDB file to be loaded
    target_residues : int or List[int]
        Sequence position(s) within the specified chain (1-indexed for that chain)
    distance_threshold : float, optional
        Maximum distance (in Angstroms) to consider a residue as "nearby"
        Default is 3.0 Angstroms
    chain : str, optional
        Chain identifier to analyze. Default is 'A'

    Returns:
    --------
    List of tuples containing:
    - Original target residue number (in chain)
    - Nearby residue sequence position (in chain)
    - Distance from the target residue
    """

    # Load the pose
    try:
        pose = pose_from_pdb(pdb_path)
    except Exception as e:
        raise ValueError(f"Error loading PDB file {pdb_path}: {e}")

    # Normalize input to a list
    if isinstance(target_residues, int):
        target_residues = [target_residues]

    # Get chain mapping
    chain_residues = {}
    current_chain = ""
    current_chain_start = 1

    for i in range(1, pose.total_residue() + 1):
        res_chain = pose.pdb_info().chain(i)

        if res_chain != current_chain:
            if current_chain:
                chain_residues[current_chain] = (current_chain_start, i - 1)
            current_chain = res_chain
            current_chain_start = i

    # Add the last chain
    chain_residues[current_chain] = (current_chain_start, pose.total_residue())

    # Validate chain exists
    if chain not in chain_residues:
        raise ValueError(
            f"Chain {chain} not found in the PDB file. Available chains: {list(chain_residues.keys())}"
        )

    # Get start and end residues for the specified chain
    chain_start, chain_end = chain_residues[chain]

    # Validate input residue numbers for the specific chain
    for res in target_residues:
        if res < 1 or res > (chain_end - chain_start + 1):
            raise ValueError(
                f"Invalid residue number {res} for chain {chain}. Chain has {chain_end - chain_start + 1} residues."
            )

    # List to store nearby residues
    nearby_residues = []

    # Iterate through each target residue
    for target_relative in target_residues:
        # Convert relative chain position to absolute pose position
        target_residue = chain_start + target_relative - 1

        # Get the 3D coordinates of the target residue's CA atom
        target_coords = pose.residue(target_residue).atom("CA").xyz()

        # Iterate through residues in the specified chain
        for res_num in range(chain_start, chain_end + 1):
            # Skip the target residue itself
            if res_num == target_residue:
                nearby_residues.append(res_num - chain_start + 1)
                continue

            # Get CA atom coordinates for the current residue
            current_coords = pose.residue(res_num).atom("CA").xyz()

            # Calculate Euclidean distance
            distance = np.linalg.norm(current_coords - target_coords)

            # Check if within distance threshold
            if distance <= distance_threshold:
                # Convert absolute residue number back to chain-relative
                nearby_residues.append(res_num - chain_start + 1)

    return np.array(nearby_residues)
