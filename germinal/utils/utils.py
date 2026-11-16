"""Germinal Utility Functions Module

This module provides a comprehensive collection of utility functions for the Germinal
protein design system. It includes functions for device management, time formatting,
PDB file processing, structural analysis, secondary structure calculation, and various
protein manipulation operations.

The module implements:
- Device detection and validation (JAX/CUDA)
- Time and timestamp utilities
- PDB file cleaning and manipulation
- Memory management for GPU operations
- Protein structure analysis and hotspot identification
- CDR position computation and framework handling
- Secondary structure analysis with DSSP
- Clash detection and structural validation
- Chain manipulation and complex assembly

Key Function Categories:
    Time/Device Utilities:
        get_clean_time, get_torch_device, get_jax_device

    Memory Management:
        clear_memory

    PDB Processing:
        clean_pdb, get_sequence_from_pdb, create_starting_structure

    Structural Analysis:
        hotspot_residues, calculate_clash_score, calc_ss_percentage

    CDR/Framework Processing:
        compute_cdr_positions, idx_from_ranges

Dependencies:
    - JAX for GPU device detection
    - PyTorch for CUDA validation
    - BioPython for PDB processing and structure analysis
    - NumPy/SciPy for numerical computations
    - DSSP for secondary structure calculation

"""

import math
import gc
from collections import defaultdict
import jax
import torch
import numpy as np
from scipy.spatial import cKDTree
from Bio import PDB
from Bio.PDB import (
    PDBParser,
    DSSP,
    Selection,
)

from Bio.SeqUtils import seq1

from germinal.utils.log import get_logger

log = get_logger(__name__)

def get_clean_time(end_time: float, start_time: float) -> str:
    """Convert elapsed time between timestamps to human-readable format.
    
    Calculates the elapsed time between two timestamps and formats it as
    a human-readable string with hours, minutes, and seconds.
    
    Args:
        end_time (float): End timestamp (typically from time.time())
        start_time (float): Start timestamp (typically from time.time())
        
    Returns:
        str: Formatted time string in format "Xh Ym Zs"
    """
    elapsed_time = end_time - start_time
    hh, remainder = divmod(elapsed_time, 3600)
    mm, ss = divmod(remainder, 60)
    return f"{int(hh)}h {int(mm)}m {int(ss)}s"


def get_torch_device() -> str:
    """Detect and return available PyTorch device.
    
    Checks for CUDA availability and returns the appropriate device
    string for PyTorch operations.
    
    Returns:
        str: 'cuda' if CUDA is available, 'cpu' otherwise
    """
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def clean_pdb(pdb_file):
    """Remove unnecessary information from PDB files for clean processing.
    
    Filters PDB files to retain only essential structural information by
    keeping ATOM, HETATM, MODEL, TER, and END records while removing
    Rosetta-specific annotations and other metadata.
    
    Args:
        pdb_file (str): Path to the PDB file to be cleaned
    """
    # Read the pdb file and filter relevant lines
    with open(pdb_file, "r") as f_in:
        relevant_lines = [
            line
            for line in f_in
            if line.startswith(("ATOM", "HETATM", "MODEL", "TER", "END"))
        ]

    # Write the cleaned lines back to the original pdb file
    with open(pdb_file, "w") as f_out:
        f_out.writelines(relevant_lines)


def clear_memory(clear_jax: bool = True):
    """Clear GPU and system memory caches for optimal performance.
    
    Performs comprehensive memory cleanup including Python garbage collection,
    PyTorch CUDA cache clearing, and JAX cache clearing. Handles cases where
    GPU libraries may not be available gracefully.
    """
    gc.collect()

    try:
        torch.cuda.empty_cache()
    except Exception as e:
        log.warning(f"Torch cache not cleared ({type(e).__name__}: {e})")

    try:
        if clear_jax:
            jax.clear_caches()
    except Exception:
        log.warning("JAX cache not cleared because it is not available")
        pass

    gc.collect()


def get_jax_device():
    """Detect and verify JAX GPU device availability.
    
    Checks for JAX GPU devices and returns available GPU device list.
    Provides comprehensive GPU availability validation for JAX operations.
    
    Returns:
        list or None: List of available JAX GPU devices, or None if there are
            none or JAX is not importable. Never raises: the caller decides
            whether a CPU-only environment is fatal.
    """

    try:
        devices = jax.devices()
        gpu_devices = [d for d in devices if d.platform == "gpu"]
        if gpu_devices:
            log.info(f"JAX GPU devices available: {gpu_devices}")
            return gpu_devices
        else:
            raise ValueError("No JAX GPU devices found")
    except Exception:
        log.warning("JAX not available")
        return None


def copy_dict(d: dict) -> dict:
    """Create a deep copy of a dictionary with all nested structures.
    
    Performs deep copying of a dictionary to ensure complete independence
    from the original, including all nested dictionaries, lists, and objects.
    
    Args:
        d (dict): Dictionary to be deep copied
        
    Returns:
        dict: Independent deep copy of the input dictionary
    """
    import copy

    return copy.deepcopy(d)


def idx_from_ranges(ranges, chain="B", offset=0):
    """Convert range string specification to zero-based index list.

    Parses a range string (e.g. 'B1-10,B15' or '1-10,15') into zero-based
    indices within a single chain.

    The returned indices are positions *within one chain*, so a specification
    spanning two chains is ambiguous and raises.

    Args:
        ranges (str): Range specification, e.g. 'B1-10,B15' or '1-10,15'.
            Chain letters may be on both endpoints of a range ('B30-B33').
        chain (str | list | None, optional): Accepted chain id(s); a
            comma-separated string such as 'A,C' is split. None accepts
            whichever single chain the specification uses. Defaults to 'B'.
        offset (int, optional): Offset applied to every residue number.
            Defaults to 0.

    Returns:
        list: Zero-based indices corresponding to the specified ranges.

    Raises:
        ValueError: if a chain letter is not among the accepted chains, or if
            the specification spans more than one chain.
    """
    if chain is None:
        allowed = None
    elif isinstance(chain, (list, tuple)):
        allowed = {str(c).strip() for c in chain if str(c).strip()}
    else:
        allowed = {c.strip() for c in str(chain).split(",") if c.strip()}

    def split_chain(token):
        """Return (chain_letter, residue_number_str) for one endpoint."""
        token = token.strip()
        if token and token[0].isalpha():
            return token[0], token[1:]
        return "", token

    rows = []
    seen_chains = set()
    for part in ranges.split(","):
        # score_interface returns "" when nothing is within the cutoff, which
        # is routine, and "".split(",") yields [""] -- so skip empty parts.
        part = part.strip()
        if not part:
            continue

        if "-" in part:
            left, right = part.split("-", 1)
            c_left, n_left = split_chain(left)
            c_right, n_right = split_chain(right)
            part_chains = {c for c in (c_left, c_right) if c}
            start, end = int(n_left) + offset, int(n_right) + offset
            indices = range(start - 1, end)
        else:
            c, n = split_chain(part)
            part_chains = {c} if c else set()
            indices = [int(n) + offset - 1]

        if allowed is not None:
            bad = part_chains - allowed
            if bad:
                raise ValueError(
                    f"idx_from_ranges: {ranges!r} names chain(s) "
                    f"{sorted(bad)}, which are not among the accepted chains "
                    f"{sorted(allowed)}. Resolving them would silently map "
                    f"their positions onto the first chain."
                )
        seen_chains |= part_chains
        rows.extend(indices)

    if len(seen_chains) > 1:
        raise ValueError(
            f"idx_from_ranges: {ranges!r} spans chains {sorted(seen_chains)}. "
            f"Indices are positions within a single chain, so a multi-chain "
            f"specification is ambiguous -- the residues preceding the second "
            f"chain are unknown here. Split the call per chain."
        )

    # Return the selected rows
    return rows


def compute_cdr_positions(
    cdr_lengths: list[int], framework_lengths: list[int]
) -> list[int]:
    """Compute CDR residue positions from framework and CDR length specifications.
    
    Calculates the absolute residue positions for all CDRs based on framework
    region lengths and CDR lengths. Uses cumulative positioning to account for
    the sequential arrangement of framework and CDR regions.
    
    Args:
        cdr_lengths (list[int]): List of CDR lengths in sequential order
        framework_lengths (list[int]): List of framework region lengths
            corresponding to each CDR position
            
    Returns:
        list[int]: Flat list of all CDR residue positions (zero-based indices)
    """

    cumulative = 0
    positions = []
    for i, cdr_length in enumerate(cdr_lengths):
        fw_len = framework_lengths[i] + cumulative
        positions.extend(range(fw_len, fw_len + cdr_length))
        cumulative = fw_len + cdr_length
    return positions


def cdr3_positions(run_settings: dict) -> list:
    """Return the zero-based positions of CDR3, the loop that dominates binding.

    For ``nb`` CDR3 is the last of three CDRs; for ``scfv`` it is the third
    of six (CDR-H3), not the last, which is the shorter CDR-L3.

    Args:
        run_settings: Run configuration, read for ``type``, ``cdr_positions``
            and ``cdr_lengths``.

    Returns:
        list[int]: Zero-based residue positions of CDR3. Callers that need PDB
        numbering add 1.

    Raises:
        ValueError: If ``type`` is neither nb nor scfv.
    """
    binder_type = run_settings["type"].lower()
    cdr_positions = run_settings["cdr_positions"]
    cdr_lengths = run_settings["cdr_lengths"]

    if binder_type == "nb":
        return list(cdr_positions[sum(cdr_lengths[:-1]) :])
    if "scfv" in binder_type:
        return list(cdr_positions[sum(cdr_lengths[:2]) : sum(cdr_lengths[:3])])
    raise ValueError(
        f"Type {run_settings['type']} not supported, select either nb or scfv"
    )


def resolve_failure_stage(filter_data: dict, default: str) -> str:
    """Pick the failure_stage label for a rejected design.

    A stage already in the filter data is more specific than the caller's
    default, so it wins. NaN counts as absent, not as a stage: a plain ``or``
    is wrong because NaN is truthy, and a DataFrame round-trip produces NaN.

    Args:
        filter_data (dict): Merged filter metrics and results for the design.
        default (str): Stage to use when the filter data names none.

    Returns:
        str: The stage label to record.
    """
    stage = filter_data.get("failure_stage")
    if stage is None:
        return default
    if isinstance(stage, float) and math.isnan(stage):
        return default
    return stage or default


def build_final_candidates(
    design_name: str,
    trajectory_sequence: str,
    abmpnn_sequences: list,
    include_codesign: bool,
) -> list:
    """Assemble the sequences to send through the final filters.

    Proposals are named ``_abmpnn_1``..``_abmpnn_N`` in ascending MPNN score,
    so ``_abmpnn_1`` is the best. The co-designed sequence always competes;
    ``include_codesign`` only picks which mechanism adds it, since the
    recycling path already scored it as one of its own candidates.

    Args:
        design_name (str): Trajectory design name, used as the name prefix.
        trajectory_sequence (str): The co-designed binder sequence.
        abmpnn_sequences (list): Dicts with ``seq``, ``score`` and ``seqid``,
            as returned by the redesign pipeline.
        include_codesign (bool): True to prepend the co-designed candidate as
            ``_abmpnn_0``; False when the caller already has it in
            ``abmpnn_sequences``.

    Returns:
        list: Dicts with ``name``, ``seq``, ``score``, ``seqid`` and ``source``
            (``"codesign"`` or ``"abmpnn"``).
    """
    candidates = []
    if include_codesign:
        candidates.append(
            {
                "name": f"{design_name}_abmpnn_0",
                "seq": trajectory_sequence,
                "score": None,
                "seqid": None,
                "source": "codesign",
            }
        )
    for j, seq_data in enumerate(abmpnn_sequences):
        candidates.append(
            {
                "name": f"{design_name}_abmpnn_{j + 1}",
                "seq": seq_data["seq"],
                "score": seq_data["score"],
                "seqid": seq_data["seqid"],
                "source": "abmpnn",
            }
        )
    return candidates


######### BIOPYTHON UTILS #########


def get_sequence_from_pdb(pdb_path: str) -> dict:
    """Extract protein sequences from PDB file organized by chain.
    
    Parses a PDB file and extracts the amino acid sequence for each chain,
    converting three-letter amino acid codes to single-letter codes.
    
    Only standard residues are included. Waters and HETATM groups are
    dropped with a warning, including modified residues such as MSE, which
    would otherwise become ``X`` in the sequence handed to AlphaFold3.

    Note this also drops modified residues recorded as HETATM, such as MSE;
    convert those to their standard parent before use.

    Args:
        pdb_path (str): Path to the PDB file to process
        
    Returns:
        dict: Dictionary mapping chain identifiers to their amino acid sequences
            in single-letter code format
    """
    parser = PDBParser()
    structure = parser.get_structure("protein", pdb_path)
    chains = {}
    for chain in structure.get_chains():
        standard, skipped = [], []
        for residue in chain:
            # hetflag is " " for standard residues, "W" for water, "H_XXX" for
            # every other heterogen.
            if residue.id[0] == " ":
                standard.append(residue.resname)
            else:
                skipped.append(residue.resname)
        if skipped:
            counts = {name: skipped.count(name) for name in sorted(set(skipped))}
            log.info(
                f"Warning: chain {chain.id} of {pdb_path} contains "
                f"{len(skipped)} non-standard residue(s) {counts}; excluded "
                f"from the sequence. Germinal does not model waters or glycans."
            )
        chains[chain.id] = seq1("".join(standard))
    return chains


def create_starting_structure(
    save_path,
    binder_pdb,
    target_pdb,
    binder_chain="B",
    start_binder_chain="A",
    target_chain="A",
):
    """Create combined PDB structure from separate binder and target files.
    
    Combines a binder PDB and target PDB into a single complex structure,
    handling chain renaming and multi-chain targets. Creates the starting
    structure needed for protein design optimization.
    
    Args:
        save_path (str): Path where the combined structure will be saved
        binder_pdb (str): Path to the binder PDB structure
        target_pdb (str): Path to the target PDB structure
        binder_chain (str, optional): Chain ID for binder in output. Defaults to 'B'.
        start_binder_chain (str, optional): Chain ID of binder in input file. Defaults to 'A'.
        target_chain (str, optional): Chain ID(s) for target (comma-separated for multiple). Defaults to 'A'.
        
    Returns:
        str: Amino acid sequence of the binder chain in single-letter code
    """
    target_chain = target_chain.split(",")

    # Set up the parser and structure objects
    parser = PDB.PDBParser(QUIET=True)
    # Load the two structures
    structure1 = parser.get_structure("structure1", target_pdb)
    structure2 = parser.get_structure("structure2", binder_pdb)
    # Create a new structure to hold the combined molecules
    combined = PDB.Structure.Structure("combined")
    # Create a model in the new structure
    model = PDB.Model.Model(0)
    combined.add(model)
    # Add chain A from first structure
    chainA = structure1[0][target_chain[0]]
    model.add(chainA)
    # Add chain from second structure as chain B
    chainB = structure2[0][start_binder_chain]
    chainB.id = binder_chain  # Rename the chain to B
    model.add(chainB)
    for chain_id in target_chain[1:]:
        chain = structure1[0][chain_id]
        chain.id = chain_id
        model.add(chain)
    # Save the combined structure
    io = PDB.PDBIO()
    io.set_structure(combined)
    io.save(save_path)
    log.info(f"Created starting PDB at: {save_path}")

    return get_sequence_from_pdb(save_path)[binder_chain]


three_to_one_map = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}


def hotspot_residues(
    trajectory_pdb, binder_chain="B", atom_distance_cutoff=4.0, target_chain="A", num_targets=1
):
    """Identify interface hotspot residues between binder and target.
    
    Analyzes a protein complex structure to identify binder residues that
    are in close contact with the target protein, defining the binding interface.
    Uses spatial proximity analysis with KD-trees for efficient computation.
    
    Args:
        trajectory_pdb (str): Path to the PDB file containing the protein complex
        binder_chain (str, optional): Chain identifier for the binder protein. Defaults to 'B'.
        atom_distance_cutoff (float, optional): Maximum distance (Å) for interface contacts. Defaults to 4.0.
        target_chain (str, optional): Chain identifier for the target protein. Defaults to 'A'.
        num_targets (int, optional): Number of targets in the complex. Defaults to 1.
    Returns:
        dict: Dictionary mapping binder residue numbers to single-letter amino acid codes
            for all residues with atoms within the distance cutoff of the target
    """
    parser = PDBParser(QUIET=True)
    interacting_residues_list = []
    for i in range(num_targets):
        if num_targets == 1:
            pdb_file = trajectory_pdb
        else:
            if "." in trajectory_pdb:
                base, ext = trajectory_pdb.rsplit(".", 1)
                pdb_file = f"{base}_target{i}.{ext}"
            else:
                pdb_file = f"{trajectory_pdb}_target{i}"

        structure = parser.get_structure("complex", pdb_file)

        binder_atoms = Selection.unfold_entities(structure[0][binder_chain], "A")
        binder_coords = np.array([atom.coord for atom in binder_atoms])

        target_atoms = Selection.unfold_entities(structure[0][target_chain], "A")
        target_coords = np.array([atom.coord for atom in target_atoms])

        binder_tree = cKDTree(binder_coords)
        target_tree = cKDTree(target_coords)

        interacting_residues = {}

        pairs = binder_tree.query_ball_tree(target_tree, atom_distance_cutoff)

        for binder_idx, close_indices in enumerate(pairs):
            binder_residue = binder_atoms[binder_idx].get_parent()
            binder_resname = binder_residue.get_resname()

            if binder_resname in three_to_one_map:
                aa_single_letter = three_to_one_map[binder_resname]
                for close_idx in close_indices:
                    interacting_residues[binder_residue.id[1]] = aa_single_letter

        interacting_residues_list.append(interacting_residues)

    # Return the minimum number of residues (by count) across all targets
    if interacting_residues_list:
        min_residues = min(interacting_residues_list, key=lambda d: len(d))
        return min_residues
    else:
        return {}


def calculate_clash_score(pdb_file, threshold, only_ca=False, num_targets=1):
    """Calculate structural clash score for protein structure validation.
    
    Analyzes a protein structure to identify and count atomic clashes based on
    distance thresholds. Provides options for CA-only analysis or full atomic
    clash detection with proper exclusions for bonded and sequential residues.
    
    Args:
        pdb_file (str): Path to the PDB file to analyze
        threshold (float): Distance threshold (Å) below which atoms are
            considered clashing. Required: callers pass
            run_settings["clash_threshold"] so the design-stage gate and the
            filter-stage metric use the same value.
        only_ca (bool, optional): If True, only analyze CA-CA distances.
            If False, analyze all heavy atoms. Defaults to False.
            
    Returns:
        int: Number of clashing atom pairs found in the structure
    """
    structures = []
    parser = PDBParser(QUIET=True)
    if num_targets == 1:
        structures.append(parser.get_structure("protein", pdb_file))
    else:
        for i in range(num_targets):
            if "." in pdb_file:
                base, ext = pdb_file.rsplit(".", 1)
                target_file = f"{base}_target{i}.{ext}"
            else:
                target_file = f"{pdb_file}_target{i}"
            structures.append(parser.get_structure("protein", target_file))

    total_pairs = []
    for structure in structures:
        atoms = []
        atom_info = []  # Detailed atom info for debugging and processing
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        if atom.element == "H":  # Skip hydrogen atoms
                            continue
                        if only_ca and atom.get_name() != "CA":
                            continue
                        atoms.append(atom.coord)
                        atom_info.append(
                            (chain.id, residue.id[1], atom.get_name(), atom.coord)
                        )

        tree = cKDTree(atoms)
        pairs = tree.query_pairs(threshold)

        valid_pairs = set()
        for i, j in pairs:
            chain_i, res_i, name_i, coord_i = atom_info[i]
            chain_j, res_j, name_j, coord_j = atom_info[j]

            # Exclude clashes within the same residue
            if chain_i == chain_j and res_i == res_j:
                continue

            # Exclude directly sequential residues in the same chain for all atoms
            if chain_i == chain_j and abs(res_i - res_j) == 1:
                continue

            # If calculating sidechain clashes, only consider clashes between different chains
            if not only_ca and chain_i == chain_j:
                continue

            valid_pairs.add((i, j))
    
        # sum across all targets
        total_pairs.append(len(valid_pairs))

    return max(total_pairs)


def calc_ss_percentage(
    pdb_file,
    advanced_settings,
    chain_id="B",
    atom_distance_cutoff=4.0,
    return_dict=False,
    target_chain="A",
):
    """Calculate secondary structure percentages with interface analysis.
    
    Analyzes protein secondary structure using DSSP and calculates percentages
    of helix, sheet, and loop content for both the entire chain and interface
    residues. Also computes confidence metrics (pLDDT) for different regions.
    
    Args:
        pdb_file (str): Path to the PDB file for analysis
        advanced_settings (dict): Settings dictionary containing 'dssp_path' key
        chain_id (str, optional): Chain to analyze. Defaults to 'B'.
        atom_distance_cutoff (float, optional): Distance cutoff (Å) for interface
            definition. Defaults to 4.0.
        return_dict (bool, optional): If True, return dict format. If False,
            return tuple format. Defaults to False.
        target_chain (str, optional): Target chain for interface calculation. Defaults to 'A'.
        
    Returns:
        dict or tuple: Secondary structure percentages and confidence metrics.
            If return_dict=True: Dictionary with keys 'alpha_', 'beta_', 'loops_',
                'alpha_i', 'beta_i', 'loops_i', 'i_plddt', 'ss_plddt'
            If return_dict=False: Tuple (alpha%, beta%, loop%, alpha_i%, beta_i%,
                loop_i%, i_plddt, ss_plddt)
    """
    # Parse the structure
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    model = structure[0]  # Consider only the first model in the structure

    # Calculate DSSP for the model
    dssp = DSSP(model, pdb_file, dssp=advanced_settings["dssp_path"])

    # Prepare to count residues
    ss_counts = defaultdict(int)
    ss_interface_counts = defaultdict(int)
    plddts_interface = []
    plddts_ss = []

    # Get chain and interacting residues once
    chain = model[chain_id]
    interacting_residues = set(
        hotspot_residues(
            pdb_file, chain_id, atom_distance_cutoff, target_chain=target_chain
        ).keys()
    )

    for residue in chain:
        residue_id = residue.id[1]
        if (chain_id, residue_id) in dssp:
            ss = dssp[(chain_id, residue_id)][2]  # Get the secondary structure
            ss_type = "loop"
            if ss in ["H", "G", "I"]:
                ss_type = "helix"
            elif ss == "E":
                ss_type = "sheet"

            ss_counts[ss_type] += 1

            if ss_type != "loop":
                # calculate secondary structure normalised pLDDT
                avg_plddt_ss = sum(atom.bfactor for atom in residue) / len(residue)
                plddts_ss.append(avg_plddt_ss)

            if residue_id in interacting_residues:
                ss_interface_counts[ss_type] += 1

                # calculate interface pLDDT
                avg_plddt_residue = sum(atom.bfactor for atom in residue) / len(residue)
                plddts_interface.append(avg_plddt_residue)

    # Calculate percentages
    total_residues = sum(ss_counts.values())
    total_interface_residues = sum(ss_interface_counts.values())

    percentages = calculate_percentages(
        total_residues, ss_counts["helix"], ss_counts["sheet"]
    )
    interface_percentages = calculate_percentages(
        total_interface_residues,
        ss_interface_counts["helix"],
        ss_interface_counts["sheet"],
    )

    i_plddt = (
        round(sum(plddts_interface) / len(plddts_interface) / 100, 2)
        if plddts_interface
        else 0
    )
    ss_plddt = round(sum(plddts_ss) / len(plddts_ss) / 100, 2) if plddts_ss else 0

    if return_dict:
        return {
            "alpha_": percentages[0],
            "beta_": percentages[1],
            "loops_": percentages[2],
            "alpha_i": interface_percentages[0],
            "beta_i": interface_percentages[1],
            "loops_i": interface_percentages[2],
            "i_plddt": i_plddt,
            "ss_plddt": ss_plddt,
        }
    else:
        return (*percentages, *interface_percentages, i_plddt, ss_plddt)


def calculate_percentages(total, helix, sheet):
    """Calculate secondary structure percentages from counts.
    
    Converts raw counts of secondary structure elements to percentages
    with proper handling of zero totals.
    
    Args:
        total (int): Total number of residues
        helix (int): Number of helical residues
        sheet (int): Number of sheet residues
        
    Returns:
        tuple: (helix_percentage, sheet_percentage, loop_percentage) rounded to 2 decimals
    """
    helix_percentage = round((helix / total) * 100, 2) if total > 0 else 0
    sheet_percentage = round((sheet / total) * 100, 2) if total > 0 else 0
    loop_percentage = (
        round(((total - helix - sheet) / total) * 100, 2) if total > 0 else 0
    )

    return helix_percentage, sheet_percentage, loop_percentage


def interface_cdrs(interface: str, cdrs: str) -> float:
    """Calculate CDR involvement in binding interface.
    
    Analyzes the overlap between interface residues and CDR regions to
    determine what fraction of the interface is composed of CDR residues.
    
    Args:
        interface (str): Interface residue specification string
        cdrs (str): All CDR residue positions

    Returns:
        float: Fraction of interface residues that fall in a CDR.
    """
    interface = idx_from_ranges(interface, chain=None)
    # cdrs = idx_from_ranges(cdrs)
    # cdr3 = idx_from_ranges(cdr3)

    # No interface residues: report zero rather than dividing by zero. Such a
    # design fails its filters anyway.
    if len(interface) == 0:
        return 0.0

    common_elements = [element for element in interface if element in cdrs]
    return len(common_elements) / len(interface)


def extract_ipsae(ipsae_raw) -> dict:
    """Normalise the ipsae value returned by the AF3 batch prediction.

    Can be a dict (from AF3), a float, or None.
    """
    if ipsae_raw is None:
        # A failed ipSAE is not a legitimate 0.0. Keep the 0.0 so the design
        # is rejected, but warn, so a tool failure is not read as weak binding.
        log.warning(
            "ipSAE unavailable for this prediction, scoring it 0.0 "
            "(design will be rejected). If every design reports ipsae=0.0, "
            "suspect the ipsae tool, not the designs."
        )
        return {"ipsae": 0.0, "pdockq2": 0.0, "lis": 0.0}
    if isinstance(ipsae_raw, dict):
        return {
            "ipsae": float(ipsae_raw.get("ipsae", 0.0)),
            "pdockq2": float(ipsae_raw.get("pdockq2", 0.0)),
            "lis": float(ipsae_raw.get("LIS", ipsae_raw.get("lis", 0.0))),
        }
    # scalar
    return {"ipsae": float(ipsae_raw), "pdockq2": 0.0, "lis": 0.0}


def stage_keys(metrics: dict, stage: str) -> dict:
    """Prefix recorded metric names with the stage that produced them.

    Columns read "<stage>.<agg>_<metric>" for a cross-leg aggregate and
    "<stage>.<metric>.<target>" for one leg.
    """
    return {f"{stage}.{k}": v for k, v in metrics.items()}

def take(queue, n):
    """Pop up to n items off the front of queue, mutating it."""
    taken = queue[:n]
    del queue[:n]
    return taken

def gate_counter(stage: str, reason: str) -> str:
    """Name a failure_counts.csv counter that is NOT a filter failure.

    A leaf starting with "_" marks it, and must not be summed with filters.
    """
    return f"{stage}._{reason}"

def chunks_by_candidate_count(slots, max_candidates):
    """Group slots so each chunk folds at most max_candidates candidates.

    A slot is never split: its candidates are scored together and its
    any_accepted is decided across all of them. One whose own candidates
    exceed the cap gets a chunk to itself. A cap of None or 0 means one chunk.
    """
    if not slots:
        return []
    cap = int(max_candidates or 0)
    if cap <= 0:
        return [list(slots)]

    chunks, current, count = [], [], 0
    for slot in slots:
        n = len(slot["candidates"])
        if current and count + n > cap:
            chunks.append(current)
            current, count = [], 0
        current.append(slot)
        count += n
    if current:
        chunks.append(current)
    return chunks
