"""
Multi-state AbMPNN redesign for multi-target binder design.

Uses the Kuhlman-Lab ProteinMPNN fork (external/proteinmpnn) which supports
multi-state design (MSD) — sampling sequences that are simultaneously scored
against multiple backbone states with configurable weights.

Every target complex is provided as a separate state
with equal weight (1.0), and the binder residues are tied across states so that
a single binder sequence is optimized to bind both targets.

Interface residues are computed per target, and their union (residues contacting
ANY target) is fixed. This preserves good contacts found for either target while
still allowing optimization of non-interface CDR positions.
"""

import os
import re
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Tuple

from germinal.utils.utils import hotspot_residues

from germinal.utils.gpu import child_env
from germinal.utils.log import get_logger

log = get_logger(__name__)


# Paths to wrapper scripts (apply patches on top of upstream proteinmpnn submodule)
_WRAPPERS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "wrappers",
)
GENERATE_JSON_SCRIPT = os.path.join(_WRAPPERS_DIR, "generate_json.py")
RUN_MPNN_SCRIPT = os.path.join(_WRAPPERS_DIR, "run_protein_mpnn.py")


def _get_interface_residues(pdb_path, binder_chain, target_chain, atom_distance_cutoff):
    """Get set of binder residue numbers (1-indexed) at the interface."""
    interface = hotspot_residues(
        pdb_path, binder_chain, target_chain=target_chain,
        atom_distance_cutoff=atom_distance_cutoff,
    )
    return set(interface.keys())


def _build_designable_res_string(target_names, binder_chain, designable_positions):
    """Build the --designable_res string in multi-state format.

    Format: target_0:{chain}{pos1},{chain}{pos2},...;target_1:{chain}{pos1},...
    Each state has the same designable positions (same binder, different targets).
    """
    pos_str = ",".join(f"{binder_chain}{pos}" for pos in sorted(designable_positions))
    state_specs = [f"{name}:{pos_str}" for name in target_names]
    return ";".join(state_specs)


def _group_contiguous(positions):
    """Group sorted positions into contiguous ranges.

    Returns list of (start, end) tuples for contiguous runs.
    E.g. [30, 31, 32, 48, 49] -> [(30, 32), (48, 49)]
    """
    sorted_pos = sorted(positions)
    if not sorted_pos:
        return []
    ranges = []
    start = sorted_pos[0]
    end = sorted_pos[0]
    for pos in sorted_pos[1:]:
        if pos == end + 1:
            end = pos
        else:
            ranges.append((start, end))
            start = pos
            end = pos
    ranges.append((start, end))
    return ranges


def _build_constraints_string(target_names, binder_chain, designable_positions, weight=1.0):
    """Build the --constraints string for multi-state design.

    Each contiguous range of designable positions becomes its own constraint group.
    Within each group, states are comma-separated; groups are semicolon-separated.

    Format per group: target_0:{chain}{start}-{chain}{end}:{weight},target_1:{chain}{start}-{chain}{end}:{weight}
    Single positions use: target_0:{chain}{pos}:{weight},target_1:{chain}{pos}:{weight}
    """
    ranges = _group_contiguous(designable_positions)
    constraint_groups = []
    for start, end in ranges:
        if start == end:
            res_spec = f"{binder_chain}{start}"
        else:
            res_spec = f"{binder_chain}{start}-{binder_chain}{end}"
        state_constraints = [f"{name}:{res_spec}:{weight}" for name in target_names]
        constraint_groups.append(",".join(state_constraints))
    return ";".join(constraint_groups)


def _parse_main_fasta(fasta_path):
    """Parse the main msd.fasta to get combined scores and seq_recovery.

    Returns list of dicts with keys: score, seq_recovery (skipping the native entry).
    """
    entries = []
    with open(fasta_path) as f:
        lines = f.readlines()

    for i in range(0, len(lines), 2):
        header = lines[i].strip()
        if not header.startswith(">"):
            continue
        # Skip native sequence header (has "fixed_chains=" not "sample=")
        if "sample=" not in header:
            continue
        # Parse: >T=0.1, sample=0, score=0.9111, seq_recovery=0.4746
        score_match = re.search(r"score=([\d.]+)", header)
        seqrec_match = re.search(r"seq_recovery=([\d.]+)", header)
        sample_match = re.search(r"sample=(\d+)", header)
        if score_match is None:
            # The score is an NLL, so 0.0 would sort best: use inf, which
            # sorts last and trips the recycle gate.
            log.warning(f"no score= in MPNN header, scoring as inf: {header}")
        entries.append({
            "sample": int(sample_match.group(1)) if sample_match else i // 2,
            "score": float(score_match.group(1)) if score_match else float("inf"),
            "seq_recovery": float(seqrec_match.group(1)) if seqrec_match else 0.0,
        })
    return entries


def _assert_binder_chain_is_last(binder_chain, target_chain):
    """Guard the positional chain assumption in _parse_state_fasta.

    The per-state FASTA lists chains in order and the binder is read as the
    last one, which no config enforces. Reversing the two would silently
    return the antigen sequence as the designed binder.

    Raises:
        ValueError: If the binder chain does not sort after the target chain.
    """
    if binder_chain is None or target_chain is None:
        return
    first_target = target_chain[0] if isinstance(target_chain, (list, tuple)) else target_chain
    if str(binder_chain) <= str(first_target):
        raise ValueError(
            f"binder_chain {binder_chain!r} does not sort after target_chain "
            f"{first_target!r}. The multi-state redesign reads the binder as "
            f"the last chain of each per-state FASTA, so this ordering would "
            f"return the antigen sequence as the designed binder. Rename the "
            f"chains so the binder sorts last (the shipped configs use target "
            f"'A', binder 'B')."
        )


def _parse_state_fasta(fasta_path, binder_chain):
    """Parse a per-state FASTA to get per-state scores and binder sequences.

    Returns list of dicts with keys: seq, state_score, sample (skipping native entry).
    """
    entries = []
    with open(fasta_path) as f:
        lines = f.readlines()

    for i in range(0, len(lines), 2):
        header = lines[i].strip()
        seq_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if not header.startswith(">"):
            continue
        # Skip native sequence header
        if "sample=" not in header:
            continue
        # Parse: >{name}, sample=0, state_score=0.9142, scores_per_chain={...}, ...
        state_score_match = re.search(r"state_score=([\d.]+)", header)
        sample_match = re.search(r"sample=(\d+)", header)

        # Chains are separated by "/", listed in original order as
        # [target_chain, binder_chain].
        chain_seqs = seq_line.split("/")
        # The binder is taken positionally as the last chain;
        # _assert_binder_chain_is_last checks that at the call site.
        binder_seq = chain_seqs[-1] if len(chain_seqs) > 1 else chain_seqs[0]

        if state_score_match is None:
            # inf, not 0.0, for the same reason as in _parse_main_fasta: this
            # is an NLL and 0.0 is its best value.
            log.warning(f"no state_score= in MPNN header, scoring as inf: {header}")
        entries.append({
            "sample": int(sample_match.group(1)) if sample_match else i // 2,
            "seq": binder_seq,
            "state_score": float(state_score_match.group(1)) if state_score_match else float("inf"),
        })
    return entries


# Failure reasons, returned third. Everything but NO_DESIGNABLE_POSITIONS
# means a broken tool, not a design with nothing to propose.
NO_DESIGNABLE_POSITIONS = "no_designable_positions"
TOOL_FAILURE_REASONS = frozenset({
    "generate_json_failed",
    "msd_pdb_missing",
    "json_specs_missing",
    "proteinmpnn_failed",
    "main_fasta_missing",
    "no_sequences_generated",
    "no_sequences_assembled",
})


def redesign(
    trajectory_pdbs_af: List[str],
    target_chain: str,
    binder_chain: str,
    run_settings: Dict[str, Any],
    atom_distance_cutoff: float = 3.0,
    exclude_sequences: set = None,
    max_sequences: int = None,
) -> Tuple[List[Dict[str, Any]], bool, str]:
    """Multi-state AbMPNN redesign pipeline for multi-target binder design.

    Uses the Kuhlman-Lab ProteinMPNN to jointly sample binder sequences against
    both target complexes. Interface residues contacting ANY target are fixed
    (union); remaining CDR positions are designable.

    Args:
        trajectory_pdbs_af: List of PDB paths (one per target, each containing
            the target-binder complex from hallucination/AF2).
        target_chain: Target chain identifier (e.g., 'A').
        binder_chain: Binder chain identifier (e.g., 'B').
        run_settings: Dictionary containing run settings including:
            - cdr_positions: CDR residue positions (0-indexed)
            - num_seqs: Number of sequences to sample
            - max_mpnn_sequences: Max sequences to return
            - sampling_temp: Sampling temperature
            - backbone_noise: Backbone noise for MPNN
            - omit_AAs: Amino acids to omit (e.g., "C")
            - mpnn_model_name: Model weight name (default "abmpnn")
        atom_distance_cutoff: Distance cutoff for interface residue detection.
        exclude_sequences: Binder sequences already tested, to be skipped.

    Returns:
        Tuple of (sequences, success_flag, failure_reason):
            - sequences: List of dicts with keys: seq, score, seqid
            - success_flag: True if redesign produced sequences
    """
    assert len(trajectory_pdbs_af) >= 2, "multi-state redesign requires at least 2 target PDBs"

    _assert_binder_chain_is_last(binder_chain, target_chain)

    # No blanket handler: redesign runs only on designs that already passed
    # the initial filter, so a failure here is systematic. Let it propagate.
    return _redesign(
        trajectory_pdbs_af, target_chain, binder_chain,
        run_settings, atom_distance_cutoff, exclude_sequences,
        max_sequences,
    )


def _redesign(
    trajectory_pdbs_af, target_chain, binder_chain,
    run_settings, atom_distance_cutoff, exclude_sequences=None,
    max_sequences=None,
):
    """Core implementation of multi-state redesign."""

    # CDR positions are 0-indexed in run_settings, convert to 1-indexed PDB numbering
    cdr_positions_1indexed = set(pos + 1 for pos in run_settings["cdr_positions"])

    # Both mpnn_fix_interface and atom_distance_cutoff are required; a zero
    # cutoff with the flag on means "not yet", which is the recycle ratchet.
    fix_interface = run_settings.get("mpnn_fix_interface", False)

    if fix_interface and atom_distance_cutoff > 0.0:
        # ========================== Interface residue intersection ==========================
        interface_sets = []
        for pdb_path in trajectory_pdbs_af:
            iface = _get_interface_residues(
                pdb_path, binder_chain, target_chain, atom_distance_cutoff
            )
            interface_sets.append(iface)
            log.info(f"Interface residues for {os.path.basename(pdb_path)}: {sorted(iface)}")

        # Union: fix residues contacting ANY target
        interface_union = set()
        for s in interface_sets:
            interface_union = interface_union | s
        log.info(f"Interface union (fixed): {sorted(interface_union)}")

        # Fixed = framework (non-CDR) + interface union
        # Designable = CDR positions that are NOT in the interface union
        designable_positions = cdr_positions_1indexed - interface_union

        if not designable_positions:
            log.warning("No designable positions after interface intersection filtering")
            return [], False, "no_designable_positions"

        log.info(f"CDR positions (1-indexed): {sorted(cdr_positions_1indexed)}")
    else:
        designable_positions = cdr_positions_1indexed

    # ========================== Prepare temp directory with PDBs ==========================
    with tempfile.TemporaryDirectory() as work_dir:
        target_names = []
        for i, pdb_path in enumerate(trajectory_pdbs_af):
            target_name = f"target_{i}"
            target_names.append(target_name)
            shutil.copy2(pdb_path, os.path.join(work_dir, f"{target_name}.pdb"))

        # ========================== Build multi-state arguments ==========================
        designable_res_str = _build_designable_res_string(
            target_names, binder_chain, designable_positions
        )
        constraints_str = _build_constraints_string(
            target_names, binder_chain, designable_positions, weight=1.0
        )

        log.info(f"Designable res: {designable_res_str}")
        log.info(f"Constraints: {constraints_str}")

        # ========================== Step 1: Generate JSON specs ==========================
        json_specs_path = os.path.join(work_dir, "proteinmpnn_res_specs.json")
        generate_cmd = [
            "python", GENERATE_JSON_SCRIPT,
            "--pdb_dir", work_dir,
            "--designable_res", designable_res_str,
            "--constraints", constraints_str,
            "--multi_state",
            "--out_path", json_specs_path,
        ]

        log.info("Running generate_json.py...")
        result = subprocess.run(
            generate_cmd, capture_output=True, text=True, cwd=work_dir,
        )
        if result.returncode != 0:
            log.warning(f"generate_json.py failed:\nstdout: {result.stdout[-500:]}\nstderr: {result.stderr[-500:]}")
            return [], False, "generate_json_failed"

        # Verify outputs exist
        msd_dir = os.path.join(work_dir, "msd")
        msd_pdb = os.path.join(msd_dir, "msd.pdb")
        if not os.path.exists(msd_pdb):
            log.error(f"msd.pdb not created at {msd_pdb}")
            return [], False, "msd_pdb_missing"
        if not os.path.exists(json_specs_path):
            log.error(f"JSON specs not created at {json_specs_path}")
            return [], False, "json_specs_missing"

        # ========================== Step 2: Run ProteinMPNN ==========================
        output_dir = os.path.join(work_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        model_name = run_settings.get("mpnn_model_name", "abmpnn")
        num_seqs = run_settings.get("num_seqs", 40)
        sampling_temp = run_settings.get("sampling_temp", 0.1)
        backbone_noise = run_settings.get("backbone_noise", 0.0)
        omit_aas = run_settings.get("omit_AAs", "")

        mpnn_cmd = [
            "python", RUN_MPNN_SCRIPT,
            "--pdb_dir", msd_dir,
            "--design_specs_json", json_specs_path,
            "--out_folder", output_dir,
            "--model_name", model_name,
            "--num_seq_per_target", str(num_seqs),
            "--batch_size", "1",
            "--sampling_temp", str(sampling_temp),
            "--backbone_noise", str(backbone_noise),
        ]
        # --global_omit_AAs REPLACES its "X" default when passed, so X must
        # be included explicitly or it becomes sampleable everywhere.
        omit_aas = "X" + "".join(a for a in omit_aas if a != "X")
        mpnn_cmd.extend(["--global_omit_AAs", omit_aas])

        log.info(f"Running multi-state ProteinMPNN with model={model_name}...")
        result = subprocess.run(
            mpnn_cmd, capture_output=True, text=True, cwd=work_dir,
            env=child_env(),
        )
        if result.returncode != 0:
            log.warning(f"run_protein_mpnn.py failed:\nstdout: {result.stdout[-500:]}\nstderr: {result.stderr[-500:]}")
            return [], False, "proteinmpnn_failed"

        # ========================== Step 3: Parse output FASTAs ==========================
        # Main FASTA has combined scores
        main_fasta = os.path.join(output_dir, "msd.fasta")
        if not os.path.exists(main_fasta):
            log.error(f"main FASTA not found at {main_fasta}")
            return [], False, "main_fasta_missing"

        main_entries = _parse_main_fasta(main_fasta)

        # Per-state FASTAs have per-target scores and binder sequences
        state_entries = {}
        for i, target_name in enumerate(target_names):
            state_fasta = os.path.join(output_dir, f"{target_name}.fasta")
            if os.path.exists(state_fasta):
                state_entries[i] = _parse_state_fasta(state_fasta, binder_chain)
            else:
                log.warning(f"per-state FASTA not found: {state_fasta}")

        if not main_entries:
            log.info("No sequences generated by ProteinMPNN (main FASTA empty or no sample entries)")
            log.info(f"  MPNN stdout (last 500): {result.stdout[-500:]}")
            log.info(f"  MPNN stderr (last 500): {result.stderr[-500:]}")
            return [], False, "no_sequences_generated"

        # ========================== Step 4: Assemble results ==========================
        # Join the two FASTAs by position, not by "sample=", which is always
        # 0 at batch_size=1.
        first_state = next(iter(state_entries.values()), [])
        unique_sequences = {}
        skipped_missing = 0
        skipped_duplicate = 0
        for idx, main_entry in enumerate(main_entries):
            binder_seq = first_state[idx]["seq"] if idx < len(first_state) else None

            if binder_seq is None:
                skipped_missing += 1
                continue
            if binder_seq in unique_sequences:
                skipped_duplicate += 1
                continue

            unique_sequences[binder_seq] = {
                "seq": binder_seq,
                "score": main_entry["score"],
                "seqid": main_entry["seq_recovery"],
            }

        # Sort by combined score (lower is better), drop already-tested
        # sequences, then limit.
        exclude = exclude_sequences or set()
        sorted_seqs = sorted(unique_sequences.values(), key=lambda x: x["score"])
        novel_seqs = [s for s in sorted_seqs if s["seq"] not in exclude]
        skipped_tested = len(sorted_seqs) - len(novel_seqs)
        # An explicit request wins over max_mpnn_sequences: the recycle loop
        # asks for 1 because it consumes one candidate per cycle.
        if max_sequences is None:
            max_sequences = run_settings.get("max_mpnn_sequences", 4)
        result_seqs = novel_seqs[:max_sequences]

        log.info(
            f"  {len(unique_sequences)} unique of {len(main_entries)} samples"
            f"{f', {skipped_missing} unmatched' if skipped_missing else ''}"
            f"{f', {skipped_duplicate} duplicate' if skipped_duplicate else ''}"
            f"{f', {skipped_tested} already tested' if skipped_tested else ''}"
            f", returning top {len(result_seqs)}"
        )
        for i, seq_data in enumerate(result_seqs):
            log.info(
                f"  Sequence {i + 1}: score={seq_data['score']:.4f}, "
                f"seqid={seq_data['seqid']:.3f}"
            )

        if not result_seqs:
            return [], False, "no_sequences_assembled"
        return result_seqs, True, ""
