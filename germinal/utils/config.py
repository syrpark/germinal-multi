"""Germinal Configuration Management Module

This module provides comprehensive configuration management for the Germinal protein
design system using Hydra framework. It handles configuration processing, validation,
and initialization of design runs with proper device checking and directory structure
setup.

The module implements:
- Hydra configuration processing and conversion
- Design run initialization with device validation
- Directory structure creation and management
- Starting structure generation and validation
- CDR position computation and binder sequence extraction

Key Functions:
    process_config: Convert Hydra configuration to system format
    initialize_germinal_run: Initialize complete design run with validation

Dependencies:
    - Hydra/OmegaConf for configuration management
    - JAX/Torch for device validation
    - Germinal utilities for structure processing
    - Bio libraries for PDB manipulation
"""

import subprocess
import os
from pathlib import Path
from typing import Dict, Any
from omegaconf import DictConfig, OmegaConf
import warnings

from germinal.utils.utils import (
    get_jax_device,
    get_torch_device,
    compute_cdr_positions,
    create_starting_structure,
    get_sequence_from_pdb,
)
from germinal.utils.io import RunLayout, IO

from germinal.utils.log import get_logger

log = get_logger(__name__)


def process_config(cfg: DictConfig) -> Dict[str, Any]:
    """Process Hydra configuration and convert to system-expected format.
    
    Takes a Hydra DictConfig object and converts it into the standardized format
    expected by the Germinal system. Separates configuration into run parameters,
    target specifications, and filtering criteria.
    
    Args:
        cfg (DictConfig): Hydra configuration object containing nested configuration
            sections including 'target', 'filter', and run parameters.
            
    Returns:
        Dict[str, Any]: Processed configuration dictionary with four main keys:
            - 'run': Run configuration parameters (excluding target and filter sections)
            - 'target': Target-specific configuration parameters
            - 'filters_initial': Initial filtering criteria for trajectory screening
            - 'filters_final': Final filtering criteria for design acceptance
    """
    # Convert OmegaConf to regular dict for compatibility
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    # Extract the main sections
    target_config = config_dict.get("target", {})
    filter_config = config_dict.get("filter", {})

    # Build run config by excluding target and filter sections
    run_config = {k: v for k, v in config_dict.items() if k not in ["target", "filter"]}

    # Extract initial and final filters
    filters_initial = filter_config.get("initial", {})
    filters_final = filter_config.get("final", {})

    # Return the four-key structure as requested
    processed_cfg = {
        "run": run_config,
        "target": target_config,
        "filters_initial": filters_initial,
        "filters_final": filters_final,
    }

    return processed_cfg


def initialize_germinal_run(
    run_settings: Dict[str, Any], target_settings: Dict[str, Any]
):
    """Initialize a complete Germinal design run with validation and setup.
    
    Performs comprehensive initialization of a Germinal protein design run including
    device validation, directory structure creation, starting structure generation,
    and configuration of all necessary parameters for the design process.
    
    The function validates computational resources (JAX/CUDA availability), creates
    the required directory structure, generates or validates starting PDB complexes,
    computes CDR positions, and configures model parameters for the design run.
    
    Args:
        run_settings (Dict[str, Any]): Run configuration parameters including:
            - project_dir: Base project directory path
            - results_dir: Results directory name
            - experiment_name: Unique experiment identifier
            - cdr_lengths: List of CDR lengths
            - fw_lengths: Framework region lengths
            - type: Binder type ('nb' for nanobody, 'scfv' for single-chain Fv)
            - use_multimer_design: Whether to use multimer models
            - bias_redesign: Bias value for redesign (negative values set to False)
            
        target_settings (Dict[str, Any]): Target-specific configuration including:
            - target_name: Name of the target protein
            - target_pdb_path: Path to target PDB structure
            - binder_chain: Chain identifier for binder (default 'B')
            - target_chain: Chain identifier for target (default 'A')
            
    Returns:
        tuple: (io, run_settings) where:
            - io (IO): Initialized I/O handler for the run with directory structure
            - run_settings (Dict[str, Any]): Updated run settings with additional fields:
                * cdr_positions: Computed CDR residue positions
                * starting_binder_seq: Extracted binder sequence from starting structure
                * starting_pdb_complex: Path to the starting PDB complex
                * design_models: List of model indices for AF2 design
                * msa_cache_dir: Run-level MSA cache directory (targets + binder frameworks, shared by every folding backend)
                
    Raises:
        AssertionError: If JAX device is not available or CUDA is not accessible
        AssertionError: If target PDB path does not exist
    """
    # Validate computational device availability for design execution
    assert get_jax_device(), "JAX device not available"
    if get_torch_device() != "cuda":
        warnings.warn("Torch device not available")
    # Construct hierarchical directory path for design outputs
    design_path = os.path.join(
        run_settings.get("project_dir", "."),
        run_settings.get("results_dir", "results"),
        run_settings.get("experiment_name", "germinal_run"),
        run_settings.get("run_config", ""),
    )
    # Initialize directory structure and I/O handler
    design_paths = RunLayout.create(design_path)
    # Init I/O handler
    io = IO(design_paths)

    # Run-level cache for target MSAs, keyed by target sequence and shared by
    # every design and array task writing into this results root.
    run_settings["msa_cache_dir"] = str(design_paths.root / "msa_cache")
    os.makedirs(run_settings["msa_cache_dir"], exist_ok=True)

    # Compute CDR residue positions from framework and CDR lengths
    cdr_str = "_".join(str(i) for i in run_settings.get("cdr_lengths"))
    cdr_positions = compute_cdr_positions(
        run_settings.get("cdr_lengths"), run_settings.get("fw_lengths")
    )
    run_settings["cdr_positions"] = cdr_positions
    # Determine path for starting PDB complex(s) based on binder type
    binder_type = run_settings.get("type", "nb")

    target_names_value = target_settings.get("target_name")
    target_paths_value = target_settings.get("target_pdb_path")
    
    target_names = list(target_names_value)
    target_paths = list(target_paths_value)

    # Derived from the configs, never declared: the per-target loops zip these
    # lists, so a declared count that disagreed would truncate silently.
    hotspots_value = target_settings.get("target_hotspots")
    target_hotspots = (
        list(hotspots_value)
        if isinstance(hotspots_value, (list, tuple))
        else [hotspots_value]
    )
    lengths = {
        "target_name": len(target_names),
        "target_pdb_path": len(target_paths),
        "target_hotspots": len(target_hotspots),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"Per-target target config fields must all have one entry per "
            f"antigen state, but their lengths differ: {lengths}. Every state "
            f"needs its own name, PDB and hotspot set."
        )
    if len(target_names) != len(set(map(str, target_names))):
        raise ValueError(
            f"target_name entries must be unique -- they key the per-target "
            f"columns in designs.csv and the AF3 output directories, so a "
            f"duplicate silently overwrites one state with another. Got "
            f"{target_names}."
        )
    target_settings["num_targets"] = len(target_paths)

    starting_pdb_paths = []
    for tname, pdb_path in zip(target_names, target_paths):
        complex_path = ensure_starting_complex_for_target(
            target_name_single=str(tname),
            pdb_path_single=str(pdb_path),
            binder_type=binder_type,
            cdr_str=cdr_str,
            run_settings=run_settings,
            target_settings=target_settings,
        )
        starting_pdb_paths.append(complex_path)

    run_settings["starting_binder_seq"] = get_sequence_from_pdb(starting_pdb_paths[0])[
        target_settings.get("binder_chain", "B")
    ]
    run_settings["starting_pdb_complex"] = starting_pdb_paths

    # Apply configuration updates and validation logic
    if run_settings.get("use_multimer_design"):
        run_settings["design_models"] = [0, 1, 2, 3, 4]
    else:
        run_settings["design_models"] = [0, 1]

    # Normalize negative bias values to False for consistency
    if run_settings.get("bias_redesign") < 0:
        run_settings["bias_redesign"] = False

    # Recovery-mode defaults: use the WT complex as initialisation only.
    # setdefault, so an explicit config value always wins.
    if run_settings.get("is_recover"):
        run_settings.setdefault("rm_template_ic", True)
        run_settings.setdefault("dgram_cce", 0)

    return io, run_settings

def validate_binder_length(
    run_settings: Dict[str, Any], target_settings: Dict[str, Any]
) -> None:
    """Assert the configured design length matches the starting complex's binder.

    A mismatch shifts every CDR window, so it is always a configuration
    error. Call this after any ``complex_path`` override has been applied.

    Raises:
        ValueError: If the configured length differs from the binder length.
    """
    binder_chain = target_settings.get("binder_chain", "B")
    cdr_lengths = list(run_settings.get("cdr_lengths"))
    fw_lengths = list(run_settings.get("fw_lengths"))
    expected_length = sum(cdr_lengths) + sum(fw_lengths)

    pdb_paths = list(run_settings.get("starting_pdb_complex"))

    for pdb_path in pdb_paths:
        actual_length = len(get_sequence_from_pdb(pdb_path)[binder_chain])
        if actual_length != expected_length:
            raise ValueError(
                f"Binder length mismatch -- the CDR windows would be shifted by "
                f"{expected_length - actual_length:+d} residues, silently "
                f"designing the wrong positions.\n"
                f"  cdr_lengths {cdr_lengths} (sum {sum(cdr_lengths)})\n"
                f"  fw_lengths  {fw_lengths} (sum {sum(fw_lengths)})\n"
                f"  configured design length = {expected_length}\n"
                f"  chain {binder_chain} of {pdb_path} has {actual_length} residues\n"
                f"Fix cdr_lengths/fw_lengths so they sum to {actual_length}, or "
                f"supply a starting complex whose binder is {expected_length} "
                f"residues."
            )


def validate_num_targets(
    run_settings: Dict[str, Any], target_settings: Dict[str, Any]
) -> None:
    """Check the per-target gradient merge is one the state count supports.

    ``"sum"`` is generic in N; ``"pcgrad"`` takes exactly two gradients, so
    above two states it is switched to ``"sum"`` with a note. The effective
    value is written back into ``run_settings``.

    Raises:
        ValueError: If multi_grad_merge_method is not "sum" or "pcgrad".
    """
    # A different selector: it merges the AF2 and IgLM gradients. colabdesign
    # has no else branch, so an unknown value silently drops IgLM's gradient.
    iglm_valid = ("scale", "pcgrad")
    iglm_method = run_settings.get("iglm_grad_merge_method", "pcgrad")
    if iglm_method not in iglm_valid:
        raise ValueError(
            f"iglm_grad_merge_method is {iglm_method!r}, expected one of "
            f"{' or '.join(repr(v) for v in iglm_valid)}. It selects how the "
            f"IgLM gradient is merged with the AlphaFold gradient; an "
            f"unrecognised value would discard the IgLM gradient silently."
        )

    valid = ("sum", "pcgrad")
    method = run_settings.get("multi_grad_merge_method", "sum")
    if method not in valid:
        raise ValueError(
            f"multi_grad_merge_method is {method!r}, expected one of "
            f"{' or '.join(repr(v) for v in valid)}. It selects how the "
            f"per-target gradients are combined; see configs/run/multi.yaml."
        )

    num_targets = target_settings["num_targets"]
    if num_targets <= 2 or method == "sum":
        return

    log.info(
        f"\n{'=' * 72}\n"
        f"NOTE: multi_grad_merge_method has been switched from 'pcgrad' to "
        f"'sum'.\n"
        f"  num_targets is {num_targets}, and pcgrad merges exactly two "
        f"gradients, so\n"
        f"  states 3+ would be reported but never optimised against. Gradients "
        f"will be\n"
        f"  summed across all {num_targets} states instead.\n"
        f"  Set multi_grad_merge_method: \"sum\" explicitly to silence this, or "
        f"use 2 states\n"
        f"  for pcgrad conflict resolution.\n"
        f"{'=' * 72}\n"
    )
    run_settings["multi_grad_merge_method"] = "sum"


def validate_redesign_submodule(repo_root=None) -> None:
    """Make sure the ProteinMPNN submodule is populated before any GPU time.

    Sequence redesign shells out to it, so an unpopulated submodule otherwise
    surfaces as a subprocess failure partway into the first trajectory.

    A plain ``git clone`` leaves it empty, which is the common case rather than
    the exotic one, so try to populate it here before giving up: it is a
    two-second fetch wherever there is network, and failing on something this
    recoverable helps nobody. Only then raise, with both what went wrong and
    the command to run by hand.

    Raises:
        FileNotFoundError: If the submodule is missing and cannot be populated.
    """
    # Parameterised only so the behaviour is testable against a stand-in
    # checkout; callers use the default.
    repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    entry = repo_root / "external" / "proteinmpnn" / "run" / "run_protein_mpnn.py"
    if entry.is_file():
        return

    manual = "    git clone --recurse-submodules <url>   # or, in this checkout:\n" \
             "    git submodule update --init external/proteinmpnn"
    if not (repo_root / ".git").exists():
        # A source tarball or an export has no submodule to init; say so rather
        # than reporting a git failure the user cannot act on.
        raise FileNotFoundError(
            f"{entry} is missing and {repo_root} is not a git checkout, so the "
            f"ProteinMPNN submodule cannot be fetched. No design can be "
            f"redesigned without it. Clone the repository instead of "
            f"downloading a snapshot:\n{manual}"
        )

    log.info("ProteinMPNN submodule is empty; running "
             "`git submodule update --init external/proteinmpnn`.")
    try:
        subprocess.run(
            ["git", "submodule", "update", "--init", "external/proteinmpnn"],
            cwd=repo_root, check=True, capture_output=True, text=True,
            timeout=600,
        )
    except Exception as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise FileNotFoundError(
            f"{entry} is missing and populating the ProteinMPNN submodule "
            f"failed, so no design can be redesigned. Most often this is a "
            f"compute node without outbound network, in which case run it once "
            f"from a login node. The failure was:\n{detail}\n{manual}"
        ) from exc

    if not entry.is_file():
        raise FileNotFoundError(
            f"`git submodule update --init` reported success but {entry} is "
            f"still missing, so the submodule is registered yet empty. Fix by "
            f"hand:\n{manual}"
        )
    log.info("ProteinMPNN submodule populated.")


def produced_metric_names() -> set:
    """The bare metric names build_filter_metrics emits, for any design.

    The vocabulary is fixed -- the same keys for every design -- so it can be
    read once without folding anything. Derived from the function itself
    rather than a hand-kept list, so a new metric needs no second edit here.
    """
    from germinal.filters import filter_utils

    class _Stub(dict):
        """Stands in for any argument: indexes, counts and computes as zero.

        build_filter_metrics does real arithmetic on its inputs, so a bare
        dict or None is not enough to walk it once for its keys.
        """

        def __missing__(self, key):
            return _Stub()

        def __len__(self):
            return 0

        def __iter__(self):
            return iter(())

        def __float__(self):
            return 0.0

        def __int__(self):
            return 0

        def __round__(self, n=0):
            return 0.0

        def __getattr__(self, name):
            return _Stub()

        def _op(self, *_args):
            return _Stub()

        __add__ = __radd__ = __sub__ = __rsub__ = _op
        __mul__ = __rmul__ = __truediv__ = __rtruediv__ = _op
        __neg__ = __abs__ = _op

    n_args = 18
    return set(filter_utils.build_filter_metrics(*[_Stub() for _ in range(n_args)]))


def validate_filter_metric_names(initial_filters, final_filters) -> None:
    """Assert both filter sets name metrics the pipeline actually produces.

    evaluate_filters already rejects an unknown key, but only once a design
    reaches that stage -- for a final-filter typo, potentially a day into the
    job. The metric vocabulary is static, so the same mistake is catchable
    before any GPU time.

    Raises:
        ValueError: If either set names a metric that is never produced.
    """
    from germinal.filters.filter_utils import strip_stage_prefix

    produced = produced_metric_names()
    problems = []
    for stage, filter_set in (("initial", initial_filters), ("final", final_filters)):
        if not filter_set:
            continue
        # strip_stage_prefix validates the "<stage>." prefix itself and raises
        # on a wrong-stage or aggregation-prefixed key.
        bare = strip_stage_prefix(dict(filter_set), stage)
        unknown = sorted(set(bare) - produced)
        if unknown:
            problems.append(
                f"  {stage} filters name metric(s) the pipeline does not "
                f"produce: {unknown}"
            )
    if problems:
        raise ValueError(
            "Filter config names unknown metric(s):\n"
            + "\n".join(problems)
            + f"\nEvery filter key must match a produced metric exactly; an "
            f"unmatched key would otherwise fail every design.\nAvailable: "
            f"{sorted(produced)}"
        )


def validate_dalphaball(run_settings: Dict[str, Any]) -> None:
    """Assert the DAlphaBall binary exists and is executable.

    PyRosetta accepts the path at init without touching it, so a wrong one
    first surfaces inside BuriedUnsatHbonds -- which only runs in the final
    filter, potentially a day into the job.

    Raises:
        FileNotFoundError: If the path is unset, missing, or not executable.
    """
    value = run_settings.get("dalphaball_path")
    if value is None or str(value).strip() in ("", "???"):
        raise FileNotFoundError(
            "dalphaball_path is unset, but the final filter needs DAlphaBall "
            "for BuriedUnsatHbonds. Set it in configs/config.yaml."
        )
    path = Path(str(value))
    if not path.is_file():
        raise FileNotFoundError(
            f"dalphaball_path does not exist: {path}\n"
            f"The final filter needs it for BuriedUnsatHbonds. Set it in "
            f"configs/config.yaml, or override on the command line with "
            f"dalphaball_path=/path/to/DAlphaBall.gcc"
        )
    if not os.access(path, os.X_OK):
        raise FileNotFoundError(
            f"dalphaball_path is not executable: {path}\n"
            f"Fix with:\n    chmod +x {path}"
        )


def validate_af3_paths(run_settings: Dict[str, Any]) -> None:
    """Assert the AF3 locations are set and present before any GPU time is spent.

    The paths ship as Hydra's ``???`` sentinel, which survives
    ``to_container(resolve=True)`` as a literal string, so it must be checked
    here. Required only if either stage names af3.

    Raises:
        ValueError: If any path is unset, still ``???``, or does not exist.
    """
    from germinal.filters import folding

    folding.validate_backends(run_settings)
    uses_af3 = any(
        folding.resolve_backend_name(run_settings, stage) == "af3"
        for stage in folding.STAGE_KEYS
    )
    if not uses_af3:
        return

    # The four AF3 locations live in the stage's folding config, so each stage
    # that names af3 is checked against its own. msa_db_dir is global: both
    # backends read it, and it is required only when a stage sets
    # msa_method: "colabfold_local" -- the shipped configs search remotely.
    path_keys = ["af3_repo_path", "af3_sif_path", "af3_model_dir", "af3_db_dir"]

    def _unset(value):
        return value is None or str(value).strip() in ("", "???")

    unset, missing = [], []
    seen = set()
    for stage in folding.STAGE_KEYS:
        if folding.resolve_backend_name(run_settings, stage) != "af3":
            continue
        folder = folding.resolve_folder(run_settings, stage)
        for key in path_keys:
            value = folder.get(key)
            label = f"{folding.STAGE_KEYS[stage]}.{key}"
            if (label, str(value)) in seen:
                continue
            seen.add((label, str(value)))
            if _unset(value):
                unset.append(label)
            elif not os.path.exists(str(value)):
                missing.append(f"{label}: {value}")

    needs_local_db = any(
        folding.resolve_msa_policy(run_settings, stage)[0] == "colabfold_local"
        for stage in ("filter", "recycle")
    )
    if needs_local_db:
        value = run_settings.get("msa_db_dir")
        if _unset(value):
            unset.append("msa_db_dir")
        elif not os.path.exists(str(value)):
            missing.append(f"msa_db_dir: {value}")

    if not unset and not missing:
        return

    lines = ["AlphaFold3 is not configured for this machine."]
    if unset:
        lines.append("  Unset (still ??? or empty): " + ", ".join(unset))
    if missing:
        lines.append("  Set but not found on disk:")
        lines.extend(f"    {entry}" for entry in missing)
    lines.append(
        "Set the AF3 locations in configs/folding/af3.yaml (both stages load "
        "that file) and msa_db_dir in configs/config.yaml -- or override the "
        "individual keys on the command line, e.g. "
        "recycle_folder.af3_sif_path=/path/to/alphafold3.sif. "
        "See docs/af3-setup.md."
    )
    raise ValueError("\n".join(lines))


def validate_single_path_redesign_cutoff(run_settings: Dict[str, Any]) -> None:
    """Assert mpnn_fix_interface has a usable cutoff when recycling is off.

    "Single path" means single-pass, not single-target: with no ratchet to
    raise it later, a non-positive cutoff protects nothing. Call this only
    when recycling is off, where 0.0 is a legitimate starting value.

    Raises:
        ValueError: If mpnn_fix_interface is on with a non-positive cutoff.
    """
    if not run_settings.get("mpnn_fix_interface", False):
        return

    cutoff = run_settings.get("atom_distance_cutoff", 0.0)
    if cutoff is None or cutoff <= 0.0:
        raise ValueError(
            f"mpnn_fix_interface is enabled but atom_distance_cutoff is "
            f"{cutoff}, which matches no contacts. With recycling off there is no "
            f"ratchet to raise it later, so interface fixing would "
            f"be a silent no-op and AbMPNN would be handed every CDR position "
            f"as designable. Set a positive cutoff "
            f"(configs/run/multi.yaml uses 2.6) or set mpnn_fix_interface: "
            f"false."
        )


def ensure_starting_complex_for_target(
        target_name_single: str,
        pdb_path_single: str,
        binder_type: str,
        cdr_str: str,
        run_settings: Dict[str, Any],
        target_settings: Dict[str, Any],
) -> str:
    """
    Ensure the existence of a starting PDB complex file for a given target and binder type.

    This function constructs the expected path for the starting complex PDB file based on the
    target name, CDR string, and binder type. If the file does not exist, it creates the complex
    by combining the target and binder structures using a template binder PDB.

    Args:
        target_name_single (str): Name of the target protein.
        pdb_path_single (str): Path to the target PDB file.
        binder_type (str): Type of binder ("nb" for nanobody, "scfv" for scFv).
        cdr_str (str): String representing the CDR configuration.
        run_settings (Dict[str, Any]): Dictionary containing run-specific settings.
        target_settings (Dict[str, Any]): Dictionary containing target-specific settings.

    Returns:
        str: Path to the ensured or newly created complex PDB file.

    Raises:
        AssertionError: If the target PDB file does not exist.
    """
    # Determine the local complex name based on binder type
    if binder_type == "nb":
        complex_name_local = f"{target_name_single}_{cdr_str}_nb"
    else:
        complex_name_local = f"{target_name_single}_{cdr_str}_scfv"

    # Construct the path for the complex PDB file
    complex_path = os.path.join(
        run_settings.get("pdb_dir", "pdbs"),
        f"{complex_name_local}.pdb"
    )

    # Validated on every call, not only on a cache miss: the target PDB is
    # read again downstream for target_sequences.
    assert os.path.exists(pdb_path_single), (
        f"Target PDB path does not exist: {pdb_path_single}"
    )

    # If the complex PDB does not exist, create it
    if not os.path.exists(complex_path):
        # The template is pdbs/{binder_type}.pdb, built by interpolation: a
        # literal search for "scfv.pdb" finds nothing, so say so here.
        template_binder_pdb = os.path.join(
            run_settings.get("pdb_dir", "pdbs"),
            f"{binder_type}.pdb"
        )
        if not os.path.exists(template_binder_pdb):
            raise FileNotFoundError(
                f"No binder template at {template_binder_pdb}. Every run of "
                f"type {binder_type!r} starts from it. Only 'scfv' ships a "
                f"template; supply your own to run another binder type."
            )
        create_starting_structure(
            complex_path,
            template_binder_pdb,
            pdb_path_single,
            binder_chain=target_settings.get("binder_chain", "B"),
            target_chain=target_settings.get("target_chain", "A"),
        )

    return complex_path

def validate_target_numbering(target_settings: Dict[str, Any]) -> None:
    """Check every target chain is numbered 1..N with no gaps.

    ColabDesign resolves a hotspot like "A70" by author residue number; the
    filters resolve it positionally. The two agree only while the chain is
    numbered 1..N with no gaps, and a gapped target fails silently.

    Note AF3 prep forces ``ignore_missing=True`` for targets, so a residue
    lacking a CA is dropped from the ColabDesign frame and creates exactly this
    skew.

    Raises:
        ValueError: if any target chain is not contiguous from 1.
    """
    chain = target_settings["target_chain"]
    for pdb_path in target_settings["target_pdb_path"]:
        seen = []
        with open(pdb_path) as fh:
            for line in fh:
                if (
                    line.startswith("ATOM")
                    and line[21] == chain
                    and line[12:16].strip() == "CA"
                ):
                    seen.append(int(line[22:26]))
        if not seen:
            raise ValueError(
                f"{pdb_path}: no CA atoms found for target chain {chain!r}."
            )
        expected = list(range(1, len(seen) + 1))
        if seen != expected:
            gaps = sorted(set(range(min(seen), max(seen) + 1)) - set(seen))
            raise ValueError(
                f"{pdb_path}: target chain {chain!r} is not numbered 1..N "
                f"contiguously (found {len(seen)} CA atoms spanning "
                f"{min(seen)}-{max(seen)}"
                + (f", missing {gaps}" if gaps else "")
                + "). Hotspots are resolved by AUTHOR number in ColabDesign and "
                "POSITIONALLY in the filters; those agree only for a gapless "
                "1..N chain, so this PDB would make the optimiser and the "
                "binder_near_hotspot gate target different residues. Renumber "
                "the chain, or make both paths agree explicitly."
            )
