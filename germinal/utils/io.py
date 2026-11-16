"""Germinal I/O Management Module

This module provides comprehensive input/output management for the Germinal protein
design system. It handles directory structure creation, trajectory data management,
file operations with concurrency safety, and result organization.

The module implements:
- Directory structure creation and management (RunLayout)
- Trajectory data storage and organization (Trajectory)
- Thread-safe file operations with locking mechanisms
- CSV-based data persistence with automatic column handling
- Failure tracking and termination condition monitoring

Key Classes:
    RunLayout: Defines and creates directory structure for design runs
    Trajectory: Stores and manages individual trajectory information
    IO: Main I/O handler with thread-safe operations

Key Features:
    - Automatic directory structure creation
    - Thread-safe CSV operations using file locks
    - Trajectory filtering and organization
    - Failure count tracking and analysis
    - Termination condition monitoring

Dependencies:
    - pandas for data manipulation
    - filelock for thread-safe file operations
    - pathlib for cross-platform path handling
"""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import re
import os
import shutil
import socket
import time
from datetime import datetime
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from filelock import FileLock

from germinal.utils.log import get_logger

log = get_logger(__name__)



TRAJECTORY_METRICS_TO_SAVE = [
    "plddt",
    "ptm",
    "i_ptm",
    "i_pae",
    "pae",
    "loss",
    "iglm_ll",
    "helix",
    "beta_strand",
]

#: Columns every design row carries, even when the code path never set them,
#: so the header does not vary with which paths a run happened to take.
REQUIRED_COLUMNS = [
    "design_name",
    "experiment_name",
    "status",
    "failure_stage",
    "terminate_reason",
    "candidate_source",
    "save_location",
    "design_time",
    "final_structure_path",
]


def _with_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add any REQUIRED_COLUMNS the row is missing, as NA."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        df = df.assign(**{c: pd.NA for c in missing})
    return df


def _plain(obj):
    """Plain Python for yaml.dump; filter sets arrive as OmegaConf nodes."""
    if obj is None:
        return None
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(obj):
            return OmegaConf.to_container(obj, resolve=True)
    except ImportError:
        pass
    return obj


INFLIGHT_DIRNAME = "inflight"

#: A record at another schema is skipped, not guessed at.
INFLIGHT_SCHEMA = 1


def _jsonable(obj):
    """Plain JSON types, recursively.

    The metric dicts carry numpy scalars and Paths, and hotspots and filter
    sets arrive as OmegaConf nodes; json.dump chokes on all three.
    """
    obj = _plain(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    return str(obj)


def _digest(obj) -> str:
    """Stable sha256 over a nested structure, for config fingerprints."""
    canonical = json.dumps(
        _jsonable(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()[:32]


def config_fingerprints(
    run_settings: Dict[str, Any],
    target_settings: Dict[str, Any],
    initial_filters: Any,
) -> Tuple[str, str]:
    """Two digests deciding how much previous work a new run may adopt.

    Split because they invalidate different amounts: a different target means
    a hallucination PDB is not the same object at all, while a different
    filter set only invalidates a verdict that was inherited rather than
    recomputed -- which is the candidate stage and nothing else. Budget knobs
    are excluded deliberately, and so are the final filters: Phase D applies
    those to every adopted design anyway.

    Returns:
        (target_fp, filter_fp)
    """
    target_fp = _digest({
        "target_name": target_settings.get("target_name"),
        "target_pdb_path": target_settings.get("target_pdb_path"),
        "target_chain": target_settings.get("target_chain"),
        "binder_chain": target_settings.get("binder_chain"),
        "target_hotspots": target_settings.get("target_hotspots"),
        "num_targets": target_settings.get("num_targets"),
        "type": run_settings.get("type"),
        "cdr_lengths": run_settings.get("cdr_lengths"),
        "fw_lengths": run_settings.get("fw_lengths"),
        "logits_only": run_settings.get("logits_only"),
    })
    # Deliberately not final_filters: no adopted record skips the final
    # filter, so Phase D always applies the current one. Including it only
    # stranded nearly-finished designs whenever a final threshold moved.
    filter_fp = _digest({
        "initial_filters": initial_filters,
        "recycling": run_settings.get("recycling"),
        "num_recycles": run_settings.get("num_recycles"),
        "recycle_ipsae_threshold": run_settings.get("recycle_ipsae_threshold"),
        "early_check_min_cdr_fraction": run_settings.get(
            "early_check_min_cdr_fraction"
        ),
        "filter_folder": run_settings.get("filter_folder"),
        "recycle_folder": run_settings.get("recycle_folder"),
    })
    return target_fp, filter_fp


def _write_header_if_absent(csv_path) -> None:
    """Create a design CSV holding only its header row."""
    csv_path = Path(csv_path)
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return
    csv_path.write_text(",".join(REQUIRED_COLUMNS) + "\n")


@dataclass
class RunLayout:
    """Directory structure definition and creation for Germinal design runs.
    
    This dataclass defines the standardized directory structure used by Germinal
    for organizing design run outputs, including trajectories, candidates, accepted
    designs, logs, and summary files. Provides methods for creating the complete
    directory hierarchy with proper initialization.
    
    Attributes:
        root (Path): Root directory for the design run
        trajectories (Path): Designs that failed before the final filters --
            hallucination, early check, initial filter, redesign or the
            recycling gate. Its structures/ is a working directory, not a
            record: the hallucination PDBs live there until they are consumed.
        redesign_candidates (Path): Every structure predicted by the filter
            model or the recycling model, plus their working files. A
            structure store with no designs.csv rows, so check_existing_seed
            must not treat it as terminal.
        filter_failed (Path): Designs that reached final filters and failed them
        accepted (Path): Directory for accepted final designs
        all_csv (Path): Path to comprehensive trajectory summary CSV
        failure_csv (Path): Path to failure count tracking CSV (stage-prefixed columns)
        final_config (Path): Path to final configuration YAML file
        inflight (Path): One JSON per design past hallucination and not yet
            terminal; an empty directory means the last run ended clean.

    Directory Structure:
        root/
        ├── trajectories/
        │   ├── structures/
        │   └── designs.csv
        ├── redesign_candidates/
        │   └── structures/
        │       ├── {design}/        filter + recycling model predictions
        │       └── _batch_fold/     batched filter-stage predictions
        │   (designs.csv exists but stays empty: structures only.
        │    Each backend stages its inputs inside its own output_dir, as
        │    copies of the shared root/msa_cache/.)
        ├── filter_failed/
        │   ├── structures/
        │   └── designs.csv
        ├── accepted/
        │   ├── structures/
        │   └── designs.csv
        ├── inflight/                one {design}.json per unfinished design
        ├── all_trajectories.csv
        ├── failure_counts.csv
        └── final_config.yaml
    """

    root: Path
    trajectories: Path
    redesign_candidates: Path
    filter_failed: Path
    accepted: Path
    all_csv: Path
    failure_csv: Path
    final_config: Path
    # Trailing and defaulted so existing positional constructions keep working.
    inflight: Path = None

    @classmethod
    def create(cls, root: Path | str):
        """Create complete directory structure for a new Germinal design run.
        
        Initializes the full directory hierarchy required for a Germinal design run,
        including all subdirectories, placeholder files, and initial CSV files.
        Creates the structure with proper permissions and initializes empty tracking files.
        
        Args:
            root (Path | str): Root directory path for the design run. Will be created
                if it doesn't exist, along with all parent directories.
                
        Returns:
            RunLayout: Initialized RunLayout instance with all paths configured
            and directory structure created.
        """
        if isinstance(root, str):
            root = Path(root)

        root.mkdir(parents=True, exist_ok=True)

        traj = root / "trajectories"
        redesign = root / "redesign_candidates"
        filter_failed = root / "filter_failed"
        accepted = root / "accepted"
        inflight = root / INFLIGHT_DIRNAME
        # Initialize empty tracking files for run management
        all_csv = root / "all_trajectories.csv"
        failure_csv = root / "failure_counts.csv"
        final_config = root / "final_config.yaml"
        # Header, not touch(): pd.read_csv raises EmptyDataError on a 0-byte
        # file, so every aggregation script had to special-case the empty ones.
        _write_header_if_absent(all_csv)
        failure_csv.touch()
        final_config.touch()
        inflight.mkdir(parents=True, exist_ok=True)

        # Create main subdirectories with internal structure
        for main_dir in (traj, redesign, filter_failed, accepted):
            main_dir.mkdir(parents=True, exist_ok=True)
            (main_dir / "structures").mkdir(exist_ok=True)
            # Initialize trajectory tracking CSV for each directory
            _write_header_if_absent(main_dir / "designs.csv")

        return cls(
            root=root,
            trajectories=traj,
            redesign_candidates=redesign,
            filter_failed=filter_failed,
            accepted=accepted,
            all_csv=all_csv,
            failure_csv=failure_csv,
            final_config=final_config,
            inflight=inflight,
        )


class Trajectory:
    """Individual trajectory data container with metrics and metadata management.
    
    This class encapsulates all information associated with a single design trajectory,
    including trajectory metrics, filtering results, structural data, and metadata.
    Provides methods for updating different metric categories and managing trajectory
    lifecycle from creation to final storage.
    
    The class organizes metrics into three categories:
    - Trajectory metrics: Core optimization metrics (pLDDT, pTM, loss, etc.)
    - Filtering metrics: Results from filtering and quality assessment
    - Other metrics: Additional analysis results and metadata
    
    Attributes:
        design_name (str): Unique identifier for this trajectory
        experiment_name (str): Name of the parent experiment
        cdr_lengths (str): CDR length specification string
        target_hotspots (str): Target hotspot residue specification
        trajectory_metrics (dict): Core trajectory optimization metrics
        filtering_metrics (dict): Quality filtering and assessment results
        other_metrics (dict): Additional analysis and metadata
        save_location (str): Target directory for saving ('trajectories', 'redesign_candidates', 'filter_failed', 'accepted')
        final_struct (str): Path to final structure file
    """

    def __init__(
        self,
        design_name: str,
        experiment_name: str,
        cdr_lengths: str,
        target_hotspots: str,
    ):
        self.design_name = design_name
        self.experiment_name = experiment_name
        self.cdr_lengths = cdr_lengths
        self.target_hotspots = target_hotspots

        self.trajectory_metrics = {}
        self.filtering_metrics = {}
        self.other_metrics = {}

        self.save_location = ""  # trajectories, redesign_candidates, filter_failed, accepted
        self.final_struct = ""

    def get_trajectory(self):
        """Consolidate all trajectory information into a single dictionary.
        
        Combines trajectory metrics, filtering metrics, and other metrics with
        basic metadata into a comprehensive dictionary representation suitable
        for CSV export and analysis.
        
        Returns:
            dict: Complete trajectory information with all metrics and metadata
                combined into a flat dictionary structure.
        """
        final_trajectory_info = {
            "design_name": self.design_name,
            "experiment_name": self.experiment_name,
            "cdr_lengths": self.cdr_lengths,
            "target_hotspots": self.target_hotspots,
            **self.trajectory_metrics,
            **self.filtering_metrics,
            **self.other_metrics,
            # A column, not just an attribute: the outcome record reads it off
            # this dict, so it was always null before.
            "save_location": self.save_location,
            "final_structure_path": self.final_struct
        }

        return final_trajectory_info

    def update_trajectory_metrics(self, trajectory_metrics: Dict[str, Any]):
        """Update core trajectory optimization metrics.
        
        Updates the trajectory metrics with values from the design optimization
        process. Only metrics defined in TRAJECTORY_METRICS_TO_SAVE are stored
        to maintain consistency across trajectories.
        
        Args:
            trajectory_metrics (Dict[str, Any]): Dictionary of trajectory metrics
                from the optimization process. Should include metrics like pLDDT,
                pTM, loss, etc., optionally under a "<stage>." prefix.
        """
        # Matched on the leaf so a stage prefix survives: these are the
        # hallucinator's own confidence, and must stay distinguishable from the
        # same quantities measured later by AF3 under initial.* and final.*.
        for key, value in trajectory_metrics.items():
            if key.rsplit(".", 1)[-1] in TRAJECTORY_METRICS_TO_SAVE:
                self.trajectory_metrics[key] = value

    def update_filtering_metrics(self, filtering_metrics: Dict[str, Any]):
        """Update quality filtering and assessment metrics.
        
        Updates metrics related to trajectory quality assessment, filtering
        criteria evaluation, and selection decisions.
        
        Args:
            filtering_metrics (Dict[str, Any]): Dictionary of filtering results
                including quality scores, pass/fail flags, and selection criteria.
        """
        self.filtering_metrics.update(filtering_metrics)

    def update_other_metrics(self, other_metrics: Dict[str, Any]):
        """Update additional analysis metrics and metadata.
        
        Updates supplementary metrics that don't fall into trajectory or filtering
        categories, such as secondary analysis results, timing information, or
        custom evaluation metrics.
        
        Args:
            other_metrics (Dict[str, Any]): Dictionary of additional metrics
                and metadata for this trajectory.
        """
        self.other_metrics.update(other_metrics)

    def set_save_location(self, save_location: str):
        """Set the target directory for trajectory storage.
        
        Specifies which directory category this trajectory should be saved to
        based on its quality and selection status.
        
        Args:
            save_location (str): Target directory name. Must be one of:
                'trajectories', 'redesign_candidates', 'filter_failed', or 'accepted'.
        """
        self.save_location = save_location

    def set_final_struct(self, final_struct: str):
        """Set the path to the final structure file for this trajectory.
        
        Args:
            final_struct (str): Path to the final PDB structure file
                representing the best result from this trajectory.
        """
        self.final_struct = final_struct

    def rename(self, new_name: str):
        """Change the design name identifier for this trajectory.
        
        Args:
            new_name (str): New unique identifier for this trajectory.
        """
        self.design_name = new_name

    def copy(self):
        """Create a deep copy of this trajectory instance.
        
        Creates a new Trajectory instance with identical data but independent
        from the original. Useful for creating variants or backups.
        
        Returns:
            Trajectory: New trajectory instance with copied data.
        """
        new_trajectory = Trajectory(
            design_name=self.design_name,
            experiment_name=self.experiment_name,
            cdr_lengths=self.cdr_lengths,
            target_hotspots=self.target_hotspots,
        )
        new_trajectory.trajectory_metrics = dict(self.trajectory_metrics)
        new_trajectory.filtering_metrics = dict(self.filtering_metrics)
        new_trajectory.other_metrics = dict(self.other_metrics)
        new_trajectory.save_location = self.save_location
        new_trajectory.final_struct = self.final_struct
        return new_trajectory

    def snapshot(self) -> dict:
        """Everything needed to rebuild this trajectory in another process."""
        return {
            "design_name": self.design_name,
            "experiment_name": self.experiment_name,
            "cdr_lengths": self.cdr_lengths,
            "target_hotspots": _jsonable(self.target_hotspots),
            "trajectory_metrics": _jsonable(self.trajectory_metrics),
            "filtering_metrics": _jsonable(self.filtering_metrics),
            "other_metrics": _jsonable(self.other_metrics),
            "save_location": self.save_location,
            "final_struct": _jsonable(self.final_struct),
        }

    @classmethod
    def from_snapshot(cls, snap: dict) -> "Trajectory":
        """Rebuild a trajectory from snapshot(). Numeric types come back as
        plain Python, which every consumer downstream of the CSV accepts."""
        traj = cls(
            design_name=snap["design_name"],
            experiment_name=snap["experiment_name"],
            cdr_lengths=snap.get("cdr_lengths", ""),
            target_hotspots=snap.get("target_hotspots", ""),
        )
        traj.trajectory_metrics = dict(snap.get("trajectory_metrics") or {})
        traj.filtering_metrics = dict(snap.get("filtering_metrics") or {})
        traj.other_metrics = dict(snap.get("other_metrics") or {})
        traj.save_location = snap.get("save_location", "")
        traj.final_struct = snap.get("final_struct", "")
        return traj


class IO:
    """Main I/O handler for Germinal design runs with thread-safe operations.
    
    This class provides comprehensive I/O management for Germinal design runs,
    including configuration persistence, trajectory saving, failure tracking,
    and termination condition monitoring. All file operations are thread-safe
    using file locks to prevent race conditions in multi-device environments.
    
    Key Features:
    - Thread-safe CSV operations with automatic column management
    - Configuration persistence and validation
    - Trajectory organization and storage
    - Failure count tracking and analysis
    - Termination condition monitoring
    - Seed tracking to prevent duplicate work
    
    Attributes:
        layout (RunLayout): Directory structure manager for the run
    """

    def __init__(self, layout: RunLayout):
        self.layout = layout

    def save_run_config(
        self,
        run_settings: Dict[str, Any],
        target_settings: Dict[str, Any],
        initial_filters: Dict[str, Any] = None,
        final_filters: Dict[str, Any] = None,
    ):
        """Save run and target configuration to persistent YAML file.
        
        Persists the complete configuration for the design run to enable
        reproducibility and debugging. Combines both run and target settings
        into a single YAML file with proper structure.
        
        Args:
            run_settings (Dict[str, Any]): Complete run configuration parameters
            target_settings (Dict[str, Any]): Target-specific configuration parameters
            initial_filters: Threshold set used by the initial filter stage.
            final_filters: Threshold set used by the final filter stage.
        """
        # The filter sets decide every acceptance, and used to be in no run
        # artifact at all -- so a snapshot could not say what a design passed.
        snapshot = {
            "run_settings": run_settings,
            "target_settings": target_settings,
            "initial_filters": _plain(initial_filters),
            "final_filters": _plain(final_filters),
        }
        self._atomic_write_text(self.layout.final_config, yaml.dump(snapshot))

        log.info(f"Run and target settings saved to {self.layout.final_config}")

    def save_run_summary(self, summary: str) -> None:
        """Write run_summary.txt for this run, atomically and under its lock.

        Every SLURM array task of a run writes this file into the same root,
        so a plain open(..., "w") lets two tasks interleave and leaves a
        half-written summary. Same treatment as the CSVs.

        Args:
            summary (str): Text to write.
        """
        self._atomic_write_text(self.layout.root / "run_summary.txt", summary)

    @staticmethod
    def _atomic_write_text(path, text: str) -> None:
        """Write text to path under the path's lock, replacing it atomically.

        Non-CSV run artifacts (final_config.yaml, run_summary.txt) live in the
        same shared results root as the CSVs and are written by every array
        task of a run. They get the same guarantee: a reader sees either the
        complete old file or the complete new one, never a truncated one.
        """
        path = Path(path)
        tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        with FileLock(f"{str(path)}.lock"):
            try:
                with open(tmp_path, "w") as fh:
                    fh.write(text)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp_path, path)
                dir_fd = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()

    def check_existing_seed(self, seed: int):
        """Has this seed already produced a RECORDED design?

        Resume has no checkpoint file; it is this check plus the deterministic
        seed list. The master CSV is authoritative, since a working directory
        is created in phase B and the row may not be written until phase D.
        The terminal output directories still count: they hold only results.

        Args:
            seed (int): Design seed to check.

        Returns:
            bool: True if the seed already has a recorded design.
        """
        # Match the delimited _s<seed> field: a bare substring test matches
        # digits inside a longer seed and burns a trajectory slot.
        seed_pattern = re.compile(rf"_s{re.escape(str(seed))}(?=[_.]|$)")

        all_csv = self.layout.all_csv
        if os.path.exists(all_csv) and all_csv.stat().st_size > 0:
            try:
                names = pd.read_csv(all_csv, usecols=["design_name"])
                for name in names["design_name"].astype(str):
                    if seed_pattern.search(name):
                        return True
            except Exception as exc:
                # Not fatal, but the fallback below is weaker, so say so.
                log.info(
                    f"WARNING: could not read {all_csv} to check seed {seed} "
                    f"({type(exc).__name__}: {exc}); falling back to the "
                    f"saved-structure directories, which miss designs whose "
                    f"only record is a CSV row."
                )

        # Terminal result directories only: the structures/ directories are
        # working dirs, created before any row is written.
        for dir in (
            f"{self.layout.filter_failed}/structures",
            f"{self.layout.accepted}/structures",
        ):
            if not os.path.exists(dir):
                continue
            for fname in os.listdir(dir):
                if seed_pattern.search(fname):
                    return True
        return False

    # Not a row in all_trajectories.csv: a row there would make
    # check_existing_seed report the design finished and would inflate the
    # max_hallucinated_trajectories budget.

    def _inflight_path(self, design_name: str) -> Path:
        return Path(self.layout.inflight) / f"{design_name}.json"

    @staticmethod
    def _absolutise(value):
        """Resolve a recorded structure path, or a list of them, against cwd.

        Stored relative, a path only resolves from the directory the run
        happened to start in, so adoption reported "its structures are gone"
        for files that were still there -- from a second checkout, or simply
        after `results_dir` was given as an absolute path.
        """
        if isinstance(value, str):
            return os.path.abspath(value) if value else value
        if isinstance(value, (list, tuple)):
            return [IO._absolutise(v) for v in value]
        return value

    def record_inflight(self, design_name: str, record: Dict[str, Any]) -> None:
        """Write or supersede this design's in-flight record."""
        payload = dict(record)
        for key in ("hallucination_pdbs", "final_struct"):
            if key in payload:
                payload[key] = self._absolutise(payload[key])
        payload["candidates"] = [
            {**c, "designed_on": self._absolutise(c["designed_on"])}
            if isinstance(c, dict) and "designed_on" in c else c
            for c in (payload.get("candidates") or [])
        ]
        payload.update({
            "schema": INFLIGHT_SCHEMA,
            "design_name": design_name,
            "written_at": time.time(),
            "pid": os.getpid(),
            "host": socket.gethostname(),
        })
        Path(self.layout.inflight).mkdir(parents=True, exist_ok=True)
        self._atomic_write_text(
            self._inflight_path(design_name),
            json.dumps(_jsonable(payload), indent=2, sort_keys=True),
        )

    def clear_inflight(self, design_name: str) -> None:
        """Drop this design's record; it has reached a terminal state."""
        path = self._inflight_path(design_name)
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            log.warning(f"could not clear in-flight record {path} ({exc}).")
        lock = Path(f"{str(path)}.lock")
        try:
            lock.unlink()
        except (FileNotFoundError, OSError):
            pass

    def read_inflight(self) -> List[Dict[str, Any]]:
        """Every readable in-flight record.

        An unreadable or foreign-schema record is skipped, never fatal: one
        bad file must not stop a run from starting.
        """
        inflight_dir = Path(self.layout.inflight)
        if not inflight_dir.exists():
            return []
        records = []
        for path in sorted(inflight_dir.glob("*.json")):
            try:
                record = json.loads(path.read_text())
            except Exception as exc:
                log.info(
                    f"WARNING: skipping unreadable in-flight record {path} "
                    f"({type(exc).__name__}: {exc})."
                )
                continue
            if not isinstance(record, dict):
                log.warning(f"skipping in-flight record {path}: not an object.")
                continue
            if record.get("schema") != INFLIGHT_SCHEMA:
                log.info(
                    f"WARNING: skipping in-flight record {path}: schema "
                    f"{record.get('schema')!r}, this build writes "
                    f"{INFLIGHT_SCHEMA}."
                )
                continue
            records.append(record)
        return records

    def _recorded_design_names(self) -> set:
        """design_name values in the master CSV, or an empty set if unreadable."""
        all_csv = self.layout.all_csv
        if not (os.path.exists(all_csv) and all_csv.stat().st_size > 0):
            return set()
        try:
            names = pd.read_csv(all_csv, usecols=["design_name"])
        except Exception as exc:
            log.info(
                f"WARNING: could not read {all_csv} ({type(exc).__name__}: "
                f"{exc}); treating every design as unrecorded, which can only "
                f"cause rework, never a silent drop."
            )
            return set()
        return set(names["design_name"].astype(str))

    def find_adoptable(
        self, target_fp: str, filter_fp: str
    ) -> List[Dict[str, Any]]:
        """Unfinished designs this run may take over, oldest first.

        A record is adoptable when no row carries its exact design_name, the
        fingerprints are compatible, its structures are still on disk, and --
        for one that reached the candidate stage -- at least one candidate has
        no row yet. That last test is not optional: a design that finished
        phase D never gets a row of its own, only candidate rows, so asking
        "does the parent have a row" would mistake every finished design for
        unfinished work.

        A record that fails any test is reported and left in place; deciding
        it is rubbish is a human's call, not this function's.

        Returns:
            The adoptable records. One that reached the candidate stage gains
            a "missing_candidates" list holding only the unscored ones.
        """
        recorded = self._recorded_design_names()
        adoptable = []
        for record in sorted(
            self.read_inflight(), key=lambda r: r.get("written_at") or 0
        ):
            name = record.get("design_name", "")
            phase = record.get("phase")
            skip = None

            if name in recorded:
                skip = "it already has a row"
            elif record.get("target_fp") != target_fp:
                skip = "it was designed against a different target"
            elif phase == "candidates_ready" and record.get("filter_fp") != filter_fp:
                # Only this phase inherits a verdict. hallucinated and
                # filtered both re-enter at the fold and are re-filtered under
                # the current config, so gating them on the filter set threw
                # away work that would have been re-evaluated correctly.
                skip = "its candidates were built under different settings"
            else:
                missing = [
                    p for p in record.get("hallucination_pdbs") or []
                    if not os.path.exists(p)
                ]
                if missing:
                    skip = f"{len(missing)} of its structures are gone"

            if skip is None and phase == "candidates_ready":
                done = {
                    n for n in recorded
                    if re.match(rf"^{re.escape(name)}_abmpnn_\d+$", n)
                }
                remaining = [
                    c for c in record.get("candidates") or []
                    if c.get("name") not in done
                ]
                # A candidate is scored against the backbone it was designed
                # on, so one whose backbone is gone cannot be finished.
                scorable = [
                    c for c in remaining
                    if all(os.path.exists(p) for p in c.get("designed_on") or [])
                ]
                for lost in [c for c in remaining if c not in scorable]:
                    log.info(f"  Dropping candidate {lost.get('name')}: the "
                          f"backbone it was designed on is gone.")
                if not remaining:
                    skip = "every candidate it produced has been scored"
                elif not scorable:
                    skip = "no remaining candidate still has its backbone"
                else:
                    record = dict(record, missing_candidates=scorable)
            elif skip is None and any(
                re.match(rf"^{re.escape(name)}_abmpnn_\d+$", n) for n in recorded
            ):
                # Candidate rows but no record of which candidates existed:
                # finishing the set would mean guessing how many there were.
                skip = "it has candidate rows but lists no candidates"

            if skip:
                log.info(f"  Not adopting {name}: {skip}.")
                continue
            adoptable.append(record)
        return adoptable

    def per_target_pdbs(self, design_name: str, num_targets: int) -> List[str]:
        """ColabDesign writes one {base}_target{idx}.pdb per antigen state."""
        base, ext = os.path.splitext(f"{design_name}.pdb")
        return [
            str(self.layout.trajectories / f"structures/{base}_target{t}{ext}")
            for t in range(num_targets)
        ]

    def clear_batch_fold_dir(self) -> None:
        """Drop the scratch directory the batched folds extract into."""
        batch_dir = self.layout.redesign_candidates / "structures" / "_batch_fold"
        if os.path.exists(batch_dir):
            shutil.rmtree(batch_dir, ignore_errors=True)

    def check_termination_conditions(
        self, run_settings: Dict[str, Any], n_trajectories: int
    ):
        """Evaluate termination conditions for the design optimization loop.
        
        Checks multiple termination criteria including maximum accepted designs,
        maximum hallucinated trajectories, and total trajectory limits to determine
        if the optimization loop should continue or terminate.
        
        Args:
            run_settings (Dict[str, Any]): Run configuration containing termination limits:
                - max_passing_designs: Maximum number of accepted designs
                - max_hallucinated_trajectories: Maximum successful trajectories
                - max_trajectories: Maximum total trajectory attempts
            n_trajectories (int): Current number of trajectory attempts
            
        Returns:
            tuple: (should_terminate, termination_reason) where:
                - should_terminate (bool): Whether to terminate the optimization loop
                - termination_reason (str): Human-readable reason for termination
        """
        # check accepted designs
        accepted_csv_path = self.layout.accepted / "designs.csv"
        if (
            not os.path.exists(accepted_csv_path)
            or accepted_csv_path.stat().st_size == 0
        ):
            n_accepted_trajectories = 0
        else:
            n_accepted_trajectories = len(pd.read_csv(accepted_csv_path))
        # Rows, deliberately: max_passing_designs counts accepted designs,
        # and each `_abmpnn_N` candidate is its own design.
        if n_accepted_trajectories >= run_settings["max_passing_designs"]:
            return True, "Max accepted designs reached. Exiting Loop."
        # check hallucinated designs
        trajectories_csv_path = self.layout.all_csv
        if (
            not os.path.exists(trajectories_csv_path)
            or trajectories_csv_path.stat().st_size == 0
        ):
            n_hallucinated_trajectories = 0
        else:
            # Count hallucinations, not rows: one hallucination yields
            # several `_abmpnn_N` candidate rows.
            all_df = pd.read_csv(trajectories_csv_path)
            if "design_name" in all_df.columns:
                n_hallucinated_trajectories = (
                    all_df["design_name"]
                    .astype(str)
                    .str.replace(r"_abmpnn_\d+$", "", regex=True)
                    .nunique()
                )
            else:
                log.info(
                    f"WARNING: {trajectories_csv_path} has no 'design_name' "
                    f"column; falling back to a row count for "
                    f"max_hallucinated_trajectories, which over-counts any "
                    f"trajectory that produced several candidates."
                )
                n_hallucinated_trajectories = len(all_df)
        if n_hallucinated_trajectories >= run_settings["max_hallucinated_trajectories"]:
            return True, "Max hallucinated trajectories reached. Exiting Loop."
        # check total trajectories
        if n_trajectories >= run_settings["max_trajectories"]:
            return True, "Max trajectories reached. Exiting Loop."
        return False, ""

    def update_failures(self, fail_column: str) -> None:
        """Thread-safe increment of failure counts in tracking CSV.
        
        Increments failure counts for specific failure types using thread-safe
        file operations. Creates the CSV file and columns as needed. Supports
        both single column updates and batch dictionary updates.
        
        Args:
            fail_column (str or dict): Either a column name to increment by 1,
                or a dictionary mapping column names to increment values.
        """
        if self.layout.failure_csv is None:
            return

        lock = FileLock(f"{str(self.layout.failure_csv)}.lock")
        with lock:
            # Absent or zero-length is the expected first-write case. Anything
            # else means the file exists but did not parse, and starting from an
            # empty frame would write back over every count accumulated so far,
            # leaving failure_counts.csv claiming the run barely failed.
            try:
                failure_df = pd.read_csv(self.layout.failure_csv)
            except (FileNotFoundError, pd.errors.EmptyDataError):
                failure_df = pd.DataFrame()
            except Exception as e:
                raise RuntimeError(
                    f"{self.layout.failure_csv} exists but could not be parsed "
                    f"({type(e).__name__}: {e}). Refusing to overwrite it with a "
                    f"fresh frame, which would discard every failure count for "
                    f"this run. Move the file aside to start a new one."
                ) from e

            def strip_model_prefix(name: str) -> str:
                parts = name.split("_")
                return "_".join(parts[1:]) if parts and parts[0].isdigit() else name

            # Prepare update dict
            if isinstance(fail_column, dict):
                updates = {}
                for k, v in fail_column.items():
                    col = strip_model_prefix(k)
                    updates[col] = updates.get(col, 0) + v
            else:
                col = strip_model_prefix(fail_column)
                updates = {col: 1}

            # If DataFrame is empty, create a single row with the updates
            if failure_df.empty:
                failure_df = pd.DataFrame(
                    [{col: count for col, count in updates.items()}]
                )
            else:
                # If DataFrame has no rows, add a row
                if len(failure_df) == 0:
                    failure_df = pd.DataFrame(
                        [{col: count for col, count in updates.items()}]
                    )
                else:
                    # Increment columns, create if missing
                    for col, count in updates.items():
                        if col in failure_df.columns:
                            failure_df.loc[:, col] = (
                                failure_df[col].fillna(0).astype(int) + count
                            )
                        else:
                            failure_df[col] = count

            # Read-modify-rewrite under the lock, so it needs the same atomic
            # write as designs.csv or a kill truncates the counters.
            self._atomic_write_df(failure_df, self.layout.failure_csv)

    @staticmethod
    def _atomic_write_df(df: pd.DataFrame, csv_path) -> None:
        """Write a DataFrame to csv_path so the file is never seen truncated.

        Writes to a sibling temp file, fsyncs it, then os.replace()s it into
        place. The rename is atomic within a filesystem, so a crash at any
        point leaves either the complete old file or the complete new one.
        A plain to_csv(path) opens O_TRUNC and streams through an 8 KiB
        buffer, leaving the file shorter than its contents for the whole
        write -- and a truncated CSV is NaN-padded by pd.read_csv rather than
        rejected, so the next read-concat-rewrite makes the loss permanent.

        The caller must already hold the file's lock.
        """
        csv_path = Path(csv_path)
        tmp_path = csv_path.with_name(f"{csv_path.name}.tmp.{os.getpid()}")

        # Sweep temp files orphaned by an earlier SIGKILL. The caller holds
        # this CSV's lock, so no other writer's temp for it can be live.
        for stale in csv_path.parent.glob(f"{csv_path.name}.tmp.*"):
            if stale != tmp_path:
                try:
                    stale.unlink()
                except OSError as e:
                    # Best-effort, but not silent: a read-only mount or a
                    # permissions problem would otherwise accumulate temp files
                    # forever with no signal.
                    log.info(
                        f"Warning: could not remove stale temp file {stale} "
                        f"({type(e).__name__}: {e})"
                    )

        try:
            with open(tmp_path, "w", newline="") as fh:
                df.to_csv(fh, index=False)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, csv_path)
            # fsync the directory so the rename itself survives a crash
            dir_fd = os.open(csv_path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def _concat_and_write_locked(self, csv_path, trajectory_df: pd.DataFrame) -> None:
        """Concat a row onto a CSV under its lock, replacing the file atomically.

        Read-concat-rewrite, so a row with new columns widens the file instead
        of being appended under a mismatched header -- but via _atomic_write_df,
        so the rewrite cannot truncate the file.
        """
        csv_path = Path(csv_path)
        with FileLock(f"{str(csv_path)}.lock"):
            if not csv_path.exists() or csv_path.stat().st_size == 0:
                self._atomic_write_df(trajectory_df, csv_path)
                return
            try:
                existing_df = pd.read_csv(csv_path)
            except Exception as e:
                # Unreadable CSV: move it aside and start a clean file rather
                # than appending this row under a header that may not match.
                quarantine = csv_path.with_name(
                    f"{csv_path.name}.corrupt.{os.getpid()}."
                    f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                )
                # log.warning, and no print kwargs: this used to be
                # log.info(..., file=..., flush=True), which raises
                # TypeError and killed the recovery before the quarantine.
                log.warning(
                    f"{csv_path} could not be parsed ({type(e).__name__}: {e}). "
                    f"Moving it to {quarantine} and starting a fresh file. "
                    f"The quarantined rows are NOT in the new CSV -- recover them by hand."
                )
                os.replace(csv_path, quarantine)
                self._atomic_write_df(trajectory_df, csv_path)
                return
            self._atomic_write_df(
                pd.concat([existing_df, trajectory_df], ignore_index=True, sort=False),
                csv_path,
            )

    def save_to_all_trajectories(self, trajectory: Trajectory) -> None:
        """Thread-safe append of trajectory data to all_trajectories.csv only.

        Writes trajectory information to the master all_trajectories.csv without
        saving to any category-specific CSV or copying structure files. Useful for
        recording designs that failed early (e.g. hallucination) where no structure
        or filter data exists.

        Args:
            trajectory (Trajectory): Trajectory object with whatever metrics are
                available at the point of failure.
        """
        trajectory_data = trajectory.get_trajectory()
        # No bucket row and no structure are written, so a save_location left
        # over from construction would point at a directory with neither.
        trajectory_data["save_location"] = ""
        trajectory_df = _with_required_columns(pd.DataFrame([trajectory_data]))
        self._append_to_all_csv(trajectory_df)

    def _append_to_all_csv(self, trajectory_df: pd.DataFrame) -> None:
        """Thread-safe append a single-row DataFrame to all_trajectories.csv."""
        self._concat_and_write_locked(self.layout.all_csv, trajectory_df)

    def save_trajectory(self, trajectory: Trajectory) -> int:
        """Save trajectory data and structure with thread-safe operations.
        
        Saves trajectory information to the appropriate directory based on its
        save_location attribute. Handles both structure file copying and CSV
        data persistence with thread-safe file locking to prevent race conditions
        in multi-device environments.
        
        The method performs the following operations:
        1. Copies structure file to appropriate directory
        2. Updates category-specific CSV file with trajectory data
        3. Updates master all_trajectories.csv with common columns
        
        Args:
            trajectory (Trajectory): Complete trajectory object containing all
                metrics, metadata, and file paths for saving.
                
        Returns:
            int: 0 on successful save operation
            
        Raises:
            ValueError: If trajectory.save_location is not one of the valid options
                ('trajectories', 'redesign_candidates', 'filter_failed', 'accepted')
        """
        trajectory_data = trajectory.get_trajectory()
        trajectory_df = _with_required_columns(pd.DataFrame([trajectory_data]))

        # Determine structure and CSV paths
        if trajectory.save_location == "trajectories":
            structure_path = (
                self.layout.trajectories
                / "structures"
                / f"{trajectory.design_name}.pdb"
            )
            csv_path = self.layout.trajectories / "designs.csv"
        elif trajectory.save_location == "redesign_candidates":
            structure_path = (
                self.layout.redesign_candidates
                / "structures"
                / f"{trajectory.design_name}.pdb"
            )
            csv_path = self.layout.redesign_candidates / "designs.csv"
        elif trajectory.save_location == "filter_failed":
            structure_path = (
                self.layout.filter_failed
                / "structures"
                / f"{trajectory.design_name}.pdb"
            )
            csv_path = self.layout.filter_failed / "designs.csv"
        elif trajectory.save_location == "accepted":
            structure_path = (
                self.layout.accepted / "structures" / f"{trajectory.design_name}.pdb"
            )
            csv_path = self.layout.accepted / "designs.csv"
        else:
            raise ValueError(f"Invalid save location: {trajectory.save_location}")

        log.info(f"Saving trajectory structure to {structure_path}")
        log.info(f"Saving trajectory data to {csv_path}")

        # Structure files are not locked; they are unique per design. A bare
        # string is normalised to a one-element list, "" to nothing.
        final_structs = trajectory.final_struct
        if isinstance(final_structs, str):
            final_structs = [final_structs] if final_structs else []
        for i,struct in enumerate(final_structs):
            new_path = str(structure_path).replace(".pdb", f"_{i}.pdb")
            shutil.copy(struct, new_path)

        self._concat_and_write_locked(csv_path, trajectory_df)

        self._append_to_all_csv(trajectory_df)

        return 0