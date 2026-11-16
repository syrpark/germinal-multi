"""
Run Protenix for antibody structure prediction.

Protenix is an open-source implementation of AlphaFold 3 by ByteDance:
https://github.com/bytedance/Protenix

Nothing here is imported unless a run config selects the protenix backend.

Attribution:
Protenix Team et al. (2026). Protenix-v1: Toward High-Accuracy Open-Source
Biomolecular Structure Prediction. bioRxiv.
https://doi.org/10.64898/2026.02.05.703733

License: Apache 2.0
"""

import subprocess
import tempfile
import os
import hashlib

from functools import lru_cache
import json
import shutil
import numpy as np
import pandas as pd

from typing import Union, List, Optional, Sequence
from Bio import PDB

from germinal.utils.gpu import child_env
from germinal.utils.log import get_logger

log = get_logger(__name__)




def _unwrap(val):
    """Unwrap Protenix's single-element list serialization.

    Protenix's ``save_json`` converts tensors via ``.tolist()``.  A shape-[1]
    tensor becomes ``[value]`` and shape-[1, N] becomes ``[[v0, v1, ...]]``.
    This helper normalises both cases so downstream code receives plain
    scalars or 1-D lists, matching what the AF3 pipeline expects.
    """
    if isinstance(val, list) and len(val) == 1 and isinstance(val[0], list):
        return val[0]          # [[v0, v1, ...]] -> [v0, v1, ...]
    if isinstance(val, list) and len(val) == 1:
        return val[0]          # [value] -> value
    return val


def create_protenix_input(
    binder_seq: str,
    target_seq: Union[str, List[str]],
    binder_chain: str,
    target_chains: Union[str, List[str]],
    design_name: str,
    msa_paths: dict = None,
) -> list:
    """
    Create input JSON data for Protenix inference.

    Protenix uses a different JSON schema from AF3: sequences are specified as
    'proteinChain' entries (not 'protein' with chain IDs). Input is a list of
    job dicts.

    Chain order matters downstream: targets are emitted first, then the binder,
    so the binder's chain index is always len(target_chains). `extract_scores`
    and the folding router both rely on that.

    Args:
        binder_seq: Amino acid sequence of the binder protein.
        target_seq: Amino acid sequence(s) of the target.
        binder_chain: Chain ID for the binder protein.
        target_chains: Chain ID(s) for the target.
        design_name: Name identifier for this design.
        msa_paths: Optional mapping of chain IDs to absolute MSA file paths.

    Returns:
        list: Protenix-compatible input JSON (list of one job dict).
    """
    if isinstance(target_chains, str):
        target_chains = target_chains.split(",")
    if isinstance(target_seq, str):
        target_seq = [target_seq]
    if msa_paths is None:
        msa_paths = {}

    sequences = []

    for i, chain_id in enumerate(target_chains):
        target_entry = {"proteinChain": {"sequence": target_seq[i], "count": 1}}
        if chain_id in msa_paths:
            target_entry["proteinChain"]["unpairedMsaPath"] = msa_paths[chain_id]
        sequences.append(target_entry)

    # Always give the binder an unpairedMsaPath, even a single-sequence stub:
    # Protenix starts a ~3 min remote search whenever any chain lacks one.
    binder_entry = {"proteinChain": {"sequence": binder_seq, "count": 1}}
    if binder_chain in msa_paths:
        binder_entry["proteinChain"]["unpairedMsaPath"] = msa_paths[binder_chain]
    elif msa_paths:
        any_msa_path = next(iter(msa_paths.values()))
        binder_msa_dir = os.path.dirname(any_msa_path)
        binder_msa_path = os.path.join(binder_msa_dir, f"binder_{design_name}.a3m")
        with open(binder_msa_path, "w") as f:
            f.write(f">binder\n{binder_seq}\n")
        binder_entry["proteinChain"]["unpairedMsaPath"] = os.path.abspath(
            binder_msa_path
        )
    sequences.append(binder_entry)

    return [{"name": design_name, "sequences": sequences}]


def get_or_generate_msas(
    target_seqs: List[str],
    target_chains: List[str],
    binder_seq: str,
    binder_chain: str,
    output_dir: str,
    msa_method: str,
    msa_scope: str,
    run_settings: dict,
) -> dict:
    """
    Get existing target MSAs or generate new ones, for Protenix input JSONs.

    Shares the AF3 backend's run-level MSA cache (``msa_cache_dir``), keyed
    by target sequence: a target folded by AF3 earlier in the run does not pay
    for a second colabfold search when Protenix folds it, and vice versa.

    Args:
        target_seqs: Target protein sequences.
        target_chains: Target chain IDs.
        binder_seq: Binder protein sequence. Gets a real MSA under the same
            msa_scope af3 uses ("binder_masked") and a
            single-sequence stub otherwise, so the two backends do not fold
            the same design against different alignments.
        binder_chain: Binder chain ID.
        output_dir: Base directory for MSA storage.
        msa_method: "colabfold_remote" or "colabfold_local".
        msa_scope: "none", "target_only" or "binder_masked".
        run_settings: Configuration with MSA database paths and the cache dir.

    Returns:
        dict: Mapping of chain IDs to absolute MSA file paths.
    """
    from germinal.filters.af3 import (
        generate_local_msa,
        call_generate_colabfold_msa_with_timeout,
        _load_cached_target_msa,
        _store_target_msa,
        resolve_binder_msa,
    )

    msa_cache_dir = run_settings.get("msa_cache_dir")
    msa_paths = {}
    msa_dir = os.path.join(output_dir, "protenix_inputs")
    os.makedirs(os.path.join(msa_dir, "msas"), exist_ok=True)

    for i, chain_id in enumerate(target_chains):
        sequence = target_seqs[i]
        # Key the filename by the SEQUENCE, not the chain id: every leg uses
        # target_chain "A", so a chain-keyed name collapses them onto one path.
        seq_digest = hashlib.sha256(sequence.encode()).hexdigest()[:16]
        design_name_msa = f"target_{seq_digest}"
        msa_filename = f"msas/{design_name_msa}.a3m"

        # Already on disk, from this backend or from the AF3 one.
        found = False
        for base_dir in (msa_dir, os.path.join(output_dir, "af3_inputs")):
            existing = os.path.join(base_dir, msa_filename)
            if os.path.exists(existing):
                msa_paths[chain_id] = os.path.abspath(existing)
                found = True
                break
        if found:
            continue

        # Run-level cache, shared with the AF3 backend and across array tasks.
        dest = os.path.join(msa_dir, msa_filename)
        if _load_cached_target_msa(msa_cache_dir, sequence, dest):
            msa_paths[chain_id] = os.path.abspath(dest)
            continue

        if msa_method == "colabfold_local":
            rel_path = generate_local_msa(
                sequence,
                design_name_msa,
                msa_dir,
                run_settings["msa_db_dir"],
                use_metagenomic_db=run_settings.get("use_metagenomic_db", False),
            )
        elif msa_method == "colabfold_remote":
            rel_path = call_generate_colabfold_msa_with_timeout(
                sequence,
                design_name_msa,
                msa_dir,
                use_metagenomic_db=run_settings.get("use_metagenomic_db", False),
            )
        else:
            continue

        if rel_path:
            full_path = os.path.join(msa_dir, rel_path)
            if os.path.exists(full_path):
                msa_paths[chain_id] = os.path.abspath(full_path)
                _store_target_msa(msa_cache_dir, sequence, full_path)

    # Binder MSA, under exactly the af3 rule: a real search for
    # "local"/"colabfold", the single-sequence stub otherwise.
    if msa_scope == "binder_masked":
        # One implementation, shared with af3 -- see af3.resolve_binder_msa.
        # Backends are a config choice; how a binder MSA is obtained is not.
        rel, _ = resolve_binder_msa(
            binder_seq=binder_seq,
            output_dir=msa_dir,
            msa_method=msa_method,
            msa_cache_dir=msa_cache_dir,
            cdr_positions=run_settings.get("cdr_positions"),
            msa_db_dir=run_settings.get("msa_db_dir"),
            use_metagenomic_db=run_settings.get("use_metagenomic_db", False),
        )
        if rel:
            msa_paths[binder_chain] = os.path.abspath(
                os.path.join(msa_dir, rel)
            )


    return msa_paths


# --- Output parsing ---


def _find_samples(results_folder: str) -> list:
    """Every sample Protenix wrote, as (ranking_score, summary, cif, full_data).

    Protenix writes one summary_confidence_*.json per (seed, diffusion sample).
    The CIF and the full-data JSON sit beside it under predictable names.
    """
    samples = []
    for root, _dirs, files in os.walk(results_folder):
        for name in files:
            if "summary_confidence" not in name or not name.endswith(".json"):
                continue
            conf_path = os.path.join(root, name)
            with open(conf_path, "r") as fh:
                conf = json.load(fh)
            cif_path = os.path.join(
                root, name.replace("_summary_confidence_", "_").replace(".json", ".cif")
            )
            full_data_path = os.path.join(
                root, name.replace("_summary_confidence_", "_full_data_")
            )
            samples.append(
                {
                    "ranking_score": conf.get("ranking_score", 0.0),
                    "summary": conf,
                    "cif": cif_path,
                    "full_data": full_data_path,
                    "seed": _seed_from_path(root),
                }
            )
    return samples


def _seed_from_path(path: str) -> int:
    """The seed a sample belongs to, from Protenix's ``seed_<n>`` directory.

    The filename carries only the rank within a seed, so the directory is
    the only place the seed can come from.

    Raises:
        ValueError: when no seed_<n> component is present. A placeholder
            would collapse every sample into one group.
    """
    for part in os.path.normpath(path).split(os.sep):
        if part.startswith("seed_") and part[len("seed_"):].isdigit():
            return int(part[len("seed_"):])
    raise ValueError(
        f"No seed_<n> directory in the Protenix output path {path!r}, so the "
        f"sample cannot be attributed to a seed. sample_select_mode and "
        f"seed_select_mode both depend on that grouping; without it the "
        f"selection silently degrades to worst-of-the-flat-pool. Check "
        f"whether this Protenix version still writes "
        f"<out_dir>/<dataset>/<name>/seed_<seed>/predictions/."
    )


def _derive_ipsae(sample: dict, pae_matrix: np.ndarray, full_data: dict):
    """Derive ipSAE, pDockQ2 and LIS from Protenix's PAE and per-atom pLDDT.

    Protenix reports no ipSAE but does report its inputs, so they are
    reshaped into what the `ipsae` CLI expects: atom_plddt rescaled 0-1 to
    0-100, token_pair_pae renamed to pae, summary_confidence to the plural.

    Returns lowercase keys matching the AF3 backend, or None if the inputs
    were not written. Both `lis` and `LIS` are set, as readers differ.
    """
    atom_plddt_raw = full_data.get("atom_plddt", [])
    if len(atom_plddt_raw) == 0 or pae_matrix.size <= 1:
        log.info(
            "WARNING: Protenix wrote no atom_plddt/PAE for this sample, so ipSAE "
            "cannot be derived. ipSAE gates the recycle loop -- if this repeats, "
            "check that need_atom_confidence is on in the Protenix config."
        )
        return None

    af3_compat_path = sample["full_data"] + ".af3compat.json"
    summary_compat_path = af3_compat_path.replace("full_data", "summary_confidences")
    try:
        with open(af3_compat_path, "w") as fh:
            json.dump(
                {
                    "atom_plddts": (np.array(atom_plddt_raw) * 100.0).tolist(),
                    "pae": pae_matrix.tolist(),
                },
                fh,
            )
        with open(summary_compat_path, "w") as fh:
            json.dump(
                {"chain_pair_iptm": sample["summary"].get("chain_pair_iptm", [])}, fh
            )

        result = subprocess.run(
            ["ipsae", af3_compat_path, sample["cif"], "10", "10"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            # af3.py uses check=True here for the same call. Raise rather than
            # fall through: the existence check below only catches this while
            # the tool writes nothing on failure, and a stale or partial
            # *_10_10.txt would otherwise be parsed as real scores.
            raise subprocess.CalledProcessError(
                result.returncode,
                ["ipsae", af3_compat_path, sample["cif"], "10", "10"],
                output=result.stdout,
                stderr=result.stderr,
            )

        out_file = os.path.join(
            os.path.dirname(os.path.abspath(sample["cif"])),
            os.path.basename(sample["cif"]).replace(".cif", "_10_10.txt"),
        )
        if not os.path.exists(out_file):
            log.info(
                "WARNING: ipsae produced no output file for this Protenix sample; "
                "recording ipSAE as unavailable."
            )
            return None

        df = pd.read_csv(out_file, sep=r"\s+")
        # .min(), matching af3.py: the ipsae CLI writes both chain
        # directions, and worse-of-the-two is the convention everywhere here.
        lis = float(df["LIS"].iloc[-1])
        return {
            "ipsae": float(df["ipSAE"].min()),
            "pdockq2": float(df["pDockQ2"].iloc[-1]),
            "lis": lis,
            "LIS": lis,
        }
    except (OSError, pd.errors.ParserError, KeyError, IndexError, ValueError) as e:
        # Narrow on purpose. A broad except here also swallows parse-side bugs
        # -- a renamed ipSAE/pDockQ2 column, an empty frame -- which would read
        # as "no ipSAE for this design" for an entire campaign rather than as
        # the code defect it is.
        log.warning(
            f"ipSAE derivation failed for Protenix output "
            f"({type(e).__name__}: {e})"
        )
        return None
    finally:
        for tmp in (af3_compat_path, summary_compat_path):
            if os.path.exists(tmp):
                os.remove(tmp)


def _scores_from_sample(sample: dict, binder_chain_idx: int) -> tuple:
    """Build an AF3-shaped scores dict for one Protenix sample.

    Returns (scores, pae_matrix, full_data) so the caller can derive ipSAE
    without re-reading the full-data JSON.
    """
    summary = sample["summary"]
    scores = {}

    # Protenix's summary "plddt" is 0-100 (atom_plddt.mean() * 100); the AF3
    # scores dict is 0-1, so normalise. "chain_plddt" is already 0-1.
    scores["plddt"] = np.float64(_unwrap(summary.get("plddt", 0))) / 100.0
    chain_plddt = _unwrap(summary.get("chain_plddt", []))
    if len(chain_plddt) > binder_chain_idx:
        scores["plddt_binder"] = np.float64(chain_plddt[binder_chain_idx])
    else:
        scores["plddt_binder"] = scores["plddt"]

    pae_matrix = np.array([[0.0]])
    token_asym_id = np.array([0])
    full_data = {}
    if sample["full_data"] and os.path.exists(sample["full_data"]):
        with open(sample["full_data"], "r") as fh:
            full_data = json.load(fh)
        if "token_pair_pae" in full_data:
            pae_matrix = np.array(full_data["token_pair_pae"])
        if "token_asym_id" in full_data:
            token_asym_id = np.array(full_data["token_asym_id"])

    scores["pae_matrix"] = pae_matrix
    scores["pae"] = np.mean(pae_matrix) if pae_matrix.size > 1 else np.float64(0.0)

    if pae_matrix.size > 1 and token_asym_id.size > 1:
        binder_mask = token_asym_id == binder_chain_idx
        scores["binder_pae"] = (
            np.mean(pae_matrix[np.ix_(binder_mask, binder_mask)])
            if binder_mask.any()
            else np.float64(0.0)
        )
    else:
        scores["binder_pae"] = np.float64(0.0)

    scores["chain_pair_pae_min"] = np.array(_unwrap(summary.get("chain_pair_gpde", [[0.0]])))
    scores["ptm"] = np.float64(_unwrap(summary.get("ptm", 0.0)))
    scores["iptm"] = np.float64(_unwrap(summary.get("iptm", 0.0)))
    scores["chain_iptm"] = np.array(_unwrap(summary.get("chain_iptm", [0.0])))
    scores["chain_ptm"] = np.array(_unwrap(summary.get("chain_ptm", [0.0])))
    scores["aggregate_score"] = np.float64(_unwrap(summary.get("ranking_score", 0.0)))

    return scores, pae_matrix, full_data


def _cif_to_pdb(cif_path: str, pdb_path: str) -> str:
    parser = PDB.MMCIFParser(QUIET=True)
    io_pdb = PDB.PDBIO()
    io_pdb.set_structure(parser.get_structure("structure", cif_path))
    io_pdb.save(pdb_path)
    return pdb_path


def extract_structure_and_scores(
    output_dir: str,
    design_name: str,
    binder_chain_idx: int = 0,
    *,
    sample_select_mode: str,
    seed_select_mode: str,
    want_all: bool = False,
    cleanup: bool = True,
    expected_seeds: Optional[Sequence[int]] = None,
) -> Union[tuple, list]:
    """
    Extract predicted structure(s) and confidence scores from Protenix output.

    Mirrors ``af3.extract_structure_and_scores`` so the two backends are
    interchangeable behind ``germinal.filters.folding``.

    Args:
        output_dir: Directory containing the Protenix output folder.
        design_name: Name identifier for the design.
        binder_chain_idx: Index of the binder chain (0-based). Targets are
            emitted first by create_protenix_input, so this is len(targets).
        sample_select_mode: "best" or "worst", within one seed's diffusion
            samples. "best": a seed's samples differ only by sampling noise.
        seed_select_mode: "best" or "worst", across seeds. "worst": seeds are
            independent attempts at the complex.
        want_all: Also return every seed's representative, for the ensemble.
        cleanup: Delete the Protenix results folder after extraction.
        expected_seeds: The seeds that were asked for. Checked against the
            seeds actually found, so a grouping that silently collapsed is an
            error rather than a quietly harsher selection.

    Returns:
        (pdb_path, scores, ipsae), or (chosen, all_samples) when want_all.
    """
    results_folder = os.path.join(output_dir, design_name)
    samples = _find_samples(results_folder)
    if not samples:
        raise FileNotFoundError(
            f"No Protenix summary confidence files found in {results_folder}"
        )

    for axis, mode in (
        ("sample_select_mode", sample_select_mode),
        ("seed_select_mode", seed_select_mode),
    ):
        if mode not in ("best", "worst"):
            raise ValueError(
                f"{axis}={mode!r} is not supported by the protenix backend; "
                f"use 'best' or 'worst'."
            )

    try:
        # One representative per seed, then one of those. Selecting over the
        # flat pool would charge the design for its unluckiest diffusion draw.
        by_seed = {}
        for sample in samples:
            by_seed.setdefault(sample["seed"], []).append(sample)
        if expected_seeds is not None and set(by_seed) != set(expected_seeds):
            raise ValueError(
                f"Protenix wrote samples for seeds {sorted(by_seed)} but "
                f"{sorted(set(expected_seeds))} were requested, for "
                f"{design_name}. Selection is per-seed, so a missing or "
                f"unexpected seed changes what sample_select_mode="
                f"{sample_select_mode!r} / seed_select_mode="
                f"{seed_select_mode!r} actually picks."
            )
        pick_sample = max if sample_select_mode == "best" else min
        per_seed = [
            pick_sample(group, key=lambda s: s["ranking_score"])
            for group in by_seed.values()
        ]
        per_seed.sort(key=lambda s: s["ranking_score"], reverse=True)
        pick_seed = max if seed_select_mode == "best" else min
        selected = pick_seed(per_seed, key=lambda s: s["ranking_score"])
        log.info(
            f"Protenix {design_name}: {sample_select_mode}-sample / "
            f"{seed_select_mode}-seed over {len(by_seed)} seed(s), "
            f"{len(samples)} prediction(s) -> ranking_score="
            f"{selected['ranking_score']:.4f}"
        )

        chosen = per_seed if want_all else [selected]
        results = []
        for rank, sample in enumerate(chosen):
            if not os.path.exists(sample["cif"]):
                raise FileNotFoundError(
                    f"Protenix summary named a CIF that does not exist: {sample['cif']}"
                )
            suffix = f"_protenix_{rank}" if want_all else "_protenix"
            pdb_path = _cif_to_pdb(
                sample["cif"], os.path.join(output_dir, f"{design_name}{suffix}.pdb")
            )
            scores, pae_matrix, full_data = _scores_from_sample(
                sample, binder_chain_idx
            )
            ipsae = _derive_ipsae(sample, pae_matrix, full_data)
            results.append((pdb_path, scores, ipsae))

        if not want_all:
            return results[0]
        # per_seed is ranked best-first, and `selected` is one of its members,
        # so the chosen result is found by identity rather than re-ranked.
        chosen_idx = next(
            i for i, s_ in enumerate(chosen) if s_ is selected
        )
        return results[chosen_idx], results
    finally:
        if cleanup:
            shutil.rmtree(results_folder, ignore_errors=True)


# --- Inference ---


@lru_cache(maxsize=4)
def _cli_capabilities(conda_env: str) -> tuple:
    """Discover the prediction subcommand and whether atom confidence is a flag.

    The subcommand is ``predict`` on PyPI 0.5.5 and ``pred`` on 2.x, so it
    is probed once per environment per process and cached.

    Returns:
        (subcommand, supports_need_atom_confidence)

    Raises:
        RuntimeError: if the CLI is unreachable or exposes neither name.
    """
    try:
        help_out = subprocess.run(
            ["conda", "run", "-n", conda_env, "protenix", "--help"],
            capture_output=True, text=True, timeout=300,
        ).stdout
    except (OSError, subprocess.SubprocessError) as e:
        raise RuntimeError(
            f"Could not run the Protenix CLI in conda env {conda_env!r}: {e}. "
            f"See environment_setup.md section 6."
        ) from e

    subcommand = next(
        (c for c in ("pred", "predict") if f"\n  {c} " in help_out), None
    )
    if subcommand is None:
        raise RuntimeError(
            f"The Protenix CLI in conda env {conda_env!r} exposes neither a "
            f"'pred' nor a 'predict' command. Got:\n{help_out}"
        )

    sub_help = subprocess.run(
        ["conda", "run", "-n", conda_env, "protenix", subcommand, "--help"],
        capture_output=True, text=True, timeout=300,
    ).stdout
    return subcommand, "--need_atom_confidence" in sub_help


def run_inference(
    input_json: list,
    output_dir: str,
    seeds: list,
    run_settings: dict,
    *,
    num_samples: int,
    num_cycles: int,
    folder: dict,
) -> str:
    """
    Execute Protenix prediction for one complex, in its own conda environment.

    One complex per call. ``run_inference_batch`` folds several in a single
    invocation; this is the path the recycle loop takes, where there is only
    ever one candidate to fold at a time.

    Args:
        input_json: Protenix-compatible input JSON (list of one job dict).
        output_dir: Directory to save prediction outputs.
        seeds: Random seeds for inference.
        run_settings: Configuration; see the protenix_* keys below.

    Returns:
        The design name whose output folder was written.

    Raises:
        subprocess.CalledProcessError: if Protenix exits non-zero.
    """
    os.makedirs(output_dir, exist_ok=True)

    input_dir = os.path.join(output_dir, "protenix_inputs")
    os.makedirs(input_dir, exist_ok=True)
    design_name = input_json[0]["name"]
    input_path = os.path.join(input_dir, f"{design_name}.json")
    with open(input_path, "w") as fh:
        json.dump(input_json, fh, indent=2)

    _invoke_protenix(
        input_path, output_dir, seeds,
        num_samples=num_samples, num_cycles=num_cycles, folder=folder,
    )
    return design_name


def _invoke_protenix(
    input_path: str,
    output_dir: str,
    seeds: list,
    *,
    num_samples: int,
    num_cycles: int,
    folder: dict,
) -> None:
    """Run the Protenix CLI once over ``input_path``.

    ``input_path`` may be a single JSON or a DIRECTORY of them: the CLI
    expands a directory itself and loads the model once for the whole set,
    which is what makes batching worth doing.

    Raises:
        subprocess.CalledProcessError: if Protenix exits non-zero.
    """
    conda_env = folder["conda_env"]
    model_name = folder["model_name"]
    subcommand, supports_atom_confidence = _cli_capabilities(conda_env)

    # Long flags only: 2.x adds short aliases, 0.5.5 does not. The recycle
    # loop wants these speed knobs low.
    run_cmds = [
        "conda", "run", "-n", conda_env, "--no-capture-output",
        "protenix", subcommand,
        "--input", input_path,
        "--out_dir", str(output_dir),
        "--seeds", ",".join(str(s) for s in seeds),
        "--model_name", model_name,
        # Backend-agnostic, resolved from this stage's folding config so AF3
        # and Protenix cannot silently run different budgets.
        "--sample", str(num_samples),
        "--cycle", str(num_cycles),
        "--step", str(folder["num_steps"]),
        "--use_msa", str(folder["use_msa"]),
    ]

    if supports_atom_confidence:
        # Not optional: it defaults to false, and without it Protenix writes
        # no atom_plddt, which is what ipSAE -- the recycle gate -- needs.
        run_cmds += ["--need_atom_confidence", "True"]
    else:
        log.info(
            "WARNING: this Protenix build has no --need_atom_confidence flag, "
            "so it may write no atom_plddt/PAE and ipSAE cannot be derived. "
            "ipSAE gates the recycle loop; upgrade Protenix (see "
            "environment_setup.md section 6) if every design scores 0.0."
        )

    log.info(f"Running Protenix: {' '.join(run_cmds)}")
    popen = subprocess.Popen(
        run_cmds,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        env=child_env(),
    )
    _stdout, stderr_data = popen.communicate()
    if popen.returncode:
        if stderr_data:
            log.info(f"Protenix stderr:\n{stderr_data}")
        raise subprocess.CalledProcessError(popen.returncode, run_cmds)


def run_protenix_batch(
    designs: list,
    output_dir: str,
    seed: list,
    run_settings: dict,
    *,
    binder_chain: str,
    target_chains: list,
    msa_method: str,
    msa_scope: str,
    want_all: bool,
    cleanup: bool,
    num_samples: int,
    num_cycles: int,
    sample_select_mode: str,
    seed_select_mode: str,
    folder: dict,
) -> list:
    """Fold several complexes in ONE Protenix invocation.

    The CLI expands a directory of input JSONs itself and loads the model once
    for the whole set, so N complexes cost one process start-up rather than N.

    Inputs are staged into a directory unique to this call. Pointing the CLI at
    the shared ``protenix_inputs/`` would re-fold every design the run has ever
    written there.

    Args:
        designs: ``(design_name, binder_seq, target_seq)`` triples, in the
            order results are expected back.

    Returns:
        One result per design, positionally. A design whose fold or extraction
        raised is ``None`` in its slot, matching the sequential path.
    """
    if not designs:
        return []

    os.makedirs(output_dir, exist_ok=True)
    batch_dir = tempfile.mkdtemp(prefix="protenix_batch_", dir=output_dir)

    staged = []
    for design_name, binder_seq, target_seq in designs:
        try:
            msa_paths = {}
            if msa_scope != "none":
                msa_paths = get_or_generate_msas(
                    target_seq, target_chains, binder_seq, binder_chain,
                    output_dir, msa_method, msa_scope, run_settings,
                )
            input_json = create_protenix_input(
                binder_seq, target_seq, binder_chain, target_chains,
                design_name, msa_paths,
            )
            with open(os.path.join(batch_dir, f"{design_name}.json"), "w") as fh:
                json.dump(input_json, fh, indent=2)
            staged.append(design_name)
        except Exception as e:
            # Staging is per design, so one bad input must not cost the batch.
            log.info(
                f"  Could not stage {design_name} for Protenix: "
                f"{type(e).__name__}: {e}"
            )

    if staged:
        try:
            _invoke_protenix(
                batch_dir, output_dir, seed,
                num_samples=num_samples, num_cycles=num_cycles, folder=folder,
            )
        except Exception as e:
            # The whole invocation failed, so no design has output. Report it
            # as every design failing rather than raising, which is what the
            # router expects from a batch backend.
            log.info(f"  Protenix batch of {len(staged)} failed: "
                     f"{type(e).__name__}: {e}")
            staged = []
        finally:
            shutil.rmtree(batch_dir, ignore_errors=True)
    else:
        shutil.rmtree(batch_dir, ignore_errors=True)

    results = []
    for design_name, _binder_seq, _target_seq in designs:
        if design_name not in staged:
            results.append(None)
            continue
        try:
            results.append(extract_structure_and_scores(
                output_dir,
                design_name,
                binder_chain_idx=len(target_chains),
                sample_select_mode=sample_select_mode,
                seed_select_mode=seed_select_mode,
                want_all=want_all,
                cleanup=cleanup,
                expected_seeds=seed,
            ))
        except Exception as e:
            log.info(f"  Could not read Protenix output for {design_name}: "
                     f"{type(e).__name__}: {e}")
            results.append(None)
    return results


def run_protenix(
    binder_seq: str,
    target_seq: Union[str, List[str]],
    target_chains: Union[List[str], str],
    output_dir: str,
    design_name: str,
    seed: Union[int, List[int]],
    run_settings: dict,
    binder_chain: str = "B",
    msa_method: str = "colabfold_remote",
    msa_scope: str = "none",
    want_all: bool = False,
    cleanup: bool = True,
    *,
    num_samples: int,
    num_cycles: int,
    sample_select_mode: str,
    seed_select_mode: str,
    folder: dict,
) -> Union[tuple, list]:
    """
    Fold one antibody-target complex with Protenix.

    High-level interface mirroring ``af3.run_structure_prediction``.

    Returns:
        (pdb_path, scores, ipsae), or (chosen, all_samples) when want_all.
    """
    if isinstance(seed, int):
        seed = [seed]
    if isinstance(target_chains, str):
        target_chains = target_chains.split(",")
    if isinstance(target_seq, str):
        target_seq = [target_seq]

    msa_paths = {}
    if msa_scope != "none":
        msa_paths = get_or_generate_msas(
            target_seq,
            target_chains,
            binder_seq,
            binder_chain,
            output_dir,
            msa_method,
            msa_scope,
            run_settings,
        )

    input_json = create_protenix_input(
        binder_seq, target_seq, binder_chain, target_chains, design_name, msa_paths
    )

    run_inference(
        input_json,
        output_dir,
        seed,
        run_settings,
        num_samples=num_samples,
        num_cycles=num_cycles,
        folder=folder,
    )

    # Targets are written first by create_protenix_input.
    return extract_structure_and_scores(
        output_dir,
        design_name,
        binder_chain_idx=len(target_chains),
        sample_select_mode=sample_select_mode,
        seed_select_mode=seed_select_mode,
        want_all=want_all,
        cleanup=cleanup,
        expected_seeds=seed,
    )
