"""
Run AlphaFold 3 for antibody structure prediction.

This script is based on the AlphaFold 3  from the AlphaFold 3 repository:
https://github.com/deepmind/alphafold/

Attribution:
If you use this code, the AlphaFold 3 model, or outputs produced by it in your research, please cite:

Abramson, J., Adler, J., Dunger, J., Evans, R., Green, T., Pritzel, A., Ronneberger, O., Willmore, L., Ballard, A. J., Bambrick, J., et al. (2024). Accurate structure prediction of biomolecular interactions with AlphaFold 3. Nature, 630(8016), 493–500. https://doi.org/10.1038/s41586-024-07487-w

Copyright 2024 DeepMind Technologies Limited.

License:
- The AlphaFold 3 source code is licensed under the Creative Commons Attribution-Non-Commercial ShareAlike International License, Version 4.0 (CC-BY-NC-SA 4.0). See: https://github.com/google-deepmind/alphafold3/blob/main/LICENSE
- The AlphaFold 3 model parameters are subject to the AlphaFold 3 Model Parameters Terms of Use: https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

You may not use this file or the model parameters except in compliance with these terms.
"""

import subprocess
import argparse
import tempfile
import os
import glob
import json
import hashlib
import multiprocessing
import shutil
import numpy as np
import pandas as pd

from filelock import FileLock
from typing import List
from colabfold.colabfold import run_mmseqs2
from Bio import PDB
from concurrent.futures import ProcessPoolExecutor, TimeoutError

from germinal.utils.gpu import child_env
from germinal.utils.log import get_logger

log = get_logger(__name__)



class MsaGenerationError(RuntimeError):
    """An MSA was requested for a sequence and could not be produced.

    Raised rather than returning an empty path: folding without an MSA does
    not fail, it quietly produces a much worse structure. msa_scope "none" is
    unaffected, since asking for no MSA is not a failure to get one.
    """


# Host variables that must not reach the AF3 container, whose jaxlib rejects
# the deprecated spelling. Our own process needs it, so strip it here.
_CONTAINER_ENV_BLOCKLIST = ("XLA_PYTHON_CLIENT_MEM_FRACTION",)


def _container_env():
    """Host environment minus the variables that break the container's JAX."""
    return child_env(_CONTAINER_ENV_BLOCKLIST)


def create_input_dict(
    binder_seq, target_seq, binder_chain, target_chains, design_name, seed
):
    """
    Create input JSON data for AlphaFold3 inference.

    Args:
        binder_seq (str): Amino acid sequence of the binder protein.
        target_seq (str): Amino acid sequence of the target protein.
        binder_chain (str): Chain ID for the binder protein.
        target_chains (Union[str, List[str]]): Chain ID(s) for the target protein.
        design_name (str): Name identifier for this design.
        seed (Union[int, List[int]]): Random seed(s) for model inference.

    Returns:
        dict: AF3-compatible input JSON structure.
    """
    if isinstance(target_chains, str):
        target_chains = [target_chains]
    if isinstance(seed, int):
        seed = [seed]

    input_json_data = {
        "name": design_name,
        "modelSeeds": seed,
        "dialect": "alphafold3",
        "version": 2,
    }

    sequences = []
    for chain_id in [binder_chain] + target_chains:
        sequences.append(
            {
                "protein": {
                    "id": [chain_id],
                    "sequence": binder_seq if chain_id == binder_chain else target_seq,
                    "unpairedMsa": "",
                    "pairedMsa": "",
                }
            }
        )
    input_json_data["sequences"] = sequences
    return input_json_data


def remove_a3m_insertions(a3m_path):
    """
    Remove insertion characters from A3M MSA file for AF3 compatibility.

    AlphaFold3 requires MSA sequences to have uniform length, so we remove
    lowercase insertion characters that indicate gaps in the alignment.

    Args:
        a3m_path (str): Path to the A3M format MSA file to process.
    """
    with open(a3m_path, "r") as a3m_file:
        lines = a3m_file.readlines()
    new_lines = []
    for line in lines:
        line = line.replace("\x00", "")
        if line.startswith("#") or line.startswith(">"):
            new_lines.append(line)
        else:
            new_lines.append("".join(c for c in line if not c.islower()))
    with open(a3m_path, "w") as a3m_file:
        a3m_file.writelines(new_lines)


def mask_cdrs_in_a3m(a3m_path: str, cdr_positions) -> int:
    """Gap the CDR columns of every homolog row in a BINDER MSA.

    The framework's homologs are what make the complex foldable, but every
    homolog's CDR is a germline loop the design deliberately is not. The
    query row is left as written: it must keep matching the input JSON's
    `sequence`. Lowercase a3m insertions occupy no query column and are
    dropped when anchored inside a CDR.

    Args:
        a3m_path: A3M file to rewrite in place.
        cdr_positions: Zero-based residue positions of every CDR in the binder,
            as run_settings["cdr_positions"].

    Returns:
        int: Number of homolog rows masked.
    """
    cdrs = {int(p) for p in (cdr_positions or [])}
    if not cdrs or not os.path.exists(a3m_path):
        return 0

    with open(a3m_path) as fh:
        lines = fh.read().splitlines()

    out, masked, seq_idx = [], 0, -1
    for line in lines:
        if line.startswith("#"):
            out.append(line)
            continue
        if line.startswith(">"):
            out.append(line)
            seq_idx += 1
            continue
        if seq_idx <= 0:
            # The query. Never altered -- see the docstring.
            out.append(line)
            continue
        chars, col = [], 0
        for ch in line:
            if ch.islower():
                # Insertion: anchored to the query column we are about to
                # reach, so it belongs to the CDR if that column does.
                if col not in cdrs:
                    chars.append(ch)
                continue
            chars.append("-" if col in cdrs else ch)
            col += 1
        out.append("".join(chars))
        masked += 1

    with open(a3m_path, "w") as fh:
        fh.write("\n".join(out) + "\n")
    return masked


class CdrMaskError(Exception):
    """A binder MSA was going to be folded without its CDRs masked."""


def _a3m_query_row(a3m_path: str):
    """The query row of an A3M, with insertions removed, or None if empty.

    The query defines the chain, so its length in upper-case/gap columns is
    the binder length the CDR positions must be indexed against.
    """
    with open(a3m_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith(">"):
                continue
            return "".join(c for c in line if not c.islower())
    return None


def mask_binder_cdrs(
    msa_path, cdr_positions, design_name: str = "", binder_seq: str = None
) -> int:
    """Apply the CDR mask to a binder MSA, whichever backend produced it.

    Both folding backends route their binder MSA through here so the two
    cannot drift. Every failure is raised, never skipped: an absent or
    misplaced mask produces a plausible number with no error anywhere.

    Args:
        msa_path: The binder A3M to rewrite in place.
        cdr_positions: Zero-based CDR residue positions, derived from the
            configured ``cdr_lengths``/``fw_lengths`` in utils/config.py.
        design_name: For messages.
        binder_seq: The binder sequence being folded. When given, the query
            row must equal it -- this is the check that catches a config
            whose cdr_lengths describe a different binder than the one on
            the bench.

    Returns:
        int: Number of homolog rows masked (0 for a single-sequence stub).

    Raises:
        CdrMaskError: on a missing path, missing/empty cdr_positions, a
            query row that disagrees with ``binder_seq``, or positions that
            fall outside the binder.
    """
    label = design_name or msa_path

    positions = sorted({int(p) for p in (cdr_positions or [])})
    if not positions:
        raise CdrMaskError(
            f"no cdr_positions supplied for binder MSA {label}. These are "
            f"derived from cdr_lengths/fw_lengths in utils/config.py; an "
            f"empty list means the binder would fold against germline CDRs."
        )
    if not msa_path:
        raise CdrMaskError(
            f"no binder MSA path for {label}, but CDR masking was requested "
            f"(msa_scope='binder_masked'). The search above did not produce "
            f"a usable alignment."
        )
    if not os.path.exists(str(msa_path)):
        raise CdrMaskError(f"binder MSA for {label} is missing at {msa_path}")

    query = _a3m_query_row(str(msa_path))
    if query is None:
        raise CdrMaskError(f"binder MSA for {label} has no sequence rows: {msa_path}")

    if binder_seq is not None and query != binder_seq:
        raise CdrMaskError(
            f"binder MSA query row for {label} is not the binder being "
            f"folded ({len(query)} residues vs {len(binder_seq)}). Masking "
            f"would gap the wrong columns."
        )
    if positions[-1] >= len(query):
        raise CdrMaskError(
            f"CDR position {positions[-1]} is outside the {len(query)}-residue "
            f"binder for {label}. cdr_lengths/fw_lengths in the run config do "
            f"not describe this binder."
        )

    n = mask_cdrs_in_a3m(str(msa_path), positions)
    if n:
        log.info(f"Masked CDR columns in {n} homolog(s) of the binder MSA "
              f"for {label}")
    else:
        # A stub is legitimate (query only); a multi-row file that masks
        # nothing is not, and used to be indistinguishable.
        log.warning(f"binder MSA for {label} has no homolog rows to mask "
              f"({msa_path}). The binder is folding single-sequence.")
    return n


def generate_local_msa(
    sequence,
    design_name,
    output_dir,
    msa_db_dir,
    use_gpu=False,
    use_gpu_server=False,
    use_metagenomic_db=False,
):
    """
    Generate an unpaired MSA for the given sequence using colabfold_search.

    Args:
        sequence (str): The sequence to generate an MSA for.
        design_name (str): The name of the design. Used to name the output file.
        output_dir (str): The directory to save the output files to.
        msa_db_dir (str): The directory containing the MSA databases.
        use_gpu (bool): Whether to use a GPU for the search.
        use_gpu_server (bool): Whether to use a GPU server for the search.
        use_metagenomic_db (bool): Whether to use the metagenomic database.
    """
    # Start GPU server processes for accelerated MSA search
    if use_gpu_server:
        log.info("Starting GPU server...")
        gpu_server_dir = os.path.join(msa_db_dir, "colabfold_envdb_202108_db")
        uniref30_db_dir = os.path.join(msa_db_dir, "uniref30_2302_db")
        gpu_server_process = subprocess.Popen(
            [
                "mmseqs",
                "gpuserver",
                gpu_server_dir,
                "--max-seqs",
                "10000",
                "--db-load-mode",
                "0",
                "--prefilter-mode",
                "1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=child_env(),
        )
        log.info(f"GPU server started at PID {gpu_server_process.pid}")
        gpu_server_process.wait()
        uniref30_server_process = subprocess.Popen(
            [
                "mmseqs",
                "gpuserver",
                uniref30_db_dir,
                "--max-seqs",
                "10000",
                "--db-load-mode",
                "0",
                "--prefilter-mode",
                "1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=child_env(),
        )
        log.info(f"Uniref30 server started at PID {uniref30_server_process.pid}")
        uniref30_server_process.wait()

    # Create temporary FASTA file for sequence input
    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_path = os.path.join(tmpdir, f"{design_name}.fasta")
        with open(fasta_path, "w") as fasta_file:
            fasta_file.write(f">{design_name}\n{sequence}\n")
        # Create output directory for MSA results
        msa_out_dir = os.path.join(output_dir, "msas")
        os.makedirs(msa_out_dir, exist_ok=True)
        # Build ColabFold search command with appropriate flags
        cmd = ["colabfold_search"]
        if use_gpu:
            cmd += ["--gpu", "1"]
        if use_gpu_server:
            cmd += ["--gpu-server", "1"]
        if not use_metagenomic_db:
            cmd += ["--use-env", "0"]
        cmd += [fasta_path, msa_db_dir, msa_out_dir]
        log.info(f"Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, env=child_env())
        except subprocess.CalledProcessError as e:
            log.info(
                f"colabfold_search failed for {design_name}: {e}."
            )
            raise MsaGenerationError(
                f"colabfold_search failed for {design_name}: {e}"
            ) from e
        # Locate generated A3M file and rename it appropriately
        a3m_file = os.path.join(msa_out_dir, "0.a3m")
        if os.path.exists(a3m_file):
            # Rename MSA file to match design name
            shutil.move(a3m_file, os.path.join(msa_out_dir, f"{design_name}.a3m"))
            a3m_file = os.path.join(msa_out_dir, f"{design_name}.a3m")
            # Process MSA to ensure uniform sequence length for AF3
            remove_a3m_insertions(a3m_file)
            return os.path.relpath(a3m_file, output_dir)
        else:
            log.info(
                f"colabfold_search failed for {design_name}: MSA not found at {a3m_file}."
            )
            raise MsaGenerationError(
                f"colabfold_search produced no MSA for {design_name}: expected "
                f"{a3m_file}"
            )


def _clear_partial_msa_output(output_dir, design_name):
    """Remove a half-built search directory left by a killed worker.

    A timed-out worker is SIGKILLed mid-download, leaving `{name}_all/` without
    its out.tar.gz. The next search reuses that directory and fails on the
    missing tarball, so the retry inherits the first attempt's corruption.
    """
    partial = os.path.join(output_dir, f"{design_name}_all")
    if os.path.isdir(partial):
        log.info(f"Clearing partial MSA output at {partial}")
        shutil.rmtree(partial, ignore_errors=True)


def call_generate_colabfold_msa_with_timeout(
    sequence, design_name, output_dir, timeout=600, use_metagenomic_db=False,
    attempts=2,
):
    """
    Generate MSA via ColabFold API with timeout protection.

    A timeout kills the worker and raises MsaGenerationError rather than
    folding without an alignment, which would read as a weak design. The 600s
    default is sized for the binder's deep framework alignment, not a target.

    Args:
        sequence (str): Protein sequence for MSA generation.
        design_name (str): Design identifier for output naming.
        output_dir (str): Directory to save MSA output.
        timeout (int): Maximum time in seconds to wait for MSA.
        use_metagenomic_db (bool): Whether to include metagenomic databases.

    Returns:
        str: Relative path to the generated MSA file.

    Raises:
        MsaGenerationError: on timeout, or if the search produced no MSA.
    """

    # Use a 'spawn' context: os.fork() corrupts JAX's thread state in the child.
    mp_context = multiprocessing.get_context("spawn")
    last_error = None

    for attempt in range(1, attempts + 1):
        _clear_partial_msa_output(output_dir, design_name)
        # NOT a `with` block. Its __exit__ calls shutdown(wait=True), which
        # joins a worker that is still running -- so a hung ColabFold fetch
        # would block there for the rest of the job, timeout or no timeout.
        exe = ProcessPoolExecutor(max_workers=1, mp_context=mp_context)
        try:
            fut = exe.submit(
                generate_colabfold_msa,
                sequence,
                design_name,
                output_dir,
                use_metagenomic_db,
            )
            try:
                return fut.result(timeout=timeout)
            except TimeoutError:
                # fut.cancel() is a no-op once the future is running and
                # shutdown(wait=False) does not signal workers, so the kill
                # is what makes this timeout real.
                _kill_pool_workers(exe)
                last_error = f"timed out after {timeout}s"
            except MsaGenerationError as e:
                # Raised inside the worker and retried on the same terms as a
                # timeout: both are usually transient.
                last_error = str(e)
        finally:
            exe.shutdown(wait=False)

        log.warning(
            f"colabfold_search failed for {design_name}: {last_error} "
            f"(attempt {attempt}/{attempts})."
        )

    raise MsaGenerationError(
        f"colabfold_search failed for {design_name} after {attempts} "
        f"attempt(s); last failure was {last_error}"
    )


def _kill_pool_workers(exe):
    """Terminate a ProcessPoolExecutor's worker processes.

    There is no public API for this: cancelling a running future does
    nothing and shutdown() either waits for the worker or leaves it running.
    `_processes` is private, hence the defensive getattr -- if a future
    Python renames it this degrades to the old behaviour rather than raising.
    """
    for proc in list(getattr(exe, "_processes", {}).values()):
        try:
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=5)
        except (OSError, ValueError) as e:
            log.warning(f"could not kill MSA worker {proc.pid}: {e}")


def generate_colabfold_msa(sequence, design_name, output_dir, use_metagenomic_db=False):
    """
    Generate unpaired MSA using ColabFold's remote API.

    This function calls the ColabFold web service to generate a multiple
    sequence alignment for the input protein sequence.

    Args:
        sequence (str): Protein sequence for MSA generation.
        design_name (str): Design identifier for output file naming.
        output_dir (str): Directory to save MSA results.
        use_metagenomic_db (bool): Whether to search metagenomic databases.

    Returns:
        str: Relative path to generated MSA file, or empty string if failed.
    """
    try:
        log.info(
            f"Running colabfold_search for {design_name} with use_env={use_metagenomic_db}"
        )
        run_mmseqs2(
            sequence,
            os.path.join(output_dir, f"{design_name}"),
            use_env=use_metagenomic_db,
        )
        log.info(f"colabfold_search finished for {design_name}")
    except Exception as e:
        log.info(
            f"colabfold_search failed for {design_name}: {e}."
        )
        raise MsaGenerationError(
            f"colabfold_search failed for {design_name}: {e}"
        ) from e

    old_msa_path = os.path.join(output_dir, f"{design_name}_all", "uniref.a3m")
    # Existence first: remove_a3m_insertions() opens the path unguarded, so
    # checking it later would raise instead of taking the fallback below.
    if os.path.exists(old_msa_path):
        # Process MSA to ensure uniform sequence length for AF3 compatibility
        remove_a3m_insertions(old_msa_path)
        new_msa_path = os.path.join(output_dir, f"msas/{design_name}.a3m")
        os.makedirs(os.path.dirname(new_msa_path), exist_ok=True)
        shutil.copyfile(old_msa_path, new_msa_path)
        shutil.rmtree(os.path.join(output_dir, f"{design_name}_all"))
        return os.path.relpath(new_msa_path, output_dir)
    else:
        log.info(
            f"colabfold_search failed for {design_name}: MSA not found at "
            f"{old_msa_path}."
        )
        raise MsaGenerationError(
            f"colabfold_search produced no MSA for {design_name}: expected "
            f"{old_msa_path}"
        )


def binder_framework_key(binder_seq: str, cdr_positions) -> str:
    """Digest of the binder FRAMEWORK -- the only part an MSA describes.

    The CDR columns of every homolog are gapped immediately after the search
    (mask_cdrs_in_a3m), and redesign only ever substitutes within
    cdr_positions, so two binders that share a framework end up folding
    against byte-identical homolog blocks. Keying the cache on the whole
    binder sequence therefore paid a fresh ~14,700-sequence colabfold_search
    for every candidate: measured on run 7, 60 searches in egfr_logits for
    ONE distinct framework, and the search that killed kp3_logits was one of
    them.

    The CDR positions are folded into the digest as well, so a run that
    slices the same sequence into different loops cannot collide with this
    entry.
    """
    cdrs = {int(p) for p in (cdr_positions or [])}
    framework = "".join(
        ch for i, ch in enumerate(binder_seq) if i not in cdrs
    )
    payload = f"{len(binder_seq)}:{sorted(cdrs)}:{framework}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _framework_msa_cache_path(msa_cache_dir, framework_key: str):
    """Path this framework's MSA would occupy in the run-level cache.

    Shares the directory the target MSAs use, so one cache serves both and
    every array task writing into the results root sees the same entries.
    """
    if not msa_cache_dir:
        return None
    return os.path.join(str(msa_cache_dir), f"binder_fw_{framework_key}.a3m")


def load_cached_framework_msa(
    msa_cache_dir, framework_key: str, binder_seq: str, dest_path: str
) -> bool:
    """Write dest_path = cached homolog block under THIS binder's query row.

    The homolog rows are reusable across a framework; the query row is not.
    It defines the chain and must equal the `sequence` field of the input
    JSON, so it is replaced rather than copied -- mask_binder_cdrs enforces
    exactly that and would raise on a stale one.

    Returns False (rather than raising) on any mismatch, so a cache that
    cannot be trusted simply costs a fresh search.
    """
    cache_path = _framework_msa_cache_path(msa_cache_dir, framework_key)
    if cache_path is None or not os.path.exists(cache_path):
        return False
    if os.path.getsize(cache_path) == 0:
        return False

    with FileLock(f"{cache_path}.lock"):
        lines = open(cache_path).read().splitlines()

    out, seq_idx, swapped = [], -1, False
    for line in lines:
        if line.startswith("#"):
            out.append(line)
            continue
        if line.startswith(">"):
            seq_idx += 1
            out.append(line)
            continue
        if seq_idx == 0 and not swapped:
            n_cols = sum(1 for c in line if not c.islower())
            if n_cols != len(binder_seq):
                log.info(
                    f"Warning: cached framework MSA {framework_key} has "
                    f"{n_cols} query columns but this binder has "
                    f"{len(binder_seq)}; searching instead of reusing."
                )
                return False
            out.append(binder_seq)
            swapped = True
            continue
        out.append(line)

    if not swapped:
        log.info(
            f"Warning: cached framework MSA {framework_key} has no query "
            f"row; searching instead of reusing."
        )
        return False

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "w") as fh:
        fh.write("\n".join(out) + "\n")
    return True


def store_framework_msa(msa_cache_dir, framework_key: str, src_path: str) -> None:
    """Publish a freshly masked binder MSA into the run-level cache.

    Stored after masking, so every later reuse inherits the gapped CDRs.
    A caching failure is warned about and swallowed: the MSA is on disk.
    """
    cache_path = _framework_msa_cache_path(msa_cache_dir, framework_key)
    if cache_path is None or not os.path.exists(src_path):
        return
    if os.path.getsize(src_path) == 0:
        return
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        tmp_path = f"{cache_path}.tmp.{os.getpid()}"
        with FileLock(f"{cache_path}.lock"):
            if os.path.exists(cache_path):
                return
            shutil.copyfile(src_path, tmp_path)
            os.replace(tmp_path, cache_path)
    except OSError as e:
        log.warning(f"could not cache framework MSA at {cache_path}: {e}")


def resolve_binder_msa(
    binder_seq: str,
    output_dir: str,
    msa_method: str,
    msa_cache_dir,
    cdr_positions,
    msa_db_dir=None,
    use_metagenomic_db: bool = False,
):
    """Get this binder's CDR-masked MSA, searching only if nothing has it.

    One implementation for every folding backend, so they cannot drift. Two
    keys: the FILE is per binder sequence, whose query row must be this exact
    binder; the CACHE is per framework, since gapped CDRs make the homolog
    block identical across binders sharing one.

    Returns (relative_path, reused), relative to output_dir as AF3 requires;
    "" means no MSA. The result is always CDR-masked, cache hit included.
    """
    framework_key = binder_framework_key(binder_seq, cdr_positions)
    binder_name = f"binder_{hashlib.sha256(binder_seq.encode()).hexdigest()[:16]}"
    relative_path = os.path.join("msas", f"{binder_name}.a3m")
    full_path = os.path.join(output_dir, relative_path)

    if os.path.exists(full_path):
        return relative_path, True

    if load_cached_framework_msa(
        msa_cache_dir, framework_key, binder_seq, full_path
    ):
        log.info(
            f"Reused cached framework MSA {framework_key} for {binder_name} "
            f"(no colabfold_search needed)"
        )
        return relative_path, True

    if msa_method == "colabfold_local":
        produced = generate_local_msa(
            binder_seq, binder_name, output_dir, msa_db_dir,
            use_metagenomic_db=use_metagenomic_db,
        )
    elif msa_method == "colabfold_remote":
        produced = call_generate_colabfold_msa_with_timeout(
            binder_seq, binder_name, output_dir,
            use_metagenomic_db=use_metagenomic_db,
        )
    else:
        raise ValueError(
            f"Unknown msa_method {msa_method!r}; expected 'colabfold_remote' "
            f"or 'colabfold_local'."
        )

    if not produced:
        log.info(
            f"WARNING: no binder MSA produced for {binder_name} by "
            f"{msa_method}; the binder will fold single-sequence."
        )
        return "", False

    produced_full = os.path.join(output_dir, produced)
    mask_binder_cdrs(
        produced_full, cdr_positions, binder_name, binder_seq=binder_seq
    )
    # Published only after masking, so every reuse inherits gapped CDRs.
    store_framework_msa(msa_cache_dir, framework_key, produced_full)
    return produced, False


def _target_msa_cache_path(msa_cache_dir, sequence: str):
    """Path this target sequence's MSA would occupy in the run-level cache.

    Keyed by the sequence itself, not by target_id or design name, since the
    results root is shared by every design and array task. None if no cache.
    """
    if not msa_cache_dir:
        return None
    digest = hashlib.sha256(sequence.encode()).hexdigest()[:16]
    return os.path.join(str(msa_cache_dir), f"target_{digest}.a3m")


def _load_cached_target_msa(msa_cache_dir, sequence: str, dest_path: str) -> bool:
    """Copy a cached target MSA into dest_path. True if the cache had one.

    The MSA is copied rather than referenced so unpairedMsaPath stays relative
    to the AF3 input directory, which is what the container is given.
    """
    cache_path = _target_msa_cache_path(msa_cache_dir, sequence)
    if cache_path is None or not os.path.exists(cache_path):
        return False
    if os.path.getsize(cache_path) == 0:
        return False
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with FileLock(f"{cache_path}.lock"):
        shutil.copyfile(cache_path, dest_path)
    return True


def _store_target_msa(msa_cache_dir, sequence: str, src_path: str) -> None:
    """Publish a freshly generated target MSA into the run-level cache.

    Temp file plus rename under the entry's lock, so a concurrent array task
    never reads a half-copied a3m. A failure here is warned about, not fatal.
    """
    cache_path = _target_msa_cache_path(msa_cache_dir, sequence)
    if cache_path is None or not os.path.exists(src_path):
        return
    if os.path.getsize(src_path) == 0:
        return
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        tmp_path = f"{cache_path}.tmp.{os.getpid()}"
        with FileLock(f"{cache_path}.lock"):
            if os.path.exists(cache_path):
                return
            shutil.copyfile(src_path, tmp_path)
            os.replace(tmp_path, cache_path)
    except OSError as e:
        log.warning(f"could not cache target MSA at {cache_path}: {e}")


def generate_msas(
    input_json_data: dict,
    msa_db_dir: str,
    output_dir: str,
    binder_chain: str,
    msa_method: str,
    msa_scope: str,
    use_metagenomic_db: bool = False,
    target_id=None,
    msa_cache_dir=None,
    cdr_positions=None,
) -> dict:
    """
    Generate Multiple Sequence Alignments (MSAs) for protein chains.

    Creates MSAs for each protein chain in the input data using the specified
    method. MSAs improve structure prediction accuracy by providing evolutionary
    context. The function supports different MSA generation modes:
    - "local": Use local ColabFold search with databases
    - "colabfold": Use ColabFold remote API
    - "target": Generate MSA only for target protein

    Args:
        input_json_data (dict): AF3 input JSON containing sequence information.
        msa_db_dir (str): Path to local MSA databases (for local mode).
        output_dir (str): Directory to save generated MSA files.
        binder_chain (str): Chain identifier for the binder protein.
        msa_method (str): "colabfold_remote" or "colabfold_local".
        msa_scope (str): "none", "target_only" or "binder_masked".
        use_metagenomic_db (bool): Include metagenomic databases in search.
        msa_cache_dir: Run-level directory in which target MSAs are cached
            across designs and processes. None disables caching.

    Returns:
        dict: Updated input JSON with MSA paths added to each sequence.
    """
    updated_sequences = []
    for seq_idx, sequence_info in enumerate(input_json_data["sequences"]):
        chain = sequence_info["protein"]["id"][0]
        sequence = sequence_info["protein"]["sequence"]
        is_target = chain != binder_chain
        if is_target:
            # Check if target MSA already exists to avoid regeneration
            design_name = "target" if target_id is None else f"target_{target_id}"
            relative_msa_path = os.path.join(f"msas/{design_name}.a3m") 
            full_msa_path = os.path.join(output_dir, relative_msa_path)

            if os.path.exists(full_msa_path):
                sequence_info["protein"]["unpairedMsaPath"] = relative_msa_path
                updated_sequences.append(sequence_info)
                continue

            # Fall back to the run-level cache before searching again.
            if _load_cached_target_msa(msa_cache_dir, sequence, full_msa_path):
                sequence_info["protein"]["unpairedMsaPath"] = relative_msa_path
                updated_sequences.append(sequence_info)
                continue
        else:
            design_name = input_json_data["name"]
            # target_only: the binder folds single-sequence.
            if msa_scope != "binder_masked":
                updated_sequences.append(sequence_info)
                continue
            if msa_scope == "none":
                updated_sequences.append(sequence_info)
                continue
            # Shared with protenix. Keyed on the binder and its framework,
            # never on the job name, which embeds the target leg.
            relative_msa_path, _ = resolve_binder_msa(
                binder_seq=sequence,
                output_dir=output_dir,
                msa_method=msa_method,
                msa_cache_dir=msa_cache_dir,
                cdr_positions=cdr_positions,
                msa_db_dir=msa_db_dir,
                use_metagenomic_db=use_metagenomic_db,
            )
            if relative_msa_path:
                sequence_info["protein"]["unpairedMsaPath"] = relative_msa_path
                log.info(
                    f"Binder MSA at {relative_msa_path} for {design_name} "
                    f"and sequence {seq_idx}"
                )
            updated_sequences.append(sequence_info)
            continue

        if msa_scope == "none":
            # Deliberately folding without an MSA: the one path that may
            # proceed with no unpairedMsaPath.
            updated_sequences.append(sequence_info)
            continue

        # Generate MSA using the specified method
        if msa_method == "colabfold_local":
            relative_msa_path = generate_local_msa(
                sequence,
                design_name,
                output_dir,
                msa_db_dir,
                use_metagenomic_db=use_metagenomic_db,
            )
        elif msa_method == "colabfold_remote":
            relative_msa_path = call_generate_colabfold_msa_with_timeout(
                sequence, design_name, output_dir, use_metagenomic_db=use_metagenomic_db
            )
        else:
            # Raise rather than fall through: `relative_msa_path` is bound
            # per iteration, so an unknown mode would reuse the last MSA.
            raise ValueError(
                f"Unknown msa_method {msa_method!r}; expected "
                f"'colabfold_remote' or 'colabfold_local'."
            )

        sequence_info["protein"]["unpairedMsaPath"] = relative_msa_path

        # The binder never reaches here: it returns above, via
        # resolve_binder_msa, which masks and caches as part of its contract.
        if is_target and relative_msa_path:
            _store_target_msa(
                msa_cache_dir, sequence, os.path.join(output_dir, relative_msa_path)
            )

        updated_sequences.append(sequence_info)
        if relative_msa_path != "":
            log.info(
                f"Generated MSA at {relative_msa_path} for {design_name} and sequence {seq_idx}"
            )
        else:
            log.info(f"No MSA generated for {design_name} and sequence {seq_idx}")
    input_json_data["sequences"] = updated_sequences
    return input_json_data


def _extract_single_sample(output_dir, design_name, af3_results_folder,
                           af3_structure, summary_confidences, full_confidences,
                           pdb_suffix="_af3", ipsae_cif=None):
    """
    Extract a single AF3 sample: convert CIF→PDB, parse confidence scores, compute ipSAE.

    Args:
        output_dir (str): Directory for output PDB.
        design_name (str): Design identifier.
        af3_results_folder (str): Root AF3 results folder.
        af3_structure (str): Path to .cif structure file (used for PDB conversion).
        summary_confidences (str): Path to summary_confidences.json.
        full_confidences (str): Path to confidences.json.
        pdb_suffix (str): Suffix for PDB filename.
        ipsae_cif (str, optional): Override CIF path for ipSAE calculation.
            If None, uses af3_structure. Used by decoupled mode to compute
            ipSAE from the worst sample's CIF while using the best structure
            for PDB conversion.

    Returns:
        tuple: (pdb_path, af3_scores, ipsae)
    """
    pdb_path = os.path.join(output_dir, f"{design_name}{pdb_suffix}.pdb")
    parser = PDB.MMCIFParser(QUIET=True)
    bio_io = PDB.PDBIO()
    structure = parser.get_structure("structure", af3_structure)
    bio_io.set_structure(structure)
    bio_io.save(pdb_path)

    af3_scores = {}
    with open(summary_confidences, "r") as f:
        summary_metrics = json.load(f)
    with open(full_confidences, "r") as f:
        full_metrics = json.load(f)

    af3_scores["plddt"] = np.mean(full_metrics["atom_plddts"]) / 100
    pae_matrix = np.array(full_metrics["pae"])
    af3_scores["pae_matrix"] = pae_matrix
    af3_scores["pae"] = np.mean(pae_matrix)
    af3_scores["ptm"] = summary_metrics["ptm"]
    af3_scores["iptm"] = summary_metrics["iptm"]
    af3_scores["aggregate_score"] = summary_metrics["ranking_score"]

    ipsae_cif_path = ipsae_cif if ipsae_cif is not None else af3_structure
    try:
        # check=True, or a broken ipsae tool degrades into a silent ipSAE of
        # 0.0 that rejects every design with no error anywhere.
        subprocess.run(
            ["ipsae", full_confidences, ipsae_cif_path, "10", "10"],
            capture_output=True,
            text=True,
            check=True,
        )
        ipsae_out_file = os.path.join(
            os.path.dirname(os.path.abspath(ipsae_cif_path)),
            os.path.basename(ipsae_cif_path).replace(".cif", "_10_10.txt"),
        )
        df = pd.read_csv(ipsae_out_file, delim_whitespace=True)
        ipsae = {
            "ipsae": df["ipSAE"].min(),
            "pdockq2": df["pDockQ2"].max(),
            "lis": df["LIS"].max(),
        }
    except subprocess.CalledProcessError as exc:
        # Report the cause: otherwise a persistently broken tool is
        # indistinguishable from a run of genuinely non-binding designs.
        log.info(
            f"ipSAE calculation failed! 'ipsae' exited {exc.returncode} for "
            f"{ipsae_cif_path}\n"
            f"  stderr: {(exc.stderr or '').strip()[:500]}\n"
            f"  stdout: {(exc.stdout or '').strip()[:500]}"
        )
        ipsae = None
    except (OSError, pd.errors.ParserError, KeyError, IndexError, ValueError) as exc:
        # Narrow on purpose: a broad except also swallows parse-side defects
        # (a renamed ipSAE/pDockQ2 column, an empty frame), which would read as
        # "no ipSAE for this design" across a whole campaign.
        log.info(
            f"ipSAE calculation failed! {type(exc).__name__}: {exc} "
            f"(cif: {ipsae_cif_path})"
        )
        ipsae = None

    return pdb_path, af3_scores, ipsae


def extract_structure_and_scores(
    output_dir,
    design_name,
    *,
    sample_select_mode="best",
    seed_select_mode="worst",
    decoupled=False,
    want_all=False,
    cleanup=True,
):
    """Extract the chosen AF3 prediction, or one per seed.

    Two decisions, in order: ``sample_select_mode`` picks within a seed
    (samples differ only by noise), then ``seed_select_mode`` picks across
    seeds (independent attempts at the complex).

    Args:
        sample_select_mode: "best" or "worst", within a seed.
        seed_select_mode: "best" or "worst", across seeds.
        decoupled: Structure for relax from the top-ranked prediction, scores
            and ipSAE from the selected one.
        want_all: Return every seed's representative instead of one, for the
            relax ensemble.
        cleanup: Delete the AF3 results folder afterwards.

    Returns:
        (pdb_path, scores, ipsae), or a list of those when want_all.
    """
    af3_results_folder = os.path.join(output_dir, design_name)
    ranking_csv = os.path.join(
        af3_results_folder, f"{design_name}_ranking_scores.csv"
    )
    ranking_df = pd.read_csv(ranking_csv)

    for axis, mode in (
        ("sample_select_mode", sample_select_mode),
        ("seed_select_mode", seed_select_mode),
    ):
        if mode not in ("best", "worst"):
            raise ValueError(
                f"{axis}={mode!r} is not a selection mode; use 'best' or 'worst'."
            )

    def _paths(seed, sample):
        subfolder = f"seed-{seed}_sample-{sample}"
        prefix = f"{design_name}_{subfolder}"
        folder = os.path.join(af3_results_folder, subfolder)
        return (
            subfolder,
            os.path.join(folder, f"{prefix}_model.cif"),
            os.path.join(folder, f"{prefix}_summary_confidences.json"),
            os.path.join(folder, f"{prefix}_confidences.json"),
        )

    # One representative per seed.
    picker = "idxmax" if sample_select_mode == "best" else "idxmin"
    per_seed = ranking_df.loc[
        getattr(ranking_df.groupby("seed")["ranking_score"], picker)()
    ].reset_index(drop=True)

    if want_all:
        result = []
        for _, row in per_seed.iterrows():
            seed, sample = int(row["seed"]), int(row["sample"])
            _, cif, summary, full = _paths(seed, sample)
            result.append(
                _extract_single_sample(
                    output_dir, design_name, af3_results_folder,
                    cif, summary, full,
                    pdb_suffix=f"_af3_s{seed}_s{sample}",
                )
            )
    else:
        idx = (
            per_seed["ranking_score"].idxmin()
            if seed_select_mode == "worst"
            else per_seed["ranking_score"].idxmax()
        )
        row = per_seed.loc[idx]
        seed, sample = int(row["seed"]), int(row["sample"])
        subfolder, cif, summary, full = _paths(seed, sample)
        log.info(
            f"Selected {sample_select_mode}-sample / {seed_select_mode}-seed "
            f"for {design_name}: {subfolder} "
            f"(ranking_score={row['ranking_score']:.4f}, "
            f"{len(per_seed)} seed(s) of {len(ranking_df)} predictions)"
        )
        if decoupled:
            # Structure from AF3's own top-level pick, scores and ipSAE from
            # the selected prediction.
            structure = os.path.join(
                af3_results_folder, f"{design_name}_model.cif"
            )
            result = _extract_single_sample(
                output_dir, design_name, af3_results_folder,
                structure, summary, full,
                pdb_suffix="_af3_decoupled", ipsae_cif=cif,
            )
        else:
            result = _extract_single_sample(
                output_dir, design_name, af3_results_folder,
                cif, summary, full,
                pdb_suffix=f"_af3_{sample_select_mode}_{seed_select_mode}",
            )

    if cleanup:
        shutil.rmtree(af3_results_folder)

    return result


def _run_af3_inference(
    input_json: dict,
    output_dir: str,
    binder_chain: str,
    msa_method: str,
    msa_scope: str,
    run_settings: dict,
    target_id: int = None,
    *,
    folder: dict,
    num_samples: int,
    num_cycles: int,
) -> str:
    """
    Execute AlphaFold3 inference via Singularity container (no extraction).

    Runs AF3 inference only. Call extract_structure_and_scores() separately
    to parse the results.

    Args:
        input_json (dict): AF3-compatible input JSON with sequence information.
        output_dir (str): Directory to save prediction outputs.
        binder_chain (str): Chain identifier for the binder protein.
        msa_method (str): "colabfold_remote" or "colabfold_local".
        msa_scope (str): "none", "target_only" or "binder_masked".
        run_settings (dict): Configuration containing all AF3 paths and settings.
        target_id (int, optional): Target index within a multi-target run.

    Returns:
        str: The design_name used for AF3 output folder naming.
    """
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        log.info(f"Directory created at {output_dir}.")

    input_dir = os.path.join(output_dir, "af3_inputs")
    os.makedirs(input_dir, exist_ok=True)
    input_path = os.path.join(input_dir, f"{input_json['name']}.json")

    if msa_scope != "none":
        input_json = generate_msas(
            input_json,
            run_settings["msa_db_dir"],
            input_dir,
            binder_chain,
            msa_method,
            msa_scope,
            use_metagenomic_db=run_settings["use_metagenomic_db"],
            target_id=target_id,
            msa_cache_dir=run_settings.get("msa_cache_dir"),
            cdr_positions=run_settings.get("cdr_positions"),
        )

    with open(input_path, "w") as f:
        json.dump(input_json, f)

    af3_repo_path = folder["af3_repo_path"]
    weights_path = folder["af3_model_dir"]
    databases_path = folder["af3_db_dir"]
    sif_path = folder["af3_sif_path"]

    for name, path in [
        ("af3_repo_path", af3_repo_path),
        ("af3_model_dir", weights_path),
        ("af3_db_dir", databases_path),
        ("af3_sif_path", sif_path),
    ]:
        if path is None:
            raise ValueError(
                f"AF3 config '{name}' is null/None. Set it in your run config YAML."
            )

    # af3_model_dir may point at the weights file itself; bind its parent directory
    if os.path.isfile(weights_path):
        weights_path = os.path.dirname(weights_path)

    run_cmds = [
        "singularity",
        "exec",
        "--nv",
        "--env",
        "LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu",
        "--bind",
        f"{output_dir}:/root/af_output",
        "--bind",
        f"{input_dir}:/root/af_input",
        "--bind",
        f"{weights_path}:/root/models",
        "--bind",
        f"{databases_path}:/root/public_databases",
        "--bind",
        f"{af3_repo_path}:/root/alphafold3",
        sif_path,
        "python",
        "/root/alphafold3/run_alphafold.py",
        "--model_dir=/root/models",
        "--db_dir=/root/public_databases",
        f"--output_dir={output_dir}",
        f"--json_path={input_path}",
        # Same as the batch path: AF3's own defaults are 5 and 10, so leaving
        # these unset does not mean "neutral", it means the container chooses.
        f"--num_diffusion_samples={num_samples}",
        f"--num_recycles={num_cycles}",
    ]

    popen = subprocess.Popen(
        run_cmds,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        universal_newlines=True,
        env=_container_env(),
    )
    for line in popen.stdout:
        continue

    popen.stdout.close()
    return_code = popen.wait()

    if return_code:
        raise subprocess.CalledProcessError(return_code, run_cmds)

    return input_json["name"]


def _run_af3_batch_inference(
    input_jsons: List[dict],
    output_dir: str,
    binder_chain: str,
    msa_method: str,
    msa_scope: str,
    run_settings: dict,
    target_ids: List[int] = None,
    *,
    folder: dict,
    num_samples: int,
    num_cycles: int,
) -> List[str]:
    """
    Execute AlphaFold3 inference for many inputs in a single Singularity call.

    Writes every input JSON into one directory and runs AF3 with --input_dir, so
    the container start-up and model load are paid once for the whole batch
    instead of once per job.

    A job whose output is missing afterwards is reported and omitted from the
    return value rather than raising, so one bad design does not discard the
    results of every other design in the batch. Callers must therefore check
    which names came back. Only a non-zero exit from AF3 itself, or a batch in
    which nothing succeeded, raises.

    Args:
        input_jsons: AF3-compatible input JSON dicts.
        output_dir: Directory to write prediction outputs into.
        binder_chain: Chain identifier for the binder protein.
        msa_method: "colabfold_remote" or "colabfold_local".
        msa_scope: "none", "target_only" or "binder_masked".
        run_settings: Configuration holding the AF3 paths and settings.
        target_ids: Per-JSON target indices, used for target MSA reuse.

    Returns:
        The design names that produced output, in input order.
    """
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        log.info(f"Directory created at {output_dir}.")

    # PIDs collide across nodes on a shared FS, and --requeue can inherit a
    # dead run's dir. Key on the SLURM job/array task too.
    _batch_key = "_".join(
        str(x)
        for x in (
            os.environ.get("SLURM_JOB_ID", "nojob"),
            os.environ.get("SLURM_ARRAY_TASK_ID", "0"),
            os.getpid(),
        )
    )
    batch_input_dir = os.path.join(output_dir, f"af3_batch_inputs_{_batch_key}")
    os.makedirs(batch_input_dir, exist_ok=True)

    # Drop stale JSONs from an earlier failed run, but keep msas/ for reuse
    for stale_json in glob.glob(os.path.join(batch_input_dir, "*.json")):
        os.remove(stale_json)

    design_names = []
    for idx, input_json in enumerate(input_jsons):
        target_id = target_ids[idx] if target_ids is not None else None

        if msa_scope != "none":
            input_json = generate_msas(
                input_json,
                run_settings["msa_db_dir"],
                batch_input_dir,
                binder_chain,
                msa_method,
                msa_scope,
                use_metagenomic_db=run_settings["use_metagenomic_db"],
                target_id=target_id,
                msa_cache_dir=run_settings.get("msa_cache_dir"),
                cdr_positions=run_settings.get("cdr_positions"),
            )

        input_path = os.path.join(batch_input_dir, f"{input_json['name']}.json")
        with open(input_path, "w") as f:
            json.dump(input_json, f)
        design_names.append(input_json["name"])

    af3_repo_path = folder["af3_repo_path"]
    weights_path = folder["af3_model_dir"]
    databases_path = folder["af3_db_dir"]
    sif_path = folder["af3_sif_path"]

    for name, path in [
        ("af3_repo_path", af3_repo_path),
        ("af3_model_dir", weights_path),
        ("af3_db_dir", databases_path),
        ("af3_sif_path", sif_path),
    ]:
        if path is None:
            raise ValueError(
                f"AF3 config '{name}' is null/None. Set it in your run config YAML."
            )

    # af3_model_dir may point at the weights file itself; bind its parent directory
    if os.path.isfile(weights_path):
        weights_path = os.path.dirname(weights_path)

    run_cmds = [
        "singularity",
        "exec",
        "--nv",
        "--env",
        "LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu",
        "--bind",
        f"{output_dir}:/root/af_output",
        "--bind",
        f"{batch_input_dir}:/root/af_input",
        "--bind",
        f"{weights_path}:/root/models",
        "--bind",
        f"{databases_path}:/root/public_databases",
        "--bind",
        f"{af3_repo_path}:/root/alphafold3",
        sif_path,
        "python",
        "/root/alphafold3/run_alphafold.py",
        "--model_dir=/root/models",
        "--db_dir=/root/public_databases",
        f"--output_dir={output_dir}",
        f"--input_dir={batch_input_dir}",
        # Passed explicitly: AF3 defaults to 5 samples and 10 recycles, so
        # omitting them lets the container choose the sampling budget.
        f"--num_diffusion_samples={num_samples}",
        f"--num_recycles={num_cycles}",
    ]

    for name in design_names:
        job_output_dir = os.path.join(output_dir, name)
        if os.path.exists(job_output_dir):
            shutil.rmtree(job_output_dir)

    # communicate(), not a manual stdout drain: AF3 logs through absl to
    # stderr, so reading one pipe at a time deadlocks once the other fills.
    timeout_s = float(
        os.environ.get("AF3_BATCH_TIMEOUT_S", max(1800, 300 * len(input_jsons)))
    )
    popen = subprocess.Popen(
        run_cmds,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        env=_container_env(),
    )
    try:
        _, stderr_output = popen.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        popen.kill()
        _, stderr_output = popen.communicate()
        log.info(
            f"AF3 batch timed out after {timeout_s:.0f}s on "
            f"{len(input_jsons)} job(s); killed the container."
        )
        if stderr_output:
            log.info(f"AF3 stderr (tail):\n{stderr_output[-2000:]}")
        raise
    return_code = popen.returncode
    if return_code:
        log.info(f"AF3 batch inference failed (exit code {return_code})")
        if stderr_output:
            log.info(f"AF3 stderr:\n{stderr_output[-2000:]}")
        raise subprocess.CalledProcessError(return_code, run_cmds)

    succeeded, missing = [], []
    for name in design_names:
        expected_cif = os.path.join(output_dir, name, f"{name}_model.cif")
        if os.path.exists(expected_cif):
            succeeded.append(name)
        else:
            missing.append(name)

    if missing:
        log.info(
            f"  AF3 batch: {len(missing)}/{len(design_names)} outputs missing, "
            f"continuing without them: {missing}"
        )
    if not succeeded:
        raise FileNotFoundError(
            f"AF3 produced no output for any of the {len(design_names)} batched job(s). "
            f"First expected path: "
            f"{os.path.join(output_dir, design_names[0], design_names[0] + '_model.cif')}"
        )

    # Remove the JSONs but keep msas/ so later batches can reuse the alignments
    for json_file in glob.glob(os.path.join(batch_input_dir, "*.json")):
        os.remove(json_file)

    return succeeded


def _run_af3(
    input_json: dict,
    output_dir: str,
    binder_chain: str,
    msa_method: str,
    msa_scope: str,
    run_settings: dict,
    target_id: int = None,
    cleanup: bool = True,
    *,
    folder: dict,
    num_samples: int,
    num_cycles: int,
    sample_select_mode: str,
    seed_select_mode: str,
) -> tuple:
    """
    Execute AlphaFold3 structure prediction via Singularity container.

    Runs AF3 inference then extracts results. Backward-compatible wrapper
    around _run_af3_inference() + extract_structure_and_scores().

    Args:
        input_json (dict): AF3-compatible input JSON with sequence information.
        output_dir (str): Directory to save prediction outputs.
        binder_chain (str): Chain identifier for the binder protein.
        msa_method (str): "colabfold_remote" or "colabfold_local".
        msa_scope (str): "none", "target_only" or "binder_masked".
        run_settings (dict): Configuration containing all AF3 paths and settings.
        target_id (int, optional): Target index within a multi-target run.
        sample_select_mode (str): Within a seed, across its diffusion samples.
        seed_select_mode (str): Across seeds, over each seed's chosen sample.
        cleanup (bool): Whether to delete AF3 results folder after extraction.

    Returns:
        tuple or list: Result from extract_structure_and_scores().
    """
    design_name = _run_af3_inference(
        input_json,
        output_dir,
        binder_chain,
        msa_method,
        msa_scope,
        run_settings,
        target_id,
        folder=folder,
        num_samples=num_samples,
        num_cycles=num_cycles,
    )

    return extract_structure_and_scores(
        output_dir, design_name,
        sample_select_mode=sample_select_mode,
        seed_select_mode=seed_select_mode,
        cleanup=cleanup,
    )


def main(
    input_json: str,
    output_dir: str,
    msa_db_dir: str,
    binder_chain: str,
    msa_method: str,
    msa_scope: str,
    af3_repo_path: str,
    af3_sif_path: str,
    af3_model_dir: str,
    af3_db_dir: str,
    use_metagenomic_db: bool,
    num_samples: int = 5,
    num_cycles: int = 10,
    sample_select_mode: str = "best",
    seed_select_mode: str = "worst",
):
    """Standalone CLI. The pipeline goes through germinal.filters.folding,
    which resolves these four from the stage config; they are arguments here
    only because this entry point has no config to read."""
    log.info("Running AF3...")

    results = pd.DataFrame(columns=["name", "plddt", "iptm"])
    with open(input_json) as f:
        input_json_data = json.load(f)
        if isinstance(input_json_data, list):
            log.info(
                f"Detected list input with {len(input_json_data)} items. Running AF3 for each."
            )
            for i, input_json_dict in enumerate(input_json_data):
                run_settings = {
                    "msa_db_dir": msa_db_dir,
                    "use_metagenomic_db": use_metagenomic_db,
                }
                folder = {
                    "af3_repo_path": af3_repo_path,
                    "af3_sif_path": af3_sif_path,
                    "af3_model_dir": af3_model_dir,
                    "af3_db_dir": af3_db_dir,
                }
                pdb_path, scores = _run_af3(
                    input_json_dict,
                    output_dir,
                    binder_chain=binder_chain,
                    msa_method=msa_method,
                    msa_scope=msa_scope,
                    run_settings=run_settings,
                    folder=folder,
                    num_samples=num_samples,
                    num_cycles=num_cycles,
                    sample_select_mode=sample_select_mode,
                    seed_select_mode=seed_select_mode,
                )
                results = results.append(
                    {
                        "name": input_json_dict["name"],
                        "plddt": scores["plddt"],
                        "iptm": scores["iptm"],
                    },
                    ignore_index=True,
                )
                # save pae matrix
                pae_matrix_path = os.path.join(
                    output_dir, f"{input_json_dict['name']}_pae_matrix.npy"
                )
                np.save(pae_matrix_path, scores["pae_matrix"])

                log.info(
                    f"Folded {i + 1}/{len(input_json_data)} AF3 structures: design {input_json_dict['name']}, iptm {scores['iptm']}, plddt {scores['plddt']}"
                )
        else:
            run_settings = {
                "msa_db_dir": msa_db_dir,
                "use_metagenomic_db": use_metagenomic_db,
            }
            folder = {
                "af3_repo_path": af3_repo_path,
                "af3_sif_path": af3_sif_path,
                "af3_model_dir": af3_model_dir,
                "af3_db_dir": af3_db_dir,
            }
            pdb_path, scores = _run_af3(
                input_json_data,
                output_dir,
                binder_chain=binder_chain,
                msa_method=msa_method,
                msa_scope=msa_scope,
                run_settings=run_settings,
                folder=folder,
                num_samples=num_samples,
                num_cycles=num_cycles,
                sample_select_mode=sample_select_mode,
                seed_select_mode=seed_select_mode,
            )
            results = results.append(
                {
                    "name": input_json_data["name"],
                    "plddt": scores["plddt"],
                    "iptm": scores["iptm"],
                },
                ignore_index=True,
            )
            # save pae matrix
            pae_matrix_path = os.path.join(
                output_dir, f"{input_json_data['name']}_pae_matrix.npy"
            )
            np.save(pae_matrix_path, scores["pae_matrix"])

    results.to_csv(os.path.join(output_dir, "results.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_json",
        "-i",
        required=True,
        help="Path to input json file. Must be absolute",
    )
    parser.add_argument(
        "--output_dir", "-o", required=True, help="Output directory for AF3 outputs."
    )
    parser.add_argument(
        "--msa_db_dir",
        "-d",
        required=False,
        help="Path to MSA database directory for ColabFold.",
    )
    parser.add_argument(
        "--binder_chain",
        "-b",
        required=False,
        default="B",
        help="Chain ID of the binder chain.",
    )
    parser.add_argument(
        "--msa_method",
        "-m",
        required=False,
        default="colabfold_remote",
        help="How to build alignments: 'colabfold_remote' (api.colabfold.com) "
             "or 'colabfold_local' (colabfold_search against msa_db_dir).",
    )
    parser.add_argument(
        "--msa_scope",
        required=False,
        default="target_only",
        help="Who gets one: 'none', 'target_only', or 'binder_masked' "
             "(target MSAs plus a CDR-masked binder MSA).",
    )
    parser.add_argument(
        "--af3_repo_path", required=False, help="Path to local AlphaFold3 repo to bind."
    )
    parser.add_argument(
        "--af3_sif_path",
        required=False,
        help="Path to the AlphaFold3 Singularity image (.sif).",
    )
    parser.add_argument(
        "--af3_model_dir", required=False, help="Path to AF3 model weights directory."
    )
    parser.add_argument(
        "--af3_db_dir", required=False, help="Path to AF3 public databases directory."
    )
    parser.add_argument(
        "--use_metagenomic_db",
        action="store_true",
        help="Use metagenomic database for MSA generation.",
    )
    args = parser.parse_args()

    main(
        input_json=args.input_json,
        output_dir=args.output_dir,
        msa_db_dir=args.msa_db_dir,
        binder_chain=args.binder_chain,
        msa_method=args.msa_method,
        msa_scope=args.msa_scope,
        af3_repo_path=args.af3_repo_path,
        af3_sif_path=args.af3_sif_path,
        af3_model_dir=args.af3_model_dir,
        af3_db_dir=args.af3_db_dir,
        use_metagenomic_db=args.use_metagenomic_db,
    )
