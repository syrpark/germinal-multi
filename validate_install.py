#!/usr/bin/env python
"""Pre-flight check for a Germinal install.

Exits non-zero when something the pipeline needs is absent. An ERROR means a
run would fail or silently degrade; a WARN means it only matters on a compute
node. The AlphaFold3 paths are config, not install state, and are checked at
run startup by ``germinal.utils.config.validate_af3_paths``.
"""
import os
import shutil
import subprocess
import sys
from typing import List

ERRORS: List[str] = []
WARNINGS: List[str] = []


def print_status(prefix: str, msg: str) -> None:
    print(f"[{prefix}] {msg}")


def error(msg: str) -> None:
    ERRORS.append(msg)
    print_status("ERROR", msg)


def warn(msg: str) -> None:
    WARNINGS.append(msg)
    print_status("WARN", msg)


def try_import(pkg: str) -> bool:
    try:
        __import__(pkg)
        return True
    except Exception as e:
        print_status("ERROR", f"Import failed: {pkg}: {e}")
        return False


def cmd_ok(cmd: List[str]) -> bool:
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception as e:
        print_status("INFO", f"Command not usable: {' '.join(cmd)} -> {e}")
        return False


def probe_jax_backend():
    """Ask a fresh interpreter which backend JAX would use.

    Returns ``(ok, device, detail)``: ``ok`` is False when jax could not be
    imported or probed at all, ``device`` is None when it imported but has no
    usable backend.

    Out of process on purpose. Importing jaxlib can abort the interpreter
    outright rather than raise -- on glibc < 2.18 the static TLS surplus is
    small enough that extension modules imported earlier (pandas, among the
    required imports above) can leave none for
    ``jaxlib/xla_extension.so``, and the dlopen dies with SIGSEGV.
    ``try``/``except`` cannot catch that, so probing in-process takes this
    script down mid-way and reports nothing about PyRosetta, ipSAE, IgLM,
    mmseqs2 or the parameters. A crashed child is just one failed check.
    """
    probe = "import jax; print(jax.default_backend())"
    try:
        proc = subprocess.run(
            [sys.executable, "-c", probe],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=300,
        )
    except Exception as e:  # the interpreter itself is unusable
        return False, None, str(e)

    if proc.returncode == 0:
        return True, proc.stdout.strip() or "unknown", ""

    stderr_lines = proc.stderr.strip().splitlines()
    detail = stderr_lines[-1] if stderr_lines else f"exit {proc.returncode}"
    if proc.returncode < 0:
        detail += (
            f" [probe killed by signal {-proc.returncode}, it crashed rather "
            "than raised. On glibc < 2.18 that is usually static TLS "
            "exhaustion; check with `python -c 'import jax'` in a fresh "
            "interpreter]"
        )
    # An absent accelerator is a login-node fact, not a broken install.
    elif "No visible GPU devices" in proc.stderr or "FAILED_PRECONDITION" in proc.stderr:
        return True, None, detail
    return False, None, detail


def check_redesign_assets() -> None:
    """Check what AbMPNN sequence redesign needs: the submodule and the weights.

    Every shipped configuration redesigns with AbMPNN, so a missing submodule is
    an error. The weights are only a warning: the wrapper downloads them on
    first use -- but that first use is inside a GPU job, and compute nodes
    usually cannot reach Zenodo, so an unfetched file is worth flagging here.
    """
    print_status("INFO", "Checking the ProteinMPNN submodule and AbMPNN weights")

    mpnn_entry = os.path.join("external", "proteinmpnn", "run", "run_protein_mpnn.py")
    if not os.path.isfile(mpnn_entry):
        error(
            f"{mpnn_entry} is missing, so the ProteinMPNN submodule is not "
            "populated. Sequence redesign shells out to it, so no design can be "
            "redesigned. Fix with:\n"
            "    git submodule update --init external/proteinmpnn"
        )
    else:
        print_status("OK", "ProteinMPNN submodule populated")

    abmpnn_weights = os.path.join("params", "abmpnn.pt")
    if not os.path.isfile(abmpnn_weights):
        warn(
            f"{abmpnn_weights} is not present. germinal/wrappers/"
            "run_protein_mpnn.py downloads it from Zenodo on first use, but that "
            "happens inside the first GPU job and most compute nodes cannot reach "
            "the internet. Pre-fetch it from a login node (~20 MB):\n"
            "    curl -fL -o params/abmpnn.pt "
            '"https://zenodo.org/records/8164693/files/abmpnn.pt?download=1"'
        )
    else:
        size_mb = os.path.getsize(abmpnn_weights) / 1e6
        # Size, not just presence: without curl's -f a rate-limited Zenodo
        # writes its 403 page to this path, and a checkpoint-shaped HTML file
        # otherwise surfaces much later as an unreadable torch load.
        if size_mb < 1:
            warn(
                f"{abmpnn_weights} is only {os.path.getsize(abmpnn_weights)} "
                f"bytes, far short of the ~20 MB checkpoint. It is most likely "
                f"an HTTP error page saved by a download that was not told to "
                f"fail: Zenodo rate-limits whole networks and answers 403. "
                f"Delete it and re-fetch WITH -f, which writes no file on an "
                f"error:\n"
                "    curl -fL -o params/abmpnn.pt "
                '"https://zenodo.org/records/8164693/files/abmpnn.pt?download=1"'
            )
        else:
            print_status("OK", f"AbMPNN weights present ({size_mb:.1f} MB)")


def check_protenix_backend() -> None:
    """Check the optional Protenix backend, only when it is asked for.

    Opt in with GERMINAL_CHECK_PROTENIX=1, optionally GERMINAL_PROTENIX_ENV.
    WARN-level only: an af3-only install is complete without Protenix.
    """
    if os.environ.get("GERMINAL_CHECK_PROTENIX", "").lower() not in ("1", "true", "yes"):
        return

    env_name = os.environ.get("GERMINAL_PROTENIX_ENV", "protenix")
    print_status("INFO", f"Checking the Protenix backend in conda env {env_name!r}")

    if shutil.which("conda") is None:
        warn(
            "conda not found on PATH, so the Protenix backend cannot be launched. "
            "Germinal invokes it as: conda run -n <env> protenix pred ..."
        )
        return

    if not cmd_ok(["conda", "run", "-n", env_name, "protenix", "--help"]):
        warn(
            f"`conda run -n {env_name} protenix --help` failed. Select "
            f"protenix for a stage (folding@recycle_folder=protenix) only once "
            f"this works; see environment_setup.md section 6."
        )
        return
    print_status("OK", f"Protenix CLI reachable in env {env_name!r}")

    # The ipsae CLI runs in the Germinal environment, not the Protenix one:
    # Protenix reports no ipSAE, so Germinal derives it.
    print_status(
        "INFO",
        "Protenix reports no ipSAE; Germinal derives it from Protenix PAE/pLDDT "
        "using the ipsae CLI in THIS environment (checked above).",
    )


def main() -> int:
    required = [
        # First on purpose. Later checks import germinal, which imports jaxlib,
        # and on glibc < 2.18 jaxlib needs static TLS that pandas' extension
        # modules will have taken if they load first -- see probe_jax_backend.
        "jax",
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "yaml",
        "filelock",
        "omegaconf",
        "hydra",  # run_germinal_multi.py is @hydra.main; without it nothing runs
        "Bio",
        "tqdm",
    ]
    print_status("INFO", "Checking Python imports")
    missing = [p for p in required if not try_import(p)]
    if missing:
        print_status("ERROR", f"Missing packages: {', '.join(missing)}")
        return 1

    print_status("INFO", "Checking GPU availability via nvidia-smi")
    if cmd_ok(["nvidia-smi"]):
        print_status("OK", "nvidia-smi detected")
    else:
        warn(
            "nvidia-smi not available. Hallucination needs a GPU; this is "
            "expected on a login node but a compute node without it cannot run."
        )

    print_status("INFO", "Checking JAX device")
    jax_ok, device, detail = probe_jax_backend()
    if not jax_ok:
        # A broken jax install is fatal; no GPU is merely a login-node fact.
        error(f"JAX could not be imported or probed: {detail}")
    elif device is None:
        warn(
            f"JAX has no usable accelerator backend ({detail}). Expected on a "
            "login node; on a compute node, check the CUDA build."
        )
    elif device == "cpu":
        warn(
            "JAX backend is CPU. Hallucination will be unusably slow. "
            'Install the CUDA build: uv pip install "jax[cuda12_pip]==0.5.3" '
            "-f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
        )
    else:
        print_status("OK", f"JAX backend: {device}")

    print_status("INFO", "Checking PyTorch CUDA")
    try:
        import torch

        print_status("OK", f"Torch: {torch.__version__}")
        if not torch.cuda.is_available():
            warn(
                "Torch reports CUDA unavailable. AbMPNN redesign and IgLM "
                "scoring both run on GPU; on a compute node this means the "
                "install is CPU-only."
            )
    except Exception as e:
        error(f"Torch check failed: {e}")

    print_status("INFO", "Checking local packages")
    for mod in ["colabdesign", "germinal"]:
        if try_import(mod):
            print_status("OK", f"Imported {mod}")
        else:
            return 2

    print_status("INFO", "Checking the ipsae command-line tool")
    if shutil.which("ipsae") is None:
        print_status(
            "ERROR",
            "ipsae not found on PATH. The filters call it as a subprocess; without\n"
            "  it ipSAE, pDockQ2 and LIS are all None and the ipsae filter rejects\n"
            "  every design. Install it with: pip install ipsae==1.0.1",
        )
        return 4
    print_status("OK", "ipsae available")

    check_protenix_backend()

    print_status("INFO", "Checking IgLM (required for the sequence prior)")
    try:
        import iglm  # type: ignore

        print_status("OK", "IgLM available")
    except Exception as e:
        print_status("ERROR", f"IgLM not found: {e}")
        return 3

    # Not optional: filter_utils imports pyrosetta_utils unconditionally, so
    # without it every run aborts on its first filter call.
    print_status("INFO", "Checking PyRosetta (required for all filtering)")
    try:
        import pyrosetta  # type: ignore

        print_status("OK", "PyRosetta available")
    except Exception as e:
        print_status(
            "ERROR",
            f"PyRosetta not found ({e}). It is imported unconditionally by\n"
            "  germinal.filters.filter_utils, so no run can reach its first filter\n"
            "  without it. It needs an academic license from Rosetta Commons.\n"
            "  Install with:\n"
            "    uv pip install pyrosetta-installer\n"
            "    python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'\n"
            "  See environment_setup.md for details.",
        )
        return 5

    # Informational, not errors: only msa_mode "local" searches local
    # databases, and the shipped configs search remotely.
    print_status("INFO", "Checking local MSA search tools (only needed for msa_mode: local)")
    for tool, how in [
        ("mmseqs", "conda install -c conda-forge -c bioconda mmseqs2"),
        ("colabfold_search", "uv pip install colabfold"),
    ]:
        if shutil.which(tool) is None:
            print_status(
                "INFO",
                f"{tool} not on PATH. Needed only for msa_mode: \"local\"; the "
                f"shipped configs use msa_mode: \"target\" and build MSAs through "
                f"the remote colabfold API. For local mode: {how}",
            )
        else:
            print_status("OK", f"{tool} available")

    print_status("INFO", "Checking the container runtime for AlphaFold3")
    runtime = shutil.which("singularity")
    if runtime is None:
        error(
            "singularity not found on PATH. AF3 is invoked as a Singularity "
            "container, so no design can be refolded without it. On an "
            "apptainer-only host, put a singularity -> apptainer shim on PATH."
        )
    else:
        print_status("OK", f"Container runtime: {runtime}")

    print_status("INFO", "Checking bundled binaries and model parameters")
    for path, what in [
        ("params/dssp", "secondary-structure assignment"),
        ("params/DAlphaBall.gcc", "Rosetta hole/packing scoring"),
    ]:
        if not os.path.isfile(path):
            error(f"{path} is missing ({what}). Run from the repository root.")
        elif not os.access(path, os.X_OK):
            error(f"{path} is not executable ({what}). Fix with: chmod +x {path}")
        else:
            print_status("OK", f"{path} present and executable")

    af_params = [f for f in os.listdir("params") if f.startswith("params_")] if os.path.isdir("params") else []
    if not af_params:
        error(
            "No AlphaFold-Multimer parameters found in params/ (expected "
            "params_*.npz). Hallucination cannot start without them, and the "
            "location is fixed -- symlink them into params/ rather than "
            "relocating. See the README for the "
            "alphafold_params_2022-12-06.tar download."
        )
    else:
        print_status("OK", f"AlphaFold params present ({len(af_params)} file(s))")

    check_redesign_assets()

    print()
    if ERRORS:
        print_status(
            "FAIL",
            f"{len(ERRORS)} error(s), {len(WARNINGS)} warning(s). "
            "This install cannot run the pipeline as configured.",
        )
        return 6
    if WARNINGS:
        print_status(
            "INCOMPLETE",
            f"No errors, but {len(WARNINGS)} warning(s) above. Everything the "
            "pipeline imports is present; re-check on a compute node before "
            "treating this as a working install.",
        )
        return 0
    print_status("OK", "All checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
