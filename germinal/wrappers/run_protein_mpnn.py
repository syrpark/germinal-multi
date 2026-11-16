"""
Wrapper around Kuhlman-Lab ProteinMPNN's run_protein_mpnn.py.

Patches MODEL_NAMES to include "abmpnn", auto-downloads the abmpnn weights
from Zenodo if missing, and symlinks them into the submodule before
delegating to the original script.
"""

import os
import subprocess
import sys
import urllib.request

# ---- Set up paths ----
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_PROTEINMPNN_RUN = os.path.join(_PROJECT_ROOT, "external", "proteinmpnn", "run")
_ABMPNN_WEIGHTS = os.path.join(_PROJECT_ROOT, "params", "abmpnn.pt")
_ABMPNN_URL = "https://zenodo.org/records/8164693/files/abmpnn.pt?download=1"

# ---- Auto-init submodule if needed ----
if not os.path.exists(os.path.join(_PROTEINMPNN_RUN, "mpnn_utils.py")):
    print("ProteinMPNN submodule not initialised, running git submodule update --init ...")
    subprocess.check_call(
        ["git", "submodule", "update", "--init", "external/proteinmpnn"],
        cwd=_PROJECT_ROOT,
    )

# Prepend the proteinmpnn run/ directory so its imports resolve
sys.path.insert(0, _PROTEINMPNN_RUN)

# ---- Patch MODEL_NAMES to recognise "abmpnn" ----
import mpnn_utils  # noqa: E402

if "abmpnn" not in mpnn_utils.MODEL_NAMES:
    mpnn_utils.MODEL_NAMES.append("abmpnn")

# ---- Auto-download abmpnn weights if missing ----
if not os.path.exists(_ABMPNN_WEIGHTS):
    os.makedirs(os.path.dirname(_ABMPNN_WEIGHTS), exist_ok=True)
    print(f"Downloading abmpnn weights from Zenodo to {_ABMPNN_WEIGHTS} ...")
    try:
        urllib.request.urlretrieve(_ABMPNN_URL, _ABMPNN_WEIGHTS)
        print(f"Downloaded abmpnn weights ({os.path.getsize(_ABMPNN_WEIGHTS) / 1e6:.1f} MB)")
    except Exception as e:
        if os.path.exists(_ABMPNN_WEIGHTS):
            os.remove(_ABMPNN_WEIGHTS)
        raise RuntimeError(f"Failed to download abmpnn weights: {e}") from e

# ---- Symlink abmpnn weights into the model_weights dir if needed ----
_MODEL_WEIGHTS_DIR = os.path.join(_PROTEINMPNN_RUN, "model_weights")
_DEST_WEIGHTS = os.path.join(_MODEL_WEIGHTS_DIR, "abmpnn.pt")

if not os.path.exists(_DEST_WEIGHTS):
    os.makedirs(_MODEL_WEIGHTS_DIR, exist_ok=True)
    os.symlink(os.path.abspath(_ABMPNN_WEIGHTS), _DEST_WEIGHTS)

# ---- Patch torch.load for PyTorch >= 2.6 (weights_only default changed to True) ----
import torch  # noqa: E402

_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

# ---- Delegate to the original entry point ----
from generate_json import FileArgumentParser  # noqa: E402
from run_protein_mpnn import run_protein_mpnn  # noqa: E402

if __name__ == "__main__":
    parser = FileArgumentParser(
        description="Wrapper: runs ProteinMPNN with abmpnn support.",
        fromfile_prefix_chars="@",
    )

    parser.add_argument("--model_name", type=str, default="v_48_020",
                        choices=mpnn_utils.MODEL_NAMES)
    parser.add_argument("--backbone_noise", type=float, default=0.00)
    parser.add_argument("--num_seq_per_target", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--sampling_temp", type=str, default="0.1")
    parser.add_argument("--destabilize", action="store_true")
    parser.add_argument("--out_folder", type=str)
    parser.add_argument("--pdb_dir", type=str, default="")
    parser.add_argument("--design_specs_json", type=str)
    parser.add_argument("--af2_formatted_output", action="store_true")
    parser.add_argument("--global_omit_AAs", type=str, default="X")
    parser.add_argument("--experimental", action="store_true")
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--bias_AA_dict", default=None, type=str)
    parser.add_argument("--bias_by_res_dict", default=None, type=str)
    parser.add_argument("--pairwise", action="store_true")
    parser.add_argument("--mcmc", action="store_true")
    parser.add_argument("--dump_probs", action="store_true")

    args = parser.parse_args()
    run_protein_mpnn(args)
