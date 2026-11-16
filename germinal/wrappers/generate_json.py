"""
Wrapper around Kuhlman-Lab ProteinMPNN's generate_json.py.

Fixes a bug where single-residue constraints in multi-state mode
don't get their chain IDs remapped (ranges do via _check_range_validity,
but single residues were missing the equivalent remapping).
"""

import math
import os
import subprocess
import sys
from typing import Dict, Sequence, Tuple

# ---- Set up paths ----
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_PROTEINMPNN_RUN = os.path.join(_PROJECT_ROOT, "external", "proteinmpnn", "run")

# ---- Auto-init submodule if needed ----
if not os.path.exists(os.path.join(_PROTEINMPNN_RUN, "generate_json.py")):
    print("ProteinMPNN submodule not initialised, running git submodule update --init ...")
    subprocess.check_call(
        ["git", "submodule", "update", "--init", "external/proteinmpnn"],
        cwd=_PROJECT_ROOT,
    )

sys.path.insert(0, _PROTEINMPNN_RUN)

# ---- Import the original module ----
from generate_json import ProteinDesignInputFormatter, get_arguments  # noqa: E402

# ---- Monkey-patch the buggy method ----
_original_check_symmetry_validity = ProteinDesignInputFormatter._check_symmetry_validity


def _patched_check_symmetry_validity(
    self, symmetric_item: str, pdb_names: Sequence[str] = None
) -> Dict[str, Sequence[Tuple[str, int]]]:
    """Patched version that remaps chain IDs for single-residue constraints."""

    if not self.multi_state:
        return _original_check_symmetry_validity(self, symmetric_item, pdb_names)

    split_item = symmetric_item.split(":")
    symmetry_dict = {}
    for subitem, pdb_name in zip(split_item, pdb_names):
        adj_subitem = subitem.replace(
            subitem[0], self.chain_dict[pdb_name][subitem[0]]
        )
        if "-" in subitem:
            res_range = self._check_range_validity(subitem, pdb_name)
            symmetry_dict[adj_subitem] = res_range
        else:
            res_ch, res_idx = self._check_res_validity(subitem)
            # Fix: remap chain ID (matches _check_range_validity behavior)
            res_ch = self.chain_dict[pdb_name][res_ch]
            symmetry_dict[adj_subitem] = [(res_ch, res_idx)]

    item_lens = [len(symmetry_dict[key]) for key in symmetry_dict]
    if math.floor(sum([length == item_lens[0] for length in item_lens]) / len(item_lens)) != 1:
        raise ValueError(
            f"Tied residues and residue ranges must be of the same "
            f"size for forcing symmetry: {symmetric_item}"
        )

    return symmetry_dict


ProteinDesignInputFormatter._check_symmetry_validity = _patched_check_symmetry_validity

# ---- Delegate to the original entry point ----
if __name__ == "__main__":
    args = get_arguments()
    pdif = ProteinDesignInputFormatter(
        args.pdb_dir,
        args.designable_res,
        args.default_design_setting,
        args.symmetric_res,
        args.cluster_center,
        args.cluster_radius,
        args.gap,
        args.validation_tries,
        args.bidirectional,
        args.constraints,
        args.multi_state,
    )
    pdif.generate_json(args.out_path)
