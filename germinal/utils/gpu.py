"""GPU environment for child processes.

Germinal holds a JAX context on its GPU for the life of a run and shells out
to other GPU programs -- AbMPNN, AF3, Protenix, mmseqs -- while holding it.
Where that works depends on the cluster's compute mode, which is a property of
the site, not of this code:

    nvidia-smi --query-gpu=compute_mode --format=csv

"Default" admits several processes per card and every child is fine. Under
"Exclusive_Process" a card admits exactly one, so the child cannot open its
own context and dies at its first allocation -- not at the availability check,
which still reports a device. torch says is_available() is True and then
fails to allocate; AF3's jaxlib says "INTERNAL: no supported devices found for
platform CUDA". Neither names the real cause.

GERMINAL_CHILD_CUDA_DEVICE pins those children to a card this process is not
using. Request two GPUs and set it to "1". The stages are sequential -- the
recycle loop folds, then redesigns, never both at once -- so one spare card
covers all of them. The exception is the mmseqs GPU server, a long-lived
process that overlaps with folding; a run using msa_method "colabfold_local"
with use_gpu_server needs a third card.

See docs/af3-setup.md.
"""

import os

_CHILD_DEVICE_VAR = "GERMINAL_CHILD_CUDA_DEVICE"


def child_env(blocklist=()):
    """Return the env for a GPU child process, or None to inherit unchanged.

    None is not a soft default: subprocess treats env=None exactly as it
    treats an omitted env, inheriting the live environment. So with
    GERMINAL_CHILD_CUDA_DEVICE unset and nothing to strip, routing a call
    through this helper cannot change what the child sees, whatever any
    library has done to the environment since startup.

    Args:
        blocklist: variable names to drop from the child's environment.

    Returns:
        dict | None: the child's environment, or None to inherit.
    """
    device = os.environ.get(_CHILD_DEVICE_VAR)
    if not device and not blocklist:
        return None

    env = os.environ.copy()
    for key in blocklist:
        env.pop(key, None)
    if device:
        env["CUDA_VISIBLE_DEVICES"] = device
    return env
