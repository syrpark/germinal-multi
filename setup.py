"""Local installation for Germinal: pip install -e ".[gpu]"

This file is the single source of truth for Germinal's PyPI dependencies.
environment_setup.md defers to it rather than restating the list, so a
version pin is changed here and nowhere else.

Use the `gpu` extra for a GPU install; plain `pip install -e .` leaves jax on
CPU. Three things cannot be declared here and are installed separately -- see
environment_setup.md:

  * PyRosetta -- ships from its own index under an academic license.
  * mmseqs2 -- not on PyPI; comes from bioconda, hence the conda step in
    environment_setup.md section 1. germinal/filters/af3.py shells out to the
    `mmseqs` binary, so without it colabfold_search cannot build MSAs and AF3
    falls back to single-sequence input.
  * colabdesign -- vendored in this repository; install with
    `pip install -e colabdesign`.

AlphaFold3 is a separate, access-gated download run as a container; its
paths are run configuration, not install state.
"""

from setuptools import find_packages, setup

setup(
    name="germinal",
    version="0.0.1",
    packages=find_packages(include=["germinal*"]),
    python_requires=">=3.10",
    install_requires=[
        # Core scientific stack. Upper-bounded against the hard jax==0.5.3 pin
        # below: numpy 2.x in particular does not work with a jax/haiku of this
        # vintage, and an unbounded install resolves to something untested.
        # Lower bounds are the versions this was validated against.
        "numpy>=1.26,<2",
        "scipy>=1.15,<2",
        "pandas>=2.2,<3",
        "matplotlib>=3.10,<4",
        "biopython>=1.84,<2",
        "pyyaml",
        "filelock",
        "tqdm",
        "joblib",
        "py3Dmol",
        # config
        "hydra-core",
        "omegaconf",
        # jax / AlphaFold, pinned: colabdesign is only validated against
        # 0.5.3. Add the platform-specific cuda extra yourself for a GPU.
        "jax==0.5.3",
        "jaxlib==0.5.3",
        "dm-haiku==0.0.13",
        # Upper-bounded for the wheel, not the API: 0.1.10 ships only
        # manylinux_2_28 wheels, so on a glibc 2.17 host (RHEL/CentOS 7, which
        # is most older clusters) pip falls back to a source build and fails
        # with "CMake must be installed". 0.1.9 has a manylinux2014 wheel,
        # valid on new and old glibc alike.
        "dm-tree<=0.1.9",
        "optax",
        "ml-collections",
        "absl-py",
        # colabfold is installed --no-deps (see environment_setup.md s.2): its
        # own pins (dm-haiku==0.0.10, pandas<2, biopython<1.83, tensorflow-cpu)
        # each contradict one above, so declaring it here makes the resolve
        # unsatisfiable. Only run_mmseqs2 and the colabfold_search CLI are
        # used; these are the transitive deps that actually matter.
        "requests",
        "appdirs",
        # torch, for IgLM and ProteinMPNN. 2.6 is also the last release with a
        # manylinux2014 (glibc 2.17) wheel, so on a RHEL/CentOS 7 host this pin
        # is what keeps Germinal installable at all -- 2.7 and above are
        # manylinux_2_28 only. Check `ldd --version` before raising it.
        "torch==2.6.*",
        "torchvision==0.21.*",
        "torchaudio==2.6.*",
        "torchtyping==0.1.5",
        "torch_geometric==2.6.*",
        # IgLM, the sequence prior. transformers must stay below 5.x, which
        # no longer imports GPT2LMHeadModel.
        "iglm",
        "transformers==4.56.2",
        # ipSAE. Provides the `ipsae` CLI that af3.py shells out to; without
        # it every ipSAE value is None and the filter rejects everything.
        "ipsae==1.0.1",
    ],
    extras_require={
        # The tests under tests/ are pytest-compatible but also runnable
        # directly with `python tests/test_design_classification.py`.
        "test": ["pytest"],
        # GPU jax. Separate from install_requires because the cuda12 wheels
        # are linux/NVIDIA-only; a CPU install wants plain `pip install -e .`.
        "gpu": ["jax[cuda12]==0.5.3"],
    },
)
