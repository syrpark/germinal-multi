# Installation Instructions

This guide sets up the **Germinal** environment with [uv](https://docs.astral.sh/uv/),
which resolves and installs considerably faster than pip. Plain `pip` works
too — substitute it for `uv pip` throughout.

Every Python dependency is declared in `setup.py`'s `install_requires`, so
installing the project installs the stack. There is no separate package list
to keep in sync.

---

## 0. Clone the Repository

Clone with submodules. Sequence redesign shells out to the Kuhlman-Lab
ProteinMPNN fork vendored at `external/proteinmpnn`, so a clone without it
cannot redesign anything:

```bash
git clone --recurse-submodules https://github.com/twidatalla/germinal-multi.git
cd germinal-multi
```

If you already cloned without the flag, populate it now rather than later —
the runner refuses to start without it, and the wrapper's own fallback runs
inside the GPU job, on a compute node that usually has no network:

```bash
git submodule update --init external/proteinmpnn
```

---

## 1. Create and Activate the Conda Environment

```bash
conda create --name germinal python=3.10 -c conda-forge
conda activate germinal
pip install uv

# Only for msa_method: "colabfold_local" -- see below.
# conda install -c bioconda mmseqs2
```

`mmseqs2` is **optional**. Both shipped folding configs use
`msa_method: "colabfold_remote"`, which builds MSAs through the ColabFold
API (`api.colabfold.com`) and needs no local search. Install `mmseqs2` only
to run `msa_method: "colabfold_local"`, which searches a local `msa_db_dir`
instead — useful on a compute node with no outbound network, and the one
path that invokes the `mmseqs` binary. It is not on PyPI, hence conda.

---

## 2. Install Germinal and ColabDesign

> **Note:** Make sure you are in the **Germinal root directory** before running this.

```bash
uv pip install --only-binary pillow -e ".[gpu,test]"
uv pip install -e colabdesign
uv pip install --no-deps colabfold==1.5.5
```

`[gpu]` pulls `jax[cuda12]==0.5.3`; drop it for a CPU-only install. `[test]`
adds pytest, needed only for section 7.

`--only-binary pillow` costs nothing where the newest wheel fits and is
required where it does not. pillow 12.3 publishes `manylinux_2_28` wheels
only, so on a glibc 2.17 host (RHEL/CentOS 7) the resolver otherwise tries to
compile it from source and stops at `The headers or library files could not be
found for jpeg`. The flag makes it fall back to 12.2, which has a
`manylinux2014` wheel. It is only needed on this first command: pillow arrives
as a torchvision dependency, and once it is installed the colabdesign step
leaves it alone.

**Install Germinal before colabdesign; the order matters.**
`colabdesign/setup.py` declares its dependencies with no version bounds, so on
its own it resolves `jax`, `numpy`, `dm-haiku` and the rest to their latest
releases — including the numpy 2.x that the `jax==0.5.3` pin below cannot work
with. Germinal first puts the pins in place, and colabdesign then adds only
what is still missing (`chex`, `immutabledict`, `toolz`).

`colabfold` is the third line, and `--no-deps` is not optional. Its own
metadata pins `dm-haiku==0.0.10`, `pandas<2.0.0`, `biopython<1.83` and
`tensorflow-cpu`, each of which contradicts a pin above, so no version of it
resolves against this project — `uv` reports the requirements as
unsatisfiable and installs nothing. Germinal uses exactly two things from it,
`run_mmseqs2` and the `colabfold_search` CLI, neither of which needs those
dependencies; the ones that do matter (`requests`, `appdirs`) are declared in
`setup.py` instead. Check it landed with:

```bash
python -c "from colabfold.colabfold import run_mmseqs2"
which colabfold_search
```

Two pins in `setup.py` are load-bearing and will not be obvious from the error
message if you loosen them:

> `transformers==4.56.2` — IgLM loads its model via
> `transformers.GPT2LMHeadModel`, which fails to import on transformers 5.x,
> and its tokenizer loses the amino-acid vocabulary on recent releases.

> `ipsae==1.0.1` — provides the `ipsae` command-line tool that the filters call
> as a subprocess. If it is missing, ipSAE, pDockQ2 and LIS are all silently
> `None` and no design can pass the `ipsae` filter. Check it with `which ipsae`.

> `jax==0.5.3` — ColabDesign and the AlphaFold-Multimer parameters are only
> validated against this version. The core scientific stack in `setup.py` is
> upper-bounded against it; numpy 2.x in particular does not work with a
> jax/haiku of this vintage.

> `torch==2.6.*` — matches the `torch_geometric` and `torchvision` builds
> pinned alongside it.

---

## 3. Install PyRosetta

Required for all filtering. It ships from its own index under an academic
license, so it cannot be declared alongside the other dependencies:

```bash
uv pip install pyrosetta-installer
python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'
python -c 'import pyrosetta'      # the installer does not check this itself
```

That third line is worth running. `pyrosetta-installer` only knows the
`ubuntu`, `mac` and `m1` builds, and the current Ubuntu wheels are linked
against **glibc 2.27**. On an older host it downloads 1.6 GB, reports success,
and then every import fails with
``ImportError: /lib64/libm.so.6: version `GLIBC_2.27' not found``. Since
`run_germinal_multi.py` imports pyrosetta at module scope, that is a run which
dies at startup rather than one that degrades.

If you are below glibc 2.27 (`ldd --version`), the `ubuntu` build line is not
an option at any version -- every wheel still mirrored, back to 2026.30, needs
2.27. Use the legacy generic `linux` build instead, which was compiled against
glibc 2.17; it was discontinued in Sept 2024, so pin the last one explicitly:

```bash
uv pip install --no-deps "https://graylab.jhu.edu/download/PyRosetta4/archive/release/PyRosetta4.Release.python310.linux.wheel/pyrosetta-2024.39%2Brelease.59628fb-cp310-cp310-linux_x86_64.whl"
```

---

## 4. Model weights

Two sets of weights are downloaded by hand. Both land under `params/`, and
neither location is configurable.

### AlphaFold-Multimer

Hallucination folds with AlphaFold-Multimer, so these are needed for every
run. The tarball is 5.6 GB, and unpacks to about the same again. From the
repository root:

```bash
# aria2c is fast but is not installed by the steps above; `wget -P params` or
# `curl -o params/<file>` on the same URL works just as well.
aria2c -x 16 -d params https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar
tar -xf params/alphafold_params_2022-12-06.tar -C params
rm params/alphafold_params_2022-12-06.tar
```

They must end up as `params/params_model_*_multimer_v3.npz`. ColabDesign is
pointed at the directory containing `params/`, which is derived from the
package location rather than configured, so to share one copy between
checkouts symlink the files into `params/` rather than relocating them.

### AbMPNN

Sequence redesign runs AbMPNN, so **every shipped configuration needs these
weights**. `germinal/wrappers/run_protein_mpnn.py` downloads them from Zenodo
to `params/abmpnn.pt` automatically on first use — but that first use is inside
a GPU job, on a compute node, which on most clusters cannot reach the internet.
Fetch them once from a login node instead (~20 MB):

```bash
curl -fL -o params/abmpnn.pt \
  "https://zenodo.org/records/8164693/files/abmpnn.pt?download=1"
```

`-f` is not cosmetic. Zenodo rate-limits by network and answers `403`,
and without it curl cheerfully writes that HTML error page to
`params/abmpnn.pt` — a 763-byte "checkpoint" that fails much later, inside
a GPU job, as an unreadable torch load. With `-f` curl exits 22 and
writes nothing. `validate_install.py` also rejects the file if it is
implausibly small.

The wrapper symlinks the file into `external/proteinmpnn/run/model_weights/`
on first run, so nothing else is needed.

---

## 5. Check the Install

```bash
python validate_install.py
```

This checks imports, the GPU, the JAX backend, ipSAE, IgLM, PyRosetta, mmseqs2,
colabfold_search, a Singularity runtime, the bundled `params/dssp`
and `params/DAlphaBall.gcc` binaries, and the AlphaFold-Multimer parameters. It
does **not** check the AF3 paths — see [docs/af3-setup.md](docs/af3-setup.md).

---

## 6. Optional: the Protenix folding backend

Germinal can fold with [Protenix](https://github.com/bytedance/Protenix) instead
of AlphaFold 3, independently at each stage:

The backend is a config group, selected once per stage in the defaults list of
`configs/config.yaml`:

```yaml
defaults:
  - folding@recycle_folder: protenix   # the hot loop
  - folding@filter_folder: af3         # the scoring stages
```

or, without editing that file, as a Hydra override:

```bash
python run_germinal_multi.py folding@recycle_folder=protenix
```

Each stage gets the backend's whole config from
`configs/folding/<backend>.yaml` -- sample and cycle counts, MSA policy,
selection modes, batch size -- rather than just a name.

Both stages default to `af3`, so **this section is only needed if you select
`protenix` for one of them.** Nothing in `germinal.filters.protenix` is
imported otherwise.

Protenix pins a different torch stack from Germinal, so it goes in its **own**
environment and Germinal shells out to it. The environment name is
`conda_env` in `configs/folding/protenix.yaml` (default `protenix`):

```bash
conda create --name protenix python=3.11 -y
conda activate protenix
pip install uv
uv pip install "protenix @ git+https://github.com/bytedance/Protenix.git@85767b811c40ed46e73a9b39519cf6bfca8701ba"
conda activate germinal          # back to the Germinal environment
```

**Install from git, not PyPI.** `uv pip install protenix` resolves to 0.5.5,
the newest release on PyPI, and that version cannot drive this pipeline: it
has no `--need_atom_confidence`, so Protenix writes no `atom_plddt`, so ipSAE
cannot be derived. ipSAE gates the recycle loop *and* the `ipsae` filter, so
every design scores 0.0 and nothing passes. Its CLI also names the
subcommands differently (`tojson` and `predict` rather than the `json` and
`pred` used below), so the pre-fetch steps in this section do not apply to it
either. The commit pinned above is 2.0.0, which is the version these
instructions and `configs/folding/protenix.yaml` are written against; `@main`
would work today but moves.

**Protenix needs glibc >= 2.28, so this section does not apply to a
RHEL/CentOS 7 host.** Check with `ldd --version`. Three of Protenix 2.0.0's
pinned dependencies -- `torch==2.7.1`, `cuequivariance-ops-torch-cu12` and
`rdkit` -- publish `manylinux_2_28` wheels only, so the install above cannot
resolve on glibc 2.17 and there is nothing to override: each one fails for
the same reason in turn. PyPI's 0.5.5 does install there, but it is the build
described above that scores every design 0.0, so it is not a fallback. On such
a host either keep both stages on `af3`, which is the default and what the
shipped thresholds are calibrated against anyway, or run Protenix from a
container with a newer glibc -- Germinal calls the backend as
`conda run -n <conda_env> protenix <sub> ...`, so a `protenix` script on PATH
inside that env which execs into the container is enough, with no code change.

Note that Germinal's own `torch==2.6.*` pin is the last release with a
glibc 2.17 wheel, so on these hosts that pin is load-bearing for Germinal
too, not only for Protenix.

Check it the way Germinal will call it:

```bash
conda run -n protenix protenix --help
```

### Weights

**Germinal uses Protenix v1 only.** `model_name` in
`configs/folding/protenix.yaml` is passed straight to the CLI's `--model_name`,
and the shipped value is `protenix_base_default_v1.0.0` — the v1 checkpoint,
whose weights are released under Apache 2.0. Do not point it at a v2
checkpoint: upstream releases those weights under a proprietary licence, so
they are not an option here.

**Pre-fetch the weights.** Protenix downloads its checkpoint and CCD data on
first use from `protenix.tos-cn-beijing.volces.com`, ~2 GB in total. Do this
once, deliberately, rather than letting the first GPU job do it: the download
takes minutes of allocated GPU time while nothing computes, and if egress is
blocked on the partition you land on, the job fails after it has already
queued and started.

It is not, however, guaranteed to fail on a compute node. A GPU node on this
cluster downloaded 343 MB of `common/components.cif` from that host without
trouble, and `api.colabfold.com` serves MSAs from compute nodes normally. So
if Protenix stalls on first use, check that partition's egress before assuming
the endpoint is blocked — and do not treat a working download as a sign that
something is misconfigured.

The fetch happens on the first call, whatever the input, so any prediction
will do. `protenix json` turns any PDB into a valid input, so a structure this
repo already ships will do:

```bash
# 1. build an input JSON from any shipped antigen
conda run -n protenix protenix json -i pdbs/egfr_diii.pdb -o /tmp/px_prefetch

# 2. one prediction, purely to trigger the fetch. `pred` on protenix 2.x,
#    `predict` on 0.5.5 -- `protenix --help` says which.
conda run -n protenix protenix pred \
    -i /tmp/px_prefetch/egfr_diii.json \
    -o /tmp/px_prefetch_out \
    -n protenix_base_default_v1.0.0
```

Step 2 only has to get far enough to download; it does not need to finish, so a
login node without a GPU is fine.

It fetches ~2 GB into `$PROTENIX_ROOT_DIR`, which defaults to `$HOME`
(`protenix/web_service/colab_request_parser.py`: `PROTENIX_ROOT_DIR =
os.environ.get("PROTENIX_ROOT_DIR", str(Path.home()))`):

| Path (relative to `$PROTENIX_ROOT_DIR`) | Size |
|---|---|
| `checkpoint/protenix_base_default_v1.0.0.pt` | 1.4 GB |
| `common/components.cif` | 469 MB |
| `common/components.cif.rdkit_mol.pkl` | 136 MB |
| `common/clusters-by-entity-40.txt` | 21 MB |
| `common/obsolete_release_date.csv` | 132 KB |

To keep them off your home quota, set `PROTENIX_ROOT_DIR` to a shared path
before the pre-fetch and in the job script. It propagates through `conda run`
to the Protenix child process, so no code change is needed.

### ipSAE

Protenix does not report ipSAE, and ipSAE is the gate the whole recycle loop
turns on. Germinal derives it from Protenix's per-atom pLDDT and token-pair PAE
by handing the same `ipsae` CLI (section 2) AF3-shaped inputs. That means **`ipsae`
must be installed in the `germinal` environment even on a protenix-only run** —
it is called from Germinal, not from the Protenix environment.

### What differs from AF3

Give Protenix the target **and** the binder framework alignment. That is a
different `msa_scope` from the AF3 default:

| backend | `msa_scope` | what gets an alignment |
|---|---|---|
| af3 | `target_only` | antigens only; the binder gets a one-sequence stub |
| protenix | `binder_masked` | antigens, plus the binder with its CDR columns gapped |

`configs/folding/protenix.yaml` already ships `binder_masked`, so selecting
protenix for a stage gets this without further configuration. In our hands
Protenix requires an alignment of the binder framework for its scores to be
reliable.

## 7. Running the tests

The suite needs no GPU, no AF3, no Protenix and no PyRosetta: the backends are
swapped for recording fakes, so what is under test is Germinal's own logic
(routing, filter evaluation, failure attribution, output layout).

`filter_utils` imports `pyrosetta_utils`, which imports pyrosetta at module
scope, so without PyRosetta three modules could not even be collected.
`tests/conftest.py` substitutes a stub in that case and says so on stderr; with
PyRosetta installed the stub is not used and the tests run against the real
module. No test exercises a Rosetta code path either way — the stub raises if
one tries.

```bash
conda run -n germinal python -m pytest tests/ -q
```

Each module is also runnable on its own, which is handy when one fails:

```bash
conda run -n germinal python tests/test_folding_router.py
```
