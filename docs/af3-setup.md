# AlphaFold3 setup

AF3 is the default for both folding stages and what the shipped filter
thresholds were calibrated against. Selecting `protenix` for both stages runs
without any AF3 install, at the cost of those thresholds no longer being
calibrated — see [environment_setup.md](../environment_setup.md) section 6.

## 1. Obtain AF3

- Request AF3 model weights from Google DeepMind. Access is **gated by
  application** and carries use terms; this is not a public download.
- Obtain the [AlphaFold3 repository](https://github.com/google-deepmind/alphafold3)
  and build its container image. Germinal invokes AF3 through
  Singularity, bind-mounting the paths below (see
  `germinal/filters/af3.py`).
- Download the AF3 genetic databases and a colabfold MSA database.

## 2. Point Germinal at your install

Edit `configs/folding/af3.yaml` and fill in the four AF3 locations. Both
folding stages load that one file, so setting them there covers the recycle and
filter stages at once:

| Key | In | What it points at |
|---|---|---|
| `af3_repo_path` | `configs/folding/af3.yaml` | your AlphaFold3 repository checkout |
| `af3_sif_path` | `configs/folding/af3.yaml` | the AF3 Singularity image |
| `af3_model_dir` | `configs/folding/af3.yaml` | the AF3 model weights |
| `af3_db_dir` | `configs/folding/af3.yaml` | the AF3 genetic databases |
| `msa_db_dir` | `configs/config.yaml` | the colabfold MSA database |

`msa_db_dir` sits in the main config rather than the AF3 one because it is not
an AF3 path: `germinal/filters/protenix.py` reads it too, so a protenix-only
run needs it. It is required only when a stage sets
`msa_method: "colabfold_local"`; the shipped configs search remotely.

You can also override any of them on the command line. The four AF3 keys are
per stage, so name the stage:

```bash
python run_germinal_multi.py run=multi target=ba2_kp3 \
    recycle_folder.af3_sif_path=/my/path/alphafold3.sif \
    filter_folder.af3_sif_path=/my/path/alphafold3.sif
```

> **These are tracked files.** Unlike the old `configs/paths/local.yaml`, your
> edits to `configs/folding/af3.yaml` will show up in `git status`. Keep them
> out of commits, or override on the command line instead.

## 3. Clusters where one GPU admits one process

Check your cluster's GPU compute mode before the first run:

```bash
nvidia-smi --query-gpu=compute_mode --format=csv,noheader
```

`Default` means several processes can share a card, and nothing below applies.
`Exclusive_Process` means a card admits exactly one process — and Germinal
holds a JAX context on its GPU for the whole run while shelling out to other
GPU programs: the AF3 container, AbMPNN redesign, Protenix, and the mmseqs GPU
server. On an exclusive-mode cluster every one of those children is refused
the card the parent is holding.

The failures do not name the cause. AF3 reports

```
XlaRuntimeError: INTERNAL: no supported devices found for platform CUDA
```

and AbMPNN redesign exits non-zero with no message, so the design is recorded
as `recycling_failed_redesign` with
`recycle_truncation_reason=redesign_proteinmpnn_failed_recycle_1`. Torch is
the more confusing of the two: `torch.cuda.is_available()` returns `True` on
the parent's card and only the first *allocation* fails.

Give the children a card of their own:

```bash
#SBATCH -G 2

export CUDA_VISIBLE_DEVICES=0          # parent: hallucination, JAX
export GERMINAL_CHILD_CUDA_DEVICE=1    # every GPU child process
```

One spare card is enough. The stages are sequential — the recycle loop folds,
then redesigns, never both at once. The exception is the mmseqs GPU server,
which is long-lived and overlaps with folding, so a run using
`msa_method: "colabfold_local"` with `use_gpu_server` needs a third card.

`GERMINAL_CHILD_CUDA_DEVICE` is opt-in. Unset, child processes inherit the
environment exactly as they always have, so nothing changes on a Default-mode
cluster. The shipped `examples/submit_job.sbatch` still requests one GPU.
