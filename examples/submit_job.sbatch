#!/bin/bash
# Example SLURM job script for Germinal multi-target design.
# Copy this file, fill in the <placeholders>, and submit with: sbatch my_job.sbatch
#
# See docs/af3-setup.md for the four AF3 path values.

#SBATCH --job-name=germinal_run
#SBATCH --partition=<your-gpu-partition>
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
# Log paths are relative to the submission directory and must already exist:
# SLURM opens them before this script body runs, so pointing them inside
# results/ -- which is gitignored and absent in a fresh clone -- makes the job
# die at launch with "Unable to open file" and write nothing at all.
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

mkdir -p results/my_experiment

# Keep ~/.local/lib/pythonX.Y/site-packages out of the environment. A stale
# user-site install of biopython, scikit-learn or pyarrow shadows the conda
# env's copy, and the mismatch surfaces inside the first trajectory rather
# than at startup.
export PYTHONNOUSERSITE=1

# --no-capture-output is not optional here. Without it `conda run` buffers the
# child's stdout and stderr until the child exits, which defeats python -u and
# means a job killed by the time limit or scancel writes a log containing
# nothing at all -- no progress, and no traceback explaining the failure.
conda run -n germinal --no-capture-output python -u run_germinal_multi.py \
    run=multi_logits \
    target=ba2_kp3 \
    filter/initial=scfv_multi \
    filter/final=scfv_multi \
    experiment_name=my_experiment \
    filter_folder.af3_repo_path=<path-to-alphafold3-repo> \
    filter_folder.af3_sif_path=<path-to-alphafold3.sif> \
    filter_folder.af3_model_dir=<path-to-af3-weights> \
    filter_folder.af3_db_dir=<path-to-af3-databases> \
    recycle_folder.af3_repo_path=<path-to-alphafold3-repo> \
    recycle_folder.af3_sif_path=<path-to-alphafold3.sif> \
    recycle_folder.af3_model_dir=<path-to-af3-weights> \
    recycle_folder.af3_db_dir=<path-to-af3-databases>
