#!/bin/bash
#SBATCH --job-name=big-lama-training
#SBATCH --output=big-lama-%j.out
#SBATCH --error=big-lama-%j.err
#SBATCH -p GPU-shared
#SBATCH --time=24:00:00
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=4

# Load necessary modules, if required
set -x
# Activate Conda environment
source activate lama

# Navigate to your project directory (if not submitting the job from there)
cd /ocean/projects/cis220039p/cherieho/map-explore/lama
export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)
export WANDB_API_KEY=97df59ceef60fe2f1a45ac4a14272e2d6ff54487
export WANDB_DEBUG=true
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# Run the Python script with all the arguments
python bin/train.py -cn big-lama \
    location=kth_train_3 \
    data.batch_size=4 \
    +trainer.kwargs.resume_from_checkpoint=$(pwd)/big-lama-with-discr-remove-loss_segm_pl.ckpt
