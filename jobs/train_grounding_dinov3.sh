#!/bin/bash
#SBATCH --job-name=dinov3_grounding
#SBATCH --partition=testing
#SBATCH --gres=gpu:1
#SBATCH --exclude=vgpu11
#SBATCH --chdir=/extra/fferrabe/Thesis26/lexicon3d
#SBATCH --cpus-per-task=4
#SBATCH --mem=110G
#SBATCH --time=14-00:00:00
#SBATCH --output=/extra/fferrabe/Thesis26/lexicon3d/logs/eval-%j.out

set -e

cd /home/fferrabe/Thesis26/lexicon3d/evals/M3DRef-CLIP
source /home/fferrabe/miniconda3/etc/profile.d/conda.sh
conda activate m3drefclip

unset LD_LIBRARY_PATH
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline
export HDF5_USE_FILE_LOCKING=FALSE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export NVIDIA_TF32_OVERRIDE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENCV_NUM_THREADS=0
export LEX3D_PREFIX=dinov3

# 6. Run Training
python train.py \
    data=scanrefer \
    experiment_name=dinov3\
    trainer.max_epochs=30 \
    trainer.precision=32 \
    data.dataloader.num_workers=4 \
    data.dataloader.batch_size=8 