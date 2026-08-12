#!/bin/bash
#SBATCH --job-name=eval_grounding
#SBATCH --partition=testing
#SBATCH --chdir=/extra/fferrabe/Thesis26/lexicon3d
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-04:00:00
#SBATCH --output=/extra/fferrabe/Thesis26/lexicon3d/logs/eval-%j.out


set -e 

cd /home/fferrabe/Thesis26/lexicon3d/evals/M3DRef-CLIP

source /home/fferrabe/miniconda3/etc/profile.d/conda.sh
conda activate m3drefclip

export WANDB_MODE=offline
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

python test.py \
    data=scanrefer \
    data.inference.split=val \
    ckpt_path=$CKPT_PATH \
    pred_path=$PRED_PATH