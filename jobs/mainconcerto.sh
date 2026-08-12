#!/bin/bash
#SBATCH --job-name=main_concerto
#SBATCH --partition=allgroups
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=120G
#SBATCH --time=5-04:00:00
#SBATCH --output=/extra/fferrabe/Thesis26/lexicon3d/logs/eval-%j.out


set -e 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export LD_LIBRARY_PATH=""
source /home/fferrabe/miniconda3/etc/profile.d/conda.sh
conda activate lexicon3d_dinov3

cd /home/fferrabe/Thesis26/lexicon3d/evals/segmentation/

python main.py \
  --model_name linear_probe_concerto \
  --dataset concerto_feature \
  --data_dir /home/fferrabe/Thesis26/data/openscene_format/scannet_3d \
  --batch_size 1 \
  --lr 1e-4 \
  --max_epochs 20 \
  --num_workers 20
