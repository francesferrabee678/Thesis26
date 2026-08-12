#!/bin/bash
#SBATCH --job-name=main_uniscene
#SBATCH --partition=testing
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=110G
#SBATCH --time=3-04:00:00
#SBATCH --output=/extra/fferrabe/Thesis26/lexicon3d/logs/eval-%j.out


set -e 

source /home/fferrabe/miniconda3/etc/profile.d/conda.sh
conda activate lexicon3d_dinov3

cd /home/fferrabe/Thesis26/lexicon3d/evals/segmentation/

python main.py \
  --model_name linear_probe_uniscene3d \
  --dataset uniscene3d_feature \
  --data_dir /home/fferrabe/Thesis26/data/openscene_format/scannet_3d \
  --batch_size 1 \
  --lr 1e-4 \
  --max_epochs 20 \
  --num_workers 4