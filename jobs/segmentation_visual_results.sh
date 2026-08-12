#!/bin/bash
#SBATCH --job-name=visuals
#SBATCH --partition=testing
#SBATCH --gres=gpu:1
#SBATCH --exclude=vgpu11
#SBATCH --chdir=/extra/fferrabe/Thesis26/lexicon3d
#SBATCH --cpus-per-task=4
#SBATCH --mem=110G
#SBATCH --time=7-00:00:00
#SBATCH --output=/extra/fferrabe/Thesis26/lexicon3d/logs/eval-%j.out

set -e

cd /home/fferrabe/Thesis26/lexicon3d/jobs
source /home/fferrabe/miniconda3/etc/profile.d/conda.sh
conda activate lexicon3d



python segmentation_visual_results.py  \
  --data_dir /home/fferrabe/Thesis26/data/openscene_format \
  --feat_dir /home/fferrabe/Thesis26/data/openscene_format/scannet_3d/lexicon3d/concerto/concerto_features \
  --model gt \
  --ckpt /home/fferrabe/Thesis26/lexicon3d/evals/segmentation/lightning_logs/concerto/checkpoints/last.ckpt \
  --scenes scene0030_00 \
  --out vis
