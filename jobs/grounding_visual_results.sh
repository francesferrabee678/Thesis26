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


python grounding_visual_results.py \
  --scene_data_dir /home/fferrabe/Thesis26/lexicon3d/evals/M3DRef-CLIP/dataset/scannetv2/val \
  --lang_json /home/fferrabe/Thesis26/lexicon3d/evals/M3DRef-CLIP/dataset/scanrefer/metadata/ScanRefer_filtered_val.json \
  --preds dinov3=/home/fferrabe/Thesis26/lexicon3d/evals/M3DRef-CLIP/output/ScanRefer/dinov3/inference/val dinov2=/home/fferrabe/Thesis26/lexicon3d/evals/M3DRef-CLIP/output/ScanRefer/dinov2/inference/val uniscene3d=/home/fferrabe/Thesis26/lexicon3d/evals/M3DRef-CLIP/output/ScanRefer/uniscene3d/inference/val ptv3ppt=/home/fferrabe/Thesis26/lexicon3d/evals/M3DRef-CLIP/output/ScanRefer/ptv3ppt/inference/val concerto=/home/fferrabe/Thesis26/lexicon3d/evals/M3DRef-CLIP/output/ScanRefer/concerto_redue/inference/val  \
  --scenes all --eval_type multiple --num_queries 1 \
  --out grounding_vis/multiple_vis


