#!/bin/bash
#SBATCH --job-name=ptv3_final_val
#SBATCH --partition=testing
#SBATCH --chdir=/extra/fferrabe/Thesis26/lexicon3d
#SBATCH --gres=gpu:NVIDIA-RTX-A6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-04:00:00
#SBATCH --output=/extra/fferrabe/Thesis26/lexicon3d/logs/eval-%j.out

set -e 

source /home/fferrabe/miniconda3/etc/profile.d/conda.sh
conda activate lexicon3d

cd /home/fferrabe/Thesis26/lexicon3d/lexicon3d/


python fusion_scannet_dinov2.py \
 --data_dir /home/fferrabe/Thesis26/data/openscene_format \
 --output_dir /home/fferrabe/Thesis26/data/openscene_format/scannet_3d/lexicon3d/dinov2/ \
 --split val \
 --hparams scan_dir=/home/fferrabe/Thesis26/data/openscene_format/scannet_2d/ voxel_size=0.02

