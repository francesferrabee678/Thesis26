#!/bin/bash
#SBATCH --job-name=fusionv3
#SBATCH --partition=allgroups
#SBATCH --gres=gpu:1
#SBATCH --nodelist=dellsrv5
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=3-04:00:00
#SBATCH --output=/extra/fferrabe/Thesis26/lexicon3d/logs/eval-%j.out


set -e 

source /home/fferrabe/miniconda3/etc/profile.d/conda.sh
conda activate lexicon3d_dinov3

cd /home/fferrabe/Thesis26/lexicon3d/lexicon3d/


python fusion_scannet_dinov3.py \
 --data_dir /home/fferrabe/Thesis26/data/openscene_format \
 --output_dir /home/fferrabe/Thesis26/data/openscene_format/scannet_3d/lexicon3d/dinov3/ \
 --split train \
 --hparams scan_dir=/home/fferrabe/Thesis26/data/openscene_format/scannet_2d/ voxel_size=0.05
