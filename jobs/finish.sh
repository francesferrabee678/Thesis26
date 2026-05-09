#!/bin/bash
#SBATCH --job-name=ptv3_final_val
#SBATCH --partition=testing
#SBATCH --chdir=/extra/fferrabe/Thesis26/lexicon3d
#SBATCH --gres=gpu:NVIDIA-RTX-A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-04:00:00
#SBATCH --output=/extra/fferrabe/Thesis26/lexicon3d/logs/eval-%j.out

set -e 

source /home/fferrabe/miniconda3/etc/profile.d/conda.sh
conda activate lexicon3d

cd /home/fferrabe/Thesis26/openscene_scripts/

python preprocess_2d_scannet.py\
 --scannet_path /home/fferrabe/Thesis26/data/scannet_raw/scans\
 --output_path /home/fferrabe/Thesis26/data/scannet_2d
