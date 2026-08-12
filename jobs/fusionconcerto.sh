#!/bin/bash
#SBATCH --job-name=fusionconcertoval
#SBATCH --partition=testing
#SBATCH --gres=gpu:1
#SBATCH --exclude=vgpu11
#SBATCH --cpus-per-task=4
#SBATCH --mem=110G
#SBATCH --time=3-04:00:00
#SBATCH --output=/extra/fferrabe/Thesis26/lexicon3d/logs/eval-%j.out


set -e 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export LD_LIBRARY_PATH=""
source /home/fferrabe/miniconda3/etc/profile.d/conda.sh
conda activate lexicon3d

cd /home/fferrabe/Thesis26/lexicon3d/lexicon3d/


python fusion_scannet_concerto.py \
 --data_dir /home/fferrabe/Thesis26/data/openscene_format \
 --output_dir /home/fferrabe/Thesis26/data/openscene_format/scannet_3d/lexicon3d/concerto2/ \
 --split val \
 --hparams scan_dir=/home/fferrabe/Thesis26/data/openscene_format/scannet_2d/ voxel_size=0.02
