#!/bin/bash
#SBATCH --job-name=job1
#SBATCH --partition=allgroups
#SBATCH --nodelist=dellsrv1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --output=/extra/fferrabe/Thesis26/lexicon3d/logs/eval-%j.out


set -e 

source /home/fferrabe/miniconda3/etc/profile.d/conda.sh
conda activate m3drefclip

conda install -y pytorch torchvision pytorch-cuda=11.7 \
  -c pytorch -c nvidia \
  --strict-channel-priority

