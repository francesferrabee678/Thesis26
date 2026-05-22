#!/bin/bash
#SBATCH --job-name=train_grounding
#SBATCH --partition=allgroups
#SBATCH --gres=gpu:1
#SBATCH --nodelist=dellsrv4
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=3-04:00:00
#SBATCH --output=/extra/fferrabe/Thesis26/lexicon3d/logs/eval-%j.out

set -e 

cd /home/fferrabe/Thesis26/lexicon3d/jobs

source /home/fferrabe/miniconda3/etc/profile.d/conda.sh
conda activate m3drefclip
