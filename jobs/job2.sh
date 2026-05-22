#!/bin/bash
#SBATCH --job-name=job2
#SBATCH --partition=allgroups
#SBATCH --nodelist=dellsrv1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --output=/extra/fferrabe/Thesis26/lexicon3d/logs/eval-%j.out


set -e 

source /home/fferrabe/miniconda3/etc/profile.d/conda.sh
conda activate m3drefclip

conda clean --all -y

conda install -y -c fvcore -c iopath -c conda-forge fvcore iopath

conda install -y pytorch3d -c pytorch3d

