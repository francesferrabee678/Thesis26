#!/bin/bash
#SBATCH --job-name=ptv3_eval
#SBATCH --partition=testing
#SBATCH --chdir=/extra/fferrabe/Thesis26/Pointcept
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/extra/fferrabe/Thesis26/Pointcept/logs/eval-%j.out
#SBATCH --error=/extra/fferrabe/Thesis26/Pointcept/logs/eval-%j.err

set -e 

# 1. Create a clean slate
conda create -n lexicon_env python=3.10 -y
conda activate lexicon_env

# 2. Install the requirements again here
cd ~/Thesis26/lexicon3d
pip install -r requirements.txt

# Install DINOv2 specifically (from Meta's repo)
pip install torch torchvision torchaudio
pip install xformers # Recommended for DINOv2 speed

