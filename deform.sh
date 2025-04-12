#!/bin/sh
#SBATCH --time=40:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=2
#SBATCH --mem=128G
#SBATCH --job-name=deform_move_down
#SBATCH --account=st-singha53-1-gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=royhe@student.ubc.ca
#SBATCH --output=deform_moveDown.txt
#SBATCH --error=deform_moveDown_error.txt

python crystal-train.py