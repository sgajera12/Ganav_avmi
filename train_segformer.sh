#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH --mem 32G
#SBATCH --job-name="segformer_avmi"
#SBATCH -o /home/sgajera/avmi/GANav-offroad/slurm_%j.out
#SBATCH -e /home/sgajera/avmi/GANav-offroad/slurm_%j.err
#SBATCH -p academic

module load slurm
module load cuda

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate segformer

cd /home/sgajera/avmi/GANav-offroad

python tools/train_segformer_hf.py
