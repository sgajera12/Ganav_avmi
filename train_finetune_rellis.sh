#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00
#SBATCH --mem 64G
#SBATCH --job-name="ft_rellis"
#SBATCH -o /home/sgajera/avmi/GANav-offroad/slurm_ft_rellis_%j.out
#SBATCH -e /home/sgajera/avmi/GANav-offroad/slurm_ft_rellis_%j.err
#SBATCH -p academic

module load slurm
module load cuda

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate segformer

cd /home/sgajera/avmi/GANav-offroad
python tools/finetune_segformer_rellis.py
