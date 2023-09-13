#!/bin/bash
#SBATCH --job-name=dmc_d4pg

#SBATCH --partition=t4v2,rtx6000,a40

#SBATCH --gres=gpu:1

#SBATCH --qos=normal

#SBATCH --cpus-per-task=4

#SBATCH --mem-per-cpu=8G

#SBATCH --output=slurm-%j.out

#SBATCH --error=slurm-%j.err

# prepare your environment here
module load cuda-11.8

# put your command here
