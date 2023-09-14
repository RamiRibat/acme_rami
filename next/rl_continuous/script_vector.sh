#!/bin/bash
#SBATCH --job-name=dmc_d4pg

#SBATCH --partition=rtx6000,a40

#SBATCH --gres=gpu:3

#SBATCH --qos=normal

#SBATCH --cpus-per-task=8

#SBATCH --mem-per-cpu=8G

#SBATCH --output=slurm-%j.out

#SBATCH --error=slurm-%j.err

# prepare your environment here
module load cuda-11.8

# put your command here

bash script.sh