#!/bin/bash
#SBATCH --job-name=q_dmc_d4pg

#SBATCH --partition=gpu

#SBATCH --gres=gpu:3

#SBATCH --qos=normal

#SBATCH --cpus-per-task=24

#SBATCH --mem-per-cpu=1G

# prepare your environment here
module load cuda-11.8

# put your command here

bash script.sh