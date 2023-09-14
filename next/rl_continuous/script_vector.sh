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

# bash script.sh

TASK_SUITES=(
    # 'gym'
    'control'
    # 'atari26'
    # 'atari52'
    # 'atari57'
)


DATETIME=$(date +'%Y%m%d:%H%M%S')

ID="vector_"$DATETIME

SEEDS=(1 2 3)

GPUS=(0 1 2)

source ~/.bashrc

conda activate acme

for SUITE in ${TASK_SUITES[*]}; do
    for s in ${!SEEDS[*]}; do
        CUDA_VISIBLE_DEVICES=${GPUS[s]} python run_d4pg.py --acme_id $ID --suite $SUITE --seed ${SEEDS[s]} &
    done
done