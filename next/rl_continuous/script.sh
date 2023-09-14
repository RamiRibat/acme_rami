#!/bin/bash

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

# for SUITE in ${TASK_SUITES[*]}; do
#     for s in ${!SEEDS[*]}; do
#         CUDA_VISIBLE_DEVICES=${GPUS[s]} python run_d4pg.py --acme_id $ID --suite $SUITE --seed ${SEEDS[s]} &
#     done
# done

for SUITE in ${TASK_SUITES[*]}; do
    echo "SEED " $1 >> ~/logdir/vector.log
    echo "CUDA " $CUDA_VISIBLE_DEVICES >> ~/logdir/vector.log
    # python run_d4pg.py --acme_id $ID --suite $SUITE --seed $1
done

conda deactivate