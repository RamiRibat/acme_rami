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

source ~/.bashrc

conda activate acme

# echo "SEED " $1 >> ~/logdir/vector.log
# echo "CUDA " $CUDA_VISIBLE_DEVICES >> ~/logdir/vector.log
python run_d4pg.py --acme_id $1 --suite $2 --seed $3

conda deactivate