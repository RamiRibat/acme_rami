#!/bin/bash

source ~/.bashrc

conda activate acme

SUITES=(
    # 'gym'
    'control'
    # 'atari'
)

LEVELS=(
    'trivial'
    'easy'
    'medium'
    'hard'
)

# echo "SEED " $3 >> ~/logdir/vector.log
# echo "CUDA " $CUDA_VISIBLE_DEVICES >> ~/logdir/vector.log


for SUITE in ${SUITES[*]}; do
    for LEVEL in ${LEVELS[*]}; do
        # for TASK in ${TASKS[*]}; do
        # echo "SUITE: " $SUITE "LEVEL: " $LEVEL
        python run_d4pg.py --acme_id $1 --seed $2 --suite $SUITE --level $LEVEL
        # done
    done
done

conda deactivate