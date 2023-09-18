#!/bin/bash

SUITES=(
    # 'gym'
    'control'
    # 'atari'
)

LEVELS=(
    'trivial'
    'easy'
    # 'medium'
    # 'hard'
)


DATETIME=$(date +'%Y%m%d:%H%M%S')

# ID="local_next_"$DATETIME
ID="local_next_20230918:045122"

SEEDS=(1 2 3)


source ~/.bashrc

conda activate acme

for SEED in ${SEEDS[*]}; do
    for SUITE in ${SUITES[*]}; do
        for LEVEL in ${LEVELS[*]}; do
            # MUJOCO_GL=egl python run_d4pg.py --acme_id $ID --seed $SEED --suite $SUITE --level $LEVEL
            # MUJOCO_GL=egl python run_d4pg.py --acme_id 'test_d4pg' --seed 0 --suite 'control' --level 'easy'
            MUJOCO_GL=egl python run_d4pg_next.py --acme_id $ID --seed $SEED --suite $SUITE --level $LEVEL
        done
    done
done 

conda deactivate