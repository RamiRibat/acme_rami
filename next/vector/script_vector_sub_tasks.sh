#!/bin/bash

source ~/.bashrc

conda activate acme

SUITES=(
    # 'gym'
    'control'
    # 'atari'
)

LEVELS=(
    # 'trivial'
    # 'easy'
    # 'medium'
    'hard'
)

ID="v_20230918:202133_256x256"

AGENT='d4pg'

SEEDS=(1 2 3)

MEM_FRACTION=0.8

for SEED in ${SEEDS[*]}; do
    for SUITE in ${SUITES[*]}; do
        for LEVEL in ${LEVELS[*]}; do
            MUJOCO_GL=egl \
            XLA_PYTHON_CLIENT_MEM_FRACTION=$MEM_FRACTION \
            python ../rl_continuous/run_$AGENT.py \
            --acme_id $ID --seed $SEED --suite $SUITE --level $LEVEL
        done
    done
done

conda deactivate