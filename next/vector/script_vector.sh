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

ID=$1
AGENT=$2
SEED=$3

for SUITE in ${SUITES[*]}; do
    for LEVEL in ${LEVELS[*]}; do
        MUJOCO_GL=egl python ../rl_continuous/run_$AGENT.py --acme_id $ID --seed $SEED --suite $SUITE --level $LEVEL
    done
done

conda deactivate