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
HP=$3
SEED=$4

MEM_FRACTION=0.75

for SUITE in ${SUITES[*]}; do
    for LEVEL in ${LEVELS[*]}; do
        MUJOCO_GL=egl \
        XLA_PYTHON_CLIENT_MEM_FRACTION=$MEM_FRACTION \
        python ../rl_continuous/run_$AGENT.py \
        --acme_id $ID --agent_id "hp_"$HP --hp $HP --seed $SEED --suite $SUITE --level $LEVEL
    done
done

conda deactivate