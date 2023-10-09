#!/bin/bash

SUITES=(
    # 'gym'
    'dmc'
    # 'atari'
)

DMC_LEVELS=(
    'trivial'
    # 'easy'
    # 'medium'
    # 'hard'
)


DATETIME=$(date +'%Y%m%d:%H%M%S')

# ID="local_"$DATETIME
ID="local_mfrl_mpo"

SEEDS=(1)

CONFIG=$1

AGENT=$2
# RR=$2
# HP=$2

MEM_FRACTION=0.75


source ~/.bashrc

conda activate acme

for SEED in ${SEEDS[*]}; do
    for SUITE in ${SUITES[*]}; do
        if [ $SUITE == 'gym' ]; then
            echo "suite: " $SUITE
            MUJOCO_GL=egl \
                XLA_PYTHON_CLIENT_MEM_FRACTION=$MEM_FRACTION \
                python run_$AGENT.py \
                --acme_id $ID --agent_id "ppo_hp_"$HP --hp $HP --seed $SEED --suite $SUITE
        fi
        if [ $SUITE == 'control' ] || [ $SUITE == 'dmc' ]; then
            echo "suite: " $SUITE
            for LEVEL in ${DMC_LEVELS[*]}; do
                MUJOCO_GL=egl \
                XLA_PYTHON_CLIENT_MEM_FRACTION=$MEM_FRACTION \
                python ../rl_continuous/run_$AGENT.py \
                --seed $SEED --config $CONFIG \
                --acme_id $ID --agent_id $AGENT \
                --suite $SUITE --level $LEVEL #\
                # --replay_ratio $RR \
                # --hp $HP
            done
        fi
    done
done 

# conda deactivate