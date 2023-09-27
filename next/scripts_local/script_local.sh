#!/bin/bash

SUITES=(
    # 'gym'
    'control'
    # 'atari'
)

LEVELS=(
    'trivial'
    # 'easy'
    # 'medium'
    # 'hard'
)


DATETIME=$(date +'%Y%m%d:%H%M%S')

# ID="local_"$DATETIME
# ID="local_mfrl_sr"
ID="local_ppo"

SEEDS=(1)

AGENT=$1
# RR=$2
# HP=$2


source ~/.bashrc

conda activate acme

for SEED in ${SEEDS[*]}; do
    for SUITE in ${SUITES[*]}; do
        if [ $SUITE == 'gym' ]; then
            echo 'gym'
            MUJOCO_GL=egl \
                XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 \
                python run_$AGENT.py \
                --acme_id $ID --agent_id "ppo_hp_"$HP --hp $HP --seed $SEED --suite $SUITE
        fi
        if [ $SUITE == 'control' ]; then
            echo 'control'
            for LEVEL in ${LEVELS[*]}; do
                # MUJOCO_GL=egl \
                # XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 \
                # python run_$AGENT.py \
                # --acme_id $ID --agent_id $AGENT"_sr_"$RR --replay_ratio $RR --seed $SEED --suite $SUITE --level $LEVEL
                MUJOCO_GL=egl \
                XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 \
                python run_$AGENT.py \
                --acme_id $ID --agent_id $AGENT --seed $SEED --suite $SUITE --level $LEVEL
            done
        fi
    done
done 

# conda deactivate