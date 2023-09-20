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
    # 'easy'
    # 'medium'
    # 'hard'
)

# ID=$1
# AGENT=$2
# SEED=$3

MEM_FRACTION=0.8

for SUITE in ${SUITES[*]}; do
    for LEVEL in ${LEVELS[*]}; do
        # for SEED in ${SEEDS[*]}; do
        # XLA_PYTHON_CLIENT_MEM_FRACTION=false

        # MUJOCO_GL=egl python ../rl_continuous/run_$AGENT.py --acme_id $ID --seed $SEED --suite $SUITE --level $LEVEL

        # MUJOCO_GL=egl \
        # XLA_PYTHON_CLIENT_MEM_FRACTION=$MEM_FRACTION \
        # python ../rl_continuous/run_$AGENT.py \
        # --acme_id $ID --seed $SEED --suite $SUITE --level $LEVEL

        # MUJOCO_GL=egl \
        # XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 \
        # python ../rl_continuous/run_ppo.py \
        # --acme_id 'v_test' --seed 0 --suite 'control' --level 'trivial' &

        MUJOCO_GL=egl \
        XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 \
        python ../rl_continuous/run_sac.py \
        --acme_id 'v_test' --seed 1 --suite 'control' --level 'trivial' &

        MUJOCO_GL=egl \
        XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 \
        python ../rl_continuous/run_sac.py \
        --acme_id 'v_test' --seed 2 --suite 'control' --level 'trivial' &

        MUJOCO_GL=egl \
        XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 \
        python ../rl_continuous/run_sac.py \
        --acme_id 'v_test' --seed 3 --suite 'control' --level 'trivial'
        # done
    done
done

conda deactivate