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

ID=$1
AGENT=$2
SEED=$3

for SUITE in ${SUITES[*]}; do
    for LEVEL in ${LEVELS[*]}; do
        # MUJOCO_GL=egl python ../rl_continuous/run_$AGENT.py --acme_id $ID --seed $SEED --suite $SUITE --level $LEVEL
        # MUJOCO_GL=egl \
        # XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
        # python ../rl_continuous/run_$AGENT.py \
        # --acme_id $ID --seed $SEED --suite $SUITE --level $LEVEL

        # MUJOCO_GL=egl \
        # # XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 \
        # python ../rl_continuous/run_ppo.py \
        # --acme_id 'v_test' --seed 0 --suite 'control' --level 'trivial' &

        MUJOCO_GL=egl \
        # XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 \
        python ../rl_continuous/run_sac.py \
        --acme_id 'v_test' --seed 0 --suite 'control' --level 'trivial' &
        
        sleep 30

        MUJOCO_GL=egl \
        # XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 \
        python ../rl_continuous/run_d4pg.py \
        --acme_id 'v_test' --seed 0 --suite 'control' --level 'trivial'
    done
done

conda deactivate