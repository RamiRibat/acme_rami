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
ID="local_test_acme"

SEEDS=(1)


source ~/.bashrc

conda activate acme

for SEED in ${SEEDS[*]}; do
    for SUITE in ${SUITES[*]}; do
        for LEVEL in ${LEVELS[*]}; do
            # MUJOCO_GL=egl python run_ppo.py --acme_id $ID --seed $SEED --suite $SUITE --level $LEVEL
            # MUJOCO_GL=egl python run_sac.py --acme_id $ID --seed $SEED --suite $SUITE --level $LEVEL
            # MUJOCO_GL=egl python run_d4pg.py --acme_id $ID --seed $SEED --suite $SUITE --level $LEVEL
            MUJOCO_GL=egl \
            XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 \
            python run_sac.py \
            --acme_id $ID --seed $SEED --suite $SUITE --level $LEVEL
        done
    done
done 

# conda deactivate