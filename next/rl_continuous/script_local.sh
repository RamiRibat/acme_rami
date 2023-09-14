#!/bin/bash

TASK_SUITES=(
    # 'gym'
    'control'
    # 'atari26'
    # 'atari52'
    # 'atari57'
)


DATETIME=$(date +'%Y%m%d:%H%M%S')

ID="local_"$DATETIME

SEEDS=(1 2 3)


source ~/.bashrc

conda activate acme

for SUITE in ${TASK_SUITES[*]}; do
    for s in ${!SEEDS[*]}; do
        # python run_d4pg.py --acme_id $ID --suite $SUITE --seed ${SEEDS[s]}
        python run_d4pg.py --acme_id $ID --suite $SUITE --level X --task X --seed ${SEEDS[s]}
    done
done

conda deactivate