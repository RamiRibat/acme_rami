#!/bin/bash

SUITES=(
    # 'gym'
    'control'
    # 'atari'
)

LEVELS=(
    # 'trivial'
    'easy'
    'medium'
    # 'hard'
)


DATETIME=$(date +'%Y%m%d:%H%M%S')

# ID="local_"$DATETIME
ID="local_20230914:041248"

SEEDS=(2 3)


source ~/.bashrc

conda activate acme

for SEED in ${SEEDS[*]}; do
    for SUITE in ${SUITES[*]}; do
        for LEVEL in ${LEVELS[*]}; do
            python run_d4pg.py --acme_id $ID --seed $SEED --suite $SUITE --level $LEVEL
        done
    done
done 

conda deactivate