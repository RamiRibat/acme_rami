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

for SUITE in ${SUITES[*]}; do
    for LEVEL in ${LEVELS[*]}; do
        python run_d4pg_v.py --acme_id $1 --seed $2 --suite $SUITE --level $LEVEL
    done
done

conda deactivate