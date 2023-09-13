#!/bin/bash

TASK_SUITES=(
    # 'gym'
    'control'
    # 'atari26'
    # 'atari52'
    # 'atari57'
)


DATETIME=$(date +'%Y%m%d:%H%M%S')

# ID="$DATETIME"
ID=$DATETIME"__vector_v"


for SUITE in ${TASK_SUITES[*]}
do
    echo "...INITIALIZE..."
    # python run_d4pg.py --helpfull
    python run_d4pg.py --acme_id $ID --suite $SUITE --seeds '1_2_3'
done