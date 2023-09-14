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
ID="vector_"$DATETIME"_test_parallel_seeds"

SEEDS=(1 2)


# for SUITE in ${TASK_SUITES[*]}
# do
#     echo "...INITIALIZE..."
#     # python run_d4pg.py --helpfull
#     python run_d4pg.py --acme_id $ID --suite $SUITE --seeds '1_2_3'
# done

for SUITE in ${TASK_SUITES[*]}
do
    echo "...INITIALIZE..."

    for seed in ${SEEDS[*]}
    do
        python run_d4pg.py --acme_id $ID --suite $SUITE --seed $seed 1
    done
done