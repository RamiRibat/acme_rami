#!/bin/bash

TASKS=(
    "cartplole_swingup" "walker_walk"
    "hopper_stand" "swimmer_swimmer6"
    "cheetah_run" "walker_run"
)

SEEDS=(1 2 3)


DATETIME=$(date +'%Y%m%d-%H%M%S')

ID="$DATETIME"


for T in ${TASKS[*]}
do
    ENVID="control_$T"
    for S in ${SEEDS[*]}
    do
        echo $ENVID $S
        python run_d4pg_2.py --acme_id $ID --env_name $ENVID --seed $S
    done
done