#!/bin/bash

TASKS=(
    "cartplole:swingup" "walker:walk"
    "hopper:stand" "swimmer:swimmer6"
    "cheetah:run" "walker:run"
)

SEEDS=(1 2 3)


DATETIME=$(date +'%Y%m%d-%H%M%S')

ID="$DATETIME"


for T in ${TASKS[*]}
do
    ENVID="control:$T"
    for S in ${SEEDS[*]}
    do
        echo $ENVID $S
        python run_d4pg_2.py --acme_id $ID --env_name $ENVID --seed $S
    done
done