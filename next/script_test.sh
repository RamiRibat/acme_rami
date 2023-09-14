#!/bin/bash

# function parse_yaml {
#    local prefix=$2
#    local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
#    sed -ne "s|^\($s\):|\1|" \
#         -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
#         -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  $1 |
#    awk -F$fs '{
#       indent = length($1)/2;
#       vname[indent] = $2;
#       for (i in vname) {if (i > indent) {delete vname[i]}}
#       if (length($3) > 0) {
#          vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
#          printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, $2, $3);
#       }
#    }'
# }

TASK_SUITES=(
    # 'gym'
    'control'
    # 'atari26'
    # 'atari52'
    # 'atari57'
)


DATETIME=$(date +'%Y%m%d:%H%M%S')

# ID="$DATETIME"
ID=$DATETIME"__test"

SEEDS=(1 2 3)
# GPUS=(0 1 2)

# for s in ${!SEEDS[*]}
# do
#     echo "GPU: " ${GPUS[s]} "Seed: " ${SEEDS[s]} &
# done


rm -f script_tmp.sh
touch script_tmp.sh
chmod 777 script_tmp.sh
echo '#!/bin/bash' >> script_tmp.sh

python configure.py

source script_tmp.sh

echo "SHOW"

for suite in "${config[@]}"; do
    echo "suite: $suite "
    declare -n levels="$suite"  # now p is a reference to a variable "$suite"
    for level in "${!levels[@]}"; do
        echo "       $level : ${levels[$level]}"
        # declare -n p="$suite"
        # for attr in "${!p[@]}"; do
        #     echo "    $attr : ${p[$attr]}"
        # done
    done
done

# echo "a " $A
# export A=1
# echo "a " $A

# python -c $"import os; print(len(os.environ))"

# config=$(
#     python -c $"import yaml,sys; config=yaml.safe_load(open('config.yaml')); print(config)"
# )
# X=('dict={'a': A}')
# echo ${X[0]}

# for SUITE in ${TASK_SUITES[*]}; do
#     for s in ${!SEEDS[*]}; do
#         echo "Seed: " ${SEEDS[s]}
#         echo "a " $a
#         # echo "country " ${countries[ALB]}
#         # echo "control " $control__base
#         # echo "control:trivial:tasks " ${!control[*]}
#     done
# done

# rm -f script_tmp.sh