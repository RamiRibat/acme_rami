#!/bin/bash

source ~/.bashrc

conda activate acme

if [ $3 == 1 ]; then
    sleep 30
fi

echo "SEED " $3 >> ~/logdir/vector.log
echo "CUDA " $CUDA_VISIBLE_DEVICES >> ~/logdir/vector.log
# python run_d4pg.py --acme_id $1 --suite $2 --seed $3

conda deactivate