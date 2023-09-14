#!/bin/bash

source ~/.bashrc

conda activate acme

echo "SEED " $3 >> ~/logdir/vector.log
echo "CUDA " $CUDA_VISIBLE_DEVICES >> ~/logdir/vector.log
# python run_d4pg.py --acme_id $1 --suite $2 --seed $3

conda deactivate