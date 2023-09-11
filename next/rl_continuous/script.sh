#!/bin/bash

DATETIME=$(date +'%Y%m%d-%H%M%S')

ID="$DATETIME"

python run_d4pg_2.py --acme_id $ID --seed 1
python run_d4pg_2.py --acme_id $ID --seed 2
python run_d4pg_2.py --acme_id $ID --seed 3