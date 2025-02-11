#!/bin/bash

source ~/.bashrc

conda activate acme

SUITES=(
	# 'gym'
	# 'control'
	'dmc'
	# 'atari'
)

DMC_LEVELS=(
	# 'trivial'
	# 'easy'
	# 'medium'
	'hard'
	'extra'
)

ID=$1
AGENT=$2
HP=$3
SEED=$4

MEM_FRACTION=0.75

for SUITE in ${SUITES[*]}; do

	if [ $SUITE == 'gym' ]; then
		echo $SUITE
		MUJOCO_GL=egl \
			XLA_PYTHON_CLIENT_MEM_FRACTION=$MEM_FRACTION \
			python run_$AGENT.py \
			--acme_id $ID --agent_id $AGENT --hp $HP \
			--suite $SUITE --level $LEVEL \
			--seed $SEED
	fi

	if [ $SUITE == 'control' ] || [ $SUITE == 'dmc' ]; then
		echo $SUITE
		for LEVEL in ${DMC_LEVELS[*]}; do
			MUJOCO_GL=egl \
			XLA_PYTHON_CLIENT_MEM_FRACTION=$MEM_FRACTION \
			python ../rl_continuous/run_$AGENT.py \
			--acme_id $ID --agent_id $AGENT --hp $HP \
			--suite $SUITE --level $LEVEL \
			--seed $SEED
		done
	fi

done

conda deactivate