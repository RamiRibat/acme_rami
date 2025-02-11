# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runners used for executing local agents."""

# Python
# import sys, time
from termcolor import colored
# from typing import Optional, Sequence, Tuple

# ML/DL
import jax

# ACME/DeepMind
import acme
from acme import core, specs, types
# from acme.jax import utils
from acme.jax.experiments import config
from acme.tf import savers
from acme.utils import counting

# import dm_env
# import reverb


def run_evaluation(
    experiment: config.ExperimentConfig,
	# eval_every: int = 100,
	eval_episodes: int = 1
):
	"""Runs a simple, single-threaded training loop using the default evaluators.

	It targets simplicity of the code and so only the basic features of the
	ExperimentConfig are supported.

	Arguments:
		experiment: Definition and configuration of the agent to run.
		eval_every: After how many actor steps to perform evaluation.
		num_eval_episodes: How many evaluation episodes to execute at each
		evaluation step.
	"""

	if experiment.checkpointing is not None:
		checkpointing = experiment.checkpointing

	key = jax.random.PRNGKey(experiment.seed)


	"""Environment."""
	# Create the environment and get its spec.
	environment = experiment.environment_factory(experiment.seed)
	environment_spec = experiment.environment_spec or specs.make_environment_spec(environment)


	# """Network/Policy."""
	# Create networks -> [ policy(evaluation), learner ]
	networks = experiment.network_factory(environment_spec)
	# # Create evaluation policy -> [ actor(evaluation) ]
	# eval_policy = config.make_policy(
	# 	experiment=experiment,
	# 	networks=networks,
	# 	environment_spec=environment_spec,
	# 	evaluation=True
	# )


	"""Parent Counter"""
	# Parent counter allows to (share step counts) between train and eval loops and
	# the learner, so that it is possible to plot for example evaluator's return
	# value as a function of the number of training episodes.
	counter = counting.Counter(
		parent=None,
		prefix='',
		time_delta=0.,
		return_only_prefixed=False
	)
	
	if experiment.checkpointing is not None:
		counter_ckpt = savers.Checkpointer(
			objects_to_save={'counter': counter},
			subdirectory='counter',
			time_delta_minutes=checkpointing.time_delta_minutes,
			directory=checkpointing.directory,
			add_uid=checkpointing.add_uid,
			max_to_keep=checkpointing.max_to_keep,
			keep_checkpoint_every_n_hours=checkpointing.keep_checkpoint_every_n_hours,
			checkpoint_ttl_seconds=checkpointing.checkpoint_ttl_seconds,
		)


	"""Learner."""
	# Create learner -> [ actor(evaluation) ]
	key, learner_key = jax.random.split(key)
	learner_counter = counting.Counter(
		parent=counter,
		prefix='learner',
		time_delta=0.,
		return_only_prefixed=False
	)
	# Create logger -> acme.utils.experiment_utils.py
	learner_logger = experiment.logger_factory(
		label='learner',
		# steps_key=learner_counter.get_steps_key(),
		# task_instance=0,
	)
	# Create learner -> [ actor, actor' ]
	learner = experiment.builder.make_learner(
		random_key=learner_key,
		networks=networks,
		iterator=None,
		# logger_fn=experiment.logger_factory,
		environment_spec=environment_spec,
		# replay_client=replay_client, # *
		# counter=learner_counter,
		# logger=learner_logger
	)
	
	if experiment.checkpointing is not None:
		learner_ckpt = savers.Checkpointer(
			objects_to_save={'learner': learner},
			subdirectory='learner',
			time_delta_minutes=checkpointing.time_delta_minutes,
			directory=checkpointing.directory,
			add_uid=checkpointing.add_uid,
			max_to_keep=checkpointing.max_to_keep,
			keep_checkpoint_every_n_hours=checkpointing.keep_checkpoint_every_n_hours,
			checkpoint_ttl_seconds=checkpointing.checkpoint_ttl_seconds,
		)


	"""Evaluation loop."""
	
	if eval_episodes:
		key, eval_actor_key = jax.random.split(key)
		for evaluator in experiment.get_evaluator_factories():
			eval_loop = evaluator(
				random_key=eval_actor_key,
				variable_source=learner,
				counter=counter,
				make_actor=experiment.builder.make_actor
			)

	"""Running loop(s)."""
	# if actor_counter.get_steps_key() not in counter.get_counts().keys():
	# 	actor_loop.run(num_steps=0) # init csv columns
	# 	eval_loop.run(num_episodes=eval_episodes) # eval at t=0

	eval_loop.run(num_episodes=eval_episodes)

	counter_ckpt.save(force=True)

	# Close environment.
	environment.close()



