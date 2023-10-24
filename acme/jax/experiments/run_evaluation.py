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

import sys
import time
from typing import Optional, Sequence, Tuple

import acme
from acme import core
from acme import specs
from acme import types
from acme.jax import utils
from acme.jax.experiments import config
from acme.tf import savers
from acme.utils import counting
import dm_env
import jax
import reverb


def run_evaluation(
    experiment: config.ExperimentConfig,
	# eval_every: int = 100,
	num_eval_episodes: int = 1
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

	key = jax.random.PRNGKey(experiment.seed)


	"""Environment."""
	# Create the environment and get its spec.
	environment = experiment.environment_factory(experiment.seed)
	environment_spec = experiment.environment_spec or specs.make_environment_spec(environment)


	"""Network/Policy."""
	# Create networks.
	networks = experiment.network_factory(environment_spec)
	# Create evaluation policy.
	eval_policy = config.make_policy(
		experiment=experiment,
		networks=networks,
		environment_spec=environment_spec,
		evaluation=True
	)


	"""Parent Counter"""
	# Parent counter allows to (share step counts) between train and eval loops and
	# the learner, so that it is possible to plot for example evaluator's return
	# value as a function of the number of training episodes.
	counter = counting.Counter(time_delta=0.)
	
	if experiment.checkpointing is not None:
		checkpointing = experiment.checkpointing
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
	key, learner_key = jax.random.split(key)
	learner = experiment.builder.make_learner(
		random_key=learner_key,
		networks=networks,
		iterator=None,
		logger_fn=experiment.logger_factory,
		environment_spec=environment_spec,
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


	"""Actor (evaluation)."""
	key, eval_actor_key = jax.random.split(key)
	eval_actor = experiment.builder.make_actor(
		random_key=eval_actor_key,
		policy=eval_policy,
		environment_spec=environment_spec,
		variable_source=learner,
		# no adder neede
	)


	"""Evaluation loop."""
	if 'actor_steps' not in counter.get_counts().keys():
		# init csv columns for eval_logger(eval_counter(parent_counter <- train_counter))
		# train_counter = counting.Counter(counter, prefix='actor')
		counter.get_counts().get(counter.get_steps_key(), 0)

	# Create evaluation counter/logger (~evaluator(actor)).
	eval_counter = counting.Counter(counter, prefix='evaluator', time_delta=0)
	eval_logger = experiment.logger_factory('evaluator', eval_counter.get_steps_key(), 0)

	# Create the environment loop used for evaluation.
	eval_loop = acme.EnvironmentLoop(
		environment=environment,
		actor=eval_actor,
		label='eval_loop',
		counter=eval_counter,
		logger=eval_logger,
		observers=experiment.observers
	)

	# Run evaluation loop (full episodes).
	eval_loop.run(num_episodes=num_eval_episodes)

	# Close evaluation logger.
	eval_logger.close()

	# Close environment.
	environment.close()



