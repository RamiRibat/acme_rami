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

	# Create the environment and get its spec.
	environment = experiment.environment_factory(experiment.seed)
	environment_spec = experiment.environment_spec or specs.make_environment_spec(environment)

	# Create the networks and policy.
	networks = experiment.network_factory(environment_spec)
	policy = config.make_policy(
		experiment=experiment,
		networks=networks,
		environment_spec=environment_spec,
		evaluation=False
	)

	# Create the (replay server) and grab its address.
	replay_tables = experiment.builder.make_replay_tables(environment_spec, policy)

	# Disable blocking of inserts by tables' rate limiters, as this function
	# executes learning (sampling from the table) and data generation
	# (inserting into the table) sequentially from the same thread
	# which could result in blocked insert making the algorithm hang.
	replay_tables, rate_limiters_max_diff = _disable_insert_blocking(replay_tables)

	replay_server = reverb.Server(replay_tables, port=None)
	# dfn replay_client: used by dataset(iterator), learner, and adder
	replay_client = reverb.Client(f'localhost:{replay_server.port}')

	# Parent counter allows to (share step counts) between train and eval loops and
	# the learner, so that it is possible to plot for example evaluator's return
	# value as a function of the number of training episodes.
	parent_counter = counting.Counter(time_delta=0.)

	dataset = experiment.builder.make_dataset_iterator(replay_client)
	# We always use prefetch as it provides an iterator with an additional
	# 'ready' method.
	dataset = utils.prefetch(dataset, buffer_size=1) # isn't defined b4?

	# Create actor, adder, and learner for generating, storing, and consuming
	# data respectively. (by Builder)
	# NOTE: These are created in (reverse order) as the actor needs to be given the
	# adder and the learner (as a source of variables).
	learner_key, key = jax.random.split(key)
	learner = experiment.builder.make_learner(
		random_key=learner_key,
		networks=networks,
		dataset=dataset,
		logger_fn=experiment.logger_factory,
		environment_spec=environment_spec,
		replay_client=replay_client,
		counter=counting.Counter(parent_counter, prefix='learner', time_delta=0.)
	)

	train_counter = counting.Counter( parent_counter, prefix='actor', time_delta=0.)

	checkpointer = None
	if experiment.checkpointing is not None:
		checkpointing = experiment.checkpointing
		checkpointer = savers.Checkpointer(
			objects_to_save={'learner': learner, 'counter': parent_counter},
			time_delta_minutes=checkpointing.time_delta_minutes,
			directory=checkpointing.directory,
			add_uid=checkpointing.add_uid,
			max_to_keep=checkpointing.max_to_keep,
			keep_checkpoint_every_n_hours=checkpointing.keep_checkpoint_every_n_hours,
			checkpoint_ttl_seconds=checkpointing.checkpoint_ttl_seconds,
		)

	# Create the evaluation actor and loop.
	eval_actor_key, key = jax.random.split(key)#jax.random.PRNGKey(experiment.seed)
	eval_counter = counting.Counter(parent_counter, prefix='evaluator', time_delta=0.)
	eval_logger = experiment.logger_factory('evaluator', eval_counter.get_steps_key(), 0)
	eval_policy = config.make_policy(
		experiment=experiment,
		networks=networks,
		environment_spec=environment_spec,
		evaluation=True
	)
	eval_actor = experiment.builder.make_actor(
		# random_key=jax.random.PRNGKey(experiment.seed),
		random_key=eval_actor_key,
		policy=eval_policy,
		environment_spec=environment_spec,
		variable_source=learner
	)
	eval_loop = acme.EnvironmentLoop(
		environment=environment,
		actor=eval_actor,
		label='eval_loop',
		counter=eval_counter,
		logger=eval_logger,
		observers=experiment.observers
	)

	if 'actor_steps' not in parent_counter.get_counts().keys():
		parent_counter.get_counts().get(train_counter.get_steps_key(), 0)

	eval_loop.run(num_episodes=num_eval_episodes)

	# close eval_logger
	eval_logger.close()

	# close environment
	environment.close()


def _disable_insert_blocking(
    tables: Sequence[reverb.Table]
) -> Tuple[Sequence[reverb.Table], Sequence[int]]:
	"""Disables blocking of insert operations for a given collection of tables."""
	modified_tables = []
	sample_sizes = []
	for table in tables:
		rate_limiter_info = table.info.rate_limiter_info
		rate_limiter = reverb.rate_limiters.RateLimiter(
			samples_per_insert=rate_limiter_info.samples_per_insert,
			min_size_to_sample=rate_limiter_info.min_size_to_sample,
			min_diff=rate_limiter_info.min_diff,
			max_diff=sys.float_info.max)
		modified_tables.append(table.replace(rate_limiter=rate_limiter))
		# Target the middle of the rate limiter's insert-sample balance window.
		sample_sizes.append(
			max(1, int(
				(rate_limiter_info.max_diff - rate_limiter_info.min_diff) / 2)))
	return modified_tables, sample_sizes
