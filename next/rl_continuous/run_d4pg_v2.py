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

"""Example running D4PG on continuous control tasks."""

import os, yaml, json
from absl import flags

import launchpad as lp

# import helpers
import helpers_v2 as helpers

from absl import app
from acme.jax import experiments
from acme.utils import lp_utils
from acme.utils import observers as observers_lib

from acme.agents.jax import d4pg

import warnings
warnings.filterwarnings('ignore')


hyperparams_1 = {
	'policy_arch': (256, 256),
	'critic_arch': (256, 256),
	'batch_size': 256,
	'learning_rate': 3e-4,
	'discount': 0.99,
	'n_step': 5,  # The D4PG agent learns from n-step transitions.
	'n_atoms': 51, # Atoms used by the categorical distributional critic.
	# 'critic_atoms' = jnp.linspace(-150., 150., num_atoms)
	'sigma': 0.2, # exploration noise
	'target_update_period': 100,
	'samples_per_insert': 32.0, # Controls the relative rate of sampled vs inserted items. In this case, items are n-step transitions.
	'num_sgd_steps_per_step': 1,
	'reset_interval': 0,
	# 'reset_interval': 320000,
	# 'reset_interval': 2560000,
	'replay_ratio': 0.125
}

d4pg_hp_list = [
	hyperparams_1,
	# hyperparams_2
]

FLAGS = flags.FLAGS

flags.DEFINE_string('suites', '_suites', 'Suite Configurations')
flags.DEFINE_string('acme_id', None, 'Experiment identifier.')
flags.DEFINE_string('agent_id', 'd4pg', 'What agent in use.')
flags.DEFINE_string('suite', 'dmc:state', 'Suite name {main:sub}.')
flags.DEFINE_string('task', 'trivial/walker:walk', 'What task to run')
flags.DEFINE_integer('num_steps', 500_000, 'Number of actor steps to run.')
flags.DEFINE_integer('eval_every', 50_000, 'How often to run evaluation.')
flags.DEFINE_integer('eval_episodes', 5, 'Evaluation episodes.')
flags.DEFINE_integer('seed', 0, 'Random seed.')

flags.DEFINE_bool(
	'run_distributed', False, 'Should an agent be executed in a distributed '
	'way. If False, will run single-threaded.')
flags.DEFINE_integer(
	'num_distributed_actors', 4,
	'Number of actors to use in the distributed setting.')


def build_experiment_config(suite, task, env_config):
	"""Builds D4PG experiment config which can be executed in different ways."""

	# Create an environment, grab the spec, and use it to create networks.
	environment_factory = lambda seed: helpers.make_environment(
		suite=suite,
		task=task,
		**env_config
	)

	# # Bound of the distributional critic. The reward for control environments is
	# # normalized, not for gym locomotion environments hence the different scales.
	# vmax_values = {
	# 	'gym': 1000.,
	# 	'dmc': 150.,
	# }
	# vmax = vmax_values[suite]

	d4pg_hyperparams = hyperparams_1

	# replay_ratio = d4pg_hyperparams['replay_ratio'] # eval(FLAGS.replay_ratio)
	# d4pg_hyperparams['samples_per_insert'] = int(replay_ratio * d4pg_hyperparams['batch_size'])
	# d4pg_hyperparams['num_sgd_steps_per_step'] = int(replay_ratio * (d4pg_hyperparams['batch_size']/d4pg_hyperparams['samples_per_insert']))

	def network_factory(spec) -> d4pg.D4PGNetworks:
		return d4pg.make_networks(
			spec=spec,
			policy_layer_sizes=(256, 256),
			critic_layer_sizes=(256, 256),
			vmin=-150.0, vmax=150, num_atoms=51
		)

	# Configure the agent.
	d4pg_config = d4pg.D4PGConfig(**d4pg_hyperparams)
	d4pg_builder = d4pg.D4PGBuilder(config=d4pg_config)

	return experiments.ExperimentConfig(
		builder=d4pg_builder,
		environment_factory=environment_factory,
		network_factory=network_factory,
		seed=FLAGS.seed,
		max_num_actor_steps=FLAGS.num_steps
    )


def main(_):
	path = os.path.join(os.path.dirname(os.getcwd())+f'/{FLAGS.suites}.yaml')
	SUITES = yaml.safe_load(open(path))

	suite = FLAGS.suite.split(':') # recieve a single domain_main:domain_sub at a time
				
	if 'dmc' in suite:
		suite_main, suite_sub = suite
		env_config = SUITES[suite_main]['env']
		run_config = SUITES[suite_main]['run']
		env_config.update(SUITES[suite_main][suite_sub]['env'])
		run_config.update(SUITES[suite_main][suite_sub]['run'])

		if FLAGS.task: # run a single task if not None
			level, task = FLAGS.task.split('/')
			if level in SUITES[suite_main][suite_sub].keys():
				experiment_cfg = build_experiment_config(suite, task, env_config)
				run_config.update(SUITES[suite_main][suite_sub][level]['run']) # only run variations
				FLAGS.num_steps = run_config['steps']
				eval_every = run_config['steps']//run_config['eval_frequency']
				evaluation_episodes = run_config['eval_episodes']
				if FLAGS.run_distributed:
					program = experiments.make_distributed_experiment(
						experiment=experiment_cfg,
						num_actors=FLAGS.num_distributed_actors
					)
					lp.launch(program, xm_resources=lp_utils.make_xm_docker_resources(program))
				else:
					experiments.run_experiment(
						experiment=experiment_cfg,
						eval_every=eval_every,
						num_eval_episodes=evaluation_episodes)
					
		else: # run several tasks ~ levels
			for level in SUITES[suite_main][suite_sub].keys():
				experiment_cfg = build_experiment_config(suite, task, env_config)
				run_config.update(SUITES[suite_main][suite_sub][level]['run']) # only run variations
				FLAGS.num_steps = run_config['steps']
				eval_every = run_config['steps']//run_config['eval_frequency']
				evaluation_episodes = run_config['eval_episodes']
				for task in SUITES[suite_main][suite_sub][level]['tasks']:
					FLAGS.task = f'{level}/{task}'
					if FLAGS.run_distributed:
						program = experiments.make_distributed_experiment(
							experiment=experiment_cfg,
							num_actors=FLAGS.num_distributed_actors
						)
						lp.launch(program, xm_resources=lp_utils.make_xm_docker_resources(program))
					else:
						experiments.run_experiment(
							experiment=experiment_cfg,
							eval_every=eval_every,
							num_eval_episodes=evaluation_episodes)
						
	else:
		return


if __name__ == '__main__':
	app.run(main)