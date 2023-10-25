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

import helpers

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

flags.DEFINE_string('config', 'config', 'Configurations')
flags.DEFINE_string('acme_id', None, 'Experiment identifier to use for Acme.')
flags.DEFINE_string('agent_id', 'd4pg', 'What agent in use.')
flags.DEFINE_string('suite', 'dmc:state', 'Suite')
# flags.DEFINE_string('level', 'trivial', "Task level")
flags.DEFINE_string('task', 'trivial/walker:walk', 'What environment to run')
flags.DEFINE_integer('num_steps', 500_000, 'Number of env steps to run.')
flags.DEFINE_integer('eval_every', 25_000, 'How often to run evaluation.')
flags.DEFINE_integer('evaluation_episodes', 5, 'Evaluation episodes.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('gpu', None, 'GPU')
flags.DEFINE_integer('hp', 1, 'Hyper-parameters.')

flags.DEFINE_bool(
    'run_distributed', False, 'Should an agent be executed in a distributed '
    'way. If False, will run single-threaded.')
flags.DEFINE_integer('num_distributed_actors', 4,
                     'Number of actors to use in the distributed setting.')


def build_experiment_config():
	"""Builds D4PG experiment config which can be executed in different ways."""

	# Create an environment, grab the spec, and use it to create networks.
	suite, task = FLAGS.suite, FLAGS.task
	environment_factory = lambda seed: helpers.make_environment(
		suite=suite,
		task=task
	)

	# Bound of the distributional critic. The reward for control environments is
	# normalized, not for gym locomotion environments hence the different scales.
	vmax_values = {
		'gym': 1000.,
		'dmc': 150.,
	}
	vmax = vmax_values[suite]

	d4pg_hyperparams = d4pg_hp_list[FLAGS.hp - 1]

	replay_ratio = d4pg_hyperparams['replay_ratio'] # eval(FLAGS.replay_ratio)
	d4pg_hyperparams['samples_per_insert'] = int(replay_ratio * d4pg_hyperparams['batch_size'])
	d4pg_hyperparams['num_sgd_steps_per_step'] = int(replay_ratio * (d4pg_hyperparams['batch_size']/d4pg_hyperparams['samples_per_insert']))

	def network_factory(spec) -> d4pg.D4PGNetworks:
		return d4pg.make_networks(
			spec=spec,
			policy_layer_sizes=d4pg_hyperparams['policy_arch'],
			critic_layer_sizes=d4pg_hyperparams['critic_arch'],
			vmin=-vmax, vmax=vmax, num_atoms=d4pg_hyperparams['n_atoms']
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
	path = os.path.join(os.path.dirname(os.getcwd())+f'/{FLAGS.config}.yaml')
	config = yaml.safe_load(open(path))

	if FLAGS.suite == 'gym':
		FLAGS.num_steps = config[FLAGS.suite]['run']['steps']
		FLAGS.eval_every = FLAGS.num_steps//20
		for task in config[FLAGS.suite]['tasks']:
			FLAGS.task = task
			experiment_cfg = build_experiment_config()
			if FLAGS.run_distributed:
				program = experiments.make_distributed_experiment(
					experiment=experiment_cfg,
					num_actors=FLAGS.num_distributed_actors
				)
				lp.launch(program, xm_resources=lp_utils.make_xm_docker_resources(program))
			else:
				experiments.run_experiment(
					experiment=experiment_cfg,
					eval_every=FLAGS.eval_every,
					num_eval_episodes=FLAGS.evaluation_episodes)
				
	elif FLAGS.suite == 'dmc':
		if FLAGS.level in config[FLAGS.suite].keys():
			level_info = config[FLAGS.suite][FLAGS.level]
			FLAGS.num_steps = level_info['run']['steps']
			FLAGS.eval_every = FLAGS.num_steps//20
			for task in level_info['tasks']:
				FLAGS.task = task
				experiment_cfg = build_experiment_config()
				if FLAGS.run_distributed:
					program = experiments.make_distributed_experiment(
						experiment=experiment_cfg,
						num_actors=FLAGS.num_distributed_actors
					)
					lp.launch(program, xm_resources=lp_utils.make_xm_docker_resources(program))
				else:
					experiments.run_experiment(
						experiment=experiment_cfg,
						eval_every=FLAGS.eval_every,
						num_eval_episodes=FLAGS.evaluation_episodes)
		else:
			return
		
	else:
		return


if __name__ == '__main__':
	app.run(main)