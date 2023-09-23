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

"""Example running PPO on continuous control tasks."""

import os, yaml, json
from absl import flags
from acme.agents.jax import ppo
import helpers
from absl import app
from acme.jax import experiments
from acme.utils import lp_utils
import launchpad as lp

import warnings
warnings.filterwarnings('ignore')


ppo_hyperparams_v1 = {
	# replay = n_epochs x mini_batches = 16
	'num_epochs': 2,
	'num_minibatches': 8,
	# full batch = batch_size x unroll_length = 2,048
	'batch_size': 256,
	'unroll_length': 8,
	'learning_rate': 3e-4,
	'gae_lambda': 0.95,
	'discount': 0.99,
	'hidden_layer_sizes': (256, 256),
	'normalize_advantage': True,
	'normalize_value': True,
	'reset_interval': 0,
}

ppo_hyperparams_v2 = {
	# replay = n_epochs x mini_batches = 64 (rr=1/128)
	'num_epochs': 4,
	'num_minibatches': 16,
	# full batch = batch_size x unroll_length = 8,192
	'batch_size': 16,
	'unroll_length': 512,
	'learning_rate': 3e-4,
	'gae_lambda': 0.95,
	'discount': 0.99,
	'hidden_layer_sizes': (256, 256),
	'normalize_advantage': True,
	'normalize_value': True,
	'reset_interval': 0,
}

ppo_hyperparams_v3 = {
	# replay = n_epochs x mini_batches = 128 (rr=1/64)
	'num_epochs': 8,
	'num_minibatches': 16,
	# full batch = batch_size x unroll_length = 8,192
	'batch_size': 16,
	'unroll_length': 512,
	'learning_rate': 3e-4,
	'gae_lambda': 0.95,
	'discount': 0.99,
	'hidden_layer_sizes': (256, 256),
	'normalize_advantage': True,
	'normalize_value': True,
	'reset_interval': 0,
}

ppo_hyperparams_v4 = {
	# replay = n_epochs x mini_batches = 256 (rr=1/32)
	'num_epochs': 16,
	'num_minibatches': 16,
	# full batch = batch_size x unroll_length = 8,192
	'batch_size': 16,
	'unroll_length': 512,
	'learning_rate': 3e-4,
	'gae_lambda': 0.95,
	'discount': 0.99,
	'hidden_layer_sizes': (256, 256),
	'normalize_advantage': True,
	'normalize_value': True,
	'reset_interval': 0,
}



ppo_hyperparams_list = [
    ppo_hyperparams_v1,
    ppo_hyperparams_v2,
    ppo_hyperparams_v3,
    ppo_hyperparams_v4,
]

FLAGS = flags.FLAGS

flags.DEFINE_string('acme_id', None, 'Experiment identifier to use for Acme.')
flags.DEFINE_string('agent_id', 'ppo', 'What agent in use.')
flags.DEFINE_string('suite', 'control', 'Suite')
flags.DEFINE_string('level', 'trivial', "Task level")
flags.DEFINE_string('task', 'walker:walk', 'What environment to run')
flags.DEFINE_integer('num_steps', 500_000, 'Number of env steps to run.')
flags.DEFINE_integer('eval_every', 25_000, 'How often to run evaluation.')
flags.DEFINE_integer('evaluation_episodes', 5, 'Evaluation episodes.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('gpu', None, 'Random seed.')

flags.DEFINE_bool(
    'run_distributed', True, 'Should an agent be executed in a distributed '
    'way. If False, will run single-threaded.')
flags.DEFINE_integer('num_distributed_actors', 4,
                     'Number of actors to use in the distributed setting.')


def build_experiment_config():
	"""Builds PPO experiment config which can be executed in different ways."""
	# Create an environment, grab the spec, and use it to create networks.
	suite, task = FLAGS.suite, FLAGS.task
	environment_factory = lambda seed: helpers.make_environment(suite, task)

	ppo_hyperparams = ppo_hyperparams_list[FLAGS.hp - 1]

	def network_factory(spec) -> ppo.PPONetworks:
		return ppo.make_networks(
			spec=spec,
			hidden_layer_sizes=ppo_hyperparams['hidden_layer_sizes']
		)

	ppo_config = ppo.PPOConfig(
		**ppo_hyperparams,
		obs_normalization_fns_factory=ppo.build_mean_std_normalizer)
	ppo_builder = ppo.PPOBuilder(ppo_config)

	return experiments.ExperimentConfig(
		builder=ppo_builder,
		environment_factory=environment_factory,
		network_factory=network_factory,
		seed=FLAGS.seed,
		max_num_actor_steps=FLAGS.num_steps
	)


def main(_):
	path = os.path.join(os.path.dirname(os.getcwd())+'/config.yaml')
	config = yaml.safe_load(open(path))

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


if __name__ == '__main__':
	app.run(main)
