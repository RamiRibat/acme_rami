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

"""Example running SAC on continuous control tasks."""

import os, yaml, json
from absl import flags
from acme import specs
from acme.agents.jax import normalization
from acme.agents.jax import sac
from acme.agents.jax.sac import builder
import helpers
from absl import app
from acme.jax import experiments
from acme.utils import lp_utils
import launchpad as lp

from acme.utils import observers as observers_lib

import warnings
warnings.filterwarnings('ignore')


hyperparams_1 = {
	'hidden_layer_sizes': (256, 256),
	'discount': 0.99,
	'tau': 0.005,
	'n_step': 1,  # ??
	'batch_size': 256,
	'learning_rate': 3e-4,
	'prefetch_size': 4,
	'samples_per_insert': 256,
	'num_sgd_steps_per_step': 1,
	'reset_interval': 0, #2560000,
	'replay_ratio': 1
}

sac_hp_list = [
	hyperparams_1,
	# hyperparams_2
]

FLAGS = flags.FLAGS

flags.DEFINE_string('config', 'config', 'Suite')
flags.DEFINE_string('acme_id', None, 'Experiment identifier to use for Acme.')
flags.DEFINE_string('agent_id', 'sac', 'What agent in use.')
flags.DEFINE_string('suite', 'control', 'Suite')
flags.DEFINE_string('level', 'trivial', "Task level")
flags.DEFINE_string('task', 'walker:walk', 'What environment to run')
flags.DEFINE_integer('num_steps', 500_000, 'Number of env steps to run.')
flags.DEFINE_integer('eval_every', 25_000, 'How often to run evaluation.')
flags.DEFINE_integer('evaluation_episodes', 5, 'Evaluation episodes.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('gpu', None, 'Random seed.')
flags.DEFINE_integer('hp', 1, 'Hyper-parameters.')

flags.DEFINE_bool(
    'run_distributed', False, 'Should an agent be executed in a distributed '
    'way. If False, will run single-threaded.')
flags.DEFINE_integer('num_distributed_actors', 4,
                     'Number of actors to use in the distributed setting.')



def build_experiment_config():
	"""Builds SAC experiment config which can be executed in different ways."""
	# Create an environment, grab the spec, and use it to create networks.

	# Create an environment, grab the spec, and use it to create networks.
	# suite, task = FLAGS.env_name.split(':', 1)
	suite, task = FLAGS.suite, FLAGS.task
	environment_factory = lambda seed: helpers.make_environment(suite, task)

	sac_hyperparams = sac_hp_list[FLAGS.hp - 1]

	replay_ratio = sac_hyperparams['replay_ratio'] # eval(FLAGS.replay_ratio)
	sac_hyperparams['samples_per_insert'] = int(replay_ratio * sac_hyperparams['batch_size'])
	sac_hyperparams['num_sgd_steps_per_step'] = int(replay_ratio * (sac_hyperparams['batch_size']/sac_hyperparams['samples_per_insert']))

	def network_factory(spec) -> sac.SACNetworks:
		return sac.make_networks(
			spec=spec,
			hidden_layer_sizes=sac_hyperparams['hidden_layer_sizes']
		)

	# Construct the agent.
	sac_config = sac.SACConfig(
		**sac_hyperparams,
		# target_entropy=sac.target_entropy_from_env_spec(environment_spec), # TODO(rami): Add if
		input_normalization=normalization.NormalizationConfig()
		)
	sac_builder = sac.SACBuilder(config=sac_config)

	return experiments.ExperimentConfig(
		builder=sac_builder,
		environment_factory=environment_factory,
		network_factory=network_factory,
		seed=FLAGS.seed,
		max_num_actor_steps=FLAGS.num_steps)


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
	elif FLAGS.suite in ('control', 'dmc'):
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
