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

"""Example running MPO on continuous control tasks."""

import os, yaml, json
from absl import flags
from acme import specs
from acme.agents.jax import mpo
from acme.agents.jax.mpo import types as mpo_types
import helpers
from absl import app
from acme.jax import experiments
from acme.utils import lp_utils
import launchpad as lp

import warnings
warnings.filterwarnings('ignore')


hyperparams_1 = {
	'policy_arch': (256, 256),
	'critic_arch': (256, 256),
	'critic_type': mpo.CriticType.NONDISTRIBUTIONAL,
	'batch_size': 256,
	'learning_rate': 3e-4,
	'discount': 0.99,
	'n_step': 5,  # The D4PG agent learns from n-step transitions.
	'n_atoms': 51, # Atoms used by the categorical distributional critic.
	# 'critic_atoms' = jnp.linspace(-150., 150., num_atoms)
	'target_update_period': 100,
	'samples_per_insert': 64.0, # Controls the relative rate of sampled vs inserted items. In this case, items are n-step transitions.
	'sgd_steps_per_learner_step': 1,
	# 'reset_interval': 320000,
	# 'reset_interval': 2560000,
}

hyperparams_2 = {
	'policy_arch': (256, 256),
	'critic_arch': (256, 256),
	'critic_type': mpo.CriticType.CATEGORICAL,
	'batch_size': 256,
	'learning_rate': 3e-4,
	'discount': 0.99,
	'n_step': 5,  # The D4PG agent learns from n-step transitions.
	'n_atoms': 51, # Atoms used by the categorical distributional critic.
	# 'critic_atoms' = jnp.linspace(-150., 150., num_atoms)
	'sigma': 0.2, # exploration noise
	'target_update_period': 100,
	'samples_per_insert': 64.0, # Controls the relative rate of sampled vs inserted items. In this case, items are n-step transitions.
	'sgd_steps_per_learner_step': 1,
	# 'reset_interval': 320000,
	# 'reset_interval': 2560000,
}

mpo_hp_list = [
	hyperparams_1,
	hyperparams_2
]


FLAGS = flags.FLAGS

flags.DEFINE_string('config', 'config', 'Suite')
flags.DEFINE_string('acme_id', None, 'Experiment identifier to use for Acme.')
flags.DEFINE_string('agent_id', 'mpo', 'What agent in use.')
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
	"""Builds MPO experiment config which can be executed in different ways."""

	# Create an environment, grab the spec, and use it to create networks.
	suite, task = FLAGS.suite, FLAGS.task
	environment_factory = lambda seed: helpers.make_environment(suite, task)

	mpo_hyperparams = mpo_hp_list[FLAGS.hp - 1]

	# Bound of the distributional critic. The reward for control environments is
	# normalized, not for gym locomotion environments hence the different scales.
	# vmax_values = {
	# 	'gym': 1000.,
	# 	'control': 150.,
	# 	'dmc': 150.,
	# }
	# vmax = vmax_values[suite]

	def network_factory(spec: specs.EnvironmentSpec) -> mpo.MPONetworks:
		# TODO(rami): replace w/ mpo.make_networks
		return mpo.make_control_networks(
			environment_spec=spec,
			policy_layer_sizes=mpo_hyperparams['policy_arch'],
			critic_layer_sizes=mpo_hyperparams['critic_arch'],
			policy_init_scale=0.5,
			critic_type=mpo_hyperparams['critic_type']
		)

	# Configure and construct the agent builder.
	mpo_config = mpo.MPOConfig(
		critic_type=mpo_hyperparams['critic_type'],
		policy_loss_config=mpo_types.GaussianPolicyLossConfig(epsilon_mean=0.01),
		samples_per_insert=mpo_hyperparams['samples_per_insert'],
		learning_rate=mpo_hyperparams['learning_rate'],
		experience_type=mpo_types.FromTransitions(n_step=4)
	)
	mpo_builder = mpo.MPOBuilder(mpo_config)

	return experiments.ExperimentConfig(
		builder=mpo_builder,
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
