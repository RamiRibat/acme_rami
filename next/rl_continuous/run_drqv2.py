"""
Author: https://github.com/ethanluoyc
"""

import os, yaml, json
from absl import app, flags


import launchpad as lp

import helpers
from absl import app
from acme.jax import experiments
from acme.utils import lp_utils

from acme.agents.jax import drqv2


import warnings
warnings.filterwarnings('ignore')


# hyperparams_1 = {
# 	# 'policy_arch': (256, 256),
# 	# 'critic_arch': (256, 256),
# 	'batch_size': 256,
# 	'learning_rate': 1e-4,
# 	'discount': 0.99,
# 	'n_step': 3,
# 	'sigma': 0.2, # exploration noise
# 	# 'target_update_period': 100,
# 	'samples_per_insert': 128.0, # Controls the relative rate of sampled vs inserted items. In this case, items are n-step transitions.
# 	'num_sgd_steps_per_step': 1,
# 	# 'reset_interval': 320000,
# 	# 'reset_interval': 2560000,
# 	# 'replay_ratio': 0.125
# }

hyperparams_1 = {
	# 'policy_arch': (256, 256),
	# 'critic_arch': (256, 256),
	'batch_size': 256,
	'learning_rate': 1e-4,
	'discount': 0.99,
	'n_step': 3,
	'sigma': 0.2, # exploration noise
	# 'target_update_period': 100,
	'samples_per_insert': 128.0, # Controls the relative rate of sampled vs inserted items. In this case, items are n-step transitions.
	'num_sgd_steps_per_step': 1,
	# 'reset_interval': 320000,
	# 'reset_interval': 2560000,
	# 'replay_ratio': 0.125
	'env': {
		'num_action_repeat': 2, # og = 2
		'num_stacked_frames': 4,
		'grayscaling': False,
	}
}

drqv2_hp_list = [
	hyperparams_1,
	# hyperparams_2
]


FLAGS = flags.FLAGS

flags.DEFINE_string('config', 'config', 'Suite')
flags.DEFINE_string('acme_id', None, 'Experiment identifier to use for Acme.')
flags.DEFINE_string('agent_id', 'drqv2', 'What agent in use.')
flags.DEFINE_string('suite', 'dmc_pixel', 'Suite')
flags.DEFINE_string('level', 'trivial', "Task level")
flags.DEFINE_string('task', 'walker:walk', 'What environment to run')
flags.DEFINE_integer('num_steps', 250_000, 'Number of actor steps to run.')
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
	"""Builds D4PG experiment config which can be executed in different ways."""

	# Create an environment, grab the spec, and use it to create networks.
	suite, task = FLAGS.suite, FLAGS.task

	drqv2_hyperparams = drqv2_hp_list[FLAGS.hp - 1]
	print('drqv2_hyperparams: ', drqv2_hyperparams)

	environment_factory = lambda seed: helpers.make_environment(
		suite=suite,
		task=task,
		from_pixels=True,
		# num_action_repeat=1, # og = 2
		# num_stacked_frames=4, # og = 3
		camera_id=0,
		scale_dims=(64, 64), # og = (84, 84)
		flatten_frame_stack=True, # og = True
		to_float=False, # atari = True
		# grayscaling=False
		**drqv2_hyperparams['env']
	)

	# del drqv2_hyperparams['env']

	# replay_ratio = drqv2_hyperparams['replay_ratio'] # eval(FLAGS.replay_ratio)
	# drqv2_hyperparams['samples_per_insert'] = int(replay_ratio * drqv2_hyperparams['batch_size'])
	# drqv2_hyperparams['num_sgd_steps_per_step'] = int(replay_ratio * (drqv2_hyperparams['batch_size']/drqv2_hyperparams['samples_per_insert']))

	def network_factory(spec) -> drqv2.DrQV2Networks:
		return drqv2.make_networks(
			spec=spec,
			# latent_size=drqv2_hyperparams['policy_arch'],
			# hidden_size=drqv2_hyperparams['critic_arch'],
		)
	
	# Configure the agent.
	drqv2_config = drqv2.DrQV2Config(**drqv2_hyperparams)
	drqv2_builder = drqv2.DrQV2Builder(config=drqv2_config)

	return experiments.ExperimentConfig(
		builder=drqv2_builder,
		environment_factory=environment_factory,
		network_factory=network_factory,
		max_num_actor_steps=FLAGS.num_steps,
		seed=FLAGS.seed,
    )



def main(_):
	path = os.path.join(os.path.dirname(os.getcwd())+f'/{FLAGS.config}.yaml')
	config = yaml.safe_load(open(path))

	if FLAGS.suite == 'gym':
		FLAGS.num_steps = config[FLAGS.suite]['run']['steps']
		FLAGS.eval_every = FLAGS.num_steps//10
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
				
	elif FLAGS.suite in ('dmc', 'dmc_pixel'):
		if FLAGS.level in config[FLAGS.suite].keys():
			level_info = config[FLAGS.suite][FLAGS.level]
			# FLAGS.num_steps = level_info['run']['steps']
			FLAGS.eval_every = FLAGS.num_steps//10
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

