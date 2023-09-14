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
from acme.agents.jax import d4pg
import helpers
from absl import app
from acme.jax import experiments
from acme.utils import lp_utils
import launchpad as lp

from acme.utils import observers as observers_lib

import warnings
warnings.filterwarnings('ignore')

import jax

d4pg_hyperparams = {
  'batch_size': 256,
  'learning_rate': 3e-4,
  'discount': 0.99,
  'n_step': 5,  # The D4PG agent learns from n-step transitions.
  'sigma': 0.2,
  'target_update_period': 100,
  'samples_per_insert': 32.0, # Controls the relative rate of sampled vs inserted items. In this case, items are n-step transitions.
  'n_atoms': 51, # Atoms used by the categorical distributional critic.
  'policy_arch': (256, 256, 256),
  'critic_arch': (256, 256, 256),
  # 'critic_atoms' = jnp.linspace(-150., 150., num_atoms)
}


FLAGS = flags.FLAGS


flags.DEFINE_bool(
    'run_distributed', False, 'Should an agent be executed in a distributed '
    'way. If False, will run single-threaded.')

flags.DEFINE_string('acme_id', None, 'Experiment identifier to use for Acme.')
flags.DEFINE_string('agent', 'd4pg', 'What agent in use.')
flags.DEFINE_string('suite', 'control', 'Suite')
flags.DEFINE_string('level', 'trivial', "Task level")
flags.DEFINE_string('task', 'walker:walk', 'What environment to run')
flags.DEFINE_integer('num_steps', 500_000, 'Number of env steps to run.')
flags.DEFINE_integer('eval_every', 20, 'How often to run evaluation.')
flags.DEFINE_integer('evaluation_episodes', 5, 'Evaluation episodes.')
# flags.DEFINE_multi_integer('seed', [0], 'Random seed.')
flags.DEFINE_string('seeds', '0', 'Random seed(s).')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('gpu', None, 'Random seed.')






def build_experiment_config():
  """Builds D4PG experiment config which can be executed in different ways."""

  # Create an environment, grab the spec, and use it to create networks.
  # suite, task = FLAGS.env_name.split(':', 1)
  suite, level, task = FLAGS.suite, FLAGS.level, FLAGS.task

  # Bound of the distributional critic. The reward for control environments is
  # normalized, not for gym locomotion environments hence the different scales.
  vmax_values = {
      'gym': 1000.,
      'control': 150.,
  }
  vmax = vmax_values[suite]

  environment_factory = lambda seed: helpers.make_environment(suite, task)

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
      max_num_actor_steps=FLAGS.num_steps)


def main(_):
  # if FLAGS.gpu:
  #   GPUS = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
  #   os.environ['CUDA_VISIBLE_DEVICES'] = GPUS[FLAGS.gpu]
  #   jax.default_devic = jax.device('gpu')[FLAGS.gpu]
    
  path = os.path.join(os.path.dirname(os.getcwd())+'/config.yaml')
  config = yaml.safe_load(open(path))
  # SEEDS = FLAGS.seeds.split('_')
  # print('tasks:\n', json.dumps(tasks, indent=4, sort_keys=False))
  print('id: ', FLAGS.acme_id)
  print('suite: ', FLAGS.suite)
  # for level_k, level_v in config[FLAGS.suite].items():
  #   print('level: ', level_k)
  #   FLAGS.level = level_k
  level_info = config[FLAGS.suite][FLAGS.level]
  FLAGS.num_steps = level_info['run']['steps']
  FLAGS.eval_every = FLAGS.num_steps//20
  for task in level_info['tasks']:
    # print('task: ', task)
    FLAGS.task = task
    experiment_cfg = build_experiment_config()
    experiments.run_experiment(
        experiment=experiment_cfg,
        eval_every=FLAGS.eval_every,
        num_eval_episodes=FLAGS.evaluation_episodes)


if __name__ == '__main__':
  app.run(main)