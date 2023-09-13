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
from absl import flags
from acme.agents.jax import d4pg
import helpers
from absl import app
from acme.jax import experiments
from acme.utils import lp_utils
import launchpad as lp

from acme.utils import observers as observers_lib

DMC_TASKS = {
  "cartpole:swingup": 'trivial',
  "walker:walk": 'trivial',
  "hopper:stand": 'easy',
  "swimmer:swimmer6": 'easy',
  "cheetah:run": 'medium',
  "walker:run": 'medium'
}

DMC_STEPS ={
  'trivial': 500_000,
  'easy': 1_000_000,
  'medium': 2_000_000
}


FLAGS = flags.FLAGS

# flags.DEFINE_bool(
#     'run_distributed', True, 'Should an agent be executed in a distributed '
#     'way. If False, will run single-threaded.')
flags.DEFINE_bool(
    'run_distributed', False, 'Should an agent be executed in a distributed '
    'way. If False, will run single-threaded.')
# flags.DEFINE_string('env_name', 'gym:HalfCheetah-v2', 'What environment to run')
flags.DEFINE_string('env_name', 'control:walker:walk', 'What environment to run')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('num_steps', 500_000, 'Number of env steps to run.')
flags.DEFINE_integer('eval_every', 20, 'How often to run evaluation.')
flags.DEFINE_integer('evaluation_episodes', 5, 'Evaluation episodes.')

flags.DEFINE_string('agent', 'd4pg', 'What agent in use.')

flags.DEFINE_string('acme_id', None, 'Experiment identifier to use for Acme.')

# FLAGS.append_flags_into_file()


def build_experiment_config():
  """Builds D4PG experiment config which can be executed in different ways."""

  # Create an environment, grab the spec, and use it to create networks.
  suite, task = FLAGS.env_name.split(':', 1)

  dmc_level = DMC_TASKS[task]
  FLAGS.num_steps = DMC_STEPS[dmc_level]
  FLAGS.eval_every = FLAGS.num_steps//20

  # Bound of the distributional critic. The reward for control environments is
  # normalized, not for gym locomotion environments hence the different scales.
  vmax_values = {
      'gym': 1000.,
      'control': 150.,
  }
  vmax = vmax_values[suite]

  def network_factory(spec) -> d4pg.D4PGNetworks:
    return d4pg.make_networks(
        spec,
        policy_layer_sizes=(256, 256, 256),
        critic_layer_sizes=(256, 256, 256),
        vmin=-vmax,
        vmax=vmax,
    )

  # Configure the agent.
  d4pg_config = d4pg.D4PGConfig(learning_rate=3e-4, sigma=0.2)

  return experiments.ExperimentConfig(
      builder=d4pg.D4PGBuilder(d4pg_config),
      environment_factory=lambda seed: helpers.make_environment(suite, task),
      network_factory=network_factory,
      seed=FLAGS.seed,
      max_num_actor_steps=FLAGS.num_steps)


def main(_):
  config = build_experiment_config()
  if FLAGS.run_distributed:
    program = experiments.make_distributed_experiment(
        experiment=config, num_actors=4)
    lp.launch(program, xm_resources=lp_utils.make_xm_docker_resources(program))
  else:
    experiments.run_experiment(
        experiment=config,
        eval_every=FLAGS.eval_every,
        num_eval_episodes=FLAGS.evaluation_episodes)


if __name__ == '__main__':
  app.run(main)