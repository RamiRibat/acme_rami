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




import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import pandas as pd
import optax
import rlax
import tensorflow as tf

import reverb
import matplotlib.pyplot as plt


from absl import flags
from acme.agents.jax import d4pg
import helpers
from absl import app
from acme.jax import experiments
from acme.utils import lp_utils
import launchpad as lp

from typing import Optional
import collections
from acme.utils import loggers

"""""""
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import pandas as pd
import optax
import rlax
import tensorflow as tf

import reverb
import matplotlib.pyplot as plt

import acme
from acme import specs
from acme import wrappers
from acme.adders import reverb as reverb_adders
from acme.agents.jax import actors
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax.d4pg import learning
from acme.datasets import reverb as datasets
from acme.jax import utils, variable_utils
from acme.jax import networks as networks_lib
from acme.jax.experiments.run_experiment import _disable_insert_blocking, _LearningActor
from acme.utils import counting
from acme.utils import loggers



FLAGS = flags.FLAGS

flags.DEFINE_bool(
    'run_distributed', False, 'Should an agent be executed in a distributed '
    'way. If False, will run single-threaded.')

# flags.DEFINE_string('env_name', 'gym:HalfCheetah-v2', 'What environment to run')
flags.DEFINE_string('env_name', 'control:walker:walk', 'What environment to run')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('num_steps', 500_000, 'Number of env steps to run.')
flags.DEFINE_integer('eval_every', 10_000, 'How often to run evaluation.')
flags.DEFINE_integer('evaluation_episodes', 5, 'Evaluation episodes.')


d4pg_hyperparams = {
  'batch_size': 256,
  'learning_rate': 1e-4,
  'discount': 0.99,
  'n_step': 5,  # The D4PG agent learns from n-step transitions.
  'exploration_sigma': 0.3,
  'target_update_period': 100,
  'samples_per_insert': 32.0, # Controls the relative rate of sampled vs inserted items. In this case, items are n-step transitions.
  'num_atoms': 51, # Atoms used by the categorical distributional critic.
  # 'critic_atoms' = jnp.linspace(-150., 150., num_atoms)
}


def build_experiment_config(key, d4pg_hyperparams: dict):
  """Builds D4PG experiment config which can be executed in different ways."""

  # Create an environment, grab the spec, and use it to create networks.
  suite, task = FLAGS.env_name.split(':', 1)

  # Bound of the distributional critic. The reward for control environments is
  # normalized, not for gym locomotion environments hence the different scales.
  vmax_values = {
      'gym': 1000.,
      'control': 150.,
  }
  vmax = vmax_values[suite]

  # Create the environment/pull spec.
  environment = helpers.make_environment(suite, task)
  environment_spec = specs.make_environment_spec(environment)

  # Calculate how big the last layer should be based on total # of actions.
  action_spec = environment_spec.actions
  action_size = np.prod(action_spec.shape, dtype=int)

  # Create the deterministic policy network.
  def policy_fn(obs: networks_lib.Observation) -> jnp.ndarray:
    x = obs
    x = networks_lib.LayerNormMLP([256, 256, 256], activate_final=True)(x)
    x = networks_lib.NearZeroInitializedLinear(action_size)(x)
    x = networks_lib.TanhToSpec(action_spec)(x)
    return x

  # Create the distributional critic network.
  def critic_fn(
      obs: networks_lib.Observation,
      action: networks_lib.Action,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    x = jnp.concatenate([obs, action], axis=-1)
    x = networks_lib.LayerNormMLP(layer_sizes=[256, 256, 256, d4pg_hyperparams['num_atoms']])(x)
    critic_atoms = jnp.linspace(-vmax, vmax, d4pg_hyperparams['num_atoms'])
    return x, critic_atoms
  
  policy = hk.without_apply_rng(hk.transform(policy_fn))
  critic = hk.without_apply_rng(hk.transform(critic_fn))

  # Create dummy observations and actions to create network parameters.
  dummy_action = utils.zeros_like(environment_spec.actions)
  dummy_obs = utils.zeros_like(environment_spec.observations)

  # Prebind dummy observations and actions so they are not needed in the learner.
  policy_network = networks_lib.FeedForwardNetwork(
      init=lambda rng: policy.init(rng, dummy_obs),
      apply=policy.apply)
  critic_network = networks_lib.FeedForwardNetwork(
      init=lambda rng: critic.init(rng, dummy_obs, dummy_action),
      apply=critic.apply)
  
  def exploration_policy(
      params: networks_lib.Params,
      key: networks_lib.PRNGKey,
      observation: networks_lib.Observation,
    ) -> networks_lib.Action:
    sigma = d4pg_hyperparams['exploration_sigma']
    action = policy_network.apply(params, observation)
    if sigma:
      action = rlax.add_gaussian_noise(key, action, sigma)
    return action

  def network_factory(spec) -> d4pg.D4PGNetworks:
    return d4pg.make_networks(
        spec,
        policy_layer_sizes=(256, 256, 256),
        critic_layer_sizes=(256, 256, 256),
        vmin=-vmax,
        vmax=vmax,
    )
  
  # ++ Create the logger.
  logger_dict = collections.defaultdict(loggers.InMemoryLogger)
  def logger_factory(
      name: str,
      steps_key: Optional[str] = None,
      task_id: Optional[int] = None,
    ) -> loggers.Logger:
    del steps_key, task_id
    return logger_dict[name] # name: 'evaluator', 'actor', 'learner'

  # Configure the agent.
  d4pg_config = d4pg.D4PGConfig(learning_rate=3e-4, sigma=0.2)
  d4pg_builder = d4pg.D4PGBuilder(d4pg_config)

  return experiments.ExperimentConfig(
      builder=d4pg_builder,
      environment_factory=environment,
      network_factory=network_factory,
      logger_factory=logger_factory,
      seed=FLAGS.seed,
      max_num_actor_steps=FLAGS.num_steps)


def main(_):
  key = jax.random.PRNGKey(FLAGS.seed)
  # config = build_experiment_config(key, d4pg_hyperparams)

  # Create an environment, grab the spec, and use it to create networks.
  suite, task = FLAGS.env_name.split(':', 1)

  # Bound of the distributional critic. The reward for control environments is
  # normalized, not for gym locomotion environments hence the different scales.
  vmax_values = {
      'gym': 1000.,
      'control': 150.,
  }
  vmax = vmax_values[suite]



  """Load the environment."""
  # Create the environment/pull spec.
  environment = helpers.make_environment(suite, task)
  environment_spec = specs.make_environment_spec(environment)



  """Create the Haiku networks."""
  # Calculate how big the last layer should be based on total # of actions.
  action_spec = environment_spec.actions
  action_size = np.prod(action_spec.shape, dtype=int)

  # Create the deterministic policy network.
  def policy_fn(obs: networks_lib.Observation) -> jnp.ndarray:
    x = obs
    x = networks_lib.LayerNormMLP([256, 256, 256], activate_final=True)(x)
    x = networks_lib.NearZeroInitializedLinear(action_size)(x)
    x = networks_lib.TanhToSpec(action_spec)(x)
    return x

  # Create the distributional critic network.
  def critic_fn(
      obs: networks_lib.Observation,
      action: networks_lib.Action,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    x = jnp.concatenate([obs, action], axis=-1)
    x = networks_lib.LayerNormMLP(layer_sizes=[256, 256, 256, d4pg_hyperparams['num_atoms']])(x)
    critic_atoms = jnp.linspace(-vmax, vmax, d4pg_hyperparams['num_atoms'])
    return x, critic_atoms
  
  policy = hk.without_apply_rng(hk.transform(policy_fn))
  critic = hk.without_apply_rng(hk.transform(critic_fn))

  # Create dummy observations and actions to create network parameters.
  dummy_action = utils.zeros_like(environment_spec.actions)
  dummy_obs = utils.zeros_like(environment_spec.observations)

  # Prebind dummy observations and actions so they are not needed in the learner.
  policy_network = networks_lib.FeedForwardNetwork(
      init=lambda rng: policy.init(rng, dummy_obs),
      apply=policy.apply)
  critic_network = networks_lib.FeedForwardNetwork(
      init=lambda rng: critic.init(rng, dummy_obs, dummy_action),
      apply=critic.apply)
  
  def exploration_policy(
      params: networks_lib.Params,
      key: networks_lib.PRNGKey,
      observation: networks_lib.Observation,
    ) -> networks_lib.Action:
    sigma = d4pg_hyperparams['exploration_sigma']
    action = policy_network.apply(params, observation)
    if sigma:
      action = rlax.add_gaussian_noise(key, action, sigma)
    return action
  


  """Create a D4PG agent components."""

  ## Create a central counter. ##
  # This is the parent counter to which all other component counters will synchronize
  # their counts (of their corresponding steps, walltimes, etc).
  parent_counter = counting.Counter(time_delta=0.)


  ## Create the replay table. ##
  # Manages the data flow by limiting the sample and insert calls.
  rate_limiter = reverb.rate_limiters.SampleToInsertRatio(
      min_size_to_sample=1000,
      samples_per_insert=d4pg_hyperparams['samples_per_insert'],
      error_buffer=2*d4pg_hyperparams['batch_size'])

  # Create a replay table to store previous experience.
  replay_tables = [
      reverb.Table(
          name='priority_table',
          sampler=reverb.selectors.Uniform(),
          remover=reverb.selectors.Fifo(),
          max_size=1_000_000,
          rate_limiter=rate_limiter,
          signature=reverb_adders.NStepTransitionAdder.signature(
              environment_spec))
    ]
  

  ## (Single Process Extra) ##
  # NOTE: This is the first of three code cells that are specific to
  # single-process execution. (This is done for you when you use an agent
  # `Builder` and `run_experiment`.) Everything else is logic shared between the
  # two.
  replay_tables, rate_limiters_max_diff = _disable_insert_blocking(replay_tables)

  # Create replay server/client.
  replay_server = reverb.Server(replay_tables, port=None)
  replay_client = reverb.Client(f'localhost:{replay_server.port}')


  ## Create the learner's dataset iterator. ##
  # Pull data from the Reverb server into a TF dataset the agent can consume.
  dataset = datasets.make_reverb_dataset(
      table='priority_table',
      server_address=replay_client.server_address,
      batch_size=d4pg_hyperparams['batch_size'],
    )


  # # We use multi_device_put here in case this code is run on a machine with
  # # multiple accelerator devices, but this works fine with single-device learners
  # # as long as their step functions are pmapped.
  # dataset = utils.multi_device_put(dataset.as_numpy_iterator(), jax.devices())

  ## (Single Process Extra) ##
  # NOTE: This is the second of three code cells that are specific to
  # single-process execution. (This is done for you when you use an agent
  # `Builder` and `run_experiment`.) Everything else is logic shared between the
  # two.
  dataset = utils.prefetch(dataset, buffer_size=1)


  # Create actor, adder, and learner for generating, storing, and consuming
  # data respectively.
  # NOTE: These are created in reverse order as the actor needs to be given the
  # adder and the learner (as a source of variables).

  ## Create the learner. ##
  key, learner_key = jax.random.split(key)

  # The learner updates the parameters (and initializes them).
  learner = learning.D4PGLearner(
      policy_network=policy_network,
      critic_network=critic_network,
      random_key=learner_key,
      policy_optimizer=optax.adam(d4pg_hyperparams['learning_rate']),
      critic_optimizer=optax.adam(d4pg_hyperparams['learning_rate']),
      discount=d4pg_hyperparams['discount'],
      target_update_period=d4pg_hyperparams['target_update_period'],
      iterator=dataset,
      # A simple counter object that can periodically sync with a parent counter.
      counter=counting.Counter(parent_counter, prefix='learner', time_delta=0.),
    )
  

  ## Create the adder. ##
  # Handles preprocessing of data and insertion into replay tables.
  adder = reverb_adders.NStepTransitionAdder(
      priority_fns={'priority_table': None},
      client=replay_client,
      n_step=d4pg_hyperparams['n_step'],
      discount=d4pg_hyperparams['discount'])
  

  ## Create the actor. ##
  key, actor_key = jax.random.split(key)

  # A convenience adaptor from FeedForwardPolicy to ActorCore.
  actor_core = actor_core_lib.batched_feed_forward_to_actor_core(
      exploration_policy)

  # A variable client for updating variables from a remote source.
  variable_client = variable_utils.VariableClient(learner, 'policy', device='cpu')
  actor = actors.GenericActor(
      actor=actor_core,
      random_key=actor_key,
      variable_client=variable_client,
      adder=adder,
      backend='cpu')
  
  ## (Single Process Extra) ##
  # NOTE: This is the third of three code cells that are specific to
  # single-process execution. (This is done for you when you use an agent
  # `Builder` and `run_experiment`.) Everything else is logic shared between the
  # two.
  actor = _LearningActor(actor, learner, dataset, replay_tables,
                        rate_limiters_max_diff, checkpointer=None)
  
  # Create logger factory.
  logger_dict = collections.defaultdict(loggers.InMemoryLogger)
  def logger_factory(
      name: str,
      steps_key: Optional[str] = None,
      task_id: Optional[int] = None,
    ) -> loggers.Logger:
    del steps_key, task_id
    return logger_dict[name]
  
  # Create the environment loop used for training.
  train_counter = counting.Counter(
      parent_counter, prefix='actor', time_delta=0.)
  train_logger = logger_factory(
      name='actor',  steps_key=train_counter.get_steps_key(), task_id=0)
  
  # Create train loop
  train_loop = acme.EnvironmentLoop(
      environment=environment,
      actor=actor,
      counter=train_counter,
      logger=train_logger,
      observers=())
  

  """Run the train/eval loop(s)."""
  max_num_actor_steps = (
      FLAGS.num_steps -
      parent_counter.get_counts().get(train_counter.get_steps_key(), 0))
  
  if FLAGS.evaluation_episodes == 0:
    # No evaluation. Just run the training loop.
    train_loop.run(num_steps=max_num_actor_steps)
    return
  

  ## Create the evaluation actor and loop. ##
  eval_counter = counting.Counter(
      parent_counter, prefix='evaluator', time_delta=0.)
  eval_logger = logger_factory(
      name='evaluator', steps_key=eval_counter.get_steps_key(), task_id=0)
  
  eval_policy = config.make_policy(
      experiment=experiment,
      networks=networks,
      environment_spec=environment_spec,
      evaluation=True)
  
  eval_actor = experiment.builder.make_actor(
      random_key=jax.random.PRNGKey(experiment.seed),
      policy=eval_policy,
      environment_spec=environment_spec,
      variable_source=learner)
  
  eval_loop = acme.EnvironmentLoop(
      environment=environment,
      actor=eval_actor,
      counter=eval_counter,
      logger=eval_logger,
      observers=())


  if FLAGS.run_distributed:
    program = experiments.make_distributed_experiment(
        experiment=config, num_actors=2)
    lp.launch(program, xm_resources=lp_utils.make_xm_docker_resources(program))
  else:
    experiments.run_experiment(
        experiment=config,
        eval_every=FLAGS.eval_every,
        num_eval_episodes=FLAGS.evaluation_episodes)


if __name__ == '__main__':
  app.run(main)
