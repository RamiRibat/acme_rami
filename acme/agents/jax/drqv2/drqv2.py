"""
Author: https://github.com/ethanluoyc
"""

"""Learner component for DrQV2."""
import dataclasses
from functools import partial
import time
from typing import Iterator, List, NamedTuple, Optional, Callable

from acme import adders
from acme import core
from acme import specs
from acme import types
from acme.adders import reverb as adders_reverb
from acme.jax import networks as networks_lib
from acme.jax import types as jax_types
from acme.jax import utils
from acme.jax import variable_utils
from acme import datasets
from acme.agents.jax import builders
from acme.utils import counting
from acme.utils import loggers
from acme.agents.jax import actor_core
from acme.agents.jax import actors
from reverb import rate_limiters
import jax
import jax.numpy as jnp
import optax
import reverb

import networks as drq_v2_networks

DataAugmentation = Callable[[jax_types.PRNGKey, types.NestedArray],
                            types.NestedArray]


# From https://github.com/ikostrikov/jax-rl/blob/main/jax_rl/agents/drq/augmentations.py
def random_crop(key: jax_types.PRNGKey, img, padding):
  crop_from = jax.random.randint(key, (2,), 0, 2 * padding + 1)
  crop_from = jnp.concatenate([crop_from, jnp.zeros((1,), dtype=jnp.int32)])
  padded_img = jnp.pad(
      img, ((padding, padding), (padding, padding), (0, 0)), mode="edge")
  return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


def batched_random_crop(key, imgs, padding=4):
  keys = jax.random.split(key, imgs.shape[0])
  return jax.vmap(random_crop, (0, 0, None))(keys, imgs, padding)


@dataclasses.dataclass
class DrQV2Config:
  """Configuration parameters for DrQ."""

  augmentation: DataAugmentation = batched_random_crop

  min_replay_size: int = 2_000
  max_replay_size: int = 1_000_000
  replay_table_name: str = adders_reverb.DEFAULT_PRIORITY_TABLE
  prefetch_size: int = 4

  discount: float = 0.99
  batch_size: int = 256
  n_step: int = 3

  critic_q_soft_update_rate: float = 0.01
  learning_rate: float = 1e-4
  noise_clip: float = 0.3
  sigma: float = 0.2

  samples_per_insert: float = 128.0
  samples_per_insert_tolerance_rate: float = 0.1
  num_sgd_steps_per_step: int = 1


def _soft_update(
    target_params: networks_lib.Params,
    online_params: networks_lib.Params,
    tau: float,
) -> networks_lib.Params:
  """
    Update target network using Polyak-Ruppert Averaging.
    """
  return jax.tree_multimap(lambda t, s: (1 - tau) * t + tau * s, target_params,
                           online_params)


class TrainingState(NamedTuple):
  """Holds training state for the DrQ learner."""

  policy_params: networks_lib.Params
  policy_opt_state: optax.OptState

  encoder_params: networks_lib.Params
  # There is not target encoder parameters in v2.
  encoder_opt_state: optax.OptState

  critic_params: networks_lib.Params
  critic_target_params: networks_lib.Params
  critic_opt_state: optax.OptState

  key: jax_types.PRNGKey
  steps: int


class DrQV2Learner(core.Learner):
  """Learner for DrQ-v2"""

  def __init__(
      self,
      random_key: jax_types.PRNGKey,
      dataset: Iterator[reverb.ReplaySample],
      networks: drq_v2_networks.DrQV2Networks,
      sigma_schedule: optax.Schedule,
      augmentation: DataAugmentation,
      policy_optimizer: optax.GradientTransformation,
      critic_optimizer: optax.GradientTransformation,
      encoder_optimizer: optax.GradientTransformation,
      noise_clip: float = 0.3,
      critic_soft_update_rate: float = 0.005,
      discount: float = 0.99,
      num_sgd_steps_per_step: int = 1,
      counter: Optional[counting.Counter] = None,
      logger: Optional[loggers.Logger] = None,
  ):

    def critic_loss_fn(
        critic_params: networks_lib.Params,
        encoder_params: networks_lib.Params,
        critic_target_params: networks_lib.Params,
        policy_params: networks_lib.Params,
        transitions: types.Transition,
        key: jax_types.PRNGKey,
        sigma: jnp.ndarray,
    ):
      next_encoded = networks.encoder_network.apply(
          encoder_params, transitions.next_observation)
      next_action = networks.policy_network.apply(policy_params, next_encoded)
      next_action = networks.add_policy_noise(next_action, key, sigma,
                                              noise_clip)
      next_q1, next_q2 = networks.critic_network.apply(critic_target_params,
                                                       next_encoded,
                                                       next_action)
      # Calculate q target values
      next_q = jnp.minimum(next_q1, next_q2)
      target_q = transitions.reward + transitions.discount * discount * next_q
      target_q = jax.lax.stop_gradient(target_q)
      # Calculate predicted Q
      encoded = networks.encoder_network.apply(encoder_params,
                                               transitions.observation)
      q1, q2 = networks.critic_network.apply(critic_params, encoded,
                                             transitions.action)
      loss_critic = (jnp.square(target_q - q1) +
                     jnp.square(target_q - q2)).mean(axis=0)
      return loss_critic, {"q1": q1.mean(), "q2": q2.mean()}

    def policy_loss_fn(
        policy_params: networks_lib.Params,
        critic_params: networks_lib.Params,
        encoder_params: networks_lib.Params,
        observation: types.Transition,
        sigma: jnp.ndarray,
        key,
    ):
      encoded = networks.encoder_network.apply(encoder_params, observation)
      action = networks.policy_network.apply(policy_params, encoded)
      action = networks.add_policy_noise(action, key, sigma, noise_clip)
      q1, q2 = networks.critic_network.apply(critic_params, encoded, action)
      q = jnp.minimum(q1, q2)
      policy_loss = -q.mean()
      return policy_loss, {}

    policy_grad_fn = jax.value_and_grad(policy_loss_fn, has_aux=True)
    critic_grad_fn = jax.value_and_grad(
        critic_loss_fn, argnums=(0, 1), has_aux=True)

    def update_step(
        state: TrainingState,
        transitions: types.Transition,
    ):
      key_aug1, key_aug2, key_policy, key_critic, key = jax.random.split(
          state.key, 5)
      sigma = sigma_schedule(state.steps)
      # Perform data augmentation on o_tm1 and o_t
      observation_aug = augmentation(key_aug1, transitions.observation)
      next_observation_aug = augmentation(key_aug2,
                                          transitions.next_observation)
      transitions = transitions._replace(
          observation=observation_aug,
          next_observation=next_observation_aug,
      )
      # Update critic
      (critic_loss, critic_aux), (critic_grad, encoder_grad) = critic_grad_fn(
          state.critic_params,
          state.encoder_params,
          state.critic_target_params,
          state.policy_params,
          transitions,
          key_critic,
          sigma,
      )
      encoder_update, encoder_opt_state = encoder_optimizer.update(
          encoder_grad, state.encoder_opt_state)
      critic_update, critic_opt_state = critic_optimizer.update(
          critic_grad, state.critic_opt_state)
      encoder_params = optax.apply_updates(state.encoder_params, encoder_update)
      critic_params = optax.apply_updates(state.critic_params, critic_update)
      # Update policy
      (policy_loss, policy_aux), actor_grad = policy_grad_fn(
          state.policy_params,
          critic_params,
          encoder_params,
          observation_aug,
          sigma,
          key_policy,
      )
      policy_update, policy_opt_state = policy_optimizer.update(
          actor_grad, state.policy_opt_state)
      policy_params = optax.apply_updates(state.policy_params, policy_update)

      # Update target parameters
      polyak_update_fn = partial(_soft_update, tau=critic_soft_update_rate)

      critic_target_params = polyak_update_fn(
          state.critic_target_params,
          critic_params,
      )
      metrics = {
          "policy_loss": policy_loss,
          "critic_loss": critic_loss,
          "sigma": sigma,
          **critic_aux,
          **policy_aux,
      }
      new_state = TrainingState(
          policy_params=policy_params,
          policy_opt_state=policy_opt_state,
          encoder_params=encoder_params,
          encoder_opt_state=encoder_opt_state,
          critic_params=critic_params,
          critic_target_params=critic_target_params,
          critic_opt_state=critic_opt_state,
          key=key,
          steps=state.steps + 1,
      )
      return new_state, metrics

    self._iterator = dataset
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.make_default_logger(
        label="learner",
        save_data=False,
        asynchronous=True,
        serialize_fn=utils.fetch_devicearray,
    )
    self._update_step = utils.process_multiple_batches(update_step,
                                                       num_sgd_steps_per_step)
    self._update_step = jax.jit(self._update_step)

    # Initialize training state
    def make_initial_state(key: jax_types.PRNGKey):
      key_encoder, key_critic, key_policy, key = jax.random.split(key, 4)
      encoder_init_params = networks.encoder_network.init(key_encoder)
      encoder_init_opt_state = encoder_optimizer.init(encoder_init_params)

      critic_init_params = networks.critic_network.init(key_critic)
      critic_init_opt_state = critic_optimizer.init(critic_init_params)

      policy_init_params = networks.policy_network.init(key_policy)
      policy_init_opt_state = policy_optimizer.init(policy_init_params)

      return TrainingState(
          policy_params=policy_init_params,
          policy_opt_state=policy_init_opt_state,
          encoder_params=encoder_init_params,
          critic_params=critic_init_params,
          critic_target_params=critic_init_params,
          encoder_opt_state=encoder_init_opt_state,
          critic_opt_state=critic_init_opt_state,
          key=key,
          steps=0,
      )

    # Create initial state.
    self._state = make_initial_state(random_key)

    # Do not record timestamps until after the first learning step is done.
    # This is to avoid including the time it takes for actors to come online and
    # fill the replay buffer.
    self._timestamp = None

  def step(self):
    # Get the next batch from the replay iterator
    sample = next(self._iterator)
    transitions = types.Transition(*sample.data)

    # Perform a single learner step
    self._state, metrics = self._update_step(self._state, transitions)

    # Compute elapsed time
    timestamp = time.time()
    elapsed_time = timestamp - self._timestamp if self._timestamp else 0
    self._timestamp = timestamp

    # Increment counts and record the current time
    counts = self._counter.increment(steps=1, walltime=elapsed_time)

    # Attempts to write the logs.
    self._logger.write({**metrics, **counts})

  def get_variables(self, names):
    variables = {
        "policy": {
            "encoder": self._state.encoder_params,
            "policy": self._state.policy_params,
        },
    }
    return [variables[name] for name in names]

  def save(self) -> TrainingState:
    return self._state

  def restore(self, state: TrainingState) -> None:
    self._state = state


class DrQV2Builder(builders.ActorLearnerBuilder):
  """DrQ-v2 Builder."""

  def __init__(self, config: DrQV2Config):
    self._config = config

  def make_replay_tables(
      self, environment_spec: specs.EnvironmentSpec) -> List[reverb.Table]:
    """Create tables to insert data into."""
    samples_per_insert_tolerance = (
        self._config.samples_per_insert_tolerance_rate *
        self._config.samples_per_insert)
    error_buffer = self._config.min_replay_size * samples_per_insert_tolerance
    limiter = rate_limiters.SampleToInsertRatio(
        min_size_to_sample=self._config.min_replay_size,
        samples_per_insert=self._config.samples_per_insert,
        error_buffer=error_buffer,
    )
    replay_table = reverb.Table(
        name=self._config.replay_table_name,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        max_size=self._config.max_replay_size,
        rate_limiter=limiter,
        signature=adders_reverb.NStepTransitionAdder.signature(
            environment_spec=environment_spec),
    )
    return [replay_table]

  def make_dataset_iterator(
      self, replay_client: reverb.Client) -> Iterator[reverb.ReplaySample]:
    """Create a dataset iterator to use for learning/updating the agent."""
    dataset = datasets.make_reverb_dataset(
        table=self._config.replay_table_name,
        server_address=replay_client.server_address,
        batch_size=self._config.batch_size *
        self._config.num_sgd_steps_per_step,
        prefetch_size=self._config.prefetch_size,
    )
    iterator = dataset.as_numpy_iterator()
    return utils.device_put(iterator, jax.devices()[0])

  def make_adder(self, replay_client: reverb.Client) -> Optional[adders.Adder]:
    """Create an adder which records data generated by the actor/environment.
        Args:
          replay_client: Reverb Client which points to the replay server.
    """
    return adders_reverb.NStepTransitionAdder(
        client=replay_client,
        n_step=self._config.n_step,
        discount=self._config.discount,
    )

  def make_actor(
      self,
      random_key: networks_lib.PRNGKey,
      policy_network: drq_v2_networks.DrQV2PolicyNetwork,
      adder: Optional[adders.Adder] = None,
      variable_source: Optional[core.VariableSource] = None) -> core.Actor:
    """Create an actor instance.
      Args:
        random_key: A key for random number generation.
        policy_network: Instance of a policy network; this should be a callable
          which takes as input observations and returns actions.
        adder: How data is recorded (e.g. added to replay).
        variable_source: A source providing the necessary actor parameters.
    """
    assert variable_source is not None
    variable_client = variable_utils.VariableClient(
        variable_source, "policy", device='cpu')
    variable_client.update_and_wait()
    return actors.GenericActor(
        actor_core.batched_feed_forward_to_actor_core(policy_network),
        random_key=random_key,
        variable_client=variable_client,
        adder=adder,
        backend='cpu')

  def make_learner(self,
                   random_key: networks_lib.PRNGKey,
                   networks: drq_v2_networks.DrQV2Networks,
                   dataset: Iterator[reverb.ReplaySample],
                   logger: Optional[loggers.Logger] = None,
                   replay_client: Optional[reverb.Client] = None,
                   counter: Optional[counting.Counter] = None) -> core.Learner:
    """Creates an instance of the learner.

        Args:
          random_key: A key for random number generation.
          networks: struct describing the networks needed by the learner; this can
            be specific to the learner in question.
          dataset: iterator over samples from replay.
          replay_client: client which allows communication with replay, e.g. in
            order to update priorities.
          counter: a Counter which allows for recording of counts (learner steps,
            actor steps, etc.) distributed throughout the agent.
          checkpoint: bool controlling whether the learner checkpoints itself.
        """
    del replay_client
    config = self._config
    critic_optimizer = optax.adam(config.learning_rate)
    policy_optimizer = optax.adam(config.learning_rate)
    encoder_optimizer = optax.adam(config.learning_rate)

    return DrQV2Learner(
        random_key=random_key,
        dataset=dataset,
        networks=networks,
        sigma_schedule=optax.constant_schedule(config.sigma),
        policy_optimizer=policy_optimizer,
        critic_optimizer=critic_optimizer,
        encoder_optimizer=encoder_optimizer,
        augmentation=config.augmentation,
        critic_soft_update_rate=config.critic_q_soft_update_rate,
        discount=config.discount,
        noise_clip=config.noise_clip,
        num_sgd_steps_per_step=config.num_sgd_steps_per_step,
        logger=logger,
        counter=counter,
    )
