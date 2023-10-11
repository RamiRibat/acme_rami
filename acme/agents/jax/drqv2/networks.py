"""
Author: https://github.com/ethanluoyc
"""

"""Network definitions for DrQ-v2."""
import dataclasses
from typing import Callable, Optional, Union

from acme import specs
from acme import types
from acme.agents.jax import actor_core as actor_core_lib
from acme.jax import networks as networks_lib
from acme.jax import utils

from acme.agents.jax.drqv2 import config as drqv2_config

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as onp
import optax

import jax.numpy as jnp
import numpy as np
import haiku as hk
import rlax

# Unlike standard FF-policy, in our DrQ-V2 implementation we use
# scheduled stddev parameters, the pure function for the policy
# thus needs to know the current time step of the actor to calculate
# the current stddev.
_Step = int
DrQV2PolicyNetwork = Callable[
	[networks_lib.Params,
  	networks_lib.PRNGKey,
  	types.NestedArray,
  	_Step],
	types.NestedArray
]


@dataclasses.dataclass
class DrQV2Networks:
	policy_network: networks_lib.FeedForwardNetwork
	critic_network: networks_lib.FeedForwardNetwork
	encoder_network: networks_lib.FeedForwardNetwork
	add_policy_noise: Callable[
		[types.NestedArray, networks_lib.PRNGKey, float, float],
		types.NestedArray]
  


class Encoder(hk.Module):
	"""Encoder used by DrQ-v2."""

	def __call__(self, x):
		# Floatify the image.
		x = x.astype(jnp.float32) / 255.0 - 0.5
		conv_kwargs = dict(
			kernel_shape=3,
			output_channels=32,
			padding="VALID",
			# This follows from the reference implementation, the scale accounts for
			# using the ReLU activation.
			w_init=hk.initializers.Orthogonal(jnp.sqrt(2.0)),
		)
		return hk.Sequential([
			hk.Conv2D(stride=2, **conv_kwargs),
			jax.nn.relu,
			hk.Conv2D(stride=1, **conv_kwargs),
			jax.nn.relu,
			hk.Conv2D(stride=1, **conv_kwargs),
			jax.nn.relu,
			hk.Conv2D(stride=1, **conv_kwargs),
			jax.nn.relu,
			hk.Flatten(),
		])(x)


class Actor(hk.Module):
	"""Policy network used by DrQ-v2."""

	def __init__(
		self,
		action_size: int,
		latent_size: int = 50,
		hidden_size: int = 1024,
		name: Optional[str] = None,
	):
		super().__init__(name=name)
		self.latent_size = latent_size
		self.action_size = action_size
		self.hidden_size = hidden_size
		w_init = hk.initializers.Orthogonal(1.0)
		self._trunk = hk.Sequential([
			hk.Linear(self.latent_size, w_init=w_init),
			hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
			jnp.tanh,
		])
		self._head = hk.Sequential([
			hk.Linear(self.hidden_size, w_init=w_init),
			jax.nn.relu,
			hk.Linear(self.hidden_size, w_init=w_init),
			jax.nn.relu,
			hk.Linear(self.action_size, w_init=w_init),
			# tanh is used to squash the actions into the canonical space.
			jnp.tanh,
		])

	def compute_features(self, inputs):
		return self._trunk(inputs)

	def __call__(self, inputs):
		# Use orthogonal init
		# https://github.com/facebookresearch/drqv2/blob/21e9048bf59e15f1018b49b850f727ed7b1e210d/utils.py#L54
		h = self.compute_features(inputs)
		mu = self._head(h)
		return mu


class Critic(hk.Module):
	"""Single Critic network used by DrQ-v2."""

	def __init__(self, hidden_size: int = 1024, name: Optional[str] = None):
		super().__init__(name)
		self.hidden_size = hidden_size

	def __call__(self, observation, action):
		inputs = jnp.concatenate([observation, action], axis=-1)
		# Use orthogonal init
		# https://github.com/facebookresearch/drqv2/blob/21e9048bf59e15f1018b49b850f727ed7b1e210d/utils.py#L54
		q_value = hk.nets.MLP(
			output_sizes=(self.hidden_size, self.hidden_size, 1),
			w_init=hk.initializers.Orthogonal(1.0),
			activate_final=False,
		)(inputs).squeeze(-1)
		return q_value


class DoubleCritic(hk.Module):
	"""Twin critic network used by DrQ-v2.

	This is simply two identical Critic module.
	"""

	def __init__(self, latent_size: int = 50, hidden_size: int = 1024, name=None):
		super().__init__(name)
		self.hidden_size = hidden_size
		self.latent_size = latent_size

		self._trunk = hk.Sequential([
			hk.Linear(self.latent_size, w_init=hk.initializers.Orthogonal(1.0)),
			hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
			jnp.tanh,
		])
		self._critic1 = Critic(self.hidden_size, name="critic1")
		self._critic2 = Critic(self.hidden_size, name="critic2")

	def compute_features(self, inputs):
		return self._trunk(inputs)

	def __call__(self, observation, action):
		# Use orthogonal init
		# https://github.com/facebookresearch/drqv2/blob/21e9048bf59e15f1018b49b850f727ed7b1e210d/utils.py#L54
		# The trunk is shared between the twin critics
		h = self.compute_features(observation)
		return self._critic1(h, action), self._critic2(h, action)




def get_default_behavior_policy(
	networks: DrQV2Networks,
	environment_spec: specs.BoundedArray,
	config: drqv2_config.DrQV2Config
) -> DrQV2PolicyNetwork:
	
	action_specs = environment_spec.actions

	def behavior_policy(
		params: networks_lib.Params,
		key: networks_lib.PRNGKey,
		observation: types.NestedArray,
	):
		feature_map = networks.encoder_network.apply(params["encoder"], observation)
		action = networks.policy_network.apply(params["policy"], feature_map)
		if config.sigma != 0:
			# noise = jax.random.normal(key, shape=action.shape) * config.sigma
			# action = jnp.clip(action + noise, action_specs.minimum, action_specs.maximum)
			action = rlax.add_gaussian_noise(key, action, config.sigma)
			action = jnp.clip(action, action_specs.minimum, action_specs.maximum)
		return action

	return behavior_policy


def get_default_eval_policy(
    networks: DrQV2Networks,
	environment_spec: specs.BoundedArray,
) -> actor_core_lib.FeedForwardPolicy:
	"""Selects action according to the training policy."""
	
	action_specs = environment_spec.actions

	def behavior_policy(
		params: networks_lib.Params,
		key: networks_lib.PRNGKey,
		observation: types.NestedArray,
	):
		del key
		feature_map = networks.encoder_network.apply(params["encoder"], observation)
		action = networks.policy_network.apply(params["policy"], feature_map)
		action = jnp.clip(action, action_specs.minimum, action_specs.maximum)
		return action

	return behavior_policy


def make_networks(
	spec: specs.EnvironmentSpec,
	latent_size: int = 50,
	hidden_size: int = 1024,
) -> DrQV2Networks:
	"""Create networks for the DrQ-v2 agent."""
     
	action_size = onp.prod(spec.actions.shape, dtype=int)

	def _encoder_fn(x):
		return Encoder()(x)

	def add_policy_noise(
		action: types.NestedArray,
		key: networks_lib.PRNGKey,
		sigma: float,
		noise_clip: float,
	) -> types.NestedArray:
		"""Adds action noise to bootstrapped Q-value estimate in critic loss."""
		noise = jax.random.normal(key=key, shape=spec.actions.shape) * sigma
		noise = jnp.clip(noise, -noise_clip, noise_clip)
		return jnp.clip(action + noise, spec.actions.minimum, spec.actions.maximum)

	def _policy_fn(x):
		return Actor(
			action_size=action_size,
			latent_size=latent_size,
			hidden_size=hidden_size,
		)(x)

	def _critic_fn(x, a):
		return DoubleCritic(
			latent_size=latent_size,
			hidden_size=hidden_size,
		)(x, a)

	policy = hk.without_apply_rng(hk.transform(_policy_fn, apply_rng=True))
	critic = hk.without_apply_rng(hk.transform(_critic_fn, apply_rng=True))
	encoder = hk.without_apply_rng(hk.transform(_encoder_fn, apply_rng=True))

	# Create dummy observations and actions to create network parameters.
	dummy_action = utils.zeros_like(spec.actions)
	dummy_obs = utils.zeros_like(spec.observations)
	dummy_action = utils.add_batch_dim(dummy_action)
	dummy_obs = utils.add_batch_dim(dummy_obs)
	dummy_encoded = hk.testing.transform_and_run(
		_encoder_fn, seed=0, jax_transform=jax.jit)(
			dummy_obs)

	return DrQV2Networks(
		encoder_network=networks_lib.FeedForwardNetwork(
			lambda key: encoder.init(key, dummy_obs), encoder.apply),
		policy_network=networks_lib.FeedForwardNetwork(
			lambda key: policy.init(key, dummy_encoded), policy.apply),
		critic_network=networks_lib.FeedForwardNetwork(
			lambda key: critic.init(key, dummy_encoded, dummy_action),
			critic.apply),
		add_policy_noise=add_policy_noise
	)
