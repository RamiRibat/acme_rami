"""
Author: https://github.com/ethanluoyc
"""

"""Network definitions for DrQ-v2."""
import dataclasses
from typing import Callable, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import rlax

from acme import specs
from acme import types
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.agents.jax import actor_core as actor_core_lib

from acme.agents.jax.drqv2 import config as drqv2_config

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
  

# from deepmind/haiku
# class ConvND(hk.Module):
#   """General N-dimensional convolutional."""

#   def __init__(
#       self,
#       num_spatial_dims: int,
#       output_channels: int,
#       kernel_shape: Union[int, Sequence[int]],
#       stride: Union[int, Sequence[int]] = 1,
#       rate: Union[int, Sequence[int]] = 1,
#       padding: Union[
#           str, Sequence[tuple[int, int]], hk.pad.PadFn, Sequence[hk.pad.PadFn]
#       ] = "SAME",
#       with_bias: bool = True,
#       w_init: Optional[hk.initializers.Initializer] = None,
#       b_init: Optional[hk.initializers.Initializer] = None,
#       data_format: str = "channels_last",
#       mask: Optional[jax.Array] = None,
#       feature_group_count: int = 1,
#       name: Optional[str] = None,
#   ):
#     """Initializes the module.

#     Args:
#       num_spatial_dims: The number of spatial dimensions of the input.
#       output_channels: Number of output channels.
#       kernel_shape: The shape of the kernel. Either an integer or a sequence of
#         length ``num_spatial_dims``.
#       stride: Optional stride for the kernel. Either an integer or a sequence of
#         length ``num_spatial_dims``. Defaults to 1.
#       rate: Optional kernel dilation rate. Either an integer or a sequence of
#         length ``num_spatial_dims``. 1 corresponds to standard ND convolution,
#         ``rate > 1`` corresponds to dilated convolution. Defaults to 1.
#       padding: Optional padding algorithm. Either ``VALID`` or ``SAME`` or a
#         sequence of n ``(low, high)`` integer pairs that give the padding to
#         apply before and after each spatial dimension. or a callable or sequence
#         of callables of size ``num_spatial_dims``. Any callables must take a
#         single integer argument equal to the effective kernel size and return a
#         sequence of two integers representing the padding before and after. See
#         ``haiku.pad.*`` for more details and example functions. Defaults to
#         ``SAME``. See:
#         https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
#       with_bias: Whether to add a bias. By default, true.
#       w_init: Optional weight initialization. By default, truncated normal.
#       b_init: Optional bias initialization. By default, zeros.
#       data_format: The data format of the input.  Can be either
#         ``channels_first``, ``channels_last``, ``N...C`` or ``NC...``. By
#         default, ``channels_last``. See :func:`get_channel_index`.
#       mask: Optional mask of the weights.
#       feature_group_count: Optional number of groups in group convolution.
#         Default value of 1 corresponds to normal dense convolution. If a higher
#         value is used, convolutions are applied separately to that many groups,
#         then stacked together. This reduces the number of parameters
#         and possibly the compute for a given ``output_channels``. See:
#         https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
#       name: The name of the module.
#     """
#     super().__init__(name=name)
#     if num_spatial_dims <= 0:
#       raise ValueError(
#           "We only support convolution operations for `num_spatial_dims` "
#           f"greater than 0, received num_spatial_dims={num_spatial_dims}.")

#     self.num_spatial_dims = num_spatial_dims
#     self.output_channels = output_channels
#     self.kernel_shape = (
#         utils.replicate(kernel_shape, num_spatial_dims, "kernel_shape"))
#     self.with_bias = with_bias
#     self.stride = utils.replicate(stride, num_spatial_dims, "strides")
#     self.w_init = w_init
#     self.b_init = b_init or jnp.zeros
#     self.mask = mask
#     self.feature_group_count = feature_group_count
#     self.lhs_dilation = utils.replicate(1, num_spatial_dims, "lhs_dilation")
#     self.kernel_dilation = (
#         utils.replicate(rate, num_spatial_dims, "kernel_dilation"))
#     self.data_format = data_format
#     self.channel_index = hk.get_channel_index(data_format)
#     self.dimension_numbers = to_dimension_numbers(
#         num_spatial_dims, channels_last=(self.channel_index == -1),
#         transpose=False)

#     if isinstance(padding, str):
#       self.padding = padding.upper()
#     elif hk.pad.is_padfn(padding):
#       self.padding = hk.pad.create_from_padfn(padding=padding,
#                                               kernel=self.kernel_shape,
#                                               rate=self.kernel_dilation,
#                                               n=self.num_spatial_dims)
#     else:
#       self.padding = hk.pad.create_from_tuple(padding, self.num_spatial_dims)

#   def __call__(
#       self,
#       inputs: jax.Array,
#       *,
#       precision: Optional[lax.Precision] = None,
#   ) -> jax.Array:
#     """Connects ``ConvND`` layer.

#     Args:
#       inputs: An array of shape ``[spatial_dims, C]`` and rank-N+1 if unbatched,
#         or an array of shape ``[N, spatial_dims, C]`` and rank-N+2 if batched.
#       precision: Optional :class:`jax.lax.Precision` to pass to
#         :func:`jax.lax.conv_general_dilated`.

#     Returns:
#       An array of shape ``[spatial_dims, output_channels]`` and rank-N+1 if
#         unbatched, or an array of shape ``[N, spatial_dims, output_channels]``
#         and rank-N+2 if batched.
#     """
#     unbatched_rank = self.num_spatial_dims + 1
#     allowed_ranks = [unbatched_rank, unbatched_rank + 1]
#     if inputs.ndim not in allowed_ranks:
#       raise ValueError(f"Input to ConvND needs to have rank in {allowed_ranks},"
#                        f" but input has shape {inputs.shape}.")

#     unbatched = inputs.ndim == unbatched_rank
#     if unbatched:
#       inputs = jnp.expand_dims(inputs, axis=0)

#     if inputs.shape[self.channel_index] % self.feature_group_count != 0:
#       raise ValueError(f"Inputs channels {inputs.shape[self.channel_index]} "
#                        f"should be a multiple of feature_group_count "
#                        f"{self.feature_group_count}")
#     w_shape = self.kernel_shape + (
#         inputs.shape[self.channel_index] // self.feature_group_count,
#         self.output_channels)

#     if self.mask is not None and self.mask.shape != w_shape:
#       raise ValueError("Mask needs to have the same shape as weights. "
#                        f"Shapes are: {self.mask.shape}, {w_shape}")

#     w_init = self.w_init
#     if w_init is None:
#       fan_in_shape = np.prod(w_shape[:-1])
#       stddev = 1. / np.sqrt(fan_in_shape)
#       w_init = hk.initializers.TruncatedNormal(stddev=stddev)
#     w = hk.get_parameter("w", w_shape, inputs.dtype, init=w_init)

#     if self.mask is not None:
#       w *= self.mask

#     out = lax.conv_general_dilated(inputs,
#                                    w,
#                                    window_strides=self.stride,
#                                    padding=self.padding,
#                                    lhs_dilation=self.lhs_dilation,
#                                    rhs_dilation=self.kernel_dilation,
#                                    dimension_numbers=self.dimension_numbers,
#                                    feature_group_count=self.feature_group_count,
#                                    precision=precision)

#     if self.with_bias:
#       if self.channel_index == -1:
#         bias_shape = (self.output_channels,)
#       else:
#         bias_shape = (self.output_channels,) + (1,) * self.num_spatial_dims
#       b = hk.get_parameter("b", bias_shape, inputs.dtype, init=self.b_init)
#       b = jnp.broadcast_to(b, out.shape)
#       out = out + b

#     if unbatched:
#       out = jnp.squeeze(out, axis=0)
#     return out


# class Conv2D(ConvND):
#   """Two dimensional convolution."""

#   def __init__(
#       self,
#       output_channels: int,
#       kernel_shape: Union[int, Sequence[int]],
#       stride: Union[int, Sequence[int]] = 1,
#       rate: Union[int, Sequence[int]] = 1,
#       padding: Union[
#           str, Sequence[tuple[int, int]], hk.pad.PadFn, Sequence[hk.pad.PadFn]
#       ] = "SAME",
#       with_bias: bool = True,
#       w_init: Optional[hk.initializers.Initializer] = None,
#       b_init: Optional[hk.initializers.Initializer] = None,
#       data_format: str = "NHWC",
#       mask: Optional[jax.Array] = None,
#       feature_group_count: int = 1,
#       name: Optional[str] = None,
#   ):
#     """Initializes the module.

#     Args:
#       output_channels: Number of output channels.
#       kernel_shape: The shape of the kernel. Either an integer or a sequence of
#         length 2.
#       stride: Optional stride for the kernel. Either an integer or a sequence of
#         length 2. Defaults to 1.
#       rate: Optional kernel dilation rate. Either an integer or a sequence of
#         length 2. 1 corresponds to standard ND convolution,
#         ``rate > 1`` corresponds to dilated convolution. Defaults to 1.
#       padding: Optional padding algorithm. Either ``VALID`` or ``SAME`` or
#         a callable or sequence of callables of length 2. Any callables must
#         take a single integer argument equal to the effective kernel size and
#         return a list of two integers representing the padding before and after.
#         See haiku.pad.* for more details and example functions.
#         Defaults to ``SAME``. See:
#         https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
#       with_bias: Whether to add a bias. By default, true.
#       w_init: Optional weight initialization. By default, truncated normal.
#       b_init: Optional bias initialization. By default, zeros.
#       data_format: The data format of the input. Either ``NHWC`` or ``NCHW``. By
#         default, ``NHWC``.
#       mask: Optional mask of the weights.
#       feature_group_count: Optional number of groups in group convolution.
#         Default value of 1 corresponds to normal dense convolution. If a higher
#         value is used, convolutions are applied separately to that many groups,
#         then stacked together. This reduces the number of parameters
#         and possibly the compute for a given ``output_channels``. See:
#         https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
#       name: The name of the module.
#     """
#     super().__init__(
#         num_spatial_dims=2,
#         output_channels=output_channels,
#         kernel_shape=kernel_shape,
#         stride=stride,
#         rate=rate,
#         padding=padding,
#         with_bias=with_bias,
#         w_init=w_init,
#         b_init=b_init,
#         data_format=data_format,
#         mask=mask,
#         feature_group_count=feature_group_count,
#         name=name)


class Encoder(hk.Module):
	"""Encoder used by DrQ-v2."""

	def __call__(self, x):
		# Floatify the image.
		x = x.astype(jnp.float32) / 255.0 - 0.5
		conv_kwargs = dict(
			output_channels=32,
			kernel_shape=3, # 3x3
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
	# print('env_spec: ', spec)
     
	action_size = np.prod(spec.actions.shape, dtype=int)

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
	# print(f'dummy: act: {dummy_action.shape} | obs: {dummy_obs.shape}')
	# print('dummy_encoded: ', dummy_encoded.shape)

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
