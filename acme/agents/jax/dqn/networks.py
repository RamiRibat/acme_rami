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

"""Network definitions for DQN."""
# Python
import dataclasses, functools, time
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple, Union

# ML/DL
import jax, chex, optax, rlax
import jax.numpy as jnp
import haiku as hk
import haiku.initializers as hk_init
import numpy as np
import tensorflow_probability.substrates.jax as tfp
import reverb
from reverb import rate_limiters

# ACME
from acme import specs, types
from acme.agents.jax import actor_core as actor_core_lib
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.agents.jax.dqn import config as dqn_config



Epsilon = float

EpsilonPolicy = Callable[
	[networks_lib.Params,
	networks_lib.PRNGKey,
	networks_lib.Observation,
	Epsilon],
	networks_lib.Action
]

EpsilonSampleFn = Callable[
	[networks_lib.NetworkOutput,
	networks_lib.PRNGKey, # og: types.PRNGKey
	Epsilon],
    networks_lib.Action
]

EpsilonLogProbFn = Callable[
    [networks_lib.NetworkOutput,
	 networks_lib.Action,
	 Epsilon],
    networks_lib.LogProb
]


def default_sample_fn(
    action_values: networks_lib.NetworkOutput,
    key: networks_lib.PRNGKey, # og: types.PRNGKey
    epsilon: Epsilon
) -> networks_lib.Action:
	return rlax.epsilon_greedy(epsilon).sample(key, action_values)


@dataclasses.dataclass
class DQNNetworks:
	"""The network and pure functions for the DQN agent.

	Attributes:
		policy_network: The policy network.
		sample_fn: A pure function. Samples an action based on the network output.
		log_prob: A pure function. Computes log-probability for an action.
	"""
	policy_network: networks_lib.TypedFeedForwardNetwork
	sample_fn: EpsilonSampleFn = default_sample_fn # eps_greedy
	log_prob: Optional[EpsilonLogProbFn] = None



def make_networks(
	spec: specs.EnvironmentSpec,
	config: dqn_config.D4PGConfig,
) -> DQNNetworks:
	"""Creates networks for training DQN on Atari."""

	def network(inputs):
		model = hk.Sequential([
			networks_lib.AtariTorso(),
			hk.nets.MLP([512, spec.actions.num_values]),
		])
		return model(inputs)

	network_hk = hk.without_apply_rng(hk.transform(network))

	# Create dummy observations to create network parameters.
	dummy_obs = utils.zeros_like(spec.observations)
	# Add batch dimension
	dummy_obs = utils.add_batch_dim(dummy_obs)

	# obs = utils.add_batch_dim(utils.zeros_like(spec.observations))

	network = networks_lib.FeedForwardNetwork(
		init=lambda rng: network_hk.init(rng, dummy_obs),
		apply=network_hk.apply
	)
	typed_network = networks_lib.non_stochastic_network_to_typed(network)

	return DQNNetworks(policy_network=typed_network)