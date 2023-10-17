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

"""IMPALA networks definition."""

# Python
# import dataclasses, functools, time
# from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple, Union

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
from acme.agents.jax.impala import config as impala_config



IMPALANetworks = networks_lib.UnrollableNetwork


def make_atari_networks(
	spec: specs.EnvironmentSpec,
	config: impala_config.IMPALAConfig,
) -> IMPALANetworks:
	"""Builds default IMPALA networks for Atari games."""

	def make_core_module() -> networks_lib.DeepIMPALAAtariNetwork:
		return networks_lib.DeepIMPALAAtariNetwork(spec.actions.num_values)

	return networks_lib.make_unrollable_network(spec, make_core_module)


def make_networks(
	spec: specs.EnvironmentSpec,
	config: impala_config.IMPALAConfig,
) -> IMPALANetworks:
	"""Builds default IMPALA networks for Atari games."""

	def make_core_module() -> networks_lib.DeepIMPALAAtariNetwork:
		return networks_lib.DeepIMPALAAtariNetwork(spec.actions.num_values)

	return networks_lib.make_unrollable_network(spec, make_core_module)
