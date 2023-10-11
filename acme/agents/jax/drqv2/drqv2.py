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

# DataAugmentation = Callable[[jax_types.PRNGKey, types.NestedArray],
#                             types.NestedArray]


# # From https://github.com/ikostrikov/jax-rl/blob/main/jax_rl/agents/drq/augmentations.py
# def random_crop(key: jax_types.PRNGKey, img, padding):
#   crop_from = jax.random.randint(key, (2,), 0, 2 * padding + 1)
#   crop_from = jnp.concatenate([crop_from, jnp.zeros((1,), dtype=jnp.int32)])
#   padded_img = jnp.pad(
#       img, ((padding, padding), (padding, padding), (0, 0)), mode="edge")
#   return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


# def batched_random_crop(key, imgs, padding=4):
#   keys = jax.random.split(key, imgs.shape[0])
#   return jax.vmap(random_crop, (0, 0, None))(keys, imgs, padding)



