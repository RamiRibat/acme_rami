

"""Config classes for DrQv2."""
import dataclasses
from typing import Callable, Dict, Optional, Sequence

import jax
import jax.numpy as jnp

from acme import types
from acme.adders import reverb as adders_reverb
from acme.jax import types as jax_types


DataAugmentation = Callable[[jax_types.PRNGKey, types.NestedArray], types.NestedArray]


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
	""" General hyper-parameters """
	noise_clip: float = 0.3
	sigma: float = 0.2

	""" Network/Optz hyper-parameters """
	policy_arch: Sequence[int] = (300, 200)
	critic_arch: Sequence[int] = (400, 300)
	# n_atoms: int = 51

	""" Learner (Loss) hyper-parameters """
	batch_size: int = 256
	learning_rate: float = 1e-4
	discount: float = 0.99
	n_step: int = 3
	# target_update_period: int = 100
	# clipping: bool = True # ?
	critic_q_soft_update_rate: float = 0.01

	""" Replay hyper-parameters """
	replay_table_name: str = adders_reverb.DEFAULT_PRIORITY_TABLE
	prefetch_size: int = 4 # ?
	min_replay_size: int = 2_000
	max_replay_size: int = 1_000_000
	samples_per_insert: float = 128.0
	# Rate to be used for the SampleToInsertRatio rate limitter tolerance.
	# See a formula in make_replay_tables for more details.
	samples_per_insert_tolerance_rate: float = 0.1
	# How many gradient updates to perform per step.
	num_sgd_steps_per_step: int = 1

	# replay_ratio: int = 0.125
	# reset_interval: Optional[int] = None

	""" Domain Hyper-parameters """
	augmentation: DataAugmentation = batched_random_crop
	env: Optional[Dict] = None


