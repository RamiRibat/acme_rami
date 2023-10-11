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

"""An environment wrapper to produce pixel observations from dm_control."""

import abc
import collections
from typing import Tuple, List, Optional, Sequence, Union

import numpy as np
from PIL import Image

from acme.wrappers import base
from acme.wrappers import frame_stacking

import dm_env
from dm_env import specs
from dm_control.rl import control
from dm_control.suite.wrappers import pixels  # type: ignore



class MujocoPixelWrapper(base.EnvironmentWrapper):
	"""Produces pixel observations from Mujoco environment observations."""

	def __init__(self,
				environment: control.Environment,
				*,
				height: int = 84,
				width: int = 84,
				camera_id: int = 0
	):
		render_kwargs = {'height': height, 'width': width, 'camera_id': camera_id}
		pixel_environment = pixels.Wrapper(
			environment, pixels_only=True, render_kwargs=render_kwargs)
		super().__init__(pixel_environment)

	def step(self, action) -> dm_env.TimeStep:
		return self._convert_timestep(self._environment.step(action))

	def reset(self) -> dm_env.TimeStep:
		return self._convert_timestep(self._environment.reset())

	def observation_spec(self):
		return self._environment.observation_spec()['pixels']

	def _convert_timestep(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
		"""Removes the pixel observation's OrderedDict wrapper."""
		observation: collections.OrderedDict = timestep.observation
		return timestep._replace(observation=observation['pixels'])





RGB_INDEX = 0  # Observation index holding the RGB data.
LIVES_INDEX = 1  # Observation index holding the lives count.
NUM_COLOR_CHANNELS = 3  # Number of color channels in RGB data.


class MujocoPixelWrapperV2(base.EnvironmentWrapper):
	"""Produces pixel observations from Mujoco environment observations."""

	def __init__(
		self,
		environment: control.Environment,
		*,
		# height: int = 84,
		# width: int = 84,
		camera_id: int = 0,
		scale_dims: Tuple[int, int] = (84, 84),
		# pooled_frames: int = 1,
		# num_stacked_frames: int = 4,
		# flatten_frame_stack: bool = False,
		to_float: bool = False,
		grayscaling: bool = False,
	):
		print('MujocoPixelWrapperV2')

		if scale_dims:
			self._height, self._width = scale_dims
		else:
			spec = environment.observation_spec()
			self._height, self._width = spec[RGB_INDEX].shape[:2]

		# render_kwargs = {'camera_id': camera_id}
		render_kwargs = {'height': self._height, 'width': self._width, 'camera_id': camera_id}

		pixel_environment = pixels.Wrapper(environment, pixels_only=True, render_kwargs=render_kwargs)
		super().__init__(pixel_environment)

		self._scale_dims = scale_dims
		self._to_float = to_float

		self._grayscaling = grayscaling

		self._observation_spec = self._init_observation_spec()


	def _init_observation_spec(self):
		"""Computes the observation spec for the pixel observations.

		Returns:
		An `Array` specification for the pixel observations.
		"""
        
		if self._to_float:
			pixels_dtype = float
		else:
			pixels_dtype = np.uint8

		if self._grayscaling:
			pixels_spec_shape = (self._height, self._width, 1)
			pixels_spec_name = "grayscale"
		else:
			pixels_spec_shape = (self._height, self._width, NUM_COLOR_CHANNELS)
			pixels_spec_name = "RGB"

		pixel_spec = specs.Array(
			shape=pixels_spec_shape,
			dtype=pixels_dtype,
			name=pixels_spec_name
		)
		# pixel_spec = self._frame_stacker.update_spec(pixel_spec)

		return pixel_spec


	def step(self, action) -> dm_env.TimeStep:
		return self._convert_timestep(self._environment.step(action))


	def reset(self) -> dm_env.TimeStep:
		return self._convert_timestep(self._environment.reset())


	def observation_spec(self):
		# return self._environment.observation_spec()['pixels']
		return self._observation_spec


	def _convert_timestep(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
		"""Removes the pixel observation's OrderedDict wrapper."""
		observation: collections.OrderedDict = timestep.observation
		# return timestep._replace(observation=observation['pixels'])
		observation = self._preprocess_pixels(observation['pixels'])
		return timestep._replace(observation=observation)
	

	def _preprocess_pixels(
		self,
		pixels
	):
		"""Preprocess DMC frames."""

		# # Max pooling (frameskip > 1)
		# processed_pixels = np.max(
		# 	np.stack([
		# 		s.observation[RGB_INDEX]
		# 		for s in timestep_stack[-self._pooled_frames:]
		# 	]),
		# 	axis=0
		# )
		
		processed_pixels = pixels

		# print('\n\na.processed_pixels: ', processed_pixels.shape)

		# RGB to grayscale
		if self._grayscaling:
			# processed_pixels = np.dot(
			# 	processed_pixels,
			# 	[0.299, 0.587, 1 - (0.299 + 0.587)]
			# )
			processed_pixels = np.tensordot(
				processed_pixels, # (H, W, C)
				[0.299, 0.587, 1 - (0.299 + 0.587)],
				(-1, 0)
			)

			processed_pixels = processed_pixels[:, :, None]


		# print('z.processed_pixels: ', processed_pixels.shape)

		processed_pixels = processed_pixels.astype(np.uint8, copy=False)

		# # Resize
		# if self._scale_dims != processed_pixels.shape[:2]:
		# 	processed_pixels = Image.fromarray(processed_pixels).resize(
		# 		(self._width, self._height), Image.Resampling.BILINEAR)
		# 	processed_pixels = np.array(processed_pixels, dtype=np.uint8)

		return processed_pixels



# TODO(rami): Make it like AtariWrapper
class MujocoPixelWrapperV3(base.EnvironmentWrapper):
	"""Produces pixel observations from Mujoco environment observations."""

	def __init__(
		self,
		environment: control.Environment,
		*,
		# height: int = 84,
		# width: int = 84,
		camera_id: int = 0,
		scale_dims: Tuple[int, int] = (84, 84),
		pooled_frames: int = 1,
		num_stacked_frames: int = 4,
		flatten_frame_stack: bool = False,
		to_float: bool = False,
		grayscaling: bool = True,
	):
		print('MujocoPixelWrapperV3')

		if scale_dims:
			self._height, self._width = scale_dims
		# else:
		# 	spec = environment.observation_spec()
		# 	self._height, self._width = spec[RGB_INDEX].shape[:2]

		# render_kwargs = {'camera_id': camera_id}
		render_kwargs = {'height': self._height, 'width': self._width, 'camera_id': camera_id}

		pixel_environment = pixels.Wrapper(environment, pixels_only=True, render_kwargs=render_kwargs)
		super().__init__(pixel_environment)

		self._frame_stacker = frame_stacking.FrameStacker(
			num_frames=num_stacked_frames,
			flatten=flatten_frame_stack
		)
		# self._pooled_frames = pooled_frames
		self._scale_dims = scale_dims
		self._to_float = to_float

		self._grayscaling = grayscaling

		self._observation_spec = self._init_observation_spec()


	def _init_observation_spec(self):
		"""Computes the observation spec for the pixel observations.

		Returns:
		An `Array` specification for the pixel observations.
		"""
        
		if self._to_float:
			pixels_dtype = float
		else:
			pixels_dtype = np.uint8

		if self._grayscaling:
			pixels_spec_shape = (self._height, self._width)
			pixels_spec_name = "grayscale"
		else:
			pixels_spec_shape = (self._height, self._width, NUM_COLOR_CHANNELS)
			pixels_spec_name = "RGB"

		pixel_spec = specs.Array(
			shape=pixels_spec_shape,
			dtype=pixels_dtype,
			name=pixels_spec_name
		)
		pixel_spec = self._frame_stacker.update_spec(pixel_spec)

		return pixel_spec


	def reset(self) -> dm_env.TimeStep:
		# return self._convert_timestep(self._environment.reset())
		"""Resets environment and provides the first timestep."""
		# self._reset_next_step = False
		# self._episode_len = 0
		self._frame_stacker.reset()
		timestep = self._environment.reset()
		timestep = self._convert_timestep(timestep)

		observation = self._observation_from_timestep_stack(timestep)

		timestep = timestep._replace(observation=observation)

		return self._postprocess_observation(timestep)
	

	def step(self, action) -> dm_env.TimeStep:
		# return self._convert_timestep(self._environment.step(action))
		"""Steps up to action_repeat times and returns a post-processed step."""
		
		timestep = self._environment.step([np.array([action])])
		timestep = self._convert_timestep(timestep)

		observation = self._observation_from_timestep_stack(timestep)

		timestep = timestep._replace(observation=observation)

		return self._postprocess_observation(timestep)


	def _convert_timestep(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
		"""Removes the pixel observation's OrderedDict wrapper."""
		observation: collections.OrderedDict = timestep.observation
		return timestep._replace(observation=observation['pixels'])
		# observation = self._preprocess_pixels(observation['pixels'])
		# return timestep._replace(observation=observation)
	

	def _preprocess_pixels(
		self,
		timestep_stack#: List[dm_env.TimeStep]
	):
		"""Preprocess DMC frames."""

		# # Max pooling (frameskip > 1)
		# processed_pixels = np.max(
		# 	np.stack([
		# 		s.observation[RGB_INDEX]
		# 		for s in timestep_stack[-self._pooled_frames:]
		# 	]),
		# 	axis=0
		# )
		
		processed_pixels = timestep_stack.observation
		# print('processed_pixels: ', processed_pixels)

		# # RGB to grayscale
		# if self._grayscaling:
		# 	processed_pixels = np.tensordot(
		# 		processed_pixels,
		# 		[0.299, 0.587, 1 - (0.299 + 0.587)],
		# 		(-1, 0)
		# 	)

		# # Resize
		# processed_pixels = processed_pixels.astype(np.uint8, copy=False)
		# if self._scale_dims != processed_pixels.shape[:2]:
		# 	processed_pixels = Image.fromarray(processed_pixels).resize(
		# 		(self._width, self._height), Image.Resampling.BILINEAR)
		# 	processed_pixels = np.array(processed_pixels, dtype=np.uint8)

		return processed_pixels


	def _observation_from_timestep_stack(
		self,
		timestep_stack: List[dm_env.TimeStep]
	):
		"""Compute the observation for a stack of timesteps."""
		# self._raw_observation = timestep_stack[-1].observation[RGB_INDEX].copy()
		processed_pixels = self._preprocess_pixels(timestep_stack)

		if self._to_float:
			stacked_observation = self._frame_stacker.step(processed_pixels / 255.0)
		else:
			stacked_observation = self._frame_stacker.step(processed_pixels)

		# We use last timestep for lives only.
		# observation = timestep_stack[-1].observation

		return stacked_observation


	def _postprocess_observation(
		self,
		timestep: dm_env.TimeStep
	) -> dm_env.TimeStep:
		"""Observation processing applied after action repeat consolidation."""

		# if timestep.first():
		# 	return dm_env.restart(timestep.observation)

		# reward = np.clip(timestep.reward, -self._max_abs_reward,
		# 				self._max_abs_reward)

		return timestep


	def observation_spec(self):
		# return self._environment.observation_spec()['pixels']
		return self._observation_spec
