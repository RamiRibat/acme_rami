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

"""A simple agent-environment training loop."""

import operator, time
from absl import logging
from termcolor import colored
from typing import List, Optional, Sequence

from acme import core
from acme.utils import counting, loggers, signals
from acme.utils import observers as observers_lib

import dm_env
from dm_env import specs
import numpy as np
import tree


class EnvironmentLoop(core.Worker):
	"""A simple RL environment loop.

	This takes `Environment` and `Actor` instances and coordinates their
	interaction. Agent is updated if `should_update=True`. This can be used as:

	loop = EnvironmentLoop(environment, actor)
	loop.run(num_episodes)

	A `Counter` instance can optionally be given in order to maintain counts
	between different Acme components. If not given a local Counter will be
	created to maintain counts between calls to the `run` method.

	A `Logger` instance can also be passed in order to control the output of the
	loop. If not given a platform-specific default logger will be used as defined
	by utils.loggers.make_default_logger. A string `label` can be passed to easily
	change the label associated with the default logger; this is ignored if a
	`Logger` instance is given.

	A list of 'Observer' instances can be specified to generate additional metrics
	to be logged by the logger. They have access to the 'Environment' instance,
	the current timestep datastruct and the current action.
	"""

	def __init__(
		self,
		environment: dm_env.Environment,
		actor: core.Actor,
		counter: Optional[counting.Counter] = None,
		logger: Optional[loggers.Logger] = None,
		should_update: bool = True,
		label: str = 'environment_loop',
		observers: Sequence[observers_lib.EnvLoopObserver] = (),
		iterative: Optional[bool] = True,
		# wait_eval: Optional[bool] = False,
	):
		# Internalize agent and environment.
		self._environment = environment
		self._actor = actor
		self._counter = counter or counting.Counter()
		self._logger = logger or loggers.make_default_logger(
			label, steps_key=self._counter.get_steps_key())
		self._should_update = should_update
		self._observers = observers

		self._label = label

		self._iterative = iterative


	def run_episode(self) -> loggers.LoggingData:
		"""Run one episode.

		Each episode is a loop which interacts first with the environment to get an
		observation and then give that observation to the agent in order to retrieve
		an action.

		Returns:
			An instance of `loggers.LoggingData`.
		"""
		# Reset any counts and start the environment.
		episode_start_time = time.time()
		select_action_durations: List[float] = []
		env_step_durations: List[float] = []
		episode_steps: int = 0

		# For evaluation, this keeps track of the total undiscounted reward
		# accumulated during the episode.
		episode_return = tree.map_structure(_generate_zeros_from_spec,
											self._environment.reward_spec())
		env_reset_start = time.time()
		timestep = self._environment.reset()
		env_reset_duration = time.time() - env_reset_start
		# Make the first observation.
		self._actor.observe_first(timestep)
		for observer in self._observers:
			# Initialize the observer with the current state of the env after reset
			# and the initial timestep.
			observer.observe_first(self._environment, timestep)

		# Run an episode.
		while not timestep.last():
			# Book-keeping.
			episode_steps += 1

			# Generate an action from the agent's policy.
			select_action_start = time.time()
			action = self._actor.select_action(timestep.observation)
			select_action_durations.append(time.time() - select_action_start)

			# Step the environment with the agent's selected action.
			env_step_start = time.time()
			timestep = self._environment.step(action)
			env_step_durations.append(time.time() - env_step_start)

			# Have the agent and observers observe the timestep.
			self._actor.observe(action, next_timestep=timestep)
			for observer in self._observers:
				# One environment step was completed. Observe the current state of the
				# environment, the current timestep and the action.
				observer.observe(self._environment, timestep, action)

			# # Give the actor the opportunity to update itself.
			# if self._should_update:
			# 	self._actor.update()

			# Equivalent to: episode_return += timestep.reward
			# We capture the return value because if timestep.reward is a JAX
			# DeviceArray, episode_return will not be mutated in-place. (In all other
			# cases, the returned episode_return will be the same object as the
			# argument episode_return.)
			episode_return = tree.map_structure(operator.iadd,
												episode_return,
												timestep.reward)

		# Record counts.
		counts = self._counter.increment(episodes=1, steps=episode_steps)

		# Collect the results and combine with counts.
		steps_per_second = episode_steps / (time.time() - episode_start_time)

		result = {
			'episode_length': episode_steps,
			'episode_return': episode_return,
			'steps_per_second': steps_per_second,
			'env_reset_duration_sec': env_reset_duration,
			'select_action_duration_sec': np.mean(select_action_durations),
			'env_step_duration_sec': np.mean(env_step_durations),
		}
		
		result.update(counts)
		for observer in self._observers:
			result.update(observer.get_metrics())

		return result


	def run_dummy_episode(self):
		# print(colored(f'EnvironmentLoop.run_dummy', 'dark_grey'))
		# Record counts.
		counts = self._counter.increment(episodes=0, steps=0)
		# print('counter: ', self._counter.get_counts())
		# print('logger: ', self._logger)
		
		result = {
			'episode_length': 0,
			'episode_return': 0,
			'steps_per_second': 0,
			'env_reset_duration_sec': 0,
			'select_action_duration_sec': 0,
			'env_step_duration_sec': 0,
		}
		result.update(counts)
		return result


	def run(
		self,
		num_episodes: Optional[int] = None,
		num_steps: Optional[int] = None,
	) -> int:
		"""Perform the run loop.

		Run the environment loop either for `num_episodes` episodes or for at
		least `num_steps` steps (the last episode is always run until completion,
		so the total number of steps may be slightly more than `num_steps`).
		At least one of these two arguments has to be None.

		Upon termination of an episode a new episode will be started. If the number
		of episodes and the number of steps are not given then this will interact
		with the environment infinitely.

		Args:
		num_episodes: number of episodes to run the loop for.
		num_steps: minimal number of steps to run the loop for.

		Returns:
		Actual number of steps the loop executed.

		Raises:
		ValueError: If both 'num_episodes' and 'num_steps' are not None.
		"""
		# print('\nenv_loop.run\n')

		if not (num_episodes is None or num_steps is None):
			raise ValueError('Either "num_episodes" or "num_steps" should be None.')

		def should_terminate(episode_count: int, step_count: int) -> bool:
			return (
				(num_episodes is not None and episode_count >= num_episodes) or
					(num_steps is not None and step_count >= num_steps)
			)

		episode_count: int = 0
		step_count: int = 0

		# print(colored(f'EnvironmentLoop.run ({self._label}): counter: {self._counter.get_counts()}', 'dark_grey'))
		
		# TODO(rami): make sure to run actor x 0 steps -> eval @ 0 before start
		
		# # Run eval @ 0 after 0 actor steps
		if 'eval_loop' in self._label:
			# if 'actor_steps' in self._counter.get_counts().keys():
			with signals.runtime_terminator(self._signal_handler):
				while not should_terminate(episode_count, step_count):
					episode_start = time.time()
					result = self.run_episode()
					result = {**result, **{'episode_duration': time.time() - episode_start}}
					episode_count += 1
					step_count += int(result['episode_length'])
					# Log the given episode results.
					self._logger.write(result)
		# if not run actor x 0 steps
		else:
			# init csv labels for evaluation
			if self._counter.get_steps_key() not in self._counter.get_counts().keys():
				episode_start = time.time()
				result = self.run_dummy_episode()
				result = {**result, **{'episode_duration': time.time() - episode_start}}
				episode_count += 0
				step_count += 0
				# Log the given episode results.
				self._logger.write(result)

			# iterative calls
			else:
				with signals.runtime_terminator(self._signal_handler):
					while not should_terminate(episode_count, step_count):
						episode_start = time.time()
						result = self.run_episode()
						result = {**result, **{'episode_duration': time.time() - episode_start}}
						episode_count += 1
						step_count += int(result['episode_length'])
						# Log the given episode results.
						self._logger.write(result)
		
		# print(colored(f'EnvironmentLoop.run ({self._label}): counter: {self._counter.get_counts()}', 'dark_grey'))

		return step_count
	

	# TODO(rami): Does this work?
	# Handle preemption signal.
	def _signal_handler(self):
		logging.info(
			colored(f'Caught SIGTERM: EnvironmentLoop({self._label}) forcing Logger close.', 'dark_grey')
		)
		# Close actor logger
		self._logger.close()




# class _EnvironmentLoop(core.Worker):
# 	"""A simple RL environment loop.

# 	This takes `Environment` and `Actor` instances and coordinates their
# 	interaction. Agent is updated if `should_update=True`. This can be used as:

# 	loop = EnvironmentLoop(environment, actor)
# 	loop.run(num_episodes)

# 	A `Counter` instance can optionally be given in order to maintain counts
# 	between different Acme components. If not given a local Counter will be
# 	created to maintain counts between calls to the `run` method.

# 	A `Logger` instance can also be passed in order to control the output of the
# 	loop. If not given a platform-specific default logger will be used as defined
# 	by utils.loggers.make_default_logger. A string `label` can be passed to easily
# 	change the label associated with the default logger; this is ignored if a
# 	`Logger` instance is given.

# 	A list of 'Observer' instances can be specified to generate additional metrics
# 	to be logged by the logger. They have access to the 'Environment' instance,
# 	the current timestep datastruct and the current action.
# 	"""

# 	def __init__(
# 		self,
# 		environment: dm_env.Environment,
# 		actor: core.Actor,
# 		counter: Optional[counting.Counter] = None,
# 		logger: Optional[loggers.Logger] = None,
# 		should_update: bool = True,
# 		label: str = 'environment_loop',
# 		observers: Sequence[observers_lib.EnvLoopObserver] = (),
# 		wait_eval: Optional[bool] = False,
# 	):
# 		# Internalize agent and environment.
# 		self._environment = environment
# 		self._actor = actor
# 		self._counter = counter or counting.Counter()
# 		self._logger = logger or loggers.make_default_logger(
# 			label, steps_key=self._counter.get_steps_key())
# 		self._should_update = should_update
# 		self._observers = observers

# 		self._label = label
		
# 		self._wait_eval = wait_eval


# 	def run_episode(self) -> loggers.LoggingData:
# 		"""Run one episode.

# 		Each episode is a loop which interacts first with the environment to get an
# 		observation and then give that observation to the agent in order to retrieve
# 		an action.

# 		Returns:
# 			An instance of `loggers.LoggingData`.
# 		"""
# 		# Reset any counts and start the environment.
# 		episode_start_time = time.time()
# 		select_action_durations: List[float] = []
# 		env_step_durations: List[float] = []
# 		episode_steps: int = 0

# 		# For evaluation, this keeps track of the total undiscounted reward
# 		# accumulated during the episode.
# 		episode_return = tree.map_structure(_generate_zeros_from_spec,
# 											self._environment.reward_spec())
# 		env_reset_start = time.time()
# 		timestep = self._environment.reset()
# 		env_reset_duration = time.time() - env_reset_start
# 		# Make the first observation.
# 		self._actor.observe_first(timestep)
# 		for observer in self._observers:
# 			# Initialize the observer with the current state of the env after reset
# 			# and the initial timestep.
# 			observer.observe_first(self._environment, timestep)

# 		# Run an episode.
# 		while not timestep.last():
# 			# Book-keeping.
# 			episode_steps += 1

# 			# Generate an action from the agent's policy.
# 			select_action_start = time.time()
# 			action = self._actor.select_action(timestep.observation)
# 			select_action_durations.append(time.time() - select_action_start)

# 			# Step the environment with the agent's selected action.
# 			env_step_start = time.time()
# 			timestep = self._environment.step(action)
# 			env_step_durations.append(time.time() - env_step_start)

# 			# Have the agent and observers observe the timestep.
# 			self._actor.observe(action, next_timestep=timestep)
# 			for observer in self._observers:
# 				# One environment step was completed. Observe the current state of the
# 				# environment, the current timestep and the action.
# 				observer.observe(self._environment, timestep, action)

# 			# Give the actor the opportunity to update itself.
# 			if self._should_update:
# 				self._actor.update()

# 			# Equivalent to: episode_return += timestep.reward
# 			# We capture the return value because if timestep.reward is a JAX
# 			# DeviceArray, episode_return will not be mutated in-place. (In all other
# 			# cases, the returned episode_return will be the same object as the
# 			# argument episode_return.)
# 			episode_return = tree.map_structure(operator.iadd,
# 												episode_return,
# 												timestep.reward)

# 		# Record counts.
# 		counts = self._counter.increment(episodes=1, steps=episode_steps)

# 		# Collect the results and combine with counts.
# 		steps_per_second = episode_steps / (time.time() - episode_start_time)

# 		result = {
# 			'episode_length': episode_steps,
# 			'episode_return': episode_return,
# 			'steps_per_second': steps_per_second,
# 			'env_reset_duration_sec': env_reset_duration,
# 			'select_action_duration_sec': np.mean(select_action_durations),
# 			'env_step_duration_sec': np.mean(env_step_durations),
# 		}
		
# 		result.update(counts)
# 		for observer in self._observers:
# 			result.update(observer.get_metrics())

# 		return result


# 	def run_dummy_episode(self):
# 		print(colored(f'EnvironmentLoop.run_dummy', 'dark_grey'))
# 		# Record counts.
# 		counts = self._counter.increment(episodes=0, steps=0)
# 		# print('counter: ', self._counter.get_counts())
# 		# print('logger: ', self._logger)
		
# 		result = {
# 			'episode_length': 0,
# 			'episode_return': 0,
# 			'steps_per_second': 0,
# 			'env_reset_duration_sec': 0,
# 			'select_action_duration_sec': 0,
# 			'env_step_duration_sec': 0,
# 		}
# 		result.update(counts)
# 		return result


# 	def run(
# 		self,
# 		num_episodes: Optional[int] = None,
# 		num_steps: Optional[int] = None,
# 	) -> int:
# 		"""Perform the run loop.

# 		Run the environment loop either for `num_episodes` episodes or for at
# 		least `num_steps` steps (the last episode is always run until completion,
# 		so the total number of steps may be slightly more than `num_steps`).
# 		At least one of these two arguments has to be None.

# 		Upon termination of an episode a new episode will be started. If the number
# 		of episodes and the number of steps are not given then this will interact
# 		with the environment infinitely.

# 		Args:
# 		num_episodes: number of episodes to run the loop for.
# 		num_steps: minimal number of steps to run the loop for.

# 		Returns:
# 		Actual number of steps the loop executed.

# 		Raises:
# 		ValueError: If both 'num_episodes' and 'num_steps' are not None.
# 		"""
# 		# print('\nenv_loop.run\n')

# 		if not (num_episodes is None or num_steps is None):
# 			raise ValueError('Either "num_episodes" or "num_steps" should be None.')

# 		def should_terminate(episode_count: int, step_count: int) -> bool:
# 			return (
# 				(num_episodes is not None and episode_count >= num_episodes) or
# 					(num_steps is not None and step_count >= num_steps)
# 			)

# 		episode_count: int = 0
# 		step_count: int = 0

# 		print(colored(f'EnvironmentLoop.run ({self._label}): counter: {self._counter.get_counts()}', 'dark_grey'))
		
# 		# TODO(rami): make sure to run actor x 0 steps -> eval @ 0 before start
# 		# if not run actor x 0 steps
# 		if 'actor_loop' in self._label:
# 			# init csv labels for evaluation
# 			if self._counter.get_steps_key() not in self._counter.get_counts().keys():
# 				episode_start = time.time()
# 				result = self.run_dummy_episode()
# 				result = {**result, **{'episode_duration': time.time() - episode_start}}
# 				episode_count += 0
# 				step_count += 0
# 				# Log the given episode results.
# 				self._logger.write(result)

# 				# non-iterative -> loop-to-end (distributed)
# 				if not self._iterative:
# 					if self._wait_eval:
# 						# Start running after evaluation been initialized
# 						while True:
# 							if 'evaluator' in self._counter.get_counts().keys():
# 								break

# 					with signals.runtime_terminator(self._signal_handler):
# 						while not should_terminate(episode_count, step_count):
# 							episode_start = time.time()
# 							result = self.run_episode()
# 							result = {**result, **{'episode_duration': time.time() - episode_start}}
# 							episode_count += 1
# 							step_count += int(result['episode_length'])
# 							# Log the given episode results.
# 							self._logger.write(result)

# 			else: # iterative calls
# 				with signals.runtime_terminator(self._signal_handler):
# 					while not should_terminate(episode_count, step_count):
# 						episode_start = time.time()
# 						result = self.run_episode()
# 						result = {**result, **{'episode_duration': time.time() - episode_start}}
# 						episode_count += 1
# 						step_count += int(result['episode_length'])
# 						# Log the given episode results.
# 						self._logger.write(result)

# 			# else:
# 			# 	with signals.runtime_terminator(self._signal_handler):
# 			# 		while not should_terminate(episode_count, step_count):
# 			# 			episode_start = time.time()
# 			# 			result = self.run_episode()
# 			# 			result = {**result, **{'episode_duration': time.time() - episode_start}}
# 			# 			episode_count += 1
# 			# 			step_count += int(result['episode_length'])
# 			# 			# Log the given episode results.
# 			# 			self._logger.write(result)

			
		
# 		# Run eval @ 0 after 0 actor steps
# 		if 'eval_loop' in self._label:
# 			# if 'actor_steps' in self._counter.get_counts().keys():
# 			with signals.runtime_terminator(self._signal_handler):
# 				while not should_terminate(episode_count, step_count):
# 					episode_start = time.time()
# 					result = self.run_episode()
# 					result = {**result, **{'episode_duration': time.time() - episode_start}}
# 					episode_count += 1
# 					step_count += int(result['episode_length'])
# 					# Log the given episode results.
# 					self._logger.write(result)
		
# 		print(colored(f'EnvironmentLoop.run ({self._label}): counter: {self._counter.get_counts()}', 'dark_grey'))

# 		return step_count
	

# 	# TODO(rami): Does this work?
# 	# Handle preemption signal.
# 	def _signal_handler(self):
# 		logging.info(
# 			colored(f'Caught SIGTERM: EnvironmentLoop({self._label}) forcing Logger close.', 'dark_grey')
# 		)
# 		# Close actor logger
# 		self._logger.close()





def _generate_zeros_from_spec(spec: specs.Array) -> np.ndarray:
  return np.zeros(spec.shape, spec.dtype)
