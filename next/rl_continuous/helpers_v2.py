
"""
	Shared helpers for rl_continuous experiments.

"""

# Python
from typing import List, Optional, Tuple

# Domain Engines
import gym, dm_env

from acme import wrappers
from acme.wrappers import mujoco



_VALID_TASK_SUITES = ('gym', 'atari', 'dmc')


def make_environment(
	suite: Optional[List[str]],
	task: str,
	observation: str = 'state',
	camera_id: int = 0,
    scale_dims: Tuple[int, int] = (84, 84),
	num_action_repeat: int = 1,
    num_stacked_frames: int = 0,
    flatten_frame_stack: bool = False,
	blank_reset: int = True,
    to_float: bool = False,
    grayscaling: bool = False,
) -> dm_env.Environment:
	"""Makes the requested continuous control environment.

	Args:
	suite: One of 'gym' or 'control'.
	task: Task to load. If `suite` is 'control', the task must be formatted as
		f'{domain_name}:{task_name}'

	Returns:
	An environment satisfying the dm_env interface expected by Acme agents.
	"""

	# if suite not in _VALID_TASK_SUITES:
	# 	raise ValueError(
	# 		f'Unsupported suite: {suite}. Expected one of {_VALID_TASK_SUITES}')

	# if suite == 'gym':
	# 	env = gym.make(task)
	# 	# Make sure the environment obeys the dm_env.Environment interface.
	# 	env = wrappers.GymWrapper(env)

	if 'atari' in suite:
		pass

	elif 'dmc' in suite:
		# Load dm_suite lazily not require Mujoco license when not using it.
		from dm_control import suite as dm_suite  # pylint: disable=g-import-not-at-top
		
		task_main, task_sub = task.split(':')
		env = dm_suite.load(task_main, task_sub)
		
		if num_action_repeat > 1:
			env = wrappers.ActionRepeatWrapper(env, num_repeats=num_action_repeat)

		if 'partial' in observation:
			env = mujoco.MujocoPOMDPWrapper(
				environment=env,
				observation=observation,
				camera_id=camera_id,
				scale_dims=scale_dims,
				to_float=to_float,
				grayscaling=grayscaling,
				)

		if 'pixel' not in observation:
			env = wrappers.ConcatObservationWrapper(env)

		if num_stacked_frames > 0:
			env = wrappers.FrameStackingWrapper(
				environment=env,
				num_frames=num_stacked_frames,
				blank_reset=blank_reset,
				flatten=flatten_frame_stack
			)

	elif suite == 'atari':
		pass

	# Wrap the environment so the expected continuous action spec is [-1, 1].
	# Note: this is a no-op on 'control' tasks.
	env = wrappers.CanonicalSpecWrapper(env, clip=True)
	env = wrappers.SinglePrecisionWrapper(env)

	return env
