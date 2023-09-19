"""
Author: https://github.com/ethanluoyc
"""

from absl import app
from acme import specs
from acme import wrappers
from acme.jax import experiments
from acme.utils import loggers
from acme.wrappers import mujoco
from dm_control import suite
import dm_env

# import drq_v2
# import networks as networks_lib
from acme.agents.jax import drqv2

import tensorflow as tf
import jax



def make_experiment_logger(label, steps_key, task_instance=0):
  del task_instance
  return loggers.make_default_logger(
      label, save_data=False, steps_key=steps_key)


def make_environment(domain: str,
                     task: str,
                     seed=None,
                     from_pixels: bool = False,
                     num_action_repeats: int = 1,
                     frames_to_stack: int = 0,
                     camera_id: int = 0) -> dm_env.Environment:
  """Create a dm_control suite environment."""
  environment = suite.load(domain, task, task_kwargs={"random": seed})
  if from_pixels:
    environment = mujoco.MujocoPixelWrapper(environment, camera_id=camera_id)
  else:
    environment = wrappers.ConcatObservationWrapper(environment)
  if num_action_repeats > 1:
    environment = wrappers.ActionRepeatWrapper(environment, num_action_repeats)
  if frames_to_stack > 0:
    assert from_pixels, "frame stack for state not supported"
    environment = wrappers.FrameStackingWrapper(
        environment, frames_to_stack, flatten=True)
  environment = wrappers.CanonicalSpecWrapper(environment, clip=True)
  environment = wrappers.SinglePrecisionWrapper(environment)
  return environment


def main(_):
  tf.config.set_visible_devices([], 'GPU')

  environment_factory = lambda seed: make_environment(
      domain='cheetah',
      task='run',
      seed=seed,
      from_pixels=True,
      num_action_repeats=2,
      frames_to_stack=3,
      camera_id=0)

  num_steps = int(1.5e6)

  environment = environment_factory(0)
  environment_spec = specs.make_environment_spec(environment)
  network_factory = drqv2.make_networks

  drq_config = drqv2.DrQV2Config()
  policy_factory = lambda n: drqv2.get_default_behavior_policy(
      n, environment_spec.actions, drq_config.sigma)
  eval_policy_factory = lambda n: drqv2.get_default_behavior_policy(
      n, environment_spec.actions, 0.0)

  # Construct the agent.
  builder = drqv2.DrQV2Builder(drq_config)

  experiment = experiments.Config(
      builder=builder,
      network_factory=network_factory,
      policy_network_factory=policy_factory,
      environment_factory=environment_factory,
      eval_policy_network_factory=eval_policy_factory,
      environment_spec=environment_spec,
      observers=(),
      seed=0,
      logger_factory=make_experiment_logger,
      max_number_of_steps=num_steps)

  experiments.run_experiment(
      experiment, eval_every=int(1e4), num_eval_episodes=5)


if __name__ == '__main__':
  jax.config.config_with_absl()
  app.run(main)
