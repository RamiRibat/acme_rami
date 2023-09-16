import os, yaml

import numpy as np

from dm_control import suite

from acme import wrappers


path = os.path.join(os.path.dirname(os.getcwd())+'/config_v.yaml')
config = yaml.safe_load(open(path))

for task in config['control']['extra']['tasks']:
  domain_name, task_name = task.split(':')
  print(f"\ndomain: {domain_name} | task: {task_name}")
  env = suite.load(domain_name, task_name)
  print(f"obs_spec: {env.observation_spec()}")
  env = wrappers.ConcatObservationWrapper(env)

# # Load one task:
# env = suite.load(domain_name="cartpole", task_name="swingup")

# # Iterate over a task set:
# for domain_name, task_name in suite.BENCHMARKING:
#   print(f"domain: {domain_name} | task: {task_name}")
#   env = suite.load(domain_name, task_name)
#   print(f"obs_spec: {env.observation_spec()}")

# # Step through an episode and print out reward, discount and observation.
# action_spec = env.action_spec()
# time_step = env.reset()
# while not time_step.last():
#   action = np.random.uniform(action_spec.minimum,
#                              action_spec.maximum,
#                              size=action_spec.shape)
#   time_step = env.step(action)
#   print(time_step.reward, time_step.discount, time_step.observation)