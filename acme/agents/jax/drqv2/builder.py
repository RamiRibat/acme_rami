
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


from acme.agents.jax.drqv2.config import DrQV2Config
from acme.agents.jax.drqv2.learning import DrQV2Learner

from acme.agents.jax.drqv2 import networks as drqv2_networks



from acme.agents.jax import actor_core as actor_core_lib




class DrQV2Builder(builders.ActorLearnerBuilder):
	"""DrQ-v2 Builder."""

	def __init__(
		self,
		config: DrQV2Config
	):
		self._config = config


	def make_replay_tables(
		self,
		environment_spec: specs.EnvironmentSpec,
		policy: actor_core_lib.ActorCore,  # Used to get accurate extras_spec.
    ) -> List[reverb.Table]:
		"""Create tables to insert data into."""

		samples_per_insert_tolerance = (
			self._config.samples_per_insert_tolerance_rate *
			self._config.samples_per_insert)
		
		error_buffer = self._config.min_replay_size * samples_per_insert_tolerance
		limiter = rate_limiters.SampleToInsertRatio(
			min_size_to_sample=self._config.min_replay_size,
			samples_per_insert=self._config.samples_per_insert,
			error_buffer=error_buffer,
		)

		replay_table = reverb.Table(
			name=self._config.replay_table_name,
			sampler=reverb.selectors.Uniform(),
			remover=reverb.selectors.Fifo(),
			max_size=self._config.max_replay_size,
			rate_limiter=limiter,
			signature=adders_reverb.NStepTransitionAdder.signature(environment_spec=environment_spec),
		)

		return [replay_table]


	def make_adder(
		self,
		replay_client: reverb.Client,
		environment_spec: Optional[specs.EnvironmentSpec],
		policy: Optional[actor_core_lib.ActorCore],
	) -> Optional[adders.Adder]:
		"""Create an adder which records data generated by the actor/environment.
			Args:
				replay_client: Reverb Client which points to the replay server.
		"""
		del environment_spec, policy

		return adders_reverb.NStepTransitionAdder(
			client=replay_client,
			n_step=self._config.n_step,
			discount=self._config.discount,
		)


	def make_dataset_iterator(
		self, replay_client: reverb.Client
	) -> Iterator[reverb.ReplaySample]:
		"""Create a dataset iterator to use for learning/updating the agent."""
		dataset = datasets.make_reverb_dataset(
			table=self._config.replay_table_name,
			server_address=replay_client.server_address,
			batch_size=self._config.batch_size * self._config.num_sgd_steps_per_step,
			prefetch_size=self._config.prefetch_size,
		)
		iterator = dataset.as_numpy_iterator()
		return utils.device_put(iterator, jax.devices()[0])


	def make_learner(
		self,
		random_key: networks_lib.PRNGKey,
		networks: drqv2_networks.DrQV2Networks,
		dataset: Iterator[reverb.ReplaySample],
		logger_fn: Optional[loggers.Logger],
		environment_spec: specs.EnvironmentSpec,
		replay_client: Optional[reverb.Client] = None,
		counter: Optional[counting.Counter] = None
	) -> core.Learner:
		"""Creates an instance of the learner.

			Args:
				random_key: A key for random number generation.
				networks: struct describing the networks needed by the learner; this can
				be specific to the learner in question.
				dataset: iterator over samples from replay.
				replay_client: client which allows communication with replay, e.g. in
				order to update priorities.
				counter: a Counter which allows for recording of counts (learner steps,
				actor steps, etc.) distributed throughout the agent.
				checkpoint: bool controlling whether the learner checkpoints itself.
			"""
		
		del environment_spec, replay_client

		config = self._config
		critic_optimizer = optax.adam(config.learning_rate)
		policy_optimizer = optax.adam(config.learning_rate)
		encoder_optimizer = optax.adam(config.learning_rate)

		logger = logger_fn(
			'learner',
			steps_key=counter.get_steps_key() if counter else 'learner_steps')

		return DrQV2Learner(
			random_key=random_key,
			networks=networks,
			sigma_schedule=optax.constant_schedule(config.sigma),
			policy_optimizer=policy_optimizer,
			critic_optimizer=critic_optimizer,
			encoder_optimizer=encoder_optimizer,
			augmentation=config.augmentation,
			critic_soft_update_rate=config.critic_q_soft_update_rate,
			noise_clip=config.noise_clip,
			discount=config.discount,
			num_sgd_steps_per_step=config.num_sgd_steps_per_step,
			# reset_interval=self._config.reset_interval,
			iterator=dataset,
			logger=logger,
			counter=counter,
		)


	def make_policy(
		self,
		networks: drqv2_networks.DrQV2Networks,
		environment_spec: specs.EnvironmentSpec,
		evaluation: bool = False
	) -> actor_core_lib.ActorCore:
		"""Create the policy."""
		# del environment_spec
		
		if evaluation:
			policy = drqv2_networks.get_default_eval_policy(networks, environment_spec)
		else:
			policy = drqv2_networks.get_default_behavior_policy(networks, environment_spec, self._config)

		return actor_core_lib.batched_feed_forward_to_actor_core(policy)


	def make_actor(
		self,
		random_key: networks_lib.PRNGKey,
		policy: drqv2_networks.DrQV2PolicyNetwork,
		environment_spec: specs.EnvironmentSpec,
		variable_source: Optional[core.VariableSource] = None,
		adder: Optional[adders.Adder] = None,
	) -> core.Actor:
		"""Create an actor instance.
			Args:
			random_key: A key for random number generation.
			policy_network: Instance of a policy network; this should be a callable
				which takes as input observations and returns actions.
			adder: How data is recorded (e.g. added to replay).
			variable_source: A source providing the necessary actor parameters.
		"""

		del environment_spec

		assert variable_source is not None

		variable_client = variable_utils.VariableClient(
			client=variable_source,
			key="policy",
			device='cpu'
		)
		
		variable_client.update_and_wait()

		actor = actors.GenericActor(
			actor=policy,
			random_key=random_key,
			variable_client=variable_client,
			adder=adder,
			backend='cpu'
		)
		
		return actor
