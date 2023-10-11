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

"""Implementations of a DrQv2 agent."""

from acme.agents.jax.drqv2.config import DrQV2Config
from acme.agents.jax.drqv2.builder import DrQV2Builder
from acme.agents.jax.drqv2.learning import DrQV2Learner

from acme.agents.jax.drqv2.networks import DrQV2Networks
from acme.agents.jax.drqv2.networks import get_default_behavior_policy
from acme.agents.jax.drqv2.networks import get_default_eval_policy
from acme.agents.jax.drqv2.networks import make_networks

