# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Library of components specifically for generative agents."""

from concordia.components.agent.v2 import action_spec_ignored
from concordia.components.agent.v2 import all_similar_memories
from concordia.components.agent.v2 import constant
from concordia.components.agent.v2 import identity
from concordia.components.agent.v2 import legacy_act_component
from concordia.components.agent.v2 import memory_component
from concordia.components.agent.v2 import no_op_context_processor
from concordia.components.agent.v2 import observation
from concordia.components.agent.v2 import person_by_situation
from concordia.components.agent.v2 import plan
from concordia.components.agent.v2 import report_function
from concordia.components.agent.v2 import self_perception
from concordia.components.agent.v2 import simple_act_component
from concordia.components.agent.v2 import situation_perception
