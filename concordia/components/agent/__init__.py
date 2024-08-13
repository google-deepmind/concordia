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

from concordia.components.agent import action_spec_ignored
from concordia.components.agent import all_similar_memories
from concordia.components.agent import concat_act_component
from concordia.components.agent import constant
from concordia.components.agent import instructions
from concordia.components.agent import justify_recent_voluntary_actions
from concordia.components.agent import legacy_act_component
from concordia.components.agent import memory_component
from concordia.components.agent import no_op_context_processor
from concordia.components.agent import observation
from concordia.components.agent import plan
from concordia.components.agent import question_of_query_associated_memories
from concordia.components.agent import question_of_recent_memories
from concordia.components.agent import relationships
from concordia.components.agent import report_function
from concordia.components.agent import scheduled_hint
from concordia.components.agent import simple_act_component
