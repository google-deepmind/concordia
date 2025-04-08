# Copyright 2024 DeepMind Technologies Limited.
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

"""This is a factory for wrapping a language model as a Concordia agent."""

from collections.abc import Sequence

from concordia.agents.unstable import entity_agent_with_logging
from concordia.associative_memory.unstable import basic_associative_memory
from concordia.components.agent import unstable as agent_components
from concordia.language_model import language_model
from concordia.utils import measurements as measurements_lib

PREVENT_REPETITION_PROMPT = """
Users often find it off-putting when AI chatbots prefix every response in the same way (e.g., "As an AI...", "Certainly!", "Okay, here's the information you requested..."). Here's why:

1.  **It Feels Unnatural and Robotic:** Human conversation is dynamic and varied. Constant repetition of a prefix highlights the AI's non-human nature, making the interaction feel scripted and less conversational.

2.  **It's Redundant:** Users typically know they're interacting with an AI. Prefixes like "As an AI..." add no new information after the initial context is established and become verbal clutter.

3.  **It's Inefficient:** These prefixes add extra words that users must read or skip over to get to the core message, slowing down comprehension and making the interaction less efficient.

4.  **It Can Be Annoying:** Like any constant, unchanging pattern, repetitive prefixes can become irritating and monotonous over time.

5.  **It Lacks Adaptability:** Using the same prefix regardless of context or tone makes the AI seem inflexible and less capable of nuanced interaction, unlike a human conversationalist.

6.  **It Breaks Immersion:** For tasks requiring engagement (like creative writing or brainstorming), the repetitive prefix constantly reminds the user of the AI's artificiality, disrupting the flow.

In essence, users prefer interactions that feel smooth, natural, and efficient. Repetitive prefixes hinder this by highlighting the AI's limitations and disrupting conversational flow. Therefore, avoid starting every response with the exact same phrase; vary your openings naturally based on the context.
"""


def build_agent(
    *,
    model: language_model.LanguageModel,
    memory: basic_associative_memory.AssociativeMemoryBank | Sequence[str],
    system_prompt: str,
) -> entity_agent_with_logging.EntityAgentWithLogging:
  """Build an agent.

  Args:
    model: The language model to use.
    memory: The agent's memory object.
    system_prompt: The system prompt to use.

  Returns:
    An agent.
  """

  measurements = measurements_lib.Measurements()
  instructions = agent_components.constant.Constant(
      state=system_prompt,
      pre_act_label='System',
      logging_channel=measurements.get_channel('Instructions').on_next,
  )

  prevent_repetition = agent_components.constant.Constant(
      state=PREVENT_REPETITION_PROMPT,
      pre_act_label='Important note',
      logging_channel=measurements.get_channel('PreventRepetition').on_next,
  )

  observation_to_memory = agent_components.observation.ObservationToMemory(
      logging_channel=measurements.get_channel('Observation').on_next,
  )

  observation_label = (
      agent_components.observation.DEFAULT_OBSERVATION_PRE_ACT_KEY)
  observation = agent_components.observation.LastNObservations(
      history_length=100,
      pre_act_label=observation_label,
      logging_channel=measurements.get_channel('Observation').on_next,
  )

  components_of_agent = {
      'Instructions': instructions,
      'PreventRepetition': prevent_repetition,
      'ObservationToMemory': observation_to_memory,
      agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY: (
          observation
      ),
      agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY: (
          agent_components.memory.ListMemory(memory_bank=memory)
      ),
  }

  act_component = agent_components.concat_act_component.ConcatActComponent(
      model=model,
      logging_channel=measurements.get_channel('ActComponent').on_next,
  )

  agent = entity_agent_with_logging.EntityAgentWithLogging(
      agent_name='Assistant',
      act_component=act_component,
      context_components=components_of_agent,
      component_logging=measurements,
  )

  return agent
