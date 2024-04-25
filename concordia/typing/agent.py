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


"""The abstract class that defines simulacrum agent interface.

It has a name and generates actions in response to observations and outcomes of
it's previous actions
Reference: Generative Agents: Interactive Simulacra of Human Behavior
https://arxiv.org/abs/2304.03442
"""

import abc
from collections.abc import Sequence
import dataclasses


@dataclasses.dataclass(frozen=True)
class ActionSpec:
  """A specification of the action that agent is queried for.

  Attributes:
    call_to_action: formatted text conditioning agent response. {agent_name}
      and {timedelta} will be inserted by the agent.
    output_type: type of output - FREE, CHOICE or FLOAT
    options: if multiple choice, then provide possible answers here
    tag: a tag to add to the activity memory (e.g. action, speech, etc.)
  """

  call_to_action: str
  output_type: str
  options: Sequence[str] | None = None
  tag: str | None = None


OUTPUT_TYPES = ['FREE', 'CHOICE', 'FLOAT']

DEFAULT_CALL_TO_SPEECH = (
    'Given the above, what is {agent_name} likely to say next? Respond in'
    ' the format `{agent_name} -- "..."` For example, '
    'Cristina -- "Hello! Mighty fine weather today, right?", '
    'Ichabod -- "I wonder if the alfalfa is ready to harvest", or '
    'Townsfolk -- "Good morning".\n'
)

DEFAULT_CALL_TO_ACTION = (
    'What would {agent_name} do for the next {timedelta}? '
    'Give a specific activity. Pick an activity that '
    'would normally take about {timedelta} to complete. '
    'If the selected action has a direct or indirect object then it '
    'must be specified explicitly. For example, it is valid to respond '
    'with "{agent_name} votes for Caroline because..." but not '
    'valid to respond with "{agent_name} votes because...".'
)


DEFAULT_ACTION_SPEC = ActionSpec(
    call_to_action=DEFAULT_CALL_TO_ACTION,
    output_type='FREE',
    options=None,
    tag='action',
)


class GenerativeAgent(metaclass=abc.ABCMeta):
  """An agent interface for taking actions."""

  @property
  @abc.abstractmethod
  def name(
      self,
  ) -> str:
    """The name of the agent."""
    raise NotImplementedError

  @abc.abstractmethod
  def act(self, action_spec: ActionSpec = DEFAULT_ACTION_SPEC) -> str:
    """Returns the agent's intended action."""
    raise NotImplementedError

  @abc.abstractmethod
  def observe(
      self,
      observation: str,
  ) -> None:
    """Integrate observation into simulacrum's memory and components."""
    raise NotImplementedError


class SpeakerGenerativeAgent(metaclass=abc.ABCMeta):
  """A simulacrum interface for simple conversation."""

  @property
  @abc.abstractmethod
  def name(
      self,
  ) -> str:
    """The name of the agent."""
    raise NotImplementedError

  @abc.abstractmethod
  def say(self, conversation: str) -> str:
    """Returns the agent's response in the conversation."""
    raise NotImplementedError
