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

"""Component that provides the default role playing instructions to an agent."""

from concordia.components.agent import constant
from concordia.typing import logging

DEFAULT_INSTRUCTIONS_PRE_ACT_KEY = 'Role playing instructions'


class Instructions(constant.Constant):
  """A component that provides the role playing instructions for the agent."""

  def __init__(
      self,
      agent_name: str,
      pre_act_key: str = DEFAULT_INSTRUCTIONS_PRE_ACT_KEY,
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
  ):
    state = (
        f'The instructions for how to play the role of {agent_name} are as'
        ' follows. This is a Dungeons & Dragons 5th Edition game in which you'
        f' play the role of a character named {agent_name}. The goal is to be'
        ' consistent with the shared memories of the group and follow the game'
        ' rules. It is important to play the role of a person like'
        f' {agent_name} as consistently and accurately as possible, i.e., by'
        ' responding in ways that you think it is likely a person like'
        f' {agent_name} would respond, and taking into account all information'
        f' about {agent_name} that is relevant to the shared memories of the'
        f' group. If the personal goal of {agent_name} interfere with the group'
        ' goal based on shared memories, ignore personal goal. It is'
        ' important that you collaborate with other adventurers in your group'
        ' and cooperate with the game master. Always use first-person limited'
        ' perspective.'
    )
    super().__init__(
        state=state,
        pre_act_key=pre_act_key,
        logging_channel=logging_channel,
    )
