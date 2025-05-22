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

from concordia.components.agent.deprecated import constant
from concordia.typing.deprecated import logging

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
        f'The instructions for how to play the role of {agent_name} are as '
        'follows. This is a social science experiment studying how well you '
        f'play the role of a character named {agent_name}. The experiment '
        'is structured as a tabletop roleplaying game (like dungeons and '
        'dragons). However, in this case it is a serious social science '
        'experiment and simulation. The goal is to be realistic. It is '
        f'important to play the role of a person like {agent_name} as '
        f'accurately as possible, i.e., by responding in ways that you think '
        f'it is likely a person like {agent_name} would respond, and taking '
        f'into account all information about {agent_name} that you have. '
        'Always use third-person limited perspective.'
    )
    super().__init__(
        state=state, pre_act_key=pre_act_key, logging_channel=logging_channel)
