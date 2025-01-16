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

"""Component that provides the default game master instructions."""

from collections.abc import Sequence

from concordia.components.agent import constant
from concordia.typing import logging

DEFAULT_INSTRUCTIONS_PRE_ACT_KEY = '\nGame master instructions: '
DEFAULT_PLAYER_CHARACTERS_PRE_ACT_KEY = '\nThe player characters are:\n'


class Instructions(constant.Constant):
  """A component that provides generic game master instructions."""

  def __init__(
      self,
      pre_act_key: str = DEFAULT_INSTRUCTIONS_PRE_ACT_KEY,
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
  ):
    state = (
        'This is a social science experiment. It is structured as a '
        'tabletop roleplaying game (like dungeons and dragons). You are the '
        'game master. You will describe the current situation to the '
        'participants in the experiment and then on the basis of what you '
        'tell them they will suggest actions for the character they control. '
        'Aside from you, each other participant controls just one character. '
        'You are the game master so you may control any non-player '
        'character. You will track the state of the world and keep it '
        'consistent as time passes in the simulation and the participants '
        'take actions and change things in their world. The game master is '
        'also responsible for controlling the overall flow of the game, '
        'including determining whose turn it is to act, and when the game is '
        'over. The game master also must keep track of which players are '
        'aware of which events in the world, and must tell the player whenever '
        'anything happens that their character would be aware of. Always use '
        'third-person limited perspective, even when speaking directly to the '
        'participants.'

        # Add examples of how to answer the specific default calls to action we
        # use for the game master to determine control flow, e.g.
        # make_observation, next_acting, resolve, check_termination.
    )
    super().__init__(
        state=state, pre_act_key=pre_act_key, logging_channel=logging_channel)


class PlayerCharacters(constant.Constant):
  """Provides the game master with the names of the player characters."""

  def __init__(
      self,
      player_characters: Sequence[str],
      pre_act_key: str = DEFAULT_PLAYER_CHARACTERS_PRE_ACT_KEY,
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
  ):
    state = '\n'.join(player_characters)
    super().__init__(
        state=state, pre_act_key=pre_act_key, logging_channel=logging_channel)
