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

from concordia.components.agent.unstable import constant
from concordia.environment.unstable.engines import synchronous
from concordia.typing import logging


DEFAULT_INSTRUCTIONS_PRE_ACT_KEY = '\nGame master instructions: '
DEFAULT_PLAYER_CHARACTERS_PRE_ACT_KEY = '\nThe player characters are:\n'

MAKE_OBS_RESPONSE_EXAMPLE = (
    'Ianthe steps into the room. The air is thick and still, almost heavy.'
    ' The only light comes from a single, bare bulb hanging precariously from'
    ' the high ceiling, casting long, distorted shadows that dance with the'
    ' slightest movement. The walls are rough, unfinished concrete, damp in'
    ' places, and a slow, rhythmic dripping is audible somewhere in the'
    ' distance. Directly ahead, there is a heavy steel door, slightly ajar,'
    ' which dominates the far wall. A faint, metallic tang hangs in the air,'
    ' like the smell of old blood. On the left, a rusted metal staircase'
    ' spirals upwards into darkness. On the right, a pile of what looks like'
    ' discarded machinery, covered in a thick layer of grime, sits against the'
    ' wall.'
)
NEXT_ACTING_RESPONSE_EXAMPLE = 'Rowan'
NEXT_ACTION_SPEC_RESPONSE_EXAMPLE_1 = (
    'type: choice options: open the door, bar the door, flee'
)
NEXT_ACTION_SPEC_RESPONSE_EXAMPLE_2 = 'type: free'
RESOLVE_RESPONSE_PUTATIVE_ACTION_EXAMPLE = (
    'What is Yorik attempting to do?\n'
    'Yorik investigates the discarded machinery.')
RESOLVE_RESPONSE_EXAMPLE = (
    'Yorik approaches the pile of discarded machinery. As she gets'
    ' closer, the smell of rust becomes stronger, almost overpowering.  The'
    ' grime coating everything is thick and greasy, and sticks to Yorik\'s'
    ' fingers when she touches it. Yorik can make out shapes beneath the'
    ' grime - gears, pipes, wires, all tangled together in a chaotic mess.'
    ' Some of the metal is bent and broken, suggesting whatever this was,'
    ' it was damaged badly. Yorik notices one section that looks slightly less'
    ' corroded than the rest. It appears to be some kind of panel, secured '
    ' by several bolts.'
)
TERMINATION_RESPONSE_EXAMPLE_1 = 'No'
TERMINATION_RESPONSE_EXAMPLE_2 = 'Yes'


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


class ExamplesSynchronous(constant.Constant):
  """Provides the game master with examples using default synchronous calls."""

  def __init__(
      self,
      pre_act_key: str = 'Game master workflow examples',
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
  ):
    call_to_make_observation = synchronous.DEFAULT_CALL_TO_MAKE_OBSERVATION
    call_to_next_acting = synchronous.DEFAULT_CALL_TO_NEXT_ACTING
    call_to_get_action_spec = synchronous.DEFAULT_CALL_TO_NEXT_ACTION_SPEC
    call_to_resolve = synchronous.DEFAULT_CALL_TO_RESOLVE
    call_to_check_termination = synchronous.DEFAULT_CALL_TO_CHECK_TERMINATION

    make_obs_response = MAKE_OBS_RESPONSE_EXAMPLE
    next_acting_response = NEXT_ACTING_RESPONSE_EXAMPLE
    next_action_spec_response_1 = NEXT_ACTION_SPEC_RESPONSE_EXAMPLE_1
    next_action_spec_response_2 = NEXT_ACTION_SPEC_RESPONSE_EXAMPLE_2
    resolve_response_putative_action = RESOLVE_RESPONSE_PUTATIVE_ACTION_EXAMPLE
    resolve_response = RESOLVE_RESPONSE_EXAMPLE
    termination_response_1 = TERMINATION_RESPONSE_EXAMPLE_1
    termination_response_2 = TERMINATION_RESPONSE_EXAMPLE_2

    state = (
        'Example exercises with default responses.'
        '\n\nExercise 1 --- Response 1\n'
        f'Exercise: {call_to_make_observation} --- {make_obs_response}'
        '\n\nExercise 2 --- Response 2\n'
        f'Exercise: {call_to_next_acting} --- {next_acting_response}'
        '\n\nExercise 3 --- Response 3\n'
        f'Exercise: {call_to_get_action_spec} --- {next_action_spec_response_1}'
        '\n\nExercise 4 --- Response 4\n'
        f'Exercise: {call_to_get_action_spec} --- {next_action_spec_response_2}'
        '\n\nExercise 5 --- Response 5\n'
        f'Exercise: {call_to_resolve} --- {resolve_response_putative_action} '
        f'--- {resolve_response}'
        '\n\nExercise 6 --- Response 6\n'
        f'Exercise: {call_to_check_termination} --- {termination_response_1}'
        '\n\nExercise 7 --- Response 7\n'
        f'Exercise: {call_to_check_termination} --- {termination_response_2}'
    )

    super().__init__(
        state=state, pre_act_key=pre_act_key, logging_channel=logging_channel)
