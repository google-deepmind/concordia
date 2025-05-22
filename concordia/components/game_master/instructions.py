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
import random

from concordia.components.agent import constant
from concordia.environment.engines import sequential
from concordia.typing import logging


DEFAULT_INSTRUCTIONS_PRE_ACT_LABEL = 'Game master instructions: '
DEFAULT_PLAYER_CHARACTERS_PRE_ACT_LABEL = 'The player characters are:\n'

EXAMPLE_NAMES = (
    'Ianthe',
    'Rowan',
    'Kerensa',
    'Morwenna',
    'Yorik',
)

MAKE_OBS_RESPONSE_EXAMPLE = (
    '{name} steps into the room. The air is thick and still, almost heavy.'
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
NEXT_ACTING_RESPONSE_EXAMPLE = '{name}'
NEXT_ACTION_SPEC_RESPONSE_EXAMPLE_1 = (
    'prompt: What would {name} do?;;type: choice;;options: '
    'open the door, bar the door, flee'
)
NEXT_ACTION_SPEC_RESPONSE_EXAMPLE_2 = (
    'prompt: What would {name} say?;;type: free')
RESOLVE_RESPONSE_PUTATIVE_ACTION_EXAMPLE = (
    'What is {name} attempting to do?\n'
    '{name} opens the enchanted storybook.')
RESOLVE_RESPONSE_EXAMPLE = (
    '{name} opens the colorful storybook. As she turns the pages, she notices '
    'the delightful scent of cinnamon and vanilla fills the air, warm and '
    'inviting. Sparkling illustrations twinkle merrily on the '
    'pages, and leave a pleasant tingling on {name}\'s fingers when she '
    'touches them. {name} notices one section that glows slightly more '
    'brightly than the rest. It appears to be some kind of special '
    'chapter, marked by several golden ribbons.'
)
TERMINATION_RESPONSE_EXAMPLE_1 = 'No'
TERMINATION_RESPONSE_EXAMPLE_2 = 'Yes'


class Instructions(constant.Constant):
  """A component that provides generic game master instructions."""

  def __init__(
      self,
      pre_act_label: str = DEFAULT_INSTRUCTIONS_PRE_ACT_LABEL,
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
        'participants. Try to ensure the story always moves forward and never '
        'gets stuck, even if the participants make repetitive choices.'
    )
    super().__init__(state=state, pre_act_label=pre_act_label)


class PlayerCharacters(constant.Constant):
  """Provides the game master with the names of the player characters."""

  def __init__(
      self,
      player_characters: Sequence[str],
      pre_act_label: str = DEFAULT_PLAYER_CHARACTERS_PRE_ACT_LABEL,
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
  ):
    state = '\n'.join(player_characters) + '\n'
    super().__init__(state=state, pre_act_label=pre_act_label)


class ExamplesSynchronous(constant.Constant):
  """Provides the game master with examples using default synchronous calls."""

  def __init__(
      self,
      exercises: Sequence[str] = tuple(['make_observation',
                                        'next_acting',
                                        'next_action_spec_1',
                                        'next_action_spec_2',
                                        'resolve',
                                        'check_termination_1',
                                        'check_termination_2']),
      pre_act_label: str = 'Game master workflow examples',
      rnd: random.Random | None = None,
  ):
    if rnd is None:
      rnd = random.Random()
    names = rnd.sample(EXAMPLE_NAMES, len(EXAMPLE_NAMES))

    name_for_obs = names[0]
    name_for_next_acting = names[1]
    name_for_next_action_spec_1 = names[2]
    name_for_next_action_spec_2 = names[3]
    name_for_resolve = names[4]

    call_to_make_observation = (
        sequential.DEFAULT_CALL_TO_MAKE_OBSERVATION.format(name=name_for_obs))
    call_to_next_acting = sequential.DEFAULT_CALL_TO_NEXT_ACTING
    call_to_action_spec_1 = (
        sequential.DEFAULT_CALL_TO_NEXT_ACTION_SPEC.format(
            name=name_for_next_action_spec_1))
    call_to_action_spec_2 = (
        sequential.DEFAULT_CALL_TO_NEXT_ACTION_SPEC.format(
            name=name_for_next_action_spec_2))
    call_to_resolve = sequential.DEFAULT_CALL_TO_RESOLVE
    call_to_check_termination = sequential.DEFAULT_CALL_TO_CHECK_TERMINATION

    make_obs_response = MAKE_OBS_RESPONSE_EXAMPLE.format(name=name_for_obs)
    next_acting_response = NEXT_ACTING_RESPONSE_EXAMPLE.format(
        name=name_for_next_acting)
    next_action_spec_response_1 = NEXT_ACTION_SPEC_RESPONSE_EXAMPLE_1.format(
        name=name_for_next_action_spec_1)
    next_action_spec_response_2 = NEXT_ACTION_SPEC_RESPONSE_EXAMPLE_2.format(
        name=name_for_next_action_spec_2)
    resolve_response_putative_action = (
        RESOLVE_RESPONSE_PUTATIVE_ACTION_EXAMPLE.format(name=name_for_resolve))
    resolve_response = RESOLVE_RESPONSE_EXAMPLE.format(name=name_for_resolve)
    termination_response_1 = TERMINATION_RESPONSE_EXAMPLE_1
    termination_response_2 = TERMINATION_RESPONSE_EXAMPLE_2

    state = (
        '\nExample exercises with default responses\n'
        '**--START EXAMPLES--**\n'
    )
    if 'make_observation' in exercises:
      state += (
          '\n\nExercise 1 --- Response 1\n'
          f'Exercise: {call_to_make_observation} --- {make_obs_response}'
      )
    if 'next_acting' in exercises:
      state += (
          '\n\nExercise 2 --- Response 2\n'
          f'Exercise: {call_to_next_acting} --- {next_acting_response}'
      )
    if 'next_action_spec_1' in exercises:
      state += (
          '\n\nExercise 3 --- Response 3\n'
          f'Exercise: {call_to_action_spec_1} --- {next_action_spec_response_1}'
      )
    if 'next_action_spec_2' in exercises:
      state += (
          '\n\nExercise 4 --- Response 4\n'
          f'Exercise: {call_to_action_spec_2} --- {next_action_spec_response_2}'
      )
    if 'resolve' in exercises:
      state += (
          '\n\nExercise 5 --- Response 5\n'
          f'Exercise: {call_to_resolve} --- {resolve_response_putative_action} '
          f'--- {resolve_response}'
      )
    if 'check_termination_1' in exercises:
      state += (
          '\n\nExercise 6 --- Response 6\n'
          f'Exercise: {call_to_check_termination} --- {termination_response_1}'
      )
    if 'check_termination_2' in exercises:
      state += (
          '\n\nExercise 7 --- Response 7\n'
          f'Exercise: {call_to_check_termination} --- {termination_response_2}'
      )
    state += '\n\n**--END EXAMPLES--**\n'

    super().__init__(state=state, pre_act_label=pre_act_label)
