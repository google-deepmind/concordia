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

"""Component helping a game master pick which game master to use next."""

from collections.abc import Mapping, Sequence
import functools
import re
import types

from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory as memory_component
from concordia.components.game_master import make_observation as make_observation_component
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.utils import concurrency


DEFAULT_NEXT_GAME_MASTER_COMPONENT_KEY = '__next_game_master__'

DEFAULT_NEXT_GAME_MASTER_PRE_ACT_LABEL = '\nGame Master'
DEFAULT_CALL_TO_NEXT_GAME_MASTER = (
    'Which rule set should we use for the next step?')


class NextGameMaster(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """A component that decides which game master to use next.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      map_game_master_names_to_choices: Mapping[str, str],
      call_to_action: str = DEFAULT_CALL_TO_NEXT_GAME_MASTER,
      components: Sequence[str] = (),
      pre_act_label: str = DEFAULT_NEXT_GAME_MASTER_PRE_ACT_LABEL,
  ):
    """Initializes the component.

    Args:
      model: The language model to use for the component.
      map_game_master_names_to_choices: Names of game masters (rule sets) to
        choose from, mapped to the multiple choice question option corresponding
        to that game master.
      call_to_action: The question to ask the model to select a game master.
      components: Keys of components to condition the game master selection on.
      pre_act_label: Prefix to add to the output of the component when called
        in `pre_act`.

    Raises:
      ValueError: If the component order is not None and contains duplicate
        components.
    """
    super().__init__()
    self._model = model
    self._game_master_names = list(map_game_master_names_to_choices.keys())
    self._game_master_choices = list(map_game_master_names_to_choices.values())
    self._call_to_action = call_to_action
    self._components = components
    self._pre_act_label = pre_act_label

    self._currently_active_game_master = None

  def _get_named_component_pre_act_value(self, component_name: str) -> str:
    """Returns the pre-act value of a named component of the parent entity."""
    return (
        self.get_entity().get_component(
            component_name, type_=action_spec_ignored.ActionSpecIgnored
        ).get_pre_act_value()
    )

  def _get_component_pre_act_label(self, component_name: str) -> str:
    """Returns the pre-act label of a named component of the parent entity."""
    return (
        self.get_entity().get_component(
            component_name, type_=action_spec_ignored.ActionSpecIgnored
        ).get_pre_act_label()
    )

  def _component_pre_act_display(self, key: str) -> str:
    """Returns the pre-act label and value of a named component."""
    return (
        f'{self._get_component_pre_act_label(key)}:\n'
        f'{self._get_named_component_pre_act_value(key)}')

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    result = ''
    prompt_to_log = ''
    if action_spec.output_type == entity_lib.OutputType.NEXT_GAME_MASTER:
      prompt = interactive_document.InteractiveDocument(self._model)
      component_states = '\n'.join(
          [self._component_pre_act_display(key) for key in self._components]
      )
      prompt.statement(f'{component_states}\n')
      idx = prompt.multiple_choice_question(
          question=self._call_to_action,
          answers=self._game_master_choices)
      self._currently_active_game_master = self._game_master_names[idx]
      result = self._currently_active_game_master
      prompt_to_log = prompt.view().text()

    self._logging_channel({
        'Key': self._pre_act_label,
        'Summary': result,
        'Value': result,
        'Prompt': prompt_to_log,
    })
    return result

  def get_currently_active_game_master(self) -> str | None:
    return self._currently_active_game_master

  def get_state(self) -> entity_component.ComponentState:
    """Returns the state of the component."""
    return {'currently_active_game_master': self._currently_active_game_master}

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the state of the component."""
    self._currently_active_game_master = state['currently_active_game_master']


class FormativeMemoriesInitializer(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """A component that generates a backstory for each player entity.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      next_game_master_name: str,
      player_names: Sequence[str],
      shared_memories: Sequence[str] = (),
      player_specific_memories: Mapping[
          str, Sequence[str]
      ] = types.MappingProxyType({}),
      player_specific_context: Mapping[str, str] = types.MappingProxyType({}),
      components: Sequence[str] = (),
      delimiter_symbol: str = '***',
      memory_component_key: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_KEY
      ),
      make_observation_component_key: str = (
          make_observation_component.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY
      ),
      pre_act_label: str = '',
  ):
    """A component that generates a backstory for each player entity.
    
    As this is an initializer, it should only be called once per episode. To
    achieve this, it returns the name of the next game master once it finishes.
    The idea is to use one game master (with this component) for initialization,
    and then switch to a different game master for the actual game.

    Args:
      model: The language model to use for the component.
      next_game_master_name: The name of another game master to pass control to
        after this game master is finished initializing.
      player_names: Names of the player entities in the game.
      shared_memories: specific memories all players and the game master share.
      player_specific_memories: specific memories each player shares with the
        game master.
      player_specific_context: specific context each player needs to know about
        the game master.
      components: Keys of components to condition on.
      delimiter_symbol: The symbol to use to separate episodes in the generated
        backstory.
      memory_component_key: The name of the game master's memory component.
      make_observation_component_key: The name of the MakeObservation component
        to use pass the backstory memories to the players.
      pre_act_label: Prefix to add to the output of the component when called in
        `pre_act`.

    Raises:
      ValueError: If the component order is not None and contains duplicate
        components.
    """
    super().__init__()
    self._next_game_master_name = next_game_master_name
    self._model = model
    self._player_names = player_names
    self._shared_memories = shared_memories
    self._components = components
    self._delimiter_symbol = delimiter_symbol
    self._memory_component_key = memory_component_key
    self._make_observation_component_key = make_observation_component_key
    self._pre_act_label = pre_act_label

    self._player_specific_memories = player_specific_memories
    self._player_specific_context = player_specific_context

    self._initialized = False

  def get_named_component_pre_act_value(self, component_name: str) -> str:
    """Returns the pre-act value of a named component of the parent entity."""
    return (
        self.get_entity().get_component(
            component_name, type_=action_spec_ignored.ActionSpecIgnored
        ).get_pre_act_value()
    )

  def get_component_pre_act_label(self, component_name: str) -> str:
    """Returns the pre-act label of a named component of the parent entity."""
    return (
        self.get_entity().get_component(
            component_name, type_=action_spec_ignored.ActionSpecIgnored
        ).get_pre_act_label()
    )

  def _component_pre_act_display(self, key: str) -> str:
    """Returns the pre-act label and value of a named component."""
    return (
        f'{self.get_component_pre_act_label(key)}:\n'
        f'{self.get_named_component_pre_act_value(key)}')

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    if action_spec.output_type == entity_lib.OutputType.NEXT_GAME_MASTER:
      if self._initialized:
        return self._next_game_master_name
      else:
        memory = self.get_entity().get_component(
            self._memory_component_key, type_=memory_component.Memory
        )
        for shared_memory in self._shared_memories:
          memory.add(shared_memory)
        make_observation = self.get_entity().get_component(
            self._make_observation_component_key,
            type_=make_observation_component.MakeObservation,
        )

        def _process_player(
            player_name: str,
            memory: memory_component.Memory,
            make_observation: make_observation_component.MakeObservation,
        ):
          for shared_memory in self._shared_memories:
            make_observation.add_to_queue(player_name, shared_memory)
          episodes = self.generate_backstory_episodes(player_name)
          for episode in episodes:
            make_observation.add_to_queue(player_name, episode)
            memory.add(f'{player_name} remembers: "{episode}"')
          for player_memory in self._player_specific_memories.get(
              player_name, []
          ):
            make_observation.add_to_queue(player_name, player_memory)
            memory.add(f'{player_name} remembers: "{player_memory}"')

        tasks = {
            player_name: functools.partial(
                _process_player, player_name, memory, make_observation
            )
            for player_name in self._player_names
        }

        # Run entity actions concurrently
        concurrency.run_tasks(tasks)

        self._initialized = True
        return self.get_entity().name
    return ''

  def generate_backstory_episodes(
      self, active_entity_name: str) -> Sequence[str]:
    prompt = interactive_document.InteractiveDocument(self._model)
    component_states = '\n'.join(
        [self._component_pre_act_display(key) for key in self._components]
    )
    prompt.statement(f'{component_states}\n')
    prompt.statement('----- Role Playing Master Class -----\n')
    prompt.statement("Question: What is the protagonist's name?")
    prompt.statement(f'Answer: {active_entity_name}\n')
    prompt.statement('Question: Describe the setting or background.')
    shared_memories = '\n'.join(self._shared_memories)
    prompt.statement(f'Answer: {shared_memories}\n')

    player_specific_context = '\n'.join(
        self._player_specific_context.get(active_entity_name, [])
    )
    if player_specific_context:
      prompt.statement(
          'Question: Describe the personal context of the protagonist.'
      )
      prompt.statement(f'Answer: {player_specific_context}\n')

    gender = prompt.open_question("What is the protagonist's gender?")
    date_of_birth = prompt.open_question(
        'What year was protagonist born? Respond with just the year as a '
        'number, e.g. "1990".'
    )
    question = (
        f'Write a life story for a {gender} character '
        f'named {active_entity_name} who was born in {date_of_birth}.'
    )
    question += (
        f'Begin the story when {active_entity_name} is very young and end it'
        ' when they are quite old. The story should be no more than four'
        ' paragraphs in total. The story may include details such as (but'
        ' not limited to) any of the following: what their job is or was,'
        ' what their typical day was or is like, what their goals, desires,'
        ' hopes, dreams, and aspirations are, and have been, as well as'
        ' their drives, duties, responsibilities, and obligations. It should'
        ' clarify what gives them joy and what are they afraid of. It may'
        ' include their friends and family, as well as antagonists. It'
        ' should be a complete life story for a complete person but it'
        ' should not specify how their life ends. The reader should be left'
        f' with a profound understanding of {active_entity_name}.'
    )
    backstory = prompt.open_question(
        question,
        max_tokens=4500,
        terminators=['\nQuestion', '-----'],
    )
    backstory = re.sub(r'\.\s', '.\n', backstory)

    inner_prompt = interactive_document.InteractiveDocument(self._model)
    inner_prompt.statement('Creative Writing Master Class\n')
    inner_prompt.statement('Character background story:\n\n' + backstory)
    question = (
        'Given the life story above, invent formative episodes from '
        f'the life of {active_entity_name}. '
        f'They should be memorable events for {active_entity_name} and '
        'important for establishing who they are as a person. They should '
        f"be consistent with {active_entity_name}'s personality and "
        f"circumstances. Describe each episode from {active_entity_name}'s "
        'perspective and use third-person limited point of view. Each'
        ' episode '
        'must mention their age at the time the event occurred using'
        ' language '
        f'such as "When {active_entity_name} was 5 years old, they '
        'experienced..." . Use past tense. Write no more than five sentences '
        'per episode. Separate episodes from one another by the delimiter '
        f'"{self._delimiter_symbol}". Do not apply any other '
        'special formatting besides these delimiters.'
    )
    aggregated_result = prompt.open_question(
        question=question,
        max_tokens=6000,
        terminators=[],
    )
    episodes = list(aggregated_result.split(self._delimiter_symbol))
    self._logging_channel({
        'Key': self._pre_act_label,
        'Episodes': episodes,
        'Inner Prompt': inner_prompt.view().text(),
        'Prompt': prompt.view().text(),
    })
    return episodes

  def get_state(self) -> entity_component.ComponentState:
    """Returns the state of the component."""
    return {'initialized': self._initialized}

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the state of the component."""
    pass
