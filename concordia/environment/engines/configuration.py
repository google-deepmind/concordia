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

"""Sequential (turn-based) action engine."""

from collections.abc import Mapping
from typing import Any

from concordia.components.game_master import event_resolution as event_resolution_components
from concordia.components.game_master import make_observation as make_observation_component
from concordia.components.game_master import next_acting as next_acting_components
from concordia.components.game_master import switch_act as switch_act_component
from concordia.typing import entity as entity_lib
from concordia.typing import prefab as prefab_lib
import termcolor


DEFAULT_CALL_TO_MAKE_OBSERVATION = (
    make_observation_component.DEFAULT_CALL_TO_MAKE_OBSERVATION
)
DEFAULT_CALL_TO_NEXT_ACTING = next_acting_components.DEFAULT_CALL_TO_NEXT_ACTING
DEFAULT_CALL_TO_NEXT_ACTION_SPEC = (
    next_acting_components.DEFAULT_CALL_TO_NEXT_ACTION_SPEC
)
DEFAULT_CALL_TO_RESOLVE = 'Because of all that came before, what happens next?'
DEFAULT_CALL_TO_CHECK_TERMINATION = 'Is the simulation finished?'
DEFAULT_CALL_TO_NEXT_GAME_MASTER = (
    'Which of the following game masters should be used for the simulation?'
)

DEFAULT_ACT_COMPONENT_KEY = switch_act_component.DEFAULT_ACT_COMPONENT_KEY

PUTATIVE_EVENT_TAG = event_resolution_components.PUTATIVE_EVENT_TAG
EVENT_TAG = event_resolution_components.EVENT_TAG

_PRINT_COLOR = 'cyan'


def _get_empty_log_entry():
  """Returns a dictionary to store a single log entry."""
  return {
      'terminate': {},
      'next_game_master': {},
      'make_observation': {},
      'next_acting': {},
      'next_action_spec': {},
      'resolve': {},
  }


class Configuration:
  """Sequential action (turn-based) engine.

  When this engine is used, one entity is acting at a time. The game master
  decides which entity to ask for an action on each step. The entity then
  decides what to do next, which is passed to the game master for resolution.
  The game master prepares observations for all entities in parallel.
  """

  def __init__(
      self,
      call_to_which_game_masters: str = DEFAULT_CALL_TO_NEXT_GAME_MASTER,
      max_actors: int = 5,
  ):
    """Sequential engine constructor."""
    self._log = []
    self._call_to_which_game_masters = call_to_which_game_masters
    self._max_actors = max_actors

  def which_game_masters(
      self,
      configurator: entity_lib.Entity,
      game_master_prefabs: Mapping[str, prefab_lib.Prefab],
      verbose: bool = False,
  ) -> prefab_lib.InstanceConfig:
    """Select which game master to use for the simulation."""
    game_masters_by_name = {
        name: prefab.description for name, prefab in game_master_prefabs.items()
    }

    # removing the game masters with _initializer_ in their name
    game_masters_by_name = {
        name: game_master
        for name, game_master in game_masters_by_name.items()
        if '_initializer_' not in name and 'dramaturgic' not in name
    }

    # make an observation of game masters and their descriptions
    configurator.observe(
        f'Game masters and their descriptions:\n{str(game_masters_by_name)}'
    )

    game_master_name = configurator.act(
        action_spec=entity_lib.ActionSpec(
            call_to_action=self._call_to_which_game_masters,
            output_type=entity_lib.OutputType.CHOICE,
            options=tuple(game_masters_by_name.keys()),
        )
    )
    if verbose:
      print(termcolor.colored(f'Game master: {game_master_name}', _PRINT_COLOR))
    if game_master_name not in game_masters_by_name:
      raise ValueError(
          f'Selected game master "{game_master_name}" not found in:'
          f' {game_masters_by_name.keys()}'
      )

    instance_config = prefab_lib.InstanceConfig(
        prefab=game_master_name,
        role=prefab_lib.Role.GAME_MASTER,
        params={'name': 'default rules'},
    )

    if 'locations' in game_master_prefabs[game_master_name].params:
      locations = configurator.act(
          action_spec=entity_lib.ActionSpec(
              call_to_action='What is the location of the simulation?',
              output_type=entity_lib.OutputType.FREE,
          )
      )
      instance_config = prefab_lib.InstanceConfig(
          prefab=game_master_name,
          role=prefab_lib.Role.GAME_MASTER,
          params={
              'name': 'default rules',
              'locations': locations,
          },
      )

    return instance_config

  def make_entity_instances(
      self,
      configurator: entity_lib.EntityWithLogging,
      entity_prefabs: Mapping[str, prefab_lib.Prefab],
      number_of_entities: int,
      verbose: bool = False,
  ):
    """Make an entity instance."""
    del entity_prefabs
    instances = []
    configurator.observe(
        f'The simulation will contain {number_of_entities} actors.'
    )
    for i in range(1, number_of_entities + 1):
      entity_name = configurator.act(
          action_spec=entity_lib.ActionSpec(
              call_to_action=f'What is the name of the actor number {i}?',
              output_type=entity_lib.OutputType.FREE,
          )
      )
      configurator.observe(f'The actor number {i} will be named {entity_name}')
      self._log.append(configurator.get_last_log())
      if verbose:
        print(f'Entity name: {entity_name}')
      entity_goal = configurator.act(
          action_spec=entity_lib.ActionSpec(
              call_to_action=(
                  f'What is the goal of the actor called {entity_name}?'
              ),
              output_type=entity_lib.OutputType.FREE,
          )
      )
      if verbose:
        print(f'Entity goal: {entity_goal}')

      entity_instance_config = prefab_lib.InstanceConfig(
          prefab='basic_with_plan__Entity',
          role=prefab_lib.Role.ENTITY,
          params={'name': entity_name, 'goal': entity_goal},
      )
      instances.append(entity_instance_config)
    return instances

  def make_initializer_instance(
      self,
      configurator: entity_lib.EntityWithLogging,
      verbose: bool = False,
  ):
    """Make an initializer instance."""

    shared_memories = configurator.act(
        action_spec=entity_lib.ActionSpec(
            call_to_action=(
                'What is the shared memories for all the actors of the'
                ' simulation?'
            ),
            output_type=entity_lib.OutputType.FREE,
        )
    )
    if verbose:
      print(f'Shared memories: {shared_memories}')

    return prefab_lib.InstanceConfig(
        prefab='formative_memories_initializer__GameMaster',
        role=prefab_lib.Role.INITIALIZER,
        params={
            'name': 'initial setup rules',
            'next_game_master_name': 'default rules',
            # Comma-separated list of shared memories.
            'shared_memories': [shared_memories],
        },
    )

  def run_loop(
      self,
      configurator: entity_lib.EntityWithLogging,
      premise: str,
      gm_prefabs: Mapping[str, prefab_lib.Prefab],
      entity_prefabs: Mapping[str, prefab_lib.Prefab],
      verbose: bool = False,
      log: list[Mapping[str, Any]] | None = None,
  ):
    """Run a game loop."""
    if not configurator:
      raise ValueError('No game masters provided.')

    if log:
      self._log = log
    log_entry = _get_empty_log_entry()
    if not log:
      log = [log_entry]

    if premise:
      premise = f'Premise of the simulation: {premise}'
      configurator.observe(premise)

    instances = []

    # while steps < max_steps:

    game_masters = self.which_game_masters(configurator, gm_prefabs, verbose)
    log_entry['which_game_masters'] = configurator.get_last_log()
    log.append(log_entry)

    instances.append(game_masters)
    how_many_entities_to_create = configurator.act(
        action_spec=entity_lib.ActionSpec(
            call_to_action='How many agents are required for the simulation?',
            output_type=entity_lib.OutputType.CHOICE,
            options=[str(i) for i in range(1, self._max_actors + 1)],
        )
    )

    how_many_entities_to_create = int(how_many_entities_to_create)
    if verbose:
      print(
          f'The simulation will contain {how_many_entities_to_create} actors.'
      )
    instances.extend(
        self.make_entity_instances(
            configurator,
            entity_prefabs,
            how_many_entities_to_create,
            verbose,
        )
    )
    initializer_instance = self.make_initializer_instance(configurator, verbose)
    instances.append(initializer_instance)

    return instances
