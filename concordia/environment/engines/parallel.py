# Copyright 2025 DeepMind Technologies Limited.
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

"""Engine for running questionnaires in parallel across multiple entities."""

from collections.abc import Mapping, Sequence
import functools
import json
import threading
from typing import Any, Callable, List, Tuple, cast

from concordia.agents import entity_agent
from concordia.components.game_master import event_resolution as event_resolution_components
from concordia.components.game_master import next_acting as next_acting_components
from concordia.environment import engine as engine_lib
from concordia.typing import entity as entity_lib
from concordia.utils import concurrency
import termcolor
from typing_extensions import override

PUTATIVE_EVENT_TAG = event_resolution_components.PUTATIVE_EVENT_TAG
EVENT_TAG = event_resolution_components.EVENT_TAG

DEFAULT_CALL_TO_CHECK_TERMINATION = 'Is the game/simulation finished?'
DEFAULT_CALL_TO_NEXT_GAME_MASTER = (
    'Which rule set should we use for the next step?'
)
DEFAULT_CALL_TO_NEXT_ACTING = next_acting_components.DEFAULT_CALL_TO_NEXT_ACTING
DEFAULT_CALL_TO_NEXT_ACTION_SPEC = (
    next_acting_components.DEFAULT_CALL_TO_NEXT_ACTION_SPEC
)

_PRINT_COLOR = 'cyan'


class ParallelQuestionnaireEngine(engine_lib.Engine):
  """Engine for asking all questions to all entities in parallel."""

  def __init__(
      self,
      call_to_check_termination: str = DEFAULT_CALL_TO_CHECK_TERMINATION,
      call_to_next_acting: str = DEFAULT_CALL_TO_NEXT_ACTING,
      call_to_next_action_spec: str = DEFAULT_CALL_TO_NEXT_ACTION_SPEC,
      call_to_next_game_master: str = DEFAULT_CALL_TO_NEXT_GAME_MASTER,
  ):
    """Constructor."""
    self._call_to_check_termination = call_to_check_termination
    self._call_to_next_acting = call_to_next_acting
    self._call_to_next_action_spec = call_to_next_action_spec
    self._call_to_next_game_master = call_to_next_game_master

  @override
  def next_acting(
      self,
      game_master: entity_lib.Entity,
      entities: Sequence[entity_lib.Entity],
  ) -> Sequence[entity_lib.Entity]:  # pytype: disable=signature-mismatch
    """Returns entities that should act next."""
    entities_by_name = {entity.name: entity for entity in entities}

    player_names_str = game_master.act(
        action_spec=entity_lib.ActionSpec(
            call_to_action=self._call_to_next_acting,
            output_type=entity_lib.OutputType.NEXT_ACTING,
            options=tuple(entities_by_name.keys()),
        )
    )

    next_entity_names = player_names_str.split(',')
    next_entities = [
        entities_by_name[name]
        for name in next_entity_names
        if name in entities_by_name
    ]
    return next_entities

  def next_action_spec(
      self,
      game_master: entity_lib.Entity,
      acting_entities: Sequence[entity_lib.Entity],
  ) -> List[Tuple[str, str, str]]:
    """Returns the next action spec for all questions for the acting entities."""
    if not acting_entities:
      return []

    player_names = ','.join([entity.name for entity in acting_entities])
    question_specs_json = game_master.act(
        action_spec=entity_lib.ActionSpec(
            call_to_action=player_names,  # Pass player names here
            output_type=entity_lib.OutputType.NEXT_ACTION_SPEC,
        )
    )

    question_specs_list = json.loads(question_specs_json)

    all_action_specs: List[Tuple[str, str, str]] = []
    for item in question_specs_list:
      player_name = item['player_name']
      q_id = item['question_id']
      spec_str = item['action_spec_str']
      all_action_specs.append((player_name, q_id, spec_str))

    return all_action_specs

  def terminate(
      self, game_master: entity_lib.Entity, verbose: bool = False
  ) -> bool:
    """Decide if the episode should terminate."""
    should_terminate_string = game_master.act(
        action_spec=entity_lib.ActionSpec(
            call_to_action=self._call_to_check_termination,
            output_type=entity_lib.OutputType.TERMINATE,
            options=tuple(entity_lib.BINARY_OPTIONS.values()),
        )
    )
    if verbose:
      print(
          termcolor.colored(
              f'Terminate? {should_terminate_string}', _PRINT_COLOR
          )
      )
    return should_terminate_string == entity_lib.BINARY_OPTIONS['affirmative']

  @override
  def run_loop(
      self,
      game_masters: Sequence[entity_lib.Entity | entity_lib.EntityWithLogging],
      entities: Sequence[entity_lib.Entity | entity_lib.EntityWithLogging],
      premise: str = '',
      max_steps: int = 1,  # Usually only needs 1 step
      verbose: bool = False,
      log: list[Mapping[str, Any]] | None = None,
      checkpoint_callback: Callable[[int], None] | None = None,
  ):
    if not game_masters:
      raise ValueError('No game masters provided.')
    game_master = game_masters[0]

    if self.terminate(game_master, verbose):
      return

    next_entities = self.next_acting(game_master, entities)

    if not next_entities:
      if verbose:
        print(termcolor.colored('No entities to act.', _PRINT_COLOR))
      return

    player_qid_spec_list = self.next_action_spec(game_master, next_entities)

    entity_map = {e.name: e for e in next_entities}
    entity_answers = {name: {} for name in entity_map.keys()}
    mutex = threading.Lock()

    def entity_task(player_name: str):
      entity = entity_map[player_name]

      if not isinstance(entity, entity_agent.EntityAgent):
        raise TypeError(
            'Entities used with ParallelQuestionnaireEngine must be EntityAgent'
            ' or a subclass.'
        )
      # Now safe to use entity as EntityAgent
      agent = cast(entity_agent.EntityAgent, entity)

      action_specs_for_entity: List[entity_lib.ActionSpec] = []
      qids_for_entity: List[str] = []

      for p_name, q_id, spec_str in player_qid_spec_list:
        if p_name == player_name:
          formatted_spec_str = spec_str.replace('{player_name}', player_name)
          action_specs_for_entity.append(
              engine_lib.action_spec_parser(formatted_spec_str)
          )
          qids_for_entity.append(q_id)

      if not action_specs_for_entity:
        return

      answers = agent.parallel_stateless_act(action_specs_for_entity)

      with mutex:
        entity_answers[player_name] = dict(zip(qids_for_entity, answers))

    tasks = {
        name: functools.partial(entity_task, name) for name in entity_map.keys()
    }
    if tasks:
      # Run all tasks in parallel
      concurrency.run_tasks(tasks)

    # Feed back answers to GM
    for player_name, qid_answer_map in entity_answers.items():
      for q_id, answer in qid_answer_map.items():
        observation = f'{PUTATIVE_EVENT_TAG} {player_name}: {q_id}: {answer}'
        game_master.observe(observation)

    if verbose:
      print(termcolor.colored('Questionnaire round finished.', _PRINT_COLOR))

  # Methods not needed for this engine
  @override
  def make_observation(
      self, game_master: entity_lib.Entity, entity: entity_lib.Entity
  ) -> str:
    raise NotImplementedError

  @override
  def resolve(
      self, game_master: entity_lib.Entity, putative_event: str
  ) -> None:
    raise NotImplementedError

  @override
  def next_game_master(
      self,
      game_master: entity_lib.Entity,
      game_masters: Sequence[entity_lib.Entity],
      verbose: bool = False,
  ) -> entity_lib.Entity:
    """Select which game master to use for the next step."""
    if len(game_masters) == 1:
      return game_masters[0]
    game_masters_by_name = {
        game_master_.name: game_master_ for game_master_ in game_masters
    }
    next_game_master_name = game_master.act(
        action_spec=entity_lib.ActionSpec(
            call_to_action=self._call_to_next_game_master,
            output_type=entity_lib.OutputType.NEXT_GAME_MASTER,
            options=tuple(game_masters_by_name.keys()),
        )
    )
    if verbose:
      print(
          termcolor.colored(
              f'Game master: {next_game_master_name}', _PRINT_COLOR
          )
      )
    if next_game_master_name not in game_masters_by_name:
      raise ValueError(
          f'Selected game master "{next_game_master_name}" not found in:'
          f' {game_masters_by_name.keys()}'
      )
    return game_masters_by_name[next_game_master_name]
