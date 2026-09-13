# Copyright 2026 DeepMind Technologies Limited.
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

"""One-round resource council using Concordia's standard Simultaneous engine.

This is an analogous teaching game, NOT Bellwether's three-watch game with
an engine flag changed. Choices explicitly authorize immediate contributions,
not negotiations or promises. The CLI is a programmed fixture, not live actors.
"""

import argparse
from collections.abc import Mapping
import copy
import datetime
import json
import pathlib
from typing import Any

from concordia.components.agent import human_act_component
from concordia.components.game_master import event_resolution
from concordia.components.game_master import inventory
from concordia.components.game_master import make_observation
from concordia.components.game_master import next_acting
from concordia.components.game_master import switch_act
from concordia.environment.engines import simultaneous
from concordia.examples.bellwether import game
from concordia.language_model import no_language_model
from concordia.prefabs.entity import minimal
from concordia.prefabs.simulation import generic
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.typing import prefab
import numpy as np

MEMBERS = ('Nell', 'Ivo', 'Sam')
CHOICES = ('contribute', 'keep')
GM = 'Council'
INSTRUCTIONS = (
    'Each member owns one fuel. Choose contribute to explicitly authorize and'
    ' execute donation of your own unit to Community after all choices arrive,'
    ' or keep to retain it. This is one simultaneous round. You cannot see'
    ' other members’ current choices before making your own. No promise or'
    ' majority vote can give away another member’s fuel.'
)


class CouncilLedger(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """Scenario-specific complete-batch validation, not an execution loop."""

  def __init__(self, stock):
    super().__init__()
    self.stock = stock
    self.done = False
    self.decisions = {}
    self._putative = ''
    self.result = ''
    self._resolved_input = ''

  def get_state(self):
    return copy.deepcopy({
        'done': self.done,
        'decisions': self.decisions,
        'putative': self._putative,
        'result': self.result,
        'resolved_input': self._resolved_input,
    })

  def set_state(self, state):
    self.done = state['done']
    self.decisions = copy.deepcopy(state['decisions'])
    self._putative = state['putative']
    self.result = state['result']
    self._resolved_input = state['resolved_input']

  def pre_observe(self, observation):
    if observation.startswith(event_resolution.PUTATIVE_EVENT_TAG):
      self._putative = observation[
          len(event_resolution.PUTATIVE_EVENT_TAG) :
      ].strip()
    return ''

  def pre_act(self, action_spec):
    if action_spec.output_type != entity_lib.OutputType.RESOLVE:
      return ''
    result = self.resolve_batch(self._putative)
    self._logging_channel({
        'Key': 'CouncilLedger',
        'Value': result,
        'State': self.get_state(),
        'Inventory': self.stock.get_state(),
    })
    return result

  def resolve_batch(self, text):
    if self.done:
      if text != self._resolved_input:
        raise ValueError('The council round has already resolved')
      return self.result
    decisions = {}
    for line in text.splitlines():
      name, separator, choice = line.partition(': ')
      if (
          not separator
          or name not in MEMBERS
          or name in decisions
          or choice not in CHOICES
      ):
        raise ValueError('Invalid council batch; no fuel transferred')
      decisions[name] = choice
    if set(decisions) != set(MEMBERS):
      raise ValueError('Incomplete council batch; no fuel transferred')

    def transfer(accounts):
      for name, choice in decisions.items():
        if choice == 'contribute':
          accounts[name]['fuel'] -= 1
          accounts['Community']['fuel'] += 1
      return accounts

    # A single standard validated material transaction after the whole batch.
    self.stock.apply(transfer)
    self.decisions = decisions
    self._resolved_input = text
    self.done = True
    count = self.stock.get_player_inventory('Community')['fuel']
    self.result = (
        f'{count} of 3 units explicitly contributed. '
        + '; '.join(f'{name}: {decisions[name]}' for name in MEMBERS)
        + '. No consent inferred from another member’s choice.'
    )
    return self.result


def configuration(*, human_readers=None):
  """Fresh Config and ledger; optional separately routed human readers."""
  readers = {} if human_readers is None else human_readers
  if not isinstance(readers, Mapping) or any(
      name not in MEMBERS or not callable(reader)
      for name, reader in readers.items()
  ):
    raise ValueError(
        'Human readers must be callables for known council members'
    )
  stock = game.ExplicitInventory(
      no_language_model.NoLanguageModel(),
      [
          inventory.ItemTypeConfig(
              'fuel', minimum=0, maximum=3, force_integer=True
          )
      ],
      {**{name: {'fuel': 1.0} for name in MEMBERS}, 'Community': {'fuel': 0.0}},
      lambda: datetime.datetime(2000, 1, 1),
  )
  ledger = CouncilLedger(stock)
  observations = make_observation.ObservationQueue()
  for name in MEMBERS:
    observations.add(name, INSTRUCTIONS, MEMBERS)

  class Member(prefab.Prefab):
    description = (
        'Standard minimal actor, optionally using a human input adapter.'
    )

    def build(self, model, memory_bank):
      params: dict[str, Any] = dict(self.params)
      reader = readers.get(params['name'])
      return minimal.Entity(params=params).build(
          model,
          memory_bank,
          act_component_factory=(
              (
                  lambda order: human_act_component.HumanActComponent(
                      reader, component_order=order
                  )
              )
              if reader is not None
              else None
          ),
      )

  class Moderator(prefab.Prefab):
    """Build existing GM routing and scenario accounting components."""

    description = (
        'Standard SwitchAct routing for one simultaneous council round.'
    )

    def build(self, model, memory_bank):
      components = {
          switch_act.DEFAULT_RESOLUTION_COMPONENT_KEY: ledger,
          'Inventory': stock,
          switch_act.DEFAULT_NEXT_ACTING_COMPONENT_KEY: (
              next_acting.NextActingAllEntities(MEMBERS)
          ),
          switch_act.DEFAULT_NEXT_ACTION_SPEC_COMPONENT_KEY: (
              next_acting.FixedActionSpec(
                  entity_lib.choice_action_spec(
                      call_to_action='{name}, contribute your unit or keep it?',
                      options=CHOICES,
                  )
              )
          ),
          switch_act.DEFAULT_TERMINATE_COMPONENT_KEY: game.Signal(
              ledger,
              entity_lib.OutputType.TERMINATE,
              lambda: 'Yes' if ledger.done else 'No',
          ),
          switch_act.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY: (
              make_observation.MakeObservation(
                  model,
                  player_names=MEMBERS,
                  allow_llm_fallback=False,
                  external_queue=observations,
              )
          ),
      }
      return minimal.Entity(
          params={
              'name': GM,
              'custom_instructions': INSTRUCTIONS,
              'extra_components': components,  # pyrefly: ignore[bad-assignment]
          }
      ).build(
          model,
          memory_bank,
          act_component=switch_act.SwitchAct(model, entity_names=MEMBERS),
      )

  instances = []
  for name in MEMBERS:
    params: dict[str, Any] = {
        'name': name,
        'custom_instructions': INSTRUCTIONS,
        'randomize_choices': False,
    }
    instances.append(
        prefab.InstanceConfig(
            prefab='member', role=prefab.Role.ENTITY, params=params
        )
    )
  instances.append(
      prefab.InstanceConfig(
          prefab='moderator', role=prefab.Role.GAME_MASTER, params={'name': GM}
      )
  )
  config = prefab.Config(
      prefabs={'member': Member(), 'moderator': Moderator()},
      instances=instances,
      default_max_steps=1,
  )
  return config, ledger


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--output',
      type=pathlib.Path,
      default=pathlib.Path('runs/council-fixture'),
  )
  args = parser.parse_args(argv)
  config, ledger = configuration()
  simulation = generic.Simulation(
      config,
      no_language_model.NoLanguageModel(),
      lambda text: np.zeros(8),
      engine=simultaneous.Simultaneous(),
  )
  log = simulation.play(max_steps=1)
  args.output.mkdir(parents=True, exist_ok=True)
  (args.output / 'trace.json').write_text(log.to_json(), encoding='utf-8')
  (args.output / 'trace.html').write_text(log.to_html(), encoding='utf-8')
  (args.output / 'outcome.json').write_text(
      json.dumps(
          {
              'fixture': True,
              'external_model_calls': 0,
              'engine': 'Simultaneous',
              'complete': ledger.done,
              'rounds': len(simulation.get_raw_log()),
              'decisions': ledger.decisions,
              'result': ledger.result,
              'inventory': {
                  name: ledger.stock.get_player_inventory(name)
                  for name in (*MEMBERS, 'Community')
              },
          },
          indent=2,
      ),
      encoding='utf-8',
  )
  print(ledger.result)


if __name__ == '__main__':
  main()
