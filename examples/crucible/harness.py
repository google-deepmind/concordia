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

"""Builds and plays a toy as a standard Concordia `Simulation`.

Stock and contrib pieces only: `prefabs/simulation/generic.Simulation` with
the `Sequential` engine, the contrib rulebook game master (the toy is its
rulebook) and one contrib seated entity per actor. The seat a person or a bot plays acts through `HumanActComponent`;
every other actor is a language model.
"""

from collections.abc import Mapping
import contextlib
import csv
import dataclasses
import io
import json
import os
from typing import Any

from concordia.components.agent import human_act_component
from concordia.contrib.prefabs.entity import seated
from concordia.contrib.prefabs.game_master import rulebook as rulebook_gm
from concordia.environment.engines import sequential
from concordia.language_model import language_model
from concordia.prefabs.simulation import generic as simulation_lib
from concordia.typing import human_input
from concordia.typing import prefab as prefab_lib
from concordia.utils import structured_logging

from examples.crucible import model_setup
from examples.crucible import seats
from examples.crucible import toys
from examples.crucible.toys import base

SEAT_PREFAB = 'seated__Entity'
GM_PREFAB = 'rulebook__GameMaster'
BOTS = ('skilled', 'random', 'passive')
SEATS = (*BOTS, 'human', 'model')


@dataclasses.dataclass
class Game:
  """A built game: the simulation, its toy and whoever sits in the seat."""

  sim: simulation_lib.Simulation
  toy: base.Toy
  kind: toys.ToyKind
  reader: human_input.HumanInput | None
  seat: str

  def play(self, quiet: bool = True) -> structured_logging.SimulationLog:
    """Plays the game; `Simulation.play` always prints, unless quiet."""
    if not quiet:
      return self.sim.play(max_steps=self.toy.max_steps())
    with contextlib.redirect_stdout(io.StringIO()):
      return self.sim.play(max_steps=self.toy.max_steps())

  def outcome(self) -> dict[str, Any]:
    return self.toy.outcome()


def build_game(
    toy_name: str,
    model: language_model.LanguageModel,
    seat: str = 'skilled',
    seed: int = 0,
    scenario: str = 'sandbox',
    reader: human_input.HumanInput | None = None,
    interpreter: language_model.LanguageModel | None = None,
    data: Mapping[str, Any] | None = None,
) -> Game:
  """Builds a game of a toy.

  Args:
    toy_name: A key of `toys.registry()`.
    model: Plays every actor not in the seat.
    seat: Who plays the toy's seat: a bot (`BOTS`), 'human' (`reader`, or
      the terminal) or 'model' (the model, like everyone else).
    seed: Seeds the bots' choices.
    scenario: The toy's scenario.
    reader: The person's reader for seat='human' (e.g. the phone server).
    interpreter: Reads what moves mean; defaults to `model`.
    data: Overrides the toy's data file.

  Returns:
    The game, ready to play.
  """
  if seat not in SEATS:
    raise ValueError(f'Unknown seat: {seat}')
  kind = toys.registry()[toy_name]
  toy = kind.build(
      interpreter or model,
      data or toys.load_data(toy_name),
      scenario,
      seed=seed,
  )
  if seat in BOTS:
    reader = seats.bot_reader(kind.policies[seat], toy, seed)
  elif seat == 'human' and reader is None:
    reader = seats.TerminalReader(toy)
  elif seat == 'model':
    reader = None
  instances = []
  for actor in toy.actors():
    params: dict[str, Any] = {'name': actor, 'randomize_choices': False}
    if actor == kind.seat and reader is not None:
      bound_reader = reader
      params['act_component_factory'] = (
          lambda order, r=bound_reader: human_act_component.HumanActComponent(
              r, component_order=order
          )
      )
    instances.append(
        prefab_lib.InstanceConfig(
            prefab=SEAT_PREFAB,
            role=prefab_lib.Role.ENTITY,
            params=params,
        )
    )
  instances.append(
      prefab_lib.InstanceConfig(
          prefab=GM_PREFAB,
          role=prefab_lib.Role.GAME_MASTER,
          params={'name': 'Crucible', 'rulebook': toy},
      )
  )
  config = prefab_lib.Config(
      default_premise='',
      default_max_steps=toy.max_steps(),
      prefabs={
          SEAT_PREFAB: seated.Entity(),
          GM_PREFAB: rulebook_gm.GameMaster(),
      },
      instances=instances,
  )
  sim = simulation_lib.Simulation(
      config=config,
      model=model,
      embedder=model_setup.hash_embedder,
      engine=sequential.Sequential(),
  )
  return Game(sim=sim, toy=toy, kind=kind, reader=reader, seat=seat)


def write_run(game: Game, log: Any, out_dir: str) -> dict[str, Any]:
  """Writes the run folder and returns the summary."""
  os.makedirs(out_dir, exist_ok=True)
  receipts = [
      r.to_dict()
      for gm in game.sim.get_game_masters()
      for r in getattr(gm.get_component('__resolution__'), 'results', [])  # pyrefly: ignore[missing-attribute]
  ]
  summary = {
      'toy': game.toy.name,
      'seat': game.seat,
      'outcome': game.outcome(),
      'receipts': receipts,
      'state': game.toy.get_state(),
  }
  with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
  with open(os.path.join(out_dir, 'log.html'), 'w') as f:
    f.write(log.to_html(title=f'Crucible: {game.toy.name}'))
  with open(os.path.join(out_dir, 'moves.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['turn', 'actor', 'action', 'summary'])
    for i, r in enumerate(receipts):
      writer.writerow([i + 1, r['actor'], r['action'], r['summary']])
  return summary
