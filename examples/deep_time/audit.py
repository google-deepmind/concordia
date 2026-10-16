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

"""Complexity audit: counts the game's interlocking systems.

Counts come from `data/systems.yaml` (systems, interaction-matrix cells,
feedback loops, resources, recipes, techs, reflex operators),
`data/topics.yaml` and the scenario registry, so the numbers quoted in cycle
reports are reproducible:

  python -m examples.deep_time.audit
"""

from collections.abc import Sequence
import json
from typing import Any

from absl import app
from absl import flags
from examples.deep_time import scenarios
from examples.deep_time.world import data


def registry() -> dict[str, Any]:
  return data.load('systems.yaml')


# Fields each registry entry may have. An unquoted comma in a YAML flow
# mapping splits a value and invents a key, which this check catches.
FIELDS = {
    'systems': {'key', 'name', 'module'},
    'cells': {'from', 'to', 'by'},
    'loops': {'name', 'sign', 'period_ticks', 'path', 'note'},
    'resources': {'key', 'name', 'kind'},
}
MAP_FIELDS = {
    'overlays': {'key', 'name'},
    'mobile_species': {'key', 'name', 'moves'},
    'local_interactions': {'key', 'name', 'between'},
}


def validate(reg: dict[str, Any]) -> list[str]:
  """Returns problems in the registry (unknown systems, duplicate keys)."""
  problems = []
  for section, fields in FIELDS.items():
    for item in reg.get(section, []):
      extra = sorted(set(item) - fields)
      if extra:
        problems.append(f'{section} entry has unexpected keys {extra}')
  systems = [s['key'] for s in reg['systems']]
  if len(set(systems)) != len(systems):
    problems.append('duplicate system keys')
  known = set(systems)
  seen_cells = set()
  for cell in reg['cells']:
    for end in (cell['from'], cell['to']):
      if end not in known:
        problems.append(f'cell refers to unknown system {end!r}')
    key = (cell['from'], cell['to'], cell['by'])
    if key in seen_cells:
      problems.append(f'duplicate cell {key}')
    seen_cells.add(key)
  for loop in reg['loops']:
    if loop['sign'] not in ('+', '-'):
      problems.append(f'loop {loop["name"]!r} has bad sign')
    for system in loop['path']:
      if system not in known:
        problems.append(f'loop {loop["name"]!r} uses unknown {system!r}')
  for section, fields in MAP_FIELDS.items():
    entries = reg.get('map', {}).get(section, [])
    for item in entries:
      extra = sorted(set(item) - fields)
      if extra:
        problems.append(f'map {section} entry has unexpected keys {extra}')
    keys = [item['key'] for item in entries]
    if len(set(keys)) != len(keys):
      problems.append(f'duplicate map {section} keys')
  for kind in ('resources', 'recipes', 'techs', 'reflex_operators'):
    keys = [item['key'] for item in reg.get(kind, [])]
    if len(set(keys)) != len(keys):
      problems.append(f'duplicate {kind} keys')
  return problems


def counts() -> dict[str, int]:
  """Returns the audit counts used in cycle reports."""
  reg = registry()
  return {
      'systems': len(reg['systems']),
      'matrix_cells': len({(c['from'], c['to']) for c in reg['cells']}),
      'mechanisms': len(reg['cells']),
      'resources': len(reg.get('resources', [])),
      'recipes': len(reg.get('recipes', [])),
      'techs': len(reg.get('techs', [])),
      'topics': len(data.load('topics.yaml')['topics']),
      'reflex_ops': len(reg.get('reflex_operators', [])),
      'loops': len(reg['loops']),
      'scenarios': len(scenarios.SCENARIOS),
      **map_counts(reg),
  }


def map_counts(reg: dict[str, Any]) -> dict[str, int]:
  """Stage-2 counts: tiles, biomes, overlays, mobile species, interactions."""
  tiles = data.load('tiles.yaml')
  climate = data.load('climate.yaml')['grid']
  per = int(tiles['per_region'])
  try:
    biomes = len(data.load('biomes.yaml')['biomes'])
  except FileNotFoundError:
    biomes = 0
  section = reg.get('map', {})
  return {
      'tiles': int(climate['bands']) * int(climate['sectors']) * per * per,
      'biomes': biomes,
      'overlays': len(section.get('overlays', [])),
      'mobile_species': len(section.get('mobile_species', [])),
      'local_interactions': len(section.get('local_interactions', [])),
  }


_SPEED = flags.DEFINE_bool(
    'speed', False, 'Also time every scenario in mock mode (s/tick).'
)


def speed(ticks: int = 8) -> dict[str, float]:
  """Mock seconds per tick on the full map, by scenario (the slowest wins)."""
  # Imported here: the counts above need no game, and this plays one.
  import contextlib  # pylint: disable=g-import-not-at-top
  import io  # pylint: disable=g-import-not-at-top
  import time  # pylint: disable=g-import-not-at-top
  from examples.deep_time import model_setup  # pylint: disable=g-import-not-at-top
  from examples.deep_time import simulation  # pylint: disable=g-import-not-at-top

  out = {}
  for key in sorted(scenarios.SCENARIOS):
    models = model_setup.make_models('mock')
    sim, context = simulation.build_simulation(key, models, ticks, seed=0)
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
      sim.play(max_steps=ticks)
    out[key] = round(
        (time.perf_counter() - start) / max(1, context.planet.tick), 4
    )
  return out


def main(argv: Sequence[str]) -> None:
  report: dict[str, Any] = {
      'counts': counts(),
      'problems': validate(registry()),
  }
  del argv
  if _SPEED.value:
    per_tick = speed()
    report['mock_s_per_tick'] = {'max': max(per_tick.values()), **per_tick}
  print(json.dumps(report))


if __name__ == '__main__':
  app.run(main)
