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

"""Public accounting figure using the existing Matplotlib dependency.

Input is public_account.document, never developer state or an arbitrary log.
The result is offline presentation, not geographic data or model image input.
"""

import io
import re
import threading
from xml.etree import ElementTree as etree

from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

_LOCK = threading.Lock()
_SVG = 'http://www.w3.org/2000/svg'
_WATCHES = ('Dusk', 'High Tide', 'Before Dawn')
_FACILITIES = ('beacon', 'shelter', 'cold store')
_HOLDERS = ('Generator', 'Nell', 'Ivo', 'Used')
_STATES = ('Not resolved', 'Unserved', 'Served')


def render(account: dict, *, setup_description: str = '') -> str:
  """Return an accessible self-contained SVG from the public projection only."""
  records = account['accounting']['services']
  cells = []
  for watch in _WATCHES:
    row = []
    for facility in _FACILITIES:
      record = next(
          (
              r
              for r in records
              if r['watch'] == watch and r['facility'] == facility
          ),
          None,
      )
      row.append(0 if record is None else 2 if record['served'] else 1)
    cells.append(row)
  fuel = [
      account['accounting']['inventory'][owner]['fuel'] for owner in _HOLDERS
  ]
  parts = [
      account['accounting']['inventory'][owner]['part'] for owner in _HOLDERS
  ]
  provenance = (
      'Generated from '
      + account['schema']
      + '; '
      + account['backend']
      + ' backend; '
      + account['status'].replace('_', ' ')
      + '; '
      + account['watch']
      + '. Public recorded accounting only, not predictions or a saved game.'
  )
  description = provenance + ' ' + setup_description + ' '
  description += ' '.join(
      watch
      + ': '
      + ', '.join(
          facility + ' ' + _STATES[cells[i][j]].lower()
          for j, facility in enumerate(_FACILITIES)
      )
      + '.'
      for i, watch in enumerate(_WATCHES)
  )
  description += (
      ' Stocks: '
      + '; '.join(
          f'{owner}: {fuel[i]} fuel, {parts[i]} spare parts'
          for i, owner in enumerate(_HOLDERS)
      )
      + '. Used is cumulative and includes any declared pre-play consumption.'
  )
  # Figure instances avoid pyplot state. Serialize renderer/font use without
  # changing global Matplotlib configuration or using a GUI backend.
  with _LOCK:
    figure = Figure(
        figsize=(4.2, 6.2), layout='constrained', facecolor='#ffffff'
    )
    grid, stocks = figure.subplots(
        2, 1, gridspec_kw={'height_ratios': [2.5, 1]}
    )
    figure.suptitle('Bellwether\nPublic service account', fontsize=15)
    grid.imshow(
        cells,
        cmap=ListedColormap(['#e5e7eb', '#913b32', '#22665b']),
        vmin=0,
        vmax=2,
        aspect='auto',
    )
    grid.set_xticks(range(3), ('Beacon', 'Shelter', 'Cold\nstore'))
    grid.set_yticks(range(3), ('Dusk', 'High\nTide', 'Before\nDawn'))
    grid.tick_params(length=0, labelsize=11)
    grid.set_title(
        account['backend'].capitalize()
        + ' · '
        + account['status'].replace('_', ' ')
        + '\n'
        + account['watch'],
        fontsize=12,
        pad=12,
    )
    for i, row in enumerate(cells):
      for j, state in enumerate(row):
        grid.text(
            j,
            i,
            _STATES[state].replace(' ', '\n'),
            ha='center',
            va='center',
            color='#172c36' if state == 0 else '#ffffff',
            fontsize=12,
        )
    stocks.bar(_HOLDERS, fuel, color='#365f77')
    stocks.set_ylim(0, max(1, *fuel) + 1)
    stocks.set_ylabel('Fuel units')
    part_holders = (
        ', '.join(
            f'{owner} {parts[i]}'
            for i, owner in enumerate(_HOLDERS)
            if parts[i]
        )
        or 'none'
    )
    stocks.set_xlabel('Spare parts: ' + part_holders, fontsize=10)
    stocks.set_title(
        'Recorded fuel stocks\nUsed includes pre-play consumption', fontsize=11
    )
    for i, amount in enumerate(fuel):
      stocks.text(i, amount + 0.1, str(amount), ha='center', va='bottom')
    setup = account.get('declared_setup')
    setup_caption = (
        f"Recipe: {setup['recipe']} · {setup['actor_logic']} actors\n"
        f"Initially available: {setup['available_fuel_at_start']} fuel; "
        f"pre-used: {setup['fuel_consumed_before_play']}\n"
        if setup
        else 'Setup provenance not supplied\n'
    )
    figure.supxlabel(
        setup_caption
        + 'Source: '
        + account['schema']
        + '\nRecorded outcomes only · not predictions or a saved game',
        fontsize=10,
    )
    output = io.StringIO()
    figure.savefig(
        output,
        format='svg',
        metadata={
            'Date': None,
            'Creator': 'Concordia Bellwether public accounting',
            'Title': 'Bellwether public service account',
            'Description': description,
        },
    )
    figure.clear()
  root = etree.fromstring(output.getvalue())
  # Matplotlib salts internal clip/glyph IDs. Normalize them for repeatable
  # artifacts, without changing rcParams shared by unrelated plotting code.
  identifiers = {
      node.attrib['id']: f'figure-{i}'
      for i, node in enumerate(root.iter())
      if 'id' in node.attrib
  }
  for node in root.iter():
    for key, value in list(node.attrib.items()):
      if key == 'id':
        node.set(key, identifiers[value])
      elif value.startswith('#') and value[1:] in identifiers:
        node.set(key, '#' + identifiers[value[1:]])
      else:
        node.set(
            key,
            re.sub(
                r'url\(#([^)]+)\)',
                lambda m: 'url(#' + identifiers.get(m[1], m[1]) + ')',
                value,
            ),
        )
  title = etree.Element('{' + _SVG + '}title', {'id': 'figure-title'})
  title.text = 'Bellwether public service and material account'
  desc = etree.Element('{' + _SVG + '}desc', {'id': 'figure-description'})
  desc.text = description
  root.insert(0, desc)
  root.insert(0, title)
  root.set('role', 'img')
  root.set('aria-labelledby', 'figure-title figure-description')
  root.set('width', '100%')
  root.set('height', 'auto')
  root.set('style', 'max-width:720px;height:auto;background:white')
  return etree.tostring(root, encoding='unicode') + '\n'
