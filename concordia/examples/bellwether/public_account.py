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

"""Portable public records, not executable replay or component checkpoints."""

from collections.abc import Mapping
import copy
import html
import json

from concordia.examples.bellwether import game
from concordia.examples.bellwether import public_figure
from concordia.utils import operation_service as ops

NOTICE = (
    'Public events only. Private journals, component state and service identity'
    ' are excluded. Public speech can still disclose identifying information;'
    ' this is not anonymization. This account is not a checkpoint, executable'
    ' replay, or evidence about real people.'
)


SETUP_NOTICE = (
    'Declared teaching setup only, not a complete run configuration.'
    ' Private component edits, prompts and model settings are not exported;'
    ' this does not establish deterministic replay or empirical validity.'
)


def setup_text(account: dict) -> str:
  """Human-readable provenance for both public HTML and the SVG description."""
  setup = account.get('declared_setup')
  if setup is None:
    return 'Setup provenance was not supplied. ' + SETUP_NOTICE
  return (
      f"Recipe: {setup['recipe']}. Actor prefab: {setup['actor_logic']}. "
      + 'Human roles: '
      + ', '.join(setup['human_roles'])
      + '. '
      + f"Engine: {setup['engine']}. "
      + f"Declared before play: {setup['fuel_consumed_before_play']} fuel"
      ' used; '
      + f"{setup['available_fuel_at_start']} fuel initially available. "
      + SETUP_NOTICE
  )


def document(
    world: game.StormNight,
    *,
    fixture: bool,
    phase: str,
    manifest: Mapping | None = None,
) -> dict:
  """Project an explicit public allowlist from the standard spectator view."""
  public = world.view('spectator')
  epilogue = public['epilogue']
  return {
      'schema': 'bellwether-public-account/v1',
      'scope': 'public',
      'setup_notice': SETUP_NOTICE,
      'declared_setup': (
          {
              key: copy.deepcopy(manifest[key])
              for key in (
                  'recipe',
                  'actor_logic',
                  'human_roles',
                  'engine',
                  'fuel_total_including_preconsumed',
                  'fuel_consumed_before_play',
                  'available_fuel_at_start',
                  'evidence_class',
              )
          }
          if manifest is not None
          else None
      ),
      'backend': 'fixture' if fixture else 'live',
      'status': (
          'completed'
          if phase == 'completed'
          else 'interrupted'
          if phase == 'failed'
          else 'in_progress'
      ),
      'watch': public['watch'],
      'notice': NOTICE,
      'events': [
          {
              'number': index,
              'watch': (
                  game.WATCHES[event['watch']] if event['watch'] < 3 else 'Dawn'
              ),
              'kind': event['kind'],
              'text': event['text'],
          }
          for index, event in enumerate(public['journal'], start=1)
      ],
      'accounting': {
          'inventory': {
              owner: {
                  item: public['inventory'][owner][item]
                  for item in ('fuel', 'part')
              }
              for owner in ('Generator', 'Nell', 'Ivo', 'Used')
          },
          'repair_completed': public['repair'],
          'services': [
              {
                  key: record[key]
                  for key in (
                      'watch',
                      'facility',
                      'served',
                      'consequence',
                      'demand',
                  )
                  if key in record
              }
              | (
                  {
                      'resolution': {
                          key: record['resolution'][key]
                          for key in (
                              'basis',
                              'requested',
                              'priority',
                              'fuel_before',
                              'fuel_spent',
                              'explanation',
                          )
                          if key in record['resolution']
                      }
                  }
                  if 'resolution' in record
                  else {}
              )
              for record in public['services']
          ],
      },
      'dawn': (
          {
              key: epilogue[key]
              for key in (
                  'services_maintained',
                  'services_total',
                  'fuel_used',
                  'repair_completed',
                  'dispute_settled',
              )
          }
          if epilogue
          else None
      ),
  }


def render_html(account: dict) -> str:
  """Render only a projected document as inert, offline UTF-8 HTML."""
  escape = html.escape
  events = ''.join(
      '<li><h3>'
      + escape(event['watch'])
      + '</h3><p>'
      + escape(event['text'])
      + '</p><small>'
      + escape(event['kind'])
      + '</small></li>'
      for event in account['events']
  )
  if not events:
    events = '<li>No public events recorded yet.</li>'

  def cells(values):
    return ''.join('<td>' + escape(str(value)) + '</td>' for value in values)

  material = account['accounting']
  stocks = ''.join(
      '<tr>' + cells((owner, items['fuel'], items['part'])) + '</tr>'
      for owner, items in material['inventory'].items()
  )
  services = (
      ''.join(
          '<li><strong>'
          + escape(record['watch'])
          + ' · '
          + escape(record['facility'])
          + '</strong><p>'
          + escape(record['consequence'])
          + '</p><p>'
          + escape(
              record.get('resolution', {}).get(
                  'explanation', 'Boundary details were not recorded.'
              )
          )
          + '</p></li>'
          for record in material['services']
      )
      or '<li>No watch boundary has been resolved yet.</li>'
  )
  dawn = account['dawn']
  conclusion = (
      str(dawn['services_maintained'])
      + ' of '
      + str(dawn['services_total'])
      + ' facility-watches supplied. '
      + str(dawn['fuel_used'])
      + ' fuel consumed in the cumulative Used ledger (including any declared'
      ' pre-play consumption). '
      + (
          'The beacon was repaired.'
          if dawn['repair_completed']
          else 'The beacon was not repaired.'
      )
      + ' These records do not settle the previous-storm dispute.'
      if dawn
      else 'No dawn outcome has been recorded yet.'
  )
  accounting = (
      '<table><caption>Recorded stocks</caption><thead><tr>'
      '<th scope="col">Holder</th><th scope="col">Fuel</th>'
      '<th scope="col">Spare parts</th></tr></thead><tbody>'
      + stocks
      + '</tbody></table><h3>Service consequences</h3><ul>'
      + services
      + '</ul><h3>Dawn</h3><p>'
      + escape(conclusion)
      + '</p>'
  )
  return (
      '<!doctype html><html lang="en"><meta charset="utf-8">'
      '<meta name="viewport" content="width=device-width,initial-scale=1">'
      '<meta http-equiv="Content-Security-Policy" '
      "content=\"default-src 'none'; style-src 'unsafe-inline';"
      " base-uri 'none'; form-action 'none'\">"
      '<title>Bellwether · Public account</title><style>'
      'body{font:1.1rem/1.6 system-ui,sans-serif;max-width:48rem;'
      'margin:auto;padding:1.2rem;color:#162934;background:#f8faf9;'
      'overflow-wrap:anywhere}li{margin-bottom:1.5rem}'
      'p{white-space:pre-wrap;overflow-wrap:anywhere}'
      'table{border-collapse:collapse;width:100%}'
      'td,th{text-align:left;padding:.4rem;border-bottom:1px solid #aabbb8}'
      'caption{text-align:left;font-weight:bold}small{color:#465a64}</style>'
      '<main><h1>Bellwether · Public account</h1><p>'
      + escape(account['backend'].capitalize())
      + ' · '
      + escape(account['status'].replace('_', ' ').capitalize())
      + ' · '
      + escape(account['watch'])
      + '</p><p>'
      + escape(account['notice'])
      + '</p><h2>Declared setup</h2><p>'
      + escape(setup_text(account))
      + '</p><h2>Public timeline</h2><ol>'
      + events
      + '</ol><h2>Recorded material consequences</h2>'
      + accounting
      + '</main></html>'
  )


def export(
    world: game.StormNight,
    *,
    fixture: bool,
    phase: str,
    format_name: str,
    manifest: Mapping | None = None,
) -> dict:
  """Return artifact content without transport/session envelope identifiers."""
  if format_name not in ('json', 'html', 'svg'):
    raise ops.OperationError('invalid_format', 'Choose json, html or svg.')
  account = document(world, fixture=fixture, phase=phase, manifest=manifest)
  return {
      'filename': 'bellwether-public-account.' + format_name,
      'media_type': (
          'application/json; charset=utf-8'
          if format_name == 'json'
          else 'image/svg+xml; charset=utf-8'
          if format_name == 'svg'
          else 'text/html; charset=utf-8'
      ),
      'content': (
          json.dumps(account, ensure_ascii=False, indent=2) + '\n'
          if format_name == 'json'
          else public_figure.render(
              account, setup_description=setup_text(account)
          )
          if format_name == 'svg'
          else render_html(account)
      ),
  }
