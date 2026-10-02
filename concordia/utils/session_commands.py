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

"""Shared editor/session command grammar; plans reuse existing domain actions.

Parsing never executes code, accesses paths, or mutates a session. Clients own
unsaved drafts, history, selection and file dialogs; OperationService owns
validation, saves and runtime changes.
"""

import argparse
import json
import shlex

HELP = """Session: help, state, discover, call OPERATION JSON, watch, run [--steps N], pause, step, play, reset
discover lists registered operations; call accepts a quoted JSON object of arguments.
watch reports the browser live stream; CLI watch streams until interrupted/timeout.
Run starts a fresh saved definition; play resumes a paused run. step grants one step.
Draft: save, validate, load, export, reload --discard, undo, redo
Authoring: add instance PROTOTYPE | add component TYPE OWNER | add group | add scene-type | add scene
  duplicate [ID], remove [ID], set ID FIELD JSON, references ID, replace SOURCE TARGET, move-component ID OWNER
  ID is an instance ID, simulation, or components:ID / groups:ID / scene_types:ID / scenes:ID.
  set uses a top-level record field or params.FIELD; values are JSON (quote strings).
View: select ID [COMPONENT], search TEXT, view definition|runtime, panel hierarchy|inspector|simulation|log, inspect [ID]
Runtime: edit INSTANCE COMPONENT TEXT (paused only; Instructions/Goal)
Logs: log import; log overview|entities|actions|context|step|timeline|search|memories|components|dump|bundle --source current|imported [concordia-log arguments]
  log step inspects a recorded step; step advances execution. Log dump/bundle export downloads in browser, explicit files in CLI.
Files and draft/history are local to each client; commands never silently save or discard.
"""


class CommandParser(argparse.ArgumentParser):

  def exit(self, status=0, message=None):
    raise ValueError(message or 'Use help for command syntax.')

  def error(self, message):
    raise ValueError(message)


def parse(line: str) -> dict:
  """Return a validated command plan with no shell or arbitrary-code execution."""
  words = shlex.split(line)
  if not words:
    raise ValueError('Enter a command; help lists commands.')
  name, *args = words
  if name == 'help' and not args:
    return {'kind': 'text', 'text': HELP}
  if name in (
      'state',
      'discover',
      'watch',
      'pause',
      'step',
      'play',
      'reset',
      'save',
      'validate',
      'load',
      'export',
      'undo',
      'redo',
  ):
    if args:
      raise ValueError(f'{name} takes no arguments.')
    return {'kind': 'action', 'action': name, 'args': []}
  if name == 'call':
    if len(args) != 2:
      raise ValueError('Use call OPERATION followed by a quoted JSON object.')
    values = json.loads(args[1])
    if not isinstance(values, dict):
      raise ValueError('call arguments must be a JSON object.')
    return {'kind': 'call', 'operation': args[0], 'arguments': values}
  if name == 'run':
    parser = CommandParser(add_help=False)
    parser.add_argument('--steps', type=int)
    values = parser.parse_args(args)
    return {'kind': 'action', 'action': name, 'args': [values.steps]}
  if name == 'log':
    if args == ['import']:
      return {'kind': 'action', 'action': 'log-import', 'args': []}
    if not args or args[0] not in (
        'overview',
        'entities',
        'actions',
        'context',
        'step',
        'timeline',
        'search',
        'memories',
        'components',
        'dump',
        'bundle',
    ):
      raise ValueError('Unknown log command. Use help.')
    parser = CommandParser(add_help=False)
    parser.add_argument(
        '--source', required=True, choices=('current', 'imported')
    )
    values, rest = parser.parse_known_args(args[1:])
    return {
        'kind': 'log',
        'command': args[0],
        'source': values.source,
        'args': rest,
    }
  counts = {
      'duplicate': (0, 1),
      'remove': (0, 1),
      'select': (1, 2),
      'search': (1, 1),
      'view': (1, 1),
      'panel': (1, 1),
      'inspect': (0, 1),
      'set': (3, 3),
      'references': (1, 1),
      'replace': (2, 2),
      'move-component': (2, 2),
      'edit': (3, 3),
      'reload': (1, 1),
  }
  if name == 'add':
    kinds = {
        'instance': 2,
        'component': 3,
        'group': 1,
        'scene-type': 1,
        'scene': 1,
    }
    if not args or len(args) != kinds.get(args[0]):
      raise ValueError(
          'Use add instance PROTOTYPE, add component TYPE OWNER, or add'
          ' group|scene-type|scene.'
      )
  elif (
      name not in counts or not counts[name][0] <= len(args) <= counts[name][1]
  ):
    raise ValueError('Unknown command or wrong arguments. Use help.')
  if name == 'reload' and args != ['--discard']:
    raise ValueError('reload --discard explicitly discards this client draft.')
  if name == 'view' and args[0] not in ('definition', 'runtime'):
    raise ValueError('view requires definition or runtime.')
  if name == 'panel' and args[0] not in (
      'hierarchy',
      'inspector',
      'simulation',
      'log',
  ):
    raise ValueError('Unknown editor panel.')
  return {'kind': 'action', 'action': name, 'args': args}


def operation(plan: dict, snapshot: dict) -> tuple[str, dict] | None:
  """Map shared session commands to authoritative existing operations."""
  action, args = plan.get('action'), plan.get('args', [])
  if action in ('pause', 'step', 'play'):
    return 'runtime.' + action, {}
  if action == 'reset':
    return 'project.reset', {}
  if action == 'edit':
    return 'runtime.edit', dict(
        zip(('instance_id', 'component', 'value'), args)
    )
  if action == 'run':
    if snapshot['state'] == 'paused':
      raise ValueError(
          'A run is paused. Use play to resume or reset before a fresh run.'
      )
    values = {'revision': snapshot['revision']}
    if snapshot.get('run_limits'):
      values['requested_steps'] = (
          args[0]
          if args[0] is not None
          else snapshot['run_limits']['default_requested_steps']
      )
    elif args[0] is not None:
      raise ValueError('This runner does not accept a requested length.')
    return 'project.run', values
  return None
