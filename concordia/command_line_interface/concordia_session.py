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

"""Design interactively or attach with JSON to the editor operation API.

No engine or domain behavior lives here. One-shot output is machine-readable; interactive mode adds a local prompt.
"""

import argparse
import json
import pathlib
import shlex
import sys
import tempfile
from typing import Any
import urllib.error
import urllib.request
import uuid

from concordia.utils import session_commands
from concordia.utils import session_draft


def friendly(args):
  """Execute a shared safe command; files and drafts belong to this CLI client."""
  if not args.line:
    raise ValueError('command requires --line. Use --line help.')
  for destination in (args.file, args.output):
    if (
        args.draft
        and destination
        and destination.resolve() == args.draft.resolve()
    ):
      raise ValueError(
          'The draft journal and import/export file must be different paths.'
      )
  if args.file and args.output and args.file.resolve() == args.output.resolve():
    raise ValueError('Import and output paths must be different.')
  plan = session_commands.parse(args.line)
  if plan['kind'] == 'text':
    return plan

  def request(endpoint, body=None):
    req = urllib.request.Request(
        args.url.rstrip('/') + '/api/' + endpoint,
        data=None if body is None else json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'},
    )
    try:
      with urllib.request.urlopen(req, timeout=args.timeout) as response:
        return json.loads(response.read())
    except urllib.error.HTTPError as error:
      try:
        value = json.loads(error.read())
        message = value.get('error', {}).get('message', 'Operation failed.')
      except (ValueError, AttributeError):
        message = 'Service returned a non-JSON error.'
      raise ValueError(f'HTTP {error.code}: {message}') from None

  if plan.get('action') == 'discover':
    return request('operations')['result']
  if plan.get('action') == 'watch':
    raise ValueError('Use the command entry point to stream watch events.')

  envelope = request('state')
  state = envelope['result']

  def dispatch(name, values):
    return request(
        'dispatch',
        {
            'operation': name,
            'arguments': values,
            'revision': envelope['revision'],
            'references': envelope['references'],
            'retry_key': str(uuid.uuid4()),
        },
    )['result']

  if plan['kind'] == 'log':
    imported = ''
    if plan['source'] == 'imported':
      if args.file is None:
        raise ValueError('Imported log commands require --file LOG.json.')
      imported = args.file.read_text(encoding='utf-8')
    result = dispatch(
        'log.query',
        {
            'command': plan['command'],
            'source': plan['source'],
            'arguments': json.dumps(plan['args']),
            'imported': imported,
        },
    )
    if result.get('download'):
      if args.output is None:
        raise ValueError(
            'Log export/dump/bundle requires --output DESTINATION on this CLI'
            ' client.'
        )
      args.output.write_text(
          result.pop('download')['content'], encoding='utf-8'
      )
      result['output'] = str(args.output)
    return result
  action = 'call' if plan['kind'] == 'call' else plan['action']
  journal = None
  if args.draft:
    if args.file is not None and args.file.resolve() == args.draft.resolve():
      raise ValueError(
          'The draft journal and import/export file must be different paths.'
      )
    if args.draft.exists():
      journal = json.loads(args.draft.read_text(encoding='utf-8'))
      if not isinstance(journal, dict) or not {
          'document',
          'base',
          'revision',
          'metadata',
          'selectedId',
          'session_id',
      }.issubset(journal):
        raise ValueError(
            'Expected a CLI draft journal. Import project JSON with load --file'
            ' instead.'
        )
    else:
      journal = {
          'document': state['document'],
          'base': state['document'],
          'preview_document': state['document'],
          'revision': state['revision'],
          'metadata': state['definition'],
          'selectedId': state['document']['instances'][0]['id'],
          'view': 'definition',
          'past': [],
          'future': [],
          'session_id': envelope['references']['session_id'],
      }
  if (
      journal
      and action not in ('state', 'reload', 'export')
      and journal.get('session_id') != envelope['references']['session_id']
  ):
    raise ValueError(
        'Server session changed; export your local draft or reload --discard'
        ' before continuing.'
    )
  if action == 'layout':
    raise ValueError(
        'layout sizes browser panes only. Resize your terminal window; this CLI'
        ' does not control another client.'
    )
  if action == 'panel' and plan['args'][0] in ('simulation', 'log'):
    return {
        'run': state['run'],
        'state': state['state'],
        'steps': state.get('steps', []),
        'hint': 'Use watch for events or log commands for structured analysis.',
    }
  if action == 'state':
    return {'session': state, 'client': journal}
  if plan['kind'] == 'call':
    if journal and (
        journal['document'] != journal['base']
        or journal['revision'] != state['revision']
    ):
      raise ValueError(
          'Unsaved or stale client draft: save or reload before call.'
      )
    return dispatch(plan['operation'], plan['arguments'])
  operation = session_commands.operation(plan, state)
  if operation:
    if (
        action == 'run'
        and journal
        and (
            journal['document'] != journal['base']
            or journal['revision'] != state['revision']
        )
    ):
      raise ValueError(
          'Unsaved or stale client draft: save or explicitly reload before run.'
      )
    result = dispatch(*operation)
    if action == 'run':
      return {
          'message': (
              'Run accepted; completion is not implied. Use state or watch for'
              ' waiting, progress and failure.'
          ),
          'result': result,
          'status': request('state')['result']['run'],
      }
    return result
  if action == 'log-import':
    raise ValueError(
        'CLI log imports use --file LOG.json with log COMMAND --source'
        ' imported.'
    )
  if journal is None:
    raise ValueError(
        'This client-owned action requires --draft JOURNAL.json. It cannot edit'
        ' another browser tab.'
    )
  if action == 'export':
    if args.file is None:
      raise ValueError('export requires --file DESTINATION.json.')
    args.file.write_text(
        json.dumps(journal['document'], indent=2), encoding='utf-8'
    )
    result = {'output': str(args.file)}
  elif action in ('save', 'validate'):
    values: dict[str, Any] = {'text': json.dumps(journal['document'])}
    if action == 'save':
      values['revision'] = journal['revision']
    result = dispatch('project.' + action, values)
    if action == 'save':
      journal['base'] = result['document']
      journal['preview_document'] = result['document']
      journal['revision'] = result['revision']
      journal['metadata'] = request('state')['result']['definition']
  elif action == 'reload':
    journal.update(
        document=state['document'],
        base=state['document'],
        preview_document=state['document'],
        revision=state['revision'],
        metadata=state['definition'],
        past=[],
        future=[],
        session_id=envelope['references']['session_id'],
        selectedId=state['document']['instances'][0]['id'],
        view='definition',
    )
    result = {
        'message': 'Reloaded saved definition; local draft/history discarded.'
    }
  elif action == 'load':
    if journal['document'] != journal['base']:
      raise ValueError(
          'Unsaved draft: save/export or reload --discard before load.'
      )
    if args.file is None:
      raise ValueError('load requires --file PROJECT.json.')
    text = args.file.read_text(encoding='utf-8')
    preview = dispatch('project.preview', {'text': text})
    journal['metadata'] = preview['definition']
    journal['document'] = preview['document']
    journal['preview_document'] = preview['document']
    journal['selectedId'] = preview['document']['instances'][0]['id']
    journal['view'] = 'definition'
    journal['past'] = []
    journal['future'] = []
    result = {
        'message': (
            'Loaded local draft; save explicitly to publish to the session.'
        )
    }
  else:
    if state['run']['status'] == 'active' and action not in (
        'select',
        'inspect',
        'search',
        'view',
        'panel',
        'references',
        'catalog',
        'list',
        'locate',
    ):
      raise ValueError('Authoring unavailable while a run is active.')
    applied = session_draft.apply(
        journal, plan, state.get('runtime'), state['document']
    )
    journal, result = applied['journal'], applied['result']
  # Replace the complete journal atomically: a failed write must not truncate
  # an existing unsaved draft. Temporary files stay on the same filesystem.
  temporary = None
  try:
    with tempfile.NamedTemporaryFile(
        mode='w', encoding='utf-8', dir=args.draft.parent, delete=False
    ) as stream:
      temporary = pathlib.Path(stream.name)
      json.dump(journal, stream, indent=2)
    temporary.replace(args.draft)
  finally:
    if temporary is not None:
      temporary.unlink(missing_ok=True)
  return result


def interactive(args) -> int:
  """Persistent client context, explicit journal, no shell or automatic replay."""
  print(
      'Concordia session. help for commands; files for local paths; history;'
      ' exit.\nOnly save publishes a draft. No commands are replayed after'
      ' errors.\nRun is asynchronous: use state, or watch (Ctrl-C returns'
      ' here).'
  )
  # Do not load/write readline history, or let terminal input enter its global
  # history. Only explicit submitted commands enter this loop’s memory history,
  # restored to its previous state on exit. No history file is accessed.
  try:
    import readline  # pylint: disable=import-outside-toplevel
  except ImportError:
    readline = None
  previous_history = []
  if readline is not None:
    previous_history = [
        readline.get_history_item(i + 1)
        for i in range(readline.get_current_history_length())
    ]
    readline.clear_history()
    readline.set_auto_history(False)
  history = []
  imported = None
  try:
    while True:
      try:
        line = input('concordia> ').strip()
      except EOFError:
        print(
            'Disconnected locally. Journal retained; no save/reset performed.'
        )
        return 0
      except KeyboardInterrupt:
        print('\nInput cancelled. Use exit to leave; session unchanged.')
        continue
      if not line:
        continue
      if line in ('exit', 'quit'):
        print('Journal retained; no save/reset performed.')
        return 0
      if line == 'history':
        print('\n'.join(f'{i + 1}: {value}' for i, value in enumerate(history)))
        continue
      if line == 'files':
        print(
            'load PATH | export PATH | log import PATH\nlog dump|bundle'
            ' --source current|imported --output PATH\nPaths are on this CLI'
            ' client. No shell expansion. No overwrite of journal/import.'
        )
        continue
      if line.startswith('!'):
        try:
          number = int(line[1:])
          if not 1 <= number <= len(history):
            raise ValueError()
          line = history[number - 1]
          print('Recall only (not executed): ' + line)
          print('Copy it to the prompt to execute explicitly.')
        except ValueError:
          print('Use !NUMBER to display a history entry; nothing is executed.')
        continue
      history.append(line)
      if readline is not None:
        readline.add_history(line)
      local = argparse.Namespace(**vars(args))
      local.file, local.output = None, None
      try:
        words = shlex.split(line)
        if words[:2] == ['log', 'import']:
          if len(words) != 3:
            raise ValueError('Use log import PATH.')
          candidate = pathlib.Path(words[2])
          # Parse now to report missing/invalid files without changing selection.
          value = json.loads(candidate.read_text(encoding='utf-8'))
          if not isinstance(value, dict):
            raise ValueError('Expected a structured JSON log object.')
          imported = candidate
          print('Imported log selected on this client; no project change.')
          continue
        if words[0] in ('load', 'export') and len(words) == 2:
          local.file = pathlib.Path(words.pop())
        if words[0] == 'log':
          if '--output' in words:
            index = words.index('--output')
            if index + 1 >= len(words):
              raise ValueError('--output requires a local path.')
            local.output = pathlib.Path(words[index + 1])
            del words[index : index + 2]
          if 'imported' in words:
            local.file = imported
        local.line = shlex.join(words)
        if words == ['watch']:
          main(['--url', args.url, '--timeout', str(args.timeout), 'watch'])
          print('Watch ended. No reconnect/replay; use state to refresh.')
        else:
          result = friendly(local)
          print(
              result['text']
              if isinstance(result, dict) and result.get('kind') == 'text'
              else json.dumps(result, indent=2, ensure_ascii=False)
          )
      except KeyboardInterrupt:
        print(
            '\nInterrupted. A submitted operation may have been accepted;'
            ' inspect state before retrying.'
        )
      except (OSError, ValueError, urllib.error.URLError) as error:
        print(
            f'Error: {error}\nDraft retained. Commands are not retried; inspect'
            ' state after transport failure.'
        )
  finally:
    if readline is not None:
      readline.clear_history()
      for item in previous_history:
        if item is not None:
          readline.add_history(item)
      readline.set_auto_history(True)


def main(argv=None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--url', required=True, help='Trusted local service URL')
  parser.add_argument(
      'command',
      choices=('discover', 'state', 'call', 'watch', 'command', 'interactive'),
  )
  parser.add_argument(
      '--input', default='-', help='JSON request file or - for stdin'
  )
  parser.add_argument('--line', help='Friendly command, e.g. run --steps 10')
  parser.add_argument(
      '--draft',
      type=pathlib.Path,
      help=(
          'Explicit client-owned draft journal (requires Node for authoring'
          ' actions)'
      ),
  )
  parser.add_argument(
      '--output', type=pathlib.Path, help='Explicit log dump/bundle output file'
  )
  parser.add_argument(
      '--file',
      type=pathlib.Path,
      help='Explicit client-side import/export file',
  )
  parser.add_argument('--timeout', type=float, default=15)
  args = parser.parse_args(argv)
  if args.command == 'interactive':
    if args.draft is None:
      parser.error('interactive requires --draft JOURNAL.json.')
    return interactive(args)
  if args.command == 'command':
    try:
      if session_commands.parse(args.line or '').get('action') == 'watch':
        return main(
            ['--url', args.url, '--timeout', str(args.timeout), 'watch']
        )
      print(json.dumps(friendly(args), ensure_ascii=False))
      return 0
    except (OSError, ValueError, urllib.error.URLError) as error:
      print(
          json.dumps(
              {'error': {'code': 'command_failed', 'message': str(error)}}
          ),
          file=sys.stderr,
      )
      return 2
  endpoint = {
      'discover': 'operations',
      'state': 'state',
      'call': 'dispatch',
      'watch': 'events',
  }[args.command]
  try:
    data = None
    if args.command == 'call':
      raw = (
          sys.stdin.read()
          if args.input == '-'
          else pathlib.Path(args.input).read_text(encoding='utf-8')
      )
      data = json.dumps(json.loads(raw), ensure_ascii=False).encode()
    request = urllib.request.Request(
        args.url.rstrip('/') + '/api/' + endpoint,
        data=data,
        headers={'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
      if args.command == 'watch':
        for line in response:
          if line.startswith(b'data: '):
            print(line[6:].decode().strip(), flush=True)
      else:
        print(response.read().decode())
    return 0
  except urllib.error.HTTPError as error:
    print(error.read().decode(), file=sys.stderr)
    return 2
  except (OSError, ValueError) as error:
    print(
        json.dumps(
            {'error': {'code': 'transport_or_input', 'message': str(error)}}
        ),
        file=sys.stderr,
    )
    return 3


if __name__ == '__main__':
  raise SystemExit(main())
