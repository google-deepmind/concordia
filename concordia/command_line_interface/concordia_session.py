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

"""Attach noninteractively to the same operation API used by the editor.

No engine or domain behavior lives here. JSON output is always machine-readable.
"""

import argparse
import json
import pathlib
import sys
import urllib.error
import urllib.request


def main(argv=None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--url', required=True, help='Trusted local service URL')
  parser.add_argument('command', choices=('discover', 'state', 'call', 'watch'))
  parser.add_argument(
      '--input', default='-', help='JSON request file or - for stdin'
  )
  parser.add_argument('--timeout', type=float, default=15)
  args = parser.parse_args(argv)
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
