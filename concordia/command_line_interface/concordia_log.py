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

"""CLI tool for analyzing Concordia simulation logs.

Usage:
  concordia-log overview sim.json
  concordia-log actions sim.json Alice
  concordia-log context sim.json Alice --step 3
  concordia-log step sim.json 5
  concordia-log timeline sim.json Alice
  concordia-log search sim.json "keyword"
  concordia-log memories sim.json Alice
  concordia-log components sim.json --component tension_tracker
  concordia-log entities sim.json
  concordia-log dump sim.json | jq '...'

Add --json for structured JSON output.
"""

import argparse
import json
import re
import sys

from concordia.utils import structured_logging


_IMAGE_MARKDOWN_PATTERN = re.compile(r'!\[([^\]]*)\]\(data:image/[^)]+\)')


def _strip_images(text: str) -> str:
  def _replacer(match):
    full = match.group(0)
    alt = match.group(1) or 'image'
    data_len = len(full)
    return f'[{alt}: {data_len:,} bytes]'

  return _IMAGE_MARKDOWN_PATTERN.sub(_replacer, text)


def _format_text(text: str, include_images: bool = False) -> str:
  if include_images:
    return str(text)
  return _strip_images(str(text))


def _load_log(
    path: str,
) -> structured_logging.SimulationLog:
  with open(path) as f:
    return structured_logging.SimulationLog.from_json(f.read())


def _load_interface(
    path: str,
) -> tuple[
    structured_logging.AIAgentLogInterface,
    structured_logging.SimulationLog,
]:
  log = _load_log(path)
  return structured_logging.AIAgentLogInterface(log), log


def _print_json(data, include_images: bool = False):
  def _default(obj):
    return str(obj)

  text = json.dumps(data, indent=2, default=_default, ensure_ascii=False)
  if not include_images:
    text = _strip_images(text)
  print(text)


def cmd_overview(args):
  """Show simulation overview statistics."""
  interface, _ = _load_interface(args.log_file)
  overview = interface.get_overview()
  if args.json:
    _print_json(overview)
  else:
    print(f"Steps: {overview.get('total_steps', 0)}")
    print(f"Entries: {overview.get('total_entries', 0)}")
    entities = overview.get('entities', [])
    print(f"Entities ({len(entities)}): {', '.join(entities)}")
    components = overview.get('components', [])
    if components:
      print(f"Log sources: {', '.join(components)}")
    entry_types = overview.get('entry_types', [])
    if entry_types:
      print(f"Entry types: {', '.join(entry_types)}")


def cmd_entities(args):
  """List all entity names."""
  interface, _ = _load_interface(args.log_file)
  overview = interface.get_overview()
  entities = overview.get('entities', [])
  if args.json:
    _print_json(entities)
  else:
    for name in entities:
      print(name)


def cmd_actions(args):
  """Show entity action timeline."""
  interface, _ = _load_interface(args.log_file)
  actions = interface.get_entity_actions(args.entity)
  if args.json:
    _print_json(actions, include_images=args.include_images)
  else:
    if not actions:
      print(f"No actions found for entity '{args.entity}'.", file=sys.stderr)
      return
    for a in actions:
      action_text = _format_text(
          a.get('action', ''), include_images=args.include_images
      )
      action_text = action_text.replace('\n', ' ').strip()
      print(f"Step {a.get('step', '?')}: {action_text}")


def cmd_context(args):
  """Show full action context for one step."""
  interface, _ = _load_interface(args.log_file)
  ctx = interface.get_entity_action_context(args.entity, step=args.step)
  if args.json:
    _print_json(ctx, include_images=args.include_images)
  else:
    if not ctx:
      print(
          f"No context found for '{args.entity}' at step {args.step}.",
          file=sys.stderr,
      )
      return
    action = _format_text(
        ctx.get('action', ''), include_images=args.include_images
    )
    print(f'Action: {action}')
    observations = ctx.get('observations', '')
    if observations:
      print(
          f'\nObservations:\n{_format_text(observations, include_images=args.include_images)}'
      )
    prompt = ctx.get('action_prompt', '')
    if prompt:
      print(
          f'\nPrompt:\n{_format_text(prompt, include_images=args.include_images)}'
      )
    all_components = ctx.get('all_components', {})
    if all_components:
      print('\nComponents:')
      for key, val in all_components.items():
        if key in ('__act__', '__observation__'):
          continue
        val_str = _format_text(str(val), include_images=args.include_images)
        if len(val_str) > 200:
          val_str = val_str[:200] + '...'
        print(f'  {key}: {val_str}')


def cmd_step(args):
  """Show all entries for a specific step."""
  interface, _ = _load_interface(args.log_file)
  entries = interface.get_step_summary(args.step_num, include_content=True)
  if args.json:
    _print_json(entries, include_images=args.include_images)
  else:
    if not entries:
      print(f'No entries for step {args.step_num}.', file=sys.stderr)
      return
    for e in entries:
      entity = e.get('entity_name', '?')
      entry_type = e.get('entry_type', '?')
      summary = _format_text(
          e.get('summary', ''), include_images=args.include_images
      )
      print(f'[{entity}] ({entry_type}): {summary}')


def cmd_timeline(args):
  """Show entity's full timeline."""
  interface, _ = _load_interface(args.log_file)
  timeline = interface.get_entity_timeline(
      args.entity, include_content=args.verbose
  )
  if args.json:
    _print_json(timeline, include_images=args.include_images)
  else:
    if not timeline:
      print(f"No timeline entries for '{args.entity}'.", file=sys.stderr)
      return
    for e in timeline:
      step = e.get('step', '?')
      summary = _format_text(
          e.get('summary', ''), include_images=args.include_images
      )
      print(f'Step {step}: {summary}')


def cmd_search(args):
  """Search log entries by text."""
  interface, _ = _load_interface(args.log_file)
  results = interface.search_entries(args.query)
  if args.json:
    _print_json(results, include_images=args.include_images)
  else:
    if not results:
      print(f"No entries matching '{args.query}'.", file=sys.stderr)
      return
    for e in results:
      step = e.get('step', '?')
      entity = e.get('entity_name', '?')
      summary = _format_text(
          e.get('summary', ''), include_images=args.include_images
      )
      print(f'Step {step} [{entity}]: {summary}')


def cmd_memories(args):
  """Show entity memories."""
  interface, _ = _load_interface(args.log_file)
  memories = interface.get_entity_memories(args.entity)
  if args.json:
    _print_json(memories, include_images=args.include_images)
  else:
    if not memories:
      print(f"No memories for '{args.entity}'.", file=sys.stderr)
      return
    for i, mem in enumerate(memories):
      mem_text = _format_text(str(mem), include_images=args.include_images)
      print(f'  {i + 1}. {mem_text}')


def _discover_components(log, entity_name=None, step=None):
  """Discover component names and keys from a single entry."""
  for entry in log.entries:
    if entry.entry_type != 'entity':
      continue
    if entity_name and entry.entity_name != entity_name:
      continue
    if step is not None and entry.step != step:
      continue
    full_data = log.reconstruct_value(entry.deduplicated_data)
    value_dict = full_data.get('value', {})
    if isinstance(value_dict, dict) and value_dict:
      return entry.entity_name, entry.step, value_dict
  return None, None, {}


def cmd_components(args):
  """List or extract component values."""
  interface, log = _load_interface(args.log_file)

  if args.component is None:
    entity, step, value_dict = _discover_components(
        log, entity_name=args.entity, step=args.step
    )
    if not value_dict or not isinstance(value_dict, dict):
      print('No entity entries found.', file=sys.stderr)
      return
    component_info = {}
    for comp_name, comp_val in value_dict.items():
      if isinstance(comp_val, dict):
        component_info[comp_name] = sorted(comp_val.keys())
      else:
        component_info[comp_name] = [type(comp_val).__name__]
    if args.json:
      _print_json(component_info)
    else:
      print(f'Components for {entity} at step {step}:')
      for comp_name, comp_keys in component_info.items():
        keys_str = ', '.join(comp_keys)
        print(f'  {comp_name}: {keys_str}')
    return

  if args.key is None:
    entity, step, value_dict = _discover_components(
        log, entity_name=args.entity, step=args.step
    )
    if not isinstance(value_dict, dict):
      print(f"Component '{args.component}' not found.", file=sys.stderr)
      return
    comp = value_dict.get(args.component, {})
    if not isinstance(comp, dict) or not comp:
      print(f"Component '{args.component}' not found.", file=sys.stderr)
      return
    keys = sorted(comp.keys())
    if args.json:
      _print_json(keys)
    else:
      print(f'Keys for {args.component} (step {step}, {entity}):')
      for k in keys:
        val_preview = str(comp[k])[:80]
        print(f'  {k}: {val_preview}')
    return

  step_range = None
  if args.step is not None:
    step_range = (args.step, args.step)
  if args.step_range:
    step_range = (args.step_range[0], args.step_range[1])
  values = interface.get_component_values(
      component_key=args.component,
      value_key=args.key,
      entity_name=args.entity,
      step_range=step_range,
  )
  if args.json:
    _print_json(values, include_images=args.include_images)
  else:
    if not values:
      print(f"No values for component '{args.component}'.", file=sys.stderr)
      return
    for v in values:
      step = v.get('step', '?')
      entity = v.get('entity_name', '?')
      val = _format_text(
          str(v.get('value', '')), include_images=args.include_images
      )
      if len(val) > 200:
        val = val[:200] + '...'
      print(f'Step {step} [{entity}]: {val}')


def cmd_dump(args):
  """Dump inflated JSON for jq/grep."""
  _, log = _load_interface(args.log_file)
  entries_to_dump = log.entries

  if args.step is not None:
    entries_to_dump = [e for e in entries_to_dump if e.step == args.step]
  if args.entity:
    entries_to_dump = [
        e for e in entries_to_dump if e.entity_name == args.entity
    ]

  inflated = []
  for entry in entries_to_dump:
    reconstructed = log.reconstruct_value(entry.deduplicated_data)
    inflated.append({
        'step': entry.step,
        'timestamp': entry.timestamp,
        'entity_name': entry.entity_name,
        'component_name': entry.component_name,
        'entry_type': entry.entry_type,
        'summary': entry.summary,
        'data': reconstructed,
    })

  text = json.dumps(inflated, indent=2, ensure_ascii=False, default=str)
  if not args.include_images:
    text = _strip_images(text)
  print(text)


def main(argv=None):
  parser = argparse.ArgumentParser(
      prog='concordia-log',
      description='Analyze Concordia simulation logs from the command line.',
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog=(
          'Examples:\n'
          '  concordia-log overview sim.json\n'
          '  concordia-log entities sim.json\n'
          '  concordia-log actions sim.json Alice\n'
          '  concordia-log context sim.json Alice --step 3\n'
          '  concordia-log step sim.json 5\n'
          '  concordia-log search sim.json "coffee shop"\n'
          '  concordia-log memories sim.json Alice\n'
          '  concordia-log timeline sim.json Alice\n'
          '  concordia-log components sim.json --entity Alice\n'
          '  concordia-log components sim.json --entity Alice'
          ' --component __act__\n'
          '  concordia-log components sim.json'
          ' --component __act__ --key Key\n'
          '  concordia-log components sim.json --entity Alice'
          ' --component __act__ --key Key --step 3\n'
          '  concordia-log dump sim.json | jq ".[] | .data.__act__.Value"\n'
          '  concordia-log actions sim.json Alice | grep "hello"\n'
      ),
  )
  parser.add_argument(
      '--json',
      action='store_true',
      help='Output in JSON format (for piping to jq)',
  )
  parser.add_argument(
      '--include-images',
      action='store_true',
      help='Include base64 image data instead of stripping it',
  )

  subparsers = parser.add_subparsers(dest='command', required=True)

  p = subparsers.add_parser(
      'overview', help='Show simulation overview (steps, entities, etc.)'
  )
  p.add_argument('log_file', help='Path to structured log JSON file')

  p = subparsers.add_parser('entities', help='List all entity names')
  p.add_argument('log_file', help='Path to structured log JSON file')

  p = subparsers.add_parser(
      'actions', help="Show an entity's actions across all steps"
  )
  p.add_argument('log_file', help='Path to structured log JSON file')
  p.add_argument('entity', help='Entity name')

  p = subparsers.add_parser(
      'context',
      help='Show full action context (action + observations + prompt)',
  )
  p.add_argument('log_file', help='Path to structured log JSON file')
  p.add_argument('entity', help='Entity name')
  p.add_argument('--step', type=int, required=True, help='Step number')

  p = subparsers.add_parser('step', help='Show all entries for a specific step')
  p.add_argument('log_file', help='Path to structured log JSON file')
  p.add_argument('step_num', type=int, help='Step number')

  p = subparsers.add_parser(
      'timeline', help='Show chronological timeline for an entity'
  )
  p.add_argument('log_file', help='Path to structured log JSON file')
  p.add_argument('entity', help='Entity name')
  p.add_argument(
      '-v',
      '--verbose',
      action='store_true',
      help='Include full content in timeline entries',
  )

  p = subparsers.add_parser('search', help='Search log entries by text')
  p.add_argument('log_file', help='Path to structured log JSON file')
  p.add_argument('query', help='Search text (case-insensitive)')

  p = subparsers.add_parser('memories', help='Show entity memories')
  p.add_argument('log_file', help='Path to structured log JSON file')
  p.add_argument('entity', help='Entity name')

  p = subparsers.add_parser(
      'components',
      help='List or extract component values',
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog=(
          'Discovery modes:\n'
          '  concordia-log components sim.json --entity Alice\n'
          '    List all components and their keys for Alice\n'
          '  concordia-log components sim.json --entity Alice'
          ' --component __act__\n'
          '    List keys available in the __act__ component\n'
          '\n'
          'Extraction mode:\n'
          '  concordia-log components sim.json'
          ' --component __act__ --key Value\n'
          '    Extract values across all steps'
      ),
  )
  p.add_argument('log_file', help='Path to structured log JSON file')
  p.add_argument(
      '--component',
      default=None,
      help='Component key (omit to list all components)',
  )
  p.add_argument(
      '--key',
      default=None,
      help='Value key within component (omit to list available keys)',
  )
  p.add_argument('--entity', help='Filter by entity name')
  p.add_argument(
      '--step',
      type=int,
      default=None,
      help='Step to inspect for discovery (default: first available)',
  )
  p.add_argument(
      '--step-range',
      nargs=2,
      type=int,
      metavar=('START', 'END'),
      help='Filter by step range when extracting values',
  )

  p = subparsers.add_parser(
      'dump', help='Dump inflated (de-deduplicated) JSON for jq/grep'
  )
  p.add_argument('log_file', help='Path to structured log JSON file')
  p.add_argument('--step', type=int, help='Filter to a specific step')
  p.add_argument('--entity', help='Filter to a specific entity')

  args = parser.parse_args(argv)

  commands = {
      'overview': cmd_overview,
      'entities': cmd_entities,
      'actions': cmd_actions,
      'context': cmd_context,
      'step': cmd_step,
      'timeline': cmd_timeline,
      'search': cmd_search,
      'memories': cmd_memories,
      'components': cmd_components,
      'dump': cmd_dump,
  }
  commands[args.command](args)


if __name__ == '__main__':
  main()
