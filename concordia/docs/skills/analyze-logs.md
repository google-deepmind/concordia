---
name: analyze-logs
description: >
  Helps debug Concordia simulation outputs by analyzing agent behavior, checking
  if high-level ideas worked, and identifying issues. Use when running
  simulations, verifying agent behavior, or investigating unexpected outcomes.
---

# Debug Simulation Skill

This skill helps you analyze and debug Concordia simulation outputs to verify
that high-level simulation ideas from the user have worked correctly.

## When to Use This Skill

- After running a simulation to check if it behaved as expected
- When investigating why agents made unexpected decisions
- When verifying that game masters are behaving correctly
- When comparing simulation behavior before/after code changes
- When the user asks "did it work?" about a simulation

## Primary Approach: `concordia-log` CLI Tool

The fastest way to analyze logs is via the `concordia-log` command-line tool.
It works with pipes, grep, and jq.

**Running the tool:**

- **From source:** `./concordia/command_line_interface/concordia-log <subcommand> <args>`
- **After pip install:** `concordia-log <subcommand> <args>`

### Quick Overview

Always start here to understand the scope of a simulation:

```bash
concordia-log overview sim_structured.json
# Output:
# Steps: 5
# Entries: 15
# Entities (3): Alice, Bob, default rules
# Log sources: entity_action, game_master
# Entry types: entity, step
```

### What Did an Entity Do?

```bash
concordia-log actions sim_structured.json Alice
# Output:
# Step 1: Alice said "Hello Bob, nice to meet you."
# Step 2: Alice offered to buy coffee for Bob.
# Step 3: Alice walked to the park.
```

### Why Did They Do It? (Full Context)

```bash
concordia-log context sim_structured.json Alice --step 3
# Output:
# Action: Alice walked to the park.
#
# Observations:
# Bob mentioned the park was nice today.
#
# Prompt:
# [full LLM prompt that produced the action]
#
# Components:
#   Instructions: You are Alice, a friendly baker...
#   SituationPerception: Alice is at the coffee shop...
```

### Drill Into a Specific Step

```bash
concordia-log step sim_structured.json 3
# Output:
# [Alice] (entity): Alice walked to the park.
# [Bob] (entity): Bob followed Alice to the park.
# [default rules] (step): Alice and Bob arrived at the park.
```

### Search for Keywords

```bash
concordia-log search sim_structured.json "coffee shop"
# Output:
# Step 2 [Alice]: Alice offered to buy coffee for Bob.
# Step 2 [default rules]: Alice and Bob went to the coffee shop.
```

### Entity Memories

```bash
concordia-log memories sim_structured.json Alice
# Output:
#   1. Alice loves hiking in the mountains.
#   2. Alice works as a baker.
#   3. Alice met Bob at the coffee shop.
```

### Discover Components on an Entity

```bash
# What components does Alice have?
concordia-log components sim_structured.json --entity Alice
# Output:
# Components for Alice at step 1:
#   Instructions: Key, Value
#   SituationPerception: Chain of thought, Key, State, Summary
#   __act__: Prompt, Summary, Value
#   __observation__: Key, Value

# What keys does the __act__ component have?
concordia-log components sim_structured.json --entity Alice --component __act__
# Output:
# Keys for __act__ (step 1, Alice):
#   Prompt: ['', 'Instructions:', 'The instructions for...
#   Summary: Action: Alice said hello...
#   Value: Alice said hello...
```

### Extract Component Values

```bash
concordia-log components sim_structured.json --component tension_tracker --key Key
# Output:
# Step 1 [default rules]: 0.2
# Step 2 [default rules]: 0.4
# Step 3 [default rules]: 0.6
```

### Raw Data for jq/grep

```bash
# Dump all entries as inflated JSON (references resolved)
concordia-log dump sim_structured.json | jq '.[] | {step, entity_name, action: .data.__act__.Value}'

# Filter to one entity
concordia-log dump sim_structured.json --entity Alice | jq '.'

# Filter to one step
concordia-log dump sim_structured.json --step 3 | jq '.'
```

### List Entities (Useful for Scripting)

```bash
concordia-log entities sim_structured.json
# Output:
# Alice
# Bob
# default rules
```

### Common CLI Patterns

```bash
# Pipe to grep
concordia-log actions sim_structured.json Alice | grep "hello"

# JSON output for jq
concordia-log actions sim_structured.json Alice --json | jq '.[].action'

# Compare two entities
diff <(concordia-log actions run.json Alice) <(concordia-log actions run.json Bob)

# Compare two runs
diff <(concordia-log overview run_a.json) <(concordia-log overview run_b.json)

# Loop over all entities
for entity in $(concordia-log entities run.json); do
  echo "=== $entity ==="
  concordia-log actions run.json "$entity"
done
```

### Image-Heavy Logs

Base64 image data is automatically stripped from output and replaced with
`[image: N bytes]`. Use `--include-images` to include raw image data.

### Full CLI Reference

Run `concordia-log --help` for all subcommands and options.

| Subcommand | Purpose |
|-----------|---------|
| `overview` | High-level stats (steps, entities, entry count) |
| `entities` | List all entity names |
| `actions` | Entity action timeline |
| `context` | Full action + observations + prompt at a step |
| `step` | All entries for a specific step |
| `timeline` | Chronological timeline for an entity |
| `search` | Search entries by text |
| `memories` | Entity memories |
| `components` | Extract specific component values |
| `dump` | Inflated JSON for jq/grep |

**Global flags:** `--json` (structured output),
`--include-images` (include base64)

---

## Visual Log Viewer

For **human users**, the easiest way to browse logs interactively is to open
`utils/log_viewer.html` in a browser and load the structured log JSON file.
The viewer renders inline images and supports lazy expansion of large entries.

---

## Secondary Approach: Python API

For programmatic analysis, LLM-based verification, or integration into Python
scripts, use the `AIAgentLogInterface` directly.

### Setup

```python
from concordia.utils.structured_logging import AIAgentLogInterface, SimulationLog

# After running simulation
log = simulation.play(return_structured_log=True)
interface = AIAgentLogInterface(log)

# Or load from file
log = SimulationLog.from_json(open('path/to/structured.json').read())
interface = AIAgentLogInterface(log)
```

### Understanding the Log Data Structure

> **IMPORTANT:** The structured log uses content deduplication. Long strings in
> log entries are replaced with `{'_ref': content_id}` references that point
> into a separate `content_store`. You **must** resolve these references to
> read the actual content.

The JSON has this top-level structure:
```python
{
    'content_store': { 'hash_id': 'actual_text', ... },
    'entries': [ { 'step': 2, 'entity_name': '...', 'deduplicated_data': {...} }, ... ],
    'entity_memories': { ... },
    'game_master_memories': [ ... ]
}
```

Each entry's `deduplicated_data` has a `key` and a `value`. For entity entries:

```python
{
    'key': 'Entity [Alice]',
    'value': {
        'Instructions': {'Key': "Alice's Identity", 'Value': '...'},
        '__act__': {
            'Summary': '...',
            'Value': 'Alice said hello...',  # The actual entity action
            'Prompt': '...',  # The full prompt sent to the LLM
        },
        ...
    }
}
```

For game master entries, the value is organized into **phases**. The `resolve`
phase is typically where the most interesting data lives:

```python
{
    'key': 'game_master_name',
    'value': {
        'resolve': {
            '__resolution__': {
                'Summary': '...', 'Value': '...', 'Prompt': '...',
            },
            'tension_tracker': { '__act__': { 'Value': '0.6' } },
            ...
        },
    }
}
```

### Python API: Extracting Entity Actions

```python
actions = interface.get_entity_actions('Alice')
for a in actions:
    print(f"Step {a['step']}: {str(a['action'])[:200]}")
```

### Python API: Full Action Context

```python
context = interface.get_entity_action_context('Alice', step=3)
if context:
    print(f"Action: {context['action']}")
    print(f"Observations: {context['observations']}")
    print(f"Prompt: {context['action_prompt']}")
```

### Python API: Component Values

```python
interface.get_component_values(
    component_key='Observation',
    value_key='Value',
    entity_name='Alice',
    step_range=(1, 5),
)
```

### Python API: Extracting GM Component Values

```python
for entry in log.entries:
    if entry.entry_type != 'step':
        continue
    full_data = log.reconstruct_value(entry.deduplicated_data)
    resolve = full_data.get('value', {}).get('resolve', {})
    if not resolve:
        continue

    resolution = resolve.get('__resolution__', {})
    if resolution:
        event_text = resolution.get('Value', '')
        print(f"Step {entry.step} event: {event_text[:200]}")

    tracker = resolve.get('tension_tracker', {})
    if tracker:
        act = tracker.get('__act__', tracker)
        tension_val = act.get('Value', '')
        print(f"Step {entry.step} tension: {tension_val}")
```

### LLM-Based Verification (For Complex Checks)

When verifying high-level user intentions that are too nuanced for simple string
matching, use LLM analysis:

```python
entries = interface.filter_entries(include_content=True)
log_text = '\n'.join([
    f"Step {e.get('step')}: {e.get('summary')} {e.get('content', '')}"
    for e in entries
])

prompt = f"""Analyze this simulation log and answer:
1. Did the characters demonstrate awareness of HP/Energy mechanics? [YES/NO]
2. Did stat values change during combat? [YES/NO]
3. Were the changes sensible (HP down when hit, Energy down when active)? [YES/NO]

LOG:
{log_text[:10000]}

Respond in format:
HP_AWARENESS: [YES/NO]
STATS_CHANGED: [YES/NO]
SENSIBLE: [YES/NO]
REASONING: [explanation]
"""

response = model.sample_text(prompt, max_tokens=200)
```

### Comparing Two Simulation Runs

```bash
# CLI approach (quick diff)
diff <(concordia-log actions run_a.json Alice) <(concordia-log actions run_b.json Alice)
diff <(concordia-log components run_a.json --component tension_tracker) \
     <(concordia-log components run_b.json --component tension_tracker)
```

```python
# Python approach (for detailed analysis)
def extract_trajectory(structured_path, component_name, value_key='Value'):
    with open(structured_path) as f:
        structured = json.load(f)
    log = SimulationLog.from_dict(structured)
    trajectory = []
    for entry in log.entries:
        if entry.entry_type != 'step':
            continue
        full = log.reconstruct_value(entry.deduplicated_data)
        resolve = full.get('value', {}).get('resolve', {})
        comp = resolve.get(component_name, {})
        if comp:
            act = comp.get('__act__', comp)
            trajectory.append({'step': entry.step, 'value': act.get(value_key, '')})
    return trajectory

tension_a = extract_trajectory('run_a_structured.json', 'tension_tracker')
tension_b = extract_trajectory('run_b_structured.json', 'tension_tracker')
for a, b in zip(tension_a, tension_b):
    print(f"Step {a['step']}: A={a['value']}  B={b['value']}")
```

## Common Debugging Workflow

1. **Get overview** — `concordia-log overview sim.json`
2. **Check entities** — `concordia-log entities sim.json`
3. **Scan actions** — `concordia-log actions sim.json <entity>` for each entity
4. **Drill into issues** — `concordia-log context sim.json <entity> --step N`
5. **Search for keywords** — `concordia-log search sim.json "keyword"`
6. **Examine memories** — `concordia-log memories sim.json <entity>`
7. **Raw data inspection** — `concordia-log dump sim.json --step N | jq '.'`

## Quick Reference

| Method / Command | Purpose |
|-----------------|---------|
| `concordia-log overview` | High-level stats |
| `concordia-log actions <entity>` | Concise action timeline |
| `concordia-log context <entity> --step N` | Full action + observations + prompt |
| `concordia-log step N` | All entries for one step |
| `concordia-log search "query"` | Text search across entries |
| `concordia-log dump` | Raw inflated JSON for jq |
| `get_overview()` | High-level stats (Python) |
| `get_entity_actions(name)` | Action timeline (Python) |
| `get_entity_action_context(name, step)` | Full context (Python) |
| `get_step_summary(step, include_content)` | Step entries (Python) |
| `filter_entries(...)` | Filter by entity/component/type/step (Python) |
| `search_entries(query)` | Deep text search (Python) |
| `get_component_values(...)` | Extract component values (Python) |
| `get_entity_memories(name)` | Entity memories (Python) |

## Tips

- **Start with the CLI tool** — For most debugging, `concordia-log` is faster
  than writing Python scripts
- **Use `concordia-log dump | jq`** — When you need to explore unfamiliar log
  structures
- **Verify structure before content** — Check that expected entities and game
  masters participated before analyzing what they said
- **For GM components, look in `resolve`** — The most interesting GM data lives
  in `deduplicated_data.value.resolve.<component_name>.__act__`
- **Use LLM-based analysis for high-level verification** — It's more robust than
  string matching for evaluating "did it work?"
- **Truncate log text before sending to LLM** — Logs can be very large (100KB+)
