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

## The AI Agent Log Interface

Concordia provides an `AIAgentLogInterface` specifically designed for coding
agents (not humans viewing HTML). Always use this interface instead of HTML
when debugging programmatically.

```python
from concordia.utils.structured_logging import AIAgentLogInterface

# After running simulation
log = simulation.play(return_structured_log=True)
interface = AIAgentLogInterface(log)
```

## Understanding the Log Data Structure

> **IMPORTANT:** The structured log uses content deduplication. Long strings in
> log entries are replaced with `{'_ref': content_id}` references that point
> into a separate `content_store`. You **must** resolve these references to
> read the actual content.

### How Deduplication Works

When you load a structured log from JSON:

```python
from concordia.utils.structured_logging import SimulationLog

with open('path/to/structured.json') as f:
    data = json.load(f)

log = SimulationLog.from_dict(data)
```

The JSON has this top-level structure:
```python
{
    'content_store': { 'hash_id': 'actual_text', ... },
    'entries': [ { 'step': 2, 'entity_name': '...', 'deduplicated_data': {...} }, ... ],
    'entity_memories': { ... },
    'game_master_memories': [ ... ]
}
```

Each entry's `deduplicated_data` contains nested dicts where long strings have
been replaced by `{'_ref': 'hash_id'}`. To get the actual text, use
`SimulationLog.reconstruct_value()`:

```python
entry = log.entries[0]
full_data = log.reconstruct_value(entry.deduplicated_data)
```

### Entry Data Layout

Each entry's `deduplicated_data` has a `key` and a `value`. The `value` is a
nested dict of component outputs. For entity entries, the key pattern is
`Entity [name]` and the value contains:

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

For game master entries, the value is organized into **phases**, each
containing outputs from every GM component that ran during that phase:

```python
{
    'key': 'game_master_name',
    'value': {
        'terminate': { ... },
        'next_game_master': { ... },
        'make_observation': { ... },
        'next_acting': { ... },
        'next_action_spec': { ... },
        'resolve': {           # <-- Most interesting phase
            '__act__': { ... },
            '__resolution__': {
                'Key': '...',
                'Summary': '...',   # Event resolution summary
                'Value': '...',     # Full resolved event text
                'Prompt': '...',
            },
            'LineageResourceTracker': {
                '__act__': {
                    'Key': '...',
                    'Value': '...',          # Component's text output
                    'Measurements': { ... }, # Optional numeric metrics
                },
            },
            'tension_tracker': { '__act__': { 'Value': '0.6' } },
            'ritual_propriety_observer': { '__act__': { 'Value': '...' } },
            ...
        },
    }
}
```

> **IMPORTANT:** The `resolve` phase is typically where the most interesting
> data lives — event resolutions. When analyzing GM behavior,
> always look in `resolve` first.

> **IMPORTANT:** Custom components may include a `Measurements` sub-dict
> alongside their `Value`. This contains numeric metrics (e.g.,
> `{'compound_health': 120, 'granary_stores': 0}`) that are useful for
> quantitative tracking across steps.

### Extracting Entity Actions (The Most Common Task)

The simplest way to get what entities actually did:

```python
interface = AIAgentLogInterface(log)

actions = interface.get_entity_actions('Alice')
for a in actions:
    print(f"Step {a['step']}: {str(a['action'])[:200]}")
```

To understand *why* an entity took a specific action:

```python
context = interface.get_entity_action_context('Alice', step=3)
if context:
    print(f"Action: {context['action']}")
    print(f"Observations: {context['observations']}")
    print(f"Prompt: {context['action_prompt']}")
```

For more flexible extraction, use `get_component_values()`:

```python
interface.get_component_values(
    component_key='Observation',
    value_key='Value',
    entity_name='Alice',
    step_range=(1, 5),
)
```

### Extracting GM Component Values

To extract values from specific game master components, navigate the
phase → component → `__act__` nesting:

```python
for entry in log.entries:
    if entry.entry_type != 'step':
        continue
    full_data = log.reconstruct_value(entry.deduplicated_data)
    resolve = full_data.get('value', {}).get('resolve', {})
    if not resolve:
        continue

    # Event resolution text
    resolution = resolve.get('__resolution__', {})
    if resolution:
        event_text = resolution.get('Value', '')
        print(f"Step {entry.step} event: {event_text[:200]}")

    # Custom component values (e.g., tension_tracker)
    tracker = resolve.get('tension_tracker', {})
    if tracker:
        act = tracker.get('__act__', tracker)
        tension_val = act.get('Value', '')
        print(f"Step {entry.step} tension: {tension_val}")

    # Components with Measurements
    resource = resolve.get('LineageResourceTracker', {})
    if resource:
        act = resource.get('__act__', resource)
        measurements = act.get('Measurements', {})
        if measurements:
            print(f"Step {entry.step} metrics: {measurements}")
```

To inspect the keys of the `resolve` phase:

```python
entry = next(e for e in log.entries if e.entry_type == 'step')
full_data = log.reconstruct_value(entry.deduplicated_data)
resolve_keys = list(full_data.get('value', {}).get('resolve', {}).keys())
print(f"Available GM components: {resolve_keys}")
```

### Entity Memories

The structured log also stores all entity memories accumulated during the
simulation. This is a quick way to see what each agent "knows" at the end:

```python
log_data = json.load(open('path/to/structured.json'))
memories = log_data.get('entity_memories', {})
for name, mems in memories.items():
    print(f"{name}: {len(mems)} memories")
    for m in mems[-3:]:  # Last 3 memories
        print(f"  > {str(m)[:150]}")
```

#### Manual Extraction from Raw JSON

If working directly with the raw JSON file without loading into a
`SimulationLog`, you can resolve references manually:

```python
content_store = data['content_store']

def resolve_refs(obj):
    if isinstance(obj, dict):
        if '_ref' in obj:
            return content_store.get(obj['_ref'], f'<UNRESOLVED:{obj["_ref"]}>')
        return {k: resolve_refs(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [resolve_refs(v) for v in obj]
    return obj

for entry in data['entries']:
    dd = entry.get('deduplicated_data', {})
    if not dd:
        continue
    resolved = resolve_refs(dd)
    value = resolved.get('value', {})
    entity = entry.get('entity_name', '?')
    step = entry.get('step', '?')

    if entry.get('entry_type') == 'entity' and isinstance(value, dict):
        act = value.get('__act__', {})
        if isinstance(act, dict):
            action_text = act.get('Value', '')
            if action_text and len(str(action_text)) > 10:
                print(f'Step {step} [{entity}]: {action_text[:400]}')
```

## Two Approaches to Log Analysis

### Approach 1: Exact String Matching (Simple Checks)

For basic verification, you can search for specific keywords:

```python
# Check if an entity mentioned a specific term
entries = interface.search_entries('coffee shop')
found = len(entries) > 0
```

This works for simple presence/absence checks but has limitations:

- Brittle to paraphrasing ("coffee shop" vs "café")
- Can't understand context or semantic meaning
- Can't evaluate if behavior was *appropriate*

### Approach 2: LLM-Based Interpretation (Recommended for High-Level Verification)

**For verifying whether the user's intent was achieved, LLM-based analysis is
often the best approach.** This mirrors how humans would evaluate the
simulation.

```python
# Extract relevant log content
all_entries = interface.filter_entries(include_content=True)
log_text = '\n'.join([
    f"Step {e.get('step')}: {e.get('summary')} {e.get('content', '')}"
    for e in all_entries
])

# Ask the LLM to evaluate
prompt = f"""Analyze this simulation log and answer:
1. Did the characters demonstrate awareness of HP/Energy mechanics? [YES/NO]
2. Did stat values change during combat? [YES/NO]
3. Were the changes sensible (HP down when hit, Energy down when active)? [YES/NO]

LOG:
{log_text[:10000]}  # Truncate if needed

Respond in format:
HP_AWARENESS: [YES/NO]
STATS_CHANGED: [YES/NO]
SENSIBLE: [YES/NO]
REASONING: [explanation]
"""

response = model.sample_text(prompt, max_tokens=200)
# Parse the structured response
```

**Why LLM-based analysis is often better:**

- Understands semantic meaning, not just keywords
- Can evaluate whether behavior was appropriate for the context
- Handles variation in language and phrasing
- Can assess complex multi-step narratives
- Mirrors how a human would evaluate "did it work?"

## Common Log Analysis Patterns

Choose the pattern that matches what you are trying to understand about the
simulation.

### Pattern: Fast Overview of the Simulation

Always start by getting a high-level overview of what happened:

```python
overview = interface.get_overview()
print(overview)
# Output: {
#   'total_entries': 15,
#   'total_steps': 5,
#   'entities': ['Alice', 'Bob', 'default rules'],
#   'components': ['entity_action', 'game_master'],
#   'entry_types': ['entity', 'step'],
#   ...
# }
```

This tells you:

- How many simulation steps occurred
- Which entities participated
- What types of events were logged

### Pattern: Verifying Structural Correctness

Before analyzing narrative content, verify the simulation structure matches
expectations. This catches configuration bugs before wasting time on content:

```python
overview = interface.get_overview()
entities = overview['entities']
total_steps = overview['total_steps']

# Check expected agents participated
expected_agents = ['Alice', 'Bob']
for agent in expected_agents:
    timeline = interface.get_entity_timeline(agent)
    assert len(timeline) > 0, f"{agent} has no entries!"
    print(f"{agent}: {len(timeline)} entries across steps")

# Check expected game masters ran
expected_game_masters = ['negotiation_rules', 'combat_rules']
for game_master in expected_game_masters:
    timeline = interface.get_entity_timeline(game_master)
    print(f"Game Master '{game_master}': {len(timeline)} entries")
```

### Pattern: Drilling into a Specific Step

If you want to see exactly what happened concurrently at a specific time point:

```python
# Get all entries for step 3
step_data = interface.get_step_summary(step=3, include_content=True)
for entry in step_data:
    print(f"Entity: {entry['entity_name']}")
    print(f"Event: {entry['entry_type']}")
    print(f"Summary: {entry['summary']}")
    if entry.get('prompt'):
        print(f"Prompt: {entry['prompt'][:500]}...")  # First 500 chars
    if entry.get('response'):
        print(f"Response: {entry['response']}")
```

### Pattern: Following an Entity's Timeline

To understand one agent's journey across the entire simulation:

```python
# Get Alice's complete timeline
alice_timeline = interface.get_entity_timeline('Alice', include_content=True)
for entry in alice_timeline:
    print(f"Step {entry['step']}: {entry['summary']}")
```

### Pattern: Extracting Entity Actions

The simplest way to see what each entity actually did (ignoring observations and
background events):

```python
actions = interface.get_entity_actions('Alice')
for a in actions:
    print(f"Step {a['step']}: {str(a['action'])[:200]}")
```

This resolves all `_ref` references automatically and returns just the action
text from `__act__.Value`.

### Pattern: Getting Full Action Context

To understand *why* an entity did what it did at a specific step (the prompt
to their action component):

```python
context = interface.get_entity_action_context('Alice', step=3)
if context:
    print(f"Action: {context['action']}")
    print(f"Observations: {context['observations']}")
    print(f"Prompt: {context['action_prompt']}")
```

This returns the entity's action, observations, and the full prompt sent to the
LLM at the time it produced its action.

### Pattern: Extracting Specific Component Values

Use `get_component_values()` for flexible extraction of specific game master or
entity components:

```python
actions = interface.get_component_values()
for action in actions:
    print(f"Step {action['step']} [{action['entity_name']}]: {action['value']}")

# Customize which component and key to extract
interface.get_component_values(
    component_key='Observation',
    value_key='Value',
    entity_name='Alice',
    step_range=(1, 5),
)
```

### Pattern: Filtering by Criteria

Find specific types of events or narrow down to a time range:

```python
# Find entries from a specific component
memory_entries = interface.filter_entries(component_name='memory')

# Find entries in a step range
late_game = interface.filter_entries(step_range=(10, 20))
```

### Pattern: LLM-Based Verification (For Complex Checks)

When verifying high-level user intentions that are too nuanced for simple string
matching, use LLM analysis:

```python
def verify_simulation(interface, model, checks):
    """
    Verify simulation outcomes using LLM-based analysis.

    Args:
        interface: AIAgentLogInterface instance
        model: Language model for analysis
        checks: List of verification questions

    Returns:
        dict with results for each check
    """
    # Build log text
    entries = interface.filter_entries(include_content=True)
    log_text = '\n'.join([
        f"Step {e['step']}: {e.get('summary', '')} "
        f"{extract_observations(e.get('metadata', {}))}"
        for e in entries
    ])

    results = {}
    for check_name, question in checks.items():
        prompt = f"""Based on this simulation log, answer: {question}

LOG:
{log_text[:8000]}

Answer YES or NO, then explain briefly.
"""
        response = model.sample_text(prompt, max_tokens=100)
        results[check_name] = 'YES' in response.upper()

    return results

# Example usage
checks = {
    'hp_mechanics': 'Are HP values mentioned and do they change during combat?',
    'energy_costs': 'Do actions cost Energy?',
    'narrative_progress': 'Does the story progress meaningfully?',
}
results = verify_simulation(interface, model, checks)
```

### Pattern: Searching for Keywords

For fast searching by entry summary only:

```python
matches = interface.search_summaries('coffee shop')
```

For deep searching across all reconstructed content (prompts, values,
observations, etc.):

```python
matches = interface.search_entries('coffee shop')
for e in matches:
    print(f"Step {e['step']} [{e['entity_name']}]: {e.get('summary', '')}")
```

### Pattern: Deep Inspection of a Single Entry

For deep investigation of everything attached to a specific log entry:

```python
content = interface.get_entry_content(entry_index=5)
print(content['data'])
```

### Pattern: Accessing Final Entity Memories

To quickly see what an agent accumulated during the run:

```python
memories = interface.get_entity_memories('Alice')
for m in memories[-5:]:
    print(f"  > {str(m)[:150]}")
```

### Common question: "Did the agent do X?"

**Option A: Keyword Matching (Limited Use)**

This works when actions use **choice-based action specs** where the exact action
text is known:

```python
# Only reliable when agent chose from fixed options like
# ["greet", "ignore", "attack"]
alice_timeline = interface.get_entity_timeline('Alice')
for entry in alice_timeline:
    if 'greet' in entry['summary'].lower():
        print(f"Found greeting at step {entry['step']}: {entry['summary']}")
```

> **Note:** This approach is fragile for free-response actions where the agent
can phrase things many ways (e.g., "says hello", "waves warmly", "greets them
cheerfully").

**Option B: LLM-Based Interpretation (Recommended for Free Response)**

When agents use free-form responses, use an LLM to evaluate semantically:

```python
# Get Alice's actions
alice_timeline = interface.get_entity_timeline('Alice', include_content=True)
alice_actions = '\n'.join([
    f"Step {e['step']}: {e.get('summary', '')} {e.get('response', '')}"
    for e in alice_timeline
])

# Ask LLM to interpret
prompt = f"""Based on Alice's actions below, answer: Did Alice greet Bob?

ALICE'S ACTIONS:
{alice_actions}

Consider any form of greeting: saying hello, waving, introducing herself, etc.

Answer: [YES/NO]
REASONING: [brief explanation]
"""
response = model.sample_text(prompt, max_tokens=100)
did_greet = 'YES' in response.upper()
```

### Common question: "Did the mechanic work correctly?" (LLM-based)

```python
# Use LLM to evaluate if HP/Energy mechanics worked
prompt = f"""Analyze this RPG simulation log.

{log_text}

Questions:
1. HP_MENTIONED: Are HP values shown to characters? [YES/NO]
2. HP_CHANGES: Do HP values change when combat occurs? [YES/NO]
3. ENERGY_CHANGES: Do Energy values change when actions are taken? [YES/NO]
4. CHANGES_SENSIBLE: Are changes appropriate (damage reduces HP, etc)? [YES/NO]

Respond with answers and one sentence of reasoning.
"""
response = model.sample_text(prompt, max_tokens=200)
```

### Common question: "Did the narrative progress or get stuck?"

```python
# Combat naturally has similar actions - don't flag as repetitive
prompt = f"""Analyze this simulation for narrative quality.

{log_text}

IMPORTANT: In combat, similar actions (attacking, defending) are NORMAL.
Combat is NOT repetitive if HP/Energy change or new tactics are tried.

1. STORY_PROGRESSES: Does the story advance? [YES/NO]
2. STUCK_IN_LOOP: Is it truly stuck (no consequences, no evolution)? [YES/NO]
"""
```

### Common question: "What happened at step N?"

```python
# Full breakdown of step 5
entries = interface.get_step_summary(5, include_content=True)
for e in entries:
    print(f"[{e['entity_name']}] {e['entry_type']}: {e['summary']}")
```

### Common question: "Find the bug"

When something went wrong, narrow down systematically:

1. Get overview to see the scope
2. Binary search through steps to find where behavior diverged
3. Drill into the specific step with full content
4. Examine the prompt to see what context the LLM received
5. Verify the prompt contains all expected components (e.g., memories,
observations, instructions)
6. Verify all component values are sensible, especially that they reflect a
correct flow of information between players, game masters, and the simulation's
initial specification.
7. Check if the response makes sense given the prompt

## Saving and Loading Logs

Logs can be serialized for later analysis:

```python
import json

# Save log
with open('/tmp/simulation_log.json', 'w') as f:
    f.write(log.to_json())

# Load log
from concordia.utils.structured_logging import SimulationLog
log = SimulationLog.from_json(open('/tmp/simulation_log.json').read())
interface = AIAgentLogInterface(log)
```

## Quick Reference

| Method | Purpose |
|--------|---------|
| `get_overview()` | High-level stats |
| `get_entity_actions(name)` | Concise action timeline for one entity |
| `get_entity_action_context(name, step)` | Full action + observations + prompt for one step |
| `get_step_summary(step, include_content)` | All entries for one step |
| `get_entity_timeline(entity, include_content)` | All entries for one entity |
| `filter_entries(...)` | Filter by entity/component/type/step |
| `search_summaries(query)` | Fast text search in entry summaries |
| `search_entries(query)` | Deep text search across all reconstructed content |
| `get_entry_content(index)` | Full reconstructed data for one entry |
| `get_component_values(...)` | Extract specific component values with ref resolution |
| `get_entity_memories(name)` | Get an entity's accumulated memories |
| `get_game_master_memories()` | Get game master memories |

### Common question: "How can I compare two simulation runs?"

When comparing runs (e.g., different experimental conditions), extract parallel
metrics from both structured logs and compare side-by-side:

```python
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

## Tips

- **Verify structure before content** - Check that expected entities and game
  masters participated before analyzing what they said
- **Resolve `_ref` references** - Summary fields are often generic; you must
  resolve the deduplicated data to get actual entity actions and game master
  resolutions
- **Use `__act__.Value` for entity actions** - The actual text of what an agent
  said or did lives in `deduplicated_data.value.__act__.Value` after resolving
  references
- **For GM components, look in `resolve`** - The most interesting GM data lives
  in `deduplicated_data.value.resolve.<component_name>.__act__`
- **Discover available components first** - Inspect the `resolve` phase keys
  before trying to extract specific component values
- **Use entity_memories for quick summaries** - The `entity_memories` dict in
  the structured log shows what each agent accumulated
- **Use LLM-based analysis for high-level verification** - It's more robust than
  string matching
- Start with `include_content=False` to get summaries, then drill in with `True`
- Use `filter_entries()` with multiple criteria to narrow down
- **Truncate log text before sending to LLM** - Logs can be very large (100KB+)
- **Use structured output formats** (YES/NO with REASONING) for easy parsing
