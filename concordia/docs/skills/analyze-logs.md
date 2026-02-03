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

### Extracting Content from Nested Log Structures

Log entries often have deeply nested metadata. Use recursive extraction:

```python
def extract_observations(obj):
    """Extract observation strings from nested data."""
    observations = []
    if isinstance(obj, str):
        if 'hp' in obj.lower() or 'energy' in obj.lower():
            observations.append(obj)
    elif isinstance(obj, dict):
        for key, v in obj.items():
            if 'observation' in key.lower() or key == 'Value':
                if isinstance(v, str):
                    observations.append(v)
                else:
                    observations.extend(extract_observations(v))
            else:
                observations.extend(extract_observations(v))
    elif isinstance(obj, list):
        for item in obj:
            observations.extend(extract_observations(item))
    return observations

# Use with entries
for entry in interface.filter_entries(include_content=True):
    metadata = entry.get('metadata', {})
    obs = extract_observations(metadata)
    if obs:
        print(f"Step {entry['step']}: {' '.join(obs)[:500]}")
```

## Debugging Workflow

### Step 1: Get Overview

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

### Step 2: Check Specific Steps

If you know which step is problematic, drill into it:

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

### Step 3: Follow Entity Timeline

To understand one agent's complete journey:

```python
# Get Alice's complete timeline
alice_timeline = interface.get_entity_timeline('Alice', include_content=True)
for entry in alice_timeline:
    print(f"Step {entry['step']}: {entry['summary']}")
```

### Step 4: Filter by Criteria

Find specific types of entries:

```python
# Find all observations
observations = interface.filter_entries(entry_type='observation')

# Find entries from a specific component
memory_entries = interface.filter_entries(component_name='memory')

# Find entries in a step range
late_game = interface.filter_entries(step_range=(10, 20))
```

### Step 5: LLM-Based Verification (For Complex Checks)

When verifying high-level user intentions, use LLM analysis:

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

### Step 6: Search for Keywords

When you know what text to look for:

```python
# Search for specific behavior
coffee_entries = interface.search_entries('coffee shop')
```

### Step 7: Get Full Content

For deep investigation of a specific entry:

```python
# Get full prompt/response/contexts for entry at index 5
content = interface.get_entry_content(entry_index=5)
print(content['prompt'])
print(content['response'])
```

## Common Debugging Patterns

### Pattern: "Did the agent do X?"

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

### Pattern: "Did the mechanic work correctly?" (LLM-based)

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

### Pattern: "Did the narrative progress or get stuck?"

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

### Pattern: "What happened at step N?"

```python
# Full breakdown of step 5
entries = interface.get_step_summary(5, include_content=True)
for e in entries:
    print(f"[{e['entity_name']}] {e['entry_type']}: {e['summary']}")
```

### Pattern: "Find the bug"

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
| `get_step_summary(step, include_content)` | All entries for one step |
| `get_entity_timeline(entity, include_content)` | All entries for one entity |
| `filter_entries(...)` | Filter by entity/component/type/step |
| `search_entries(query)` | Text search in summaries |
| `get_entry_content(index)` | Full prompt/response for one entry |

## Tips

- **Use LLM-based analysis for high-level verification** - It's more robust than
string matching
- Start with `include_content=False` to get summaries, then drill in with `True`
- Use `filter_entries()` with multiple criteria to narrow down
- The `summary` field gives quick insight without loading full content
- **Truncate log text before sending to LLM** - Logs can be very large (100KB+)
- **Use structured output formats** (YES/NO with REASONING) for easy parsing
- Use color-coded output to highlight key information in reports to the user.
