# Thought Chains

The `thought_chains` module provides a flexible framework for processing events,
actions, and narratives through a sequence of LLM-driven reasoning steps.

## Concept

A **Thought Chain** is a pipeline of functions where each function (a "thought")
takes a `chain_of_thought` document, a candidate event or premise, and the
active player's name. It then interacts with the LLM to refine, transform, or
analyze the event, and returns a new string to be passed to the next thought in
the chain.

Thought chains are most commonly used in **Game Master** components,
specifically for **Event Resolution**. They allow the system to "think through"
the consequences of an action step-by-step. However, they are a general-purpose
abstraction and can be used in any component (e.g., Agents) that requires
complex, multi-stage LLM reasoning.

This abstraction allows for complex narrative logic to be composed of smaller,
modular reasoning units. For example, a chain might:
1.  Determine if an action succeeds.
2.  If successful, determine the causal outcome.
3.  Reword the outcome to be more narrative.

## Core Functions

*   `run_chain_of_thought`: The main entry point. It takes a list of thought
    functions and an initial premise, and executes them sequentially.

## Available Thoughts

The module provides a library of common thought functions:

### Basic Processing

*   `identity`: Returns the input premise unchanged.
*   `extract_direct_quote`: Extracts exact quotes from an action attempt.
*   `restore_direct_quote`: Re-inserts quotes into a processed event string.

### Outcome Determination

*   `determine_success_and_why`: Asks the LLM if an action succeeds and why.
*   `attempt_to_result`: Determines the direct result of an action.
*   `attempt_to_most_likely_outcome`: Generates multiple possible outcomes and
    picks the most likely one.
*   `get_action_category_and_player_capability`: Categorizes the action (e.g.,
    Confess, Defy) and checks if the player is capable.

### Narrative Refinement

*   `result_to_causal_statement`: Rewrites an event to highlight cause and
    effect.
*   `result_to_who_what_where`: Rewrites to clarify who did what and where.
*   `result_to_effect_caused_by_active_player`: Highlights the active player's
    agency.

### Narrative Control

*   `maybe_inject_narrative_push`: Randomly introduces complications or events
    to push the story forward if it becomes repetitive.
*   `maybe_cut_to_next_scene`: Determines if the scene should end and suggests
    a time jump.

## Helper Classes

Some thoughts require state or configuration and are implemented as classes:

*   `AccountForAgencyOfOthers`: Checks if an event implies voluntary actions by
    other characters and verifies if they would actually take those actions.
*   `Conversation`: Detects if a conversation is occurring and generates the
    dialogue based on character personas.
*   `RemoveSpecificText`: A utility to clean up strings.

## Usage Example

```python
from concordia.thought_chains import thought_chains
from concordia.document import interactive_document

# Define the sequence of thoughts
thoughts = [
    thought_chains.extract_direct_quote,
    thought_chains.determine_success_and_why,
    thought_chains.restore_direct_quote,
]

# Run the chain
document = interactive_document.InteractiveDocument(model)
final_event, _ = thought_chains.run_chain_of_thought(
    thoughts=thoughts,
    premise="Alice tries to open the locked door.",
    document=document,
    active_player_name="Alice",
)
```
