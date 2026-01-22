# Concordia Prefabs

Prefabs are pre-assembled blueprints for Agents, Game Masters, and even entire
Simulations. While **Components** are the atomic building blocks, **Prefabs**
are the factory that wires them together into functional entities.

Using prefabs allows you to bootstrap a simulation quickly without needing to
manually instantiate and connect dozens of components for every agent.

## 1. Entity Prefabs (`prefabs/entity/`)

These prefabs build **Agents**—actors that represent characters in the
simulation.

*   `basic.py`: The standard "Reasoning Agent". It uses a "Chain of Thought"
    approach involving Situation Perception, Self Perception, and Person By
    Situation reasoning.
*   `basic_with_plan.py`: Extends `basic.py` with a `Plan` component, which
    acts as a long-term goal that conditions the agent's actions over time.
    Useful for agents that need to pursue objectives beyond immediate reactions.
*   `basic_scripted.py`: An agent that follows a pre-defined script for its
    actions but uses the standard reasoning components to generate internal
    thoughts and memory updates.
*   `conversational.py`: Optimized for dialogue. Takes into account conversation
    history and dynamics (e.g., whether to "converge" on a topic or "diverge" to
    a new one) to produce more engaging and natural interactions.
*   `minimal.py`: A bare-bones agent with only the absolute essentials: memory,
    instructions, and observation. Useful as a base for custom extensions or
    simple background extras.
*   `fake_assistant_with_configurable_system_prompt.py`: A wrapper around a
    language model that acts as a simple assistant, conditioned primarily by a
    system prompt rather than simulated human psychology.

## 2. Game Master Prefabs (`prefabs/game_master/`)

These prefabs build **Game Masters**—the directors that control the environment,
NPCs, and simulation logic.

### General Purpose

*   `generic.py`: A highly configurable GM that can be adapted to many scenarios
    via parameters. It supports custom thought chains for event resolution.
*   `situated.py`: A versatile GM for simulations anchored in specific
    locations. It manages a "Story so far", "Locations", and world state.
*   `situated_in_time_and_place.py`: An extension of `situated.py` that includes
    a `GenerativeClock` to track and narrate the passage of time (e.g., "It is
    now late afternoon...").

### Scenario Specific

*   `dialogic.py`: Specialized for pure conversation simulations. It includes
    logic for ending conversations when they become repetitive.
*   `dialogic_and_dramaturgic.py`: Combines conversation logic with "Scene"
    management, allowing for structured episodes (e.g., Prologue, Episode 1).
*   `game_theoretic_and_dramaturgic.py`: Designed for matrix games and social
    dilemmas. It can map joint actions to scores and observations (e.g.,
    Cooperate/Defect scenarios).
*   `marketplace.py`: A GM specifically tuned for economic simulations. It uses
    a custom experiment component to handle market transactions (buying,
    selling).
*   `psychology_experiment.py`: A generic harness for psychology experiments,
    allowing injection of custom observation and action specification
    components.
*   `interviewer.py`: Administers fixed multiple-choice questionnaires to
    players.
*   `open_ended_interviewer.py`: Administers questionnaires where answers are
    free-form text rather than multiple-choice. These answers are later mapped
    to specific categories using a sentence embedder.
*   `scripted.py`: A Game Master that follows a strictly pre-defined script,
    forcing specific actions and observations in a fixed order. Useful for
    generating Concordia agent data from a fixed script for fine-tuning.

### Utility

*   `formative_memories_initializer.py`: A special GM used at the *start* of a
    simulation to implant backing "memories" into agents before the actual
    simulation begins. It does not run the simulation loop itself but sets the
    stage.

## 3. Simulation Prefabs (`prefabs/simulation/`)

These prefabs generate the entire experimental harness, including the `Engine`
and logging infrastructure.

*   `generic.py`: The `Simulation` class here acts as a wrapper for the entire
    experiment. It:
    *   Loads Agent and GM configurations.
    *   Initializes the `AssociativeMemoryBank`.
    *   Runs the `Engine` loop.
    *   Handles checkpointing and HTML log generation.
    *   Provides a simple `.play()` API.
*   `questionnaire_simulation.py`: A specialized runner for administering
    benchmarks and psychometric questionnaires.

## 4. Advanced Configuration & Custom Prefabs
    
### Custom Prefabs
You are not limited to the pre-packaged prefabs. You can define your own by
subclassing `prefab_lib.Prefab` and implementing the `build` method. This is
demonstrated in `examples/actor_development.ipynb`, where a custom `MyAgent`
prefab is created to demonstrate adding specific components.

```python
class MyAgent(prefab_lib.Prefab):
  def build(self, model, memory_bank):
    # Custom logic to build your agent's components
    ...
```

### Passing Complex Parameters
Most prefab parameters are strings or numbers, but you can also pass executable
Python objects if needed. For example, `game_theoretic_and_dramaturgic.py`
accepts function objects for `action_to_scores` and `scores_to_observation`.
Be aware that this makes your configuration dependent on the runtime environment
and harder to serialize purely as JSON.

### The `INITIALIZER` Role
A common pattern in complex simulations is to use a specific Game Master with
the `Role.INITIALIZER`. This GM runs *before* the main loop starts. It is solely
responsible for populating the memory bank with context (e.g., "Alice and Bob
are friends"). The `formative_memories_initializer.py` prefab is the standard
tool for this task.

## 5. Usage Example

The most robust way to use prefabs is by defining `InstanceConfig` objects and
passing them to a `Simulation` runner, as seen in `examples/tutorial.ipynb`.

```python
from concordia.prefabs import entity as entity_prefabs
from concordia.prefabs import game_master as game_master_prefabs
from concordia.prefabs.simulation import generic as simulation
from concordia.typing import prefab as prefab_lib

# 1. Define Prefab Classes
# (Helper to map string names back to actual classes)
prefabs = {
    'basic__Entity': entity_prefabs.basic.Entity,
    'generic__GameMaster': game_master_prefabs.generic.GameMaster,
}

# 2. Configure Instances
instances = [
    # An Agent
    prefab_lib.InstanceConfig(
        prefab='basic__Entity',
        role=prefab_lib.Role.ENTITY,
        params={
            'name': 'Oliver',
            'goal': 'Find the treasure',
        },
    ),
    # The Game Master
    prefab_lib.InstanceConfig(
        prefab='generic__GameMaster',
        role=prefab_lib.Role.GAME_MASTER,
        params={
            'name': 'Dungeon Master',
        },
    ),
]

# 3. Create Simulation Configuration
config = prefab_lib.Config(
    default_premise='The adventure begins...',
    default_max_steps=10,
    prefabs=prefabs,
    instances=instances,
)

# 4. Run Simulation
runnable_simulation = simulation.Simulation(
    config=config,
    model=model,      # Your LanguageModel instance
    embedder=embedder # Your sentence embedder
)

results = runnable_simulation.play()
```
