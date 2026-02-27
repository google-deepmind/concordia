# Concordia Simulation Cheat Sheet

A concise guide to building and running simulations in Concordia.

---

## Core Concepts

| Concept | Description |
|---------|-------------|
| **Prefab** | A reusable recipe for building an entity (agent or game master) |
| **InstanceConfig** | Configuration specifying which prefab to use and its parameters |
| **Config** | Full simulation configuration containing prefabs, instances, premise |
| **Simulation** | The main object that orchestrates entities and game masters |
| **Entity** | An agent that can observe the world and take actions |
| **Game Master** | Controls the simulation flow, resolves actions, generates observations |
| **AssociativeMemoryBank** | Stores and retrieves memories using embeddings |

---

## Minimal Simulation Setup

```python
from concordia.prefabs import entity as entity_prefabs
from concordia.prefabs import game_master as game_master_prefabs
from concordia.prefabs.simulation import generic as simulation
from concordia.typing import prefab as prefab_lib
from concordia.utils import helper_functions

# 1. Load available prefabs
prefabs = {
    **helper_functions.get_package_classes(entity_prefabs),
    **helper_functions.get_package_classes(game_master_prefabs),
}

# 2. Define agent instances
instances = [
    prefab_lib.InstanceConfig(
        prefab="basic__Entity",
        role=prefab_lib.Role.ENTITY,
        params={"name": "Alice", "goal": "Make new friends"},
    ),
    prefab_lib.InstanceConfig(
        prefab="basic__Entity",
        role=prefab_lib.Role.ENTITY,
        params={"name": "Bob", "goal": "Find a business partner"},
    ),
]

# 3. Add a game master
instances.append(
    prefab_lib.InstanceConfig(
        prefab="dialogic__GameMaster",
        role=prefab_lib.Role.GAME_MASTER,
        params={
            "name": "conversation rules",
            "next_game_master_name": "conversation rules",
        },
    )
)

# 4. Create config
config = prefab_lib.Config(
    default_premise="Alice and Bob meet at a coffee shop.",
    default_max_steps=20,
    prefabs=prefabs,
    instances=instances,
)

# 5. Initialize and run simulation
sim = simulation.Simulation(config=config, model=model, embedder=embedder)
results = sim.play()
```

---

## Simulation Engines

Engines control the flow of time, the order of execution, and how agents update
their state.

| Engine | Execution Flow | State Handling | Best For |
| :--- | :--- | :--- | :--- |
| **Sequential** | Turn-based. One agent acts, then the GM resolves. | Agents observe every event immediately. | Narrative simulations, conversations, dependent actions. |
| **Simultaneous** | Batch-based. All agents submit actions, then GM resolves all. | Agents do not see others' actions until the round ends. | Marketplaces, voting, game-theoretic scenarios (e.g., Prisoner's Dilemma). |
| **Sequential Questionnaire** | Iterates through questions/agents one by one. | Updates agent memory after every answer. | Interviews, sequences where previous context influences the next answer. |
| **Parallel Questionnaire** | Batches questions to agents concurrently. | **Stateless**: Agents answer based on current state without updating memory during the batch. | Surveys, psychometrics, efficiently gathering independent data points. |

**Usage:**
```python
from concordia.environment.engines import sequential, simultaneous

# 1. Sequential (Default)
engine = sequential.Sequential()

# 2. Simultaneous (For fast simulations with simultaneous acting)
engine = simultaneous.Simultaneous()

simultaneous_simulation = simulation.Simulation(
    config=config,
    model=model,
    embedder=embedder,
    engine=engine,
)
```

---

## Prefab Roles

```python
from concordia.typing import prefab as prefab_lib

prefab_lib.Role.ENTITY       # Agents that act in the world
prefab_lib.Role.GAME_MASTER  # Controls simulation, generates events
prefab_lib.Role.INITIALIZER  # Runs once to set up initial state
```

---

## Built-in Prefabs

### Entity Prefabs
| Prefab | Description |
| :--- | :--- |
| `basic__Entity` | The standard "Three Key Questions" agent. Determines action by asking: *What situation is this? Who am I? What would I do?* |
| `basic_with_plan__Entity` | Extends the basic agent by generating a **Plan** before acting. |
| `basic_scripted__Entity` | Uses the "Three Key Questions" for internal thought but follows a pre-defined script for actions. |
| `conversational__Entity` | Optimized for dialogue. Explicitly balances "converging" (staying on topic) and "diverging" (introducing new ideas). |
| `minimal__Entity` | A bare-bones agent with only Memory, Instructions, and Observation. Highly configurable for custom extensions. |
| `fake_assistant_with_configurable_system_prompt__Entity` | A wrapper that makes the agent behave like a standard AI Assistant (helpful, harmless) or any custom system prompt. |

### Game Master Prefabs
| Prefab | Description |
| :--- | :--- |
| `generic__GameMaster` | A flexible, general-purpose GM. Good starting point for custom simulations. |
| `dialogic__GameMaster` | Specialized for pure conversation. Supports fixed, random, or GM-chosen turn-taking. |
| `dialogic_and_dramaturgic__GameMaster` | Manages conversations structured into **Scenes** (e.g., "Prologue", "Episode 1"). |
| `situated__GameMaster` | Manages a simulation with specific **Locations** and tracks where agents are. |
| `situated_in_time_and_place__GameMaster` | The most complex world model. Tracks **Time** (Clock) and **Locations**, supporting day/night cycles and movement. |
| `formative_memories_initializer__GameMaster` | **Initializer**: Runs once at the start to generate backing stories and childhood memories for agents. |
| `interviewer__GameMaster` | Administers multiple-choice or fixed questionnaires to agents. |
| `open_ended_interviewer__GameMaster` | Administers open-ended questionnaires, using embeddings to process answers. |
| `game_theoretic_and_dramaturgic__GameMaster` | specialized for Matrix Games (e.g., Prisoner's Dilemma) wrapped in a narrative Scene. |
| `marketplace__GameMaster` | Specialized for economic simulations. Supports buying, selling, and inventory management. |
| `psychology_experiment__GameMaster` | A generic shell for running psychology experiments defined by custom observation/action components. |
| `scripted__GameMaster` | Forces the simulation to follow a strict linear script of events. |

---

## Simulation Examples

While the "Minimal Simulation Setup" shows the code structure, here are two
common *types* of simulations you might build, with the specific prefabs and
logic required.

### 1. Narrative Simulation (e.g., Alice in Wonderland)
**Best for:** Storytelling, role-playing, and open-ended interactions where the
outcome is unknown.

**Key Components:**

*   **Entities:** `basic__Entity` (standard agents) with optional `goals`.
*   **Game Master:** `generic__GameMaster` (manages turn-taking and observations).
*   **Premise:** Sets the initial scene (e.g., "Alice sees a White Rabbit...").

In this scenario, agents act based on their character descriptions and the
`default_premise`. There are no strict "game rules" or scores, just interaction.

```python
# 1. Define Entities
alice = prefab_lib.InstanceConfig(
    prefab='basic__Entity',
    role=prefab_lib.Role.ENTITY,
    params={'name': 'Alice'},
)
rabbit = prefab_lib.InstanceConfig(
    prefab='basic__Entity',
    role=prefab_lib.Role.ENTITY,
    params={
        'name': 'White Rabbit',
        'goal': 'Get to the queen on time',
    },
)

# 2. Generic Game Master (Fixed Order)
gm = prefab_lib.InstanceConfig(
    prefab='generic__GameMaster',
    role=prefab_lib.Role.GAME_MASTER,
    params={
        'name': 'default rules',
        'acting_order': 'fixed', # options: 'fixed', 'random', or 'u-go-i-go' (mostly 2 players)
    },
)

# 3. Use generic Simulation
config = prefab_lib.Config(
    default_premise="Alice sees a White Rabbit running with a pocket watch.",
    default_max_steps=10,
    prefabs=prefabs,
    instances=[alice, rabbit, gm],
)
```

### 2. Game Theoretic Simulation (e.g., Selling Cookies)
**Best for:** Games (Prisoner's Dilemma), and scenarios with specific "moves"
and "payoffs".

**Key Components:**

*   **Entities:** `basic__Entity` or custom agents.
*   **Game Master:** `game_theoretic_and_dramaturgic__GameMaster`. This GM is crucial because it can:
    *   Enforce structured **Scenes** (e.g., specific rounds for discussion vs. decision).
    *   Calculate **Payoffs** using `action_to_scores`.
    *   Provide feedback via `scores_to_observation`.

In this scenario, we often have a "Conversation" phase (free text) followed by
a "Decision" phase (restricted choice).

```python
from concordia.typing import scene as scene_lib

# 1. Define Scenes
# Conversation Scene (Free speech)
conversation_scene = scene_lib.SceneTypeSpec(
    name='conversation',
    game_master_name='conversation rules',
    action_spec=entity_lib.free_action_spec(
        call_to_action=entity_lib.DEFAULT_CALL_TO_SPEECH
    ),
)

# Decision Scene (Binary Choice)
decision_scene = scene_lib.SceneTypeSpec(
    name='decision',
    game_master_name='decision rules',
    action_spec={
        'Alice': entity_lib.choice_action_spec(
            call_to_action='Buy cookies?',
            options=['Yes', 'No'],
        ),
    },
)

# Scene Sequence
scenes = [
    scene_lib.SceneSpec(
        scene_type=conversation_scene,
        participants=['Alice', 'Bob'],
        num_rounds=4,
        premise={'Alice': ['Bob approaches you.'],
                 'Bob': ['You approach Alice.']},
    ),
    scene_lib.SceneSpec(
        scene_type=decision_scene,
        participants=['Alice'],
        num_rounds=1,
        premise={'Alice': ['Decide whether to buy cookies.']},
    ),
]

# 2. Define Payoffs (Game Theory)
def action_to_scores(joint_action):
    # If Alice buys (Yes), she loses money (-1), Bob gains (1)
    if joint_action['Alice'] == 'Yes':
        return {'Alice': -1, 'Bob': 1}
    return {'Alice': 0, 'Bob': 0}

def scores_to_observation(scores):
    return {p: f"Resulting Score: {s}" for p, s in scores.items()}

# 3. Configure Game Masters
instances = [
    # ... Entities Alice and Bob ...
    prefab_lib.InstanceConfig(
        prefab='game_theoretic_and_dramaturgic__GameMaster',
        role=prefab_lib.Role.GAME_MASTER,
        params={
            'name': 'decision rules',
            'scenes': scenes,
            'action_to_scores': action_to_scores,
            'scores_to_observation': scores_to_observation,
        },
    ),
    prefab_lib.InstanceConfig(
        prefab='dialogic_and_dramaturgic__GameMaster',
        role=prefab_lib.Role.GAME_MASTER,
        params={
            'name': 'conversation rules',
            'scenes': scenes,
        },
    ),
    # Optional: Initializer to set initial memories/context
    prefab_lib.InstanceConfig(
        prefab='formative_memories_initializer__GameMaster',
        role=prefab_lib.Role.INITIALIZER,
        params={
            'name': 'initial setup rules',
            'next_game_master_name': 'conversation rules',
            'shared_memories': ["Alice and Bob are neighbors."],
            'player_specific_context': {
                'Alice': "You don't like cookies.",
                'Bob': "You are a cookie salesman."
            },
        },
    ),
]
```

---

## Creating a Custom Prefab

```python
import dataclasses
from concordia.typing import prefab as prefab_lib
from concordia.agents import entity_agent_with_logging
from concordia.components import agent as agent_components

@dataclasses.dataclass
class MyCustomAgent(prefab_lib.Prefab):
    description: str = "A custom agent for my simulation"

    def build(self, model, memory_bank):
        name = self.params.get("name", "Agent")
        goal = self.params.get("goal", "")

        # Create components
        memory = agent_components.memory.AssociativeMemory(
            memory_bank=memory_bank)
        instructions = agent_components.instructions.Instructions(
            agent_name=name)
        observation = agent_components.observation.LastNObservations(
            history_length=50)

        components = {
            agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY: memory,
            "Instructions": instructions,
            agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY: observation,
        }

        act_component = agent_components.concat_act_component.ConcatActComponent(
            model=model,
            component_order=list(components.keys()),
        )

        return entity_agent_with_logging.EntityAgentWithLogging(
            agent_name=name,
            act_component=act_component,
            context_components=components,
        )

# Register custom prefab
prefabs["my_custom__Entity"] = MyCustomAgent()
```

---

## Advanced Agent Architecture

For more complex behaviors, you can chain components together where the output
of one component becomes the input/context for another. This allows agents to
"think" before they act.

### Key Components for Reasoning

*   **`SituationRepresentation`**: Summarizes the current situation based on
recent observations + relevant memories.
*   **`QuestionOfRecentMemories`**: Asks a specific question to the model based
on memory. Useful for "Guiding Principles" or "Internal Monologue".

### Example: The "Reflective" Agent
This agent first builds a representation of the situation, then asks itself how
to exploit it, and *then* decides on an action.

```python
from concordia.contrib.components.agent import situation_representation_via_narrative
from concordia.components import agent as agent_components

@dataclasses.dataclass
class ReflectiveAgent(prefab_lib.Prefab):
    def build(self, model, memory_bank):
        name = self.params.get("name", "Agent")
        
        # 1. Base Components
        instructions = agent_components.instructions.Instructions(
            agent_name=name)
        observation = agent_components.observation.LastNObservations(
            history_length=100)
        memory = agent_components.memory.AssociativeMemory(
            memory_bank=memory_bank)
        
        # 2. Advanced Components (Chain of Thought)
        
        # Step A: Summarize the situation
        situation = situation_representation_via_narrative.SituationRepresentation(
            model=model,
            observation_component_key=agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY,
            declare_entity_as_protagonist=True,
        )
        
        # Step B: Apply a Guiding Principle (uses Step A context)
        principle = agent_components.question_of_recent_memories.QuestionOfRecentMemories(
            model=model,
            pre_act_label=f"{name}'s Internal Monologue",
            question=f"How can {name} best achieve their goals in this situation?",
            answer_prefix=f"{name} thinks: ",
            add_to_memory=False, # Don't clutter memory with every thought
            components=[
                "Instructions",
                "situation_representation" # <--- Depends on Step A
            ],
        )

        # 3. Assemble Components (Order matters!)
        components = {
            "Instructions": instructions,
            agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY: memory,
            agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY: observation,
            "situation_representation": situation,
            "guiding_principle": principle,
        }
        
        # The Act Component sees everything in 'components'
        act_component = agent_components.concat_act_component.ConcatActComponent(
            model=model,
            component_order=list(components.keys()),
        )
        
        return entity_agent_with_logging.EntityAgentWithLogging(
            agent_name=name,
            act_component=act_component,
            context_components=components,
        )
```

---

## Initializing Agent Memories

The recommended way to generate agent backstories and shared context is using
the `formative_memories_initializer__GameMaster`. This prefab runs once at the
start of the simulation to inject memories into agents.

```python
# Define shared facts (world building)
shared_memories = [
    "Riverbend is an idyllic rural town.",
    "The year is 2024.",
]

# Define player-specific context (backstories)
player_specific_context = {
    "Alice": "Alice is a baker who loves experiments. She is optimistic.",
    "Bob": "Bob is a skeptical journalist investigating local corruption.",
}

# Add the Initializer to your instances list
initializer = prefab_lib.InstanceConfig(
    prefab='formative_memories_initializer__GameMaster',
    role=prefab_lib.Role.INITIALIZER,
    params={
        'name': 'initial setup rules',
        # crucial: where to hand off control after initialization
        'next_game_master_name': 'conversation rules',
        'shared_memories': shared_memories,
        'player_specific_context': player_specific_context,
    },
)

# Assuming 'instances' is defined elsewhere and you want to add this to it.
# For this example, we'll just show the config.
# instances.append(initializer)
```

### Pre-loading Memories
```python
from concordia.associative_memory import basic_associative_memory

# Create memory bank
memory_bank = basic_associative_memory.AssociativeMemoryBank(
    sentence_embedder=embedder
)

# Add memories
memory_bank.add("Alice loves hiking in the mountains.")
memory_bank.add("Alice works as a software engineer.")

# Pass to instance config
instance = prefab_lib.InstanceConfig(
    prefab="basic__Entity",
    role=prefab_lib.Role.ENTITY,
    params={
        "name": "Alice",
        "memory_state": {"buffer": [], "memory_bank": memory_bank.get_state()},
    },
)
```

### Transferring Memory Between Phases
```python
import copy

# After phase 1, save memory state
source_entity = phase1_sim.entities[0]
temp_memory = copy.deepcopy(
    source_entity.get_component("__memory__").get_state()
)

# In phase 2, apply to corresponding entity
target_entity = phase2_sim.entities[0]
target_entity.get_component("__memory__").set_state(temp_memory)
```

---

## Triggering Agent Actions

```python
from concordia.typing import entity as entity_lib

# Free-form action prompt
action_spec = entity_lib.free_action_spec(
    call_to_action="What does Alice do next?"
)
response = agent.act(action_spec=action_spec)

# Agent observes something
agent.observe("Bob waves hello.")
```

---

## Parallel Simulations (Example of 2 dialogs in parallel)

```python
from concordia.utils import concurrency
import functools

def run_dyad_task(player_states, model, embedder):
    sim = create_dialog_simulation(player_states, model, embedder)
    return sim.play()

# Create parallel tasks
tasks = {
    "alice_bob": functools.partial(run_dyad_task,
        player_states={"Alice": alice_state, "Bob": bob_state},
        model=model, embedder=embedder),
    "carol_dave": functools.partial(run_dyad_task,
        player_states={"Carol": carol_state, "Dave": dave_state},
        model=model, embedder=embedder),
}

# Run all dyads in parallel
results = concurrency.run_tasks(tasks)
```

---

## Multi-Phase Workflow Pattern

```python
# Phase 1: Marketplace
marketplace_sim = simulation.Simulation(
    config=marketplace_config, model=model, embedder=embedder
)
marketplace_sim.play()

# Transfer state to Phase 2
entities_for_phase2 = []
for entity in marketplace_sim.entities:
    entities_for_phase2.append(entity)

# Phase 2: Dialogue (using transferred memories)
daily_dyads = generate_dyads(entities_for_phase2)

for p1, p2 in daily_dyads:
    dyad_sim = create_dialog_simulation(p1, p2, model, embedder)
    # Transfer memories
    for src in [p1, p2]:
        tgt = next(e for e in dyad_sim.entities if e.name == src.name)
        mem = copy.deepcopy(src.get_component("__memory__").get_state())
        tgt.get_component("__memory__").set_state(mem)

    dyad_sim.play() # see above example for running this loop in parallel
```

---

## Key Imports Reference

```python
# Core simulation
from concordia.prefabs.simulation import generic as simulation
from concordia.typing import prefab as prefab_lib
from concordia.typing import entity as entity_lib

# Prefab libraries
from concordia.prefabs import entity as entity_prefabs
from concordia.prefabs import game_master as game_master_prefabs
from concordia.utils import helper_functions

# Agent components
from concordia.components import agent as agent_components
from concordia.agents import entity_agent_with_logging

# Memory
from concordia.associative_memory import basic_associative_memory

# Concurrency
from concordia.utils import concurrency

# Engines
from concordia.environment.engines import sequential, simultaneous

# Scene management
from concordia.typing import scene as scene_lib
```

---

## Creating a Custom Game Master Component

Game Master components control simulation flow. The key method is `pre_act`,
which is called with different `action_spec.output_type` values to handle
different phases. A Game Master component can control one or many types of
action specs for the Game Master, which is specified in the Game Master prefab

```python
components_of_game_master = {
        _get_class_name(instructions): instructions,
        actor_components.memory.DEFAULT_MEMORY_COMPONENT_KEY: (
            actor_components.memory.AssociativeMemory(memory_bank=memory_bank)
        ),
        # Use the custom Game Master component for making observations
        gm_components.make_observation.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY: (
            my_gamemaster_component
        ),
        next_actor_key: next_actor,
        # Use the custom Game Master component for the next action spec
        gm_components.next_acting.DEFAULT_NEXT_ACTION_SPEC_COMPONENT_KEY: (
            my_gamemaster_component
        ),
        # Use the custom Game Master component for the resolution of actions
        gm_components.switch_act.DEFAULT_RESOLUTION_COMPONENT_KEY: (
            my_gamemaster_component
        ),
    }
```

### Basic GM Component Structure

```python
import dataclasses
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component

class MyGameMasterComponent(
    entity_component.ContextComponent,
    entity_component.ComponentWithLogging,
):
    def __init__(
        self,
        acting_player_names: list[str],
        components: list[str] = (),
        pre_act_label: str = "\nMyComponent",
    ):
        super().__init__()
        self._acting_player_names = acting_player_names
        self._components = components
        self._pre_act_label = pre_act_label
        self._state = {"round": 0}
    
    def get_pre_act_label(self) -> str:
        return self._pre_act_label
    
    def get_pre_act_value(self) -> str:
        return f"Current round: {self._state['round']}"
    
    def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
        """Handle different output types from the simulation engine."""
        output_type = action_spec.output_type
        
        if output_type == entity_lib.OutputType.MAKE_OBSERVATION:
            return self._handle_make_observation(action_spec)
        elif output_type == entity_lib.OutputType.NEXT_ACTION_SPEC:
            return self._handle_next_action_spec(action_spec)
        elif output_type == entity_lib.OutputType.NEXT_ACTING:
            return self._handle_next_acting()
        elif output_type == entity_lib.OutputType.RESOLVE:
            return self._resolve(action_spec)
        elif output_type == entity_lib.OutputType.NEXT_GAME_MASTER:
            return self._handle_next_gm()
        else:
            return ""
```

### `pre_act` Output Types

| OutputType | Purpose | What to Return |
|------------|---------|----------------|
| `MAKE_OBSERVATION` | Generate what agent sees | Observation text for agent |
| `NEXT_ACTION_SPEC` | Define agent's action format | Action spec string (JSON format, etc.) |
| `NEXT_ACTING` | Determine who acts next | Agent name string |
| `RESOLVE` | Resolve agent actions | Event outcome text |
| `NEXT_GAME_MASTER` | Hand off to another GM | Next GM's name |

### Example: Observation Handler (Marketplace-style)

```python
def _handle_make_observation(self, action_spec: entity_lib.ActionSpec) -> str:
    """Generate observation for the current agent."""
    # Extract agent name from action_spec
    agent_name = None
    for name in self._acting_player_names:
        if name in action_spec.call_to_action:
            agent_name = name
            break
    
    agent = self._agents[agent_name]
    
    # Build observation string with agent's current state
    obs = (
        f"Round: {self._state['round']+1} is starting\n"
        f"Cash: {agent.cash:.2f}\n"
        f"{agent_name}'s Inventory: {agent.inventory}\n"
        "Submit your action."
    )
    return obs
```

### Example: Action Spec Handler

```python
def _handle_next_action_spec(self, agent_name: str) -> str:
    """Define the action format for the agent."""
    call_to_action = """
    What will {name} do?
    Output your decision as JSON:
    {{"action":"buy","item":"ITEM_ID","qty":INTEGER}}
    """
    action_spec = entity_lib.free_action_spec(call_to_action=call_to_action)
    return engine_lib.action_spec_to_string(action_spec)
```

### Example: Initializer Game Master

For one-time initialization that hands off to another GM:

```python
class MyInitializer(entity_component.ContextComponent):
    def __init__(self,
                 model,
                 next_game_master_name: str,
                 player_names: list[str]):
        super().__init__()
        self._model = model
        self._next_game_master_name = next_game_master_name
        self._players = player_names
        self._initialized = False
    
    def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
        # Only respond to NEXT_GAME_MASTER queries
        if action_spec.output_type != entity_lib.OutputType.NEXT_GAME_MASTER:
            return ""
        
        if self._initialized:
            # Hand off to the dialogue/main GM
            return self._next_game_master_name
        
        # Run initialization logic (generate scene, inject observations)
        self._run_initialization()
        self._initialized = True
        
        # Return own name to let other components finish this step
        return self.get_entity().name
    
    def _run_initialization(self):
        """Generate initial scene and inject into observation queues."""
        make_obs = self.get_entity().get_component("__make_observation__")
        
        scene = self._generate_scene_with_llm()
        for player in self._players:
            make_obs.add_to_queue(player, f"[Scene] {scene}")
```

### Example: Action Resolution

```python
import json
import re

def _resolve(self, action_spec: entity_lib.ActionSpec) -> str:
    """Parse agent actions and resolve outcomes."""
    # Get the putative events from components
    component_states = "\n".join(
        [self._component_pre_act_display(key) for key in self._components]
    )
    
    events = []
    for agent_name in self._acting_player_names:
        # Find JSON action in the event string
        pattern = re.compile(
            rf"\b{re.escape(agent_name)}\b.*?(?P<JSON>\{{.*?\}})", re.DOTALL)
        match = pattern.search(component_states)
        
        if match:
            try:
                action = json.loads(match.group("JSON"))
                outcome = self._process_action(agent_name, action)
                events.append(outcome)
            except json.JSONDecodeError:
                events.append(f"{agent_name}'s action failed to parse.")
    
    self._state["round"] += 1
    return "\n".join(events)
```

### State Management (Required for Checkpointing)

```python
def get_state(self) -> entity_component.ComponentState:
    """Return serializable state for checkpointing."""
    return {
        "round": self._state["round"],
        "initialized": self._initialized,
        "agents": {
            name: dataclasses.asdict(a) for name, a in self._agents.items()
        },
    }

def set_state(self, state: entity_component.ComponentState) -> None:
    """Restore state from checkpoint."""
    self._state["round"] = state.get("round", 0)
    self._initialized = state.get("initialized", False)
```

---

## Questionnaires

This section covers running questionnaire simulations: defining questions,
configuring agents and the interviewer, and running the simulation.

## Step 1: Define Your Questionnaire

### Using a Pre-defined Questionnaire
Concordia includes an example standard questionnaire, the Depression Anxiety
Stress Scales (DASS).

```python
from concordia.contrib.data.questionnaires import depression_anxiety_stress_scale

questionnaire = depression_anxiety_stress_scale.DASSQuestionnaire()
```

### Creating a Custom Questionnaire
Subclass `QuestionnaireBase` for custom polls. Define questions with a
`statement`, `choices`, and an optional `dimension` for aggregation.

```python
from typing import Any, Dict, List
from concordia.contrib.data.questionnaires import base_questionnaire
import numpy as np
import pandas as pd

AGREEMENT_SCALE = ["Strongly disagree", "Disagree", "Agree", "Strongly agree"]

class CommunityWellbeingQuestionnaire(base_questionnaire.QuestionnaireBase):
  """4-question survey measuring community and safety dimensions."""

  def __init__(self):
    super().__init__(
        name="CommunityWellbeing",
        description="Measures community connectedness and safety.",
        questionnaire_type="multiple_choice",
        observation_preprompt="{player_name} is completing a survey.",
        questions=[
            base_questionnaire.Question(
                statement="I feel connected to my community.",
                choices=AGREEMENT_SCALE,
                dimension="community",
            ),
            base_questionnaire.Question(
                statement="My neighbors are supportive.",
                choices=AGREEMENT_SCALE,
                dimension="community",
            ),
            base_questionnaire.Question(
                statement="I feel safe in my neighborhood.",
                choices=AGREEMENT_SCALE,
                dimension="safety",
            ),
            base_questionnaire.Question(
                statement="I trust the people around me.",
                choices=AGREEMENT_SCALE,
                dimension="safety",
            ),
        ],
        dimensions=["community", "safety"],
    )

  def aggregate_results(
      self, player_answers: Dict[str, Dict[str, Any]]
  ) -> Dict[str, Any]:
    """Compute mean score for each dimension."""
    dimension_values: Dict[str, List[float]] = {}
    for question_data in player_answers.values():
      dim = question_data["dimension"]
      val = question_data["value"]
      if val is not None:
        dimension_values.setdefault(dim, []).append(val)
    return {dim: np.mean(vals) for dim, vals in dimension_values.items()}

  def get_dimension_ranges(self) -> Dict[str, tuple[float, float]]:
    """Range 0-3 for 4-point scale (indexed 0, 1, 2, 3)."""
    return {"community": (0, 3), "safety": (0, 3)}

  def plot_results(self, results_df: pd.DataFrame, **kwargs) -> None:
    pass
```

---

## Step 2: Configure Entities and Interviewer

### Create Entity (Agent) Instances

```python
persona_names = ['Alice', 'Bob', 'Charlie']

persona_instances = []
for name in persona_names:
  persona_instances.append(prefab_lib.InstanceConfig(
      prefab='basic__Entity',
      role=prefab_lib.Role.ENTITY,
      params={'name': name},
  ))
```

### Configure the Interviewer Game Master
Use `interviewer__GameMaster` for **multiple-choice** questionnaires.

```python
interviewer_config = prefab_lib.InstanceConfig(
    prefab='interviewer__GameMaster',
    role=prefab_lib.Role.GAME_MASTER,
    params={
        'name': 'interviewer',
        'player_names': persona_names,
        'questionnaires': [questionnaire],  # Your questionnaire object(s)
    },
)
```

Use `open_ended_interviewer__GameMaster` for **open-ended** questionnaires
(requires an embedder).

```python
oe_interviewer_config = prefab_lib.InstanceConfig(
    prefab='open_ended_interviewer__GameMaster',
    role=prefab_lib.Role.GAME_MASTER,
    params={
        'name': 'interviewer',
        'player_names': persona_names,
        'questionnaires': [open_ended_questionnaire],
        'embedder': embedder,  # Required for open-ended questions
    },
)
```

---

## Step 3: Run the Simulation

### Build the Config
Combine prefabs and instances into a single `Config` object.

```python
config = prefab_lib.Config(
    default_premise='',
    prefabs=prefabs,
    instances=persona_instances + [interviewer_config],  # or oe_interviewer_config
)
```

### Instantiate and Run

```python
from concordia.prefabs.simulation import questionnaire_simulation

simulation = questionnaire_simulation.QuestionnaireSimulation(
    config=config,
    model=model,
    embedder=embedder,
)

results_log = simulation.play()
```

---

### Execution Modes
Choose between parallel (faster) or sequential (context-dependent) execution.

| Mode | Engine | Best For |
|------|--------|----------|
| **Parallel** (Default) | `ParallelQuestionnaireEngine` | Speed, independent answers |
| **Sequential** | `SequentialQuestionnaireEngine` | Answers that depend on prior context |

```python
from concordia.environment.engines import parallel_questionnaire

simulation = questionnaire_simulation.QuestionnaireSimulation(
    config=config,
    model=model,
    embedder=embedder,
    engine=parallel_questionnaire.ParallelQuestionnaireEngine(max_workers=4),
)
```

### Agent Options: Randomize Choices
To avoid positional bias, enable `randomize_choices` (default: `True`) in
agent prefabs.

```python
prefab_lib.InstanceConfig(
    prefab='basic__Entity',
    role=prefab_lib.Role.ENTITY,
    params={
        'name': 'Alice',
        'randomize_choices': False,  # Disable for deterministic testing or to maintain order
    },
)
```

---

## Analyzing Simulation Logs

Concordia provides an `AIAgentLogInterface` for programmatic log analysis.
For the full debugging guide, see the **analyze-logs** skill in `docs/skills/`.
Additional skills can also be found in the `docs/` directory.

For **human users**, the easiest way to view logs is to open
`utils/log_viewer.html` in a browser and load the structured log JSON file.

### Setup for AIAgentLogInterface

```python
from concordia.utils.structured_logging import AIAgentLogInterface

log = simulation.play(return_structured_log=True)
interface = AIAgentLogInterface(log)
```

### Quick Overview

```python
overview = interface.get_overview()
# {'total_entries': 15, 'total_steps': 5,
#  'entities': ['Alice', 'Bob', 'default rules'], ...}
```

### Extracting Entity Actions

```python
actions = interface.get_component_values()
for action in actions:
    print(f"Step {action['step']} [{action['entity_name']}]: {action['value']}")

# Filter by entity and step range
interface.get_component_values(
    entity_name='Alice', step_range=(1, 5),
)
```

### Debugging Workflow

1. `get_overview()` — check entity count, steps, and participation
2. `get_entity_actions(name)` — just the action text at each step (actions only)
3. `get_entity_action_context(name, step)` — full action, observations, and prompt for one step
4. `get_entity_timeline(name)` — all log entries for an entity (actions, observations, GM events, etc.)
5. `get_step_summary(step, include_content=True)` — drill into a specific step
6. `search_summaries('keyword')` — fast text search across entry summaries
7. `search_entries('keyword')` — deep text search across all reconstructed content
8. `get_entry_content(entry_index=N)` — full prompt/response for deep inspection

### Saving and Loading Logs

```python
import json
from concordia.utils.structured_logging import SimulationLog

# Save
with open('/tmp/simulation_log.json', 'w') as f:
    f.write(log.to_json())

# Load
log = SimulationLog.from_json(open('/tmp/simulation_log.json').read())
interface = AIAgentLogInterface(log)
```

### Quick Reference

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
| `get_entry_content(index)` | Full prompt/response for one entry |
| `get_component_values(...)` | Extract specific component values with ref resolution |
| `get_entity_memories(name)` | Get an entity's accumulated memories |
| `get_game_master_memories()` | Get game master memories |

---

## CLI Tools for Log Analysis

For command-line based log analysis (ideal for scripting and piping), use
`concordia-log`:

```bash
# Quick overview of a simulation
concordia-log overview sim.json

# What did Alice do?
concordia-log actions sim.json Alice

# Why did she do it at step 3?
concordia-log context sim.json Alice --step 3

# All entries for step 5
concordia-log step sim.json 5

# Search for keywords
concordia-log search sim.json "coffee shop"

# Entity memories
concordia-log memories sim.json Alice

# List all entities (useful for scripting)
concordia-log entities sim.json

# Discover components on an entity
concordia-log components sim.json --entity Alice

# List keys for a specific component
concordia-log components sim.json --entity Alice --component __act__

# Extract value at a specific key in the log of a specific component across steps for all entities
concordia-log components sim.json --component MyComponent --key Key

# Extract value at a specific key in the log of a specific component across steps for a specific entity
concordia-log components sim.json --entity Alice --component MyComponent --key Key

# Extract value at a specific key in the log of a specific component for a specific entity on a specific step
concordia-log components sim.json --entity Alice --component MyComponent --key Key --step 3

# Full timeline for an entity
concordia-log timeline sim.json Alice

# Dump inflated JSON for jq/grep
concordia-log dump sim.json | jq '.[] | .data.__act__.Value'
```

**Common patterns:**

```bash
# Pipe to grep
concordia-log actions sim.json Alice | grep "hello"

# JSON output for jq
concordia-log actions sim.json Alice --json | jq '.[].action'

# Compare two entities
diff <(concordia-log actions run.json Alice) <(concordia-log actions run.json Bob)
```

**Running the tool:**

- **From source (dev):** `./concordia/command_line_interface/concordia-log overview sim.json`
- **After pip install:** `pip install gdm-concordia` makes `concordia-log` available globally.

Run `concordia-log --help` for all subcommands.

Image-heavy logs: base64 image data is automatically stripped from output
(replaced with `[image: N bytes]`). Use `--include-images` to include it.

---

## Common Patterns

| Pattern | Implementation |
|---------|----------------|
| Load all prefabs | `helper_functions.get_package_classes(entity_prefabs)` |
| Add custom prefab | `prefabs["myname__Entity"] = MyPrefab()` |
| Get agent memory | `entity.get_component("__memory__").get_state()` |
| Set agent memory | `entity.get_component("__memory__").set_state(state)` |
| Deep copy for transfer | `copy.deepcopy(memory_state)` |
| Run parallel tasks | `concurrency.run_tasks(tasks_dict)` |
| Free action prompt | `entity_lib.free_action_spec(call_to_action=...)` |
| Access other GM components | `self.get_entity().get_component("__make_observation__")` |
| Add observation to queue | `make_obs.add_to_queue(player_name, observation_str)` |
