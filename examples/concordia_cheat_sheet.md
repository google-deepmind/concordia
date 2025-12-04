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
        memory = agent_components.memory.AssociativeMemory(memory_bank=memory_bank)
        instructions = agent_components.instructions.Instructions(agent_name=name)
        observation = agent_components.observation.LastNObservations(history_length=50)
        
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

## Agent Memory Management

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
    def __init__(self, model, next_game_master_name: str, player_names: list[str]):
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
        pattern = re.compile(rf"\b{re.escape(agent_name)}\b.*?(?P<JSON>\{{.*?\}})", re.DOTALL)
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
        "agents": {name: dataclasses.asdict(a) for name, a in self._agents.items()},
    }

def set_state(self, state: entity_component.ComponentState) -> None:
    """Restore state from checkpoint."""
    self._state["round"] = state.get("round", 0)
    self._initialized = state.get("initialized", False)
```

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
