# A Tutorial on Concordia Components

This directory contains the modular building blocks of agents and game masters
in Concordia. The components are organized into `agent` and `game_master`
subdirectories. This tutorial provides a comprehensive guide to understanding
and using these components to build your own sophisticated simulations.

## 1. Introduction to Concordia Components

Components are the heart of Concordia's modularity. Instead of monolithic
agents, Concordia entities (both agents and game masters) are built by
composing small, reusable components. Each component has a specific
responsibility, such as managing memory, handling observations, or deciding on
the next action.

This approach offers several advantages:

*   **Flexibility:** You can mix and match components to create a wide variety
    of agent and game master behaviors.
*   **Reusability:** Components can be shared across different agents and
    simulations.
*   **Extensibility:** It's easy to create new components to add custom
    functionality.
*   **Clarity:** By breaking down complex behaviors into smaller pieces, the
    overall logic of an entity becomes easier to understand and debug.

## 2. Agent Components

Agent components are the building blocks of an agent's mind and behavior. They
can be broadly categorized into core components, action/decision-making
components, and advanced reasoning components.

### Core Components for Agent Behavior

These components are fundamental to any agent that needs to maintain a sense of
self and its environment.

*   `memory.py`: The `AssociativeMemory` component is the agent's primary
    memory system. It allows the agent to store and retrieve memories based on
    semantic similarity. This is crucial for recalling past events, knowledge,
    and experiences.
*   `observation.py`: The `LastNObservations` component keeps a history of the
    most recent observations an agent has made. This provides the agent with
    immediate context about what is happening in the simulation.
*   `instructions.py`: This component provides the agent with its name, goals,
    and any other high-level directives that should guide its behavior
    throughout the simulation.
*   `all_similar_memories.py`: An alternative to `AssociativeMemory` that
    retrieves *all* memories matching a query (up to a limit) rather than just
    the top few. It also includes a summarization step, making it powerful for
    agents that need to synthesize large amounts of context.

### Action and Decision-Making Components

These components are responsible for translating the agent's internal state
into concrete actions.

*   `concat_act_component.py`: The `ConcatActComponent` is a powerful component
    that combines the outputs of other components (like memory, observation,
    and instructions) into a single prompt for the language model. This is the
    standard way to generate an action based on a comprehensive view of the
    agent's current state.
*   `scripted_act.py`: For situations where you need deterministic behavior,
    the `ScriptedAct` component allows you to provide a predefined sequence of
    actions for the agent to follow.

### Advanced Components for Reasoning

For more sophisticated agents, Concordia provides components that enable a form
of internal monologue or chain-of-thought reasoning.

*   `question_of_recent_memories.py`: The `QuestionOfRecentMemories` component
    is a versatile tool for self-reflection. It allows an agent to ask
    questions about its recent memories and experiences, such as "What kind of
    person am I?" or "What would a person like me do in this situation?". This
    component is a key building block for creating agents with more nuanced and
    self-aware behaviors.
*   `plan.py`: The `Plan` component enables agents to formulate and follow
    multi-step plans. This is essential for agents that need to perform complex
    tasks that cannot be accomplished in a single action.

## 3. Game Master Components

Game Master (GM) components are responsible for managing the simulation
environment, controlling the flow of the narrative, and providing agents with
observations. They are the directors of the simulation.

### Simulation Flow and Control

These components manage the sequence of events and the turn-taking of agents.

*   `switch_act.py`: The `SwitchAct` component is the "brain" of standard Game
    Masters. It routes the simulation engine's requests (like "who acts next?",
    "what does X see?", "resolve action Y") to the appropriate specific
    components. It ensures the GM can handle all the different phases of the
    simulation loop.
*   `event_resolution.py`: The `EventResolution` component is the "judge" of
    the simulation. It takes a "putative" action (what an agent *wants* to do)
    and uses the LLM to determine the outcome, updating the world state and
    generating observations for witnesses.
*   `next_acting.py`: The `NextActing` component determines which agent's turn
    it is to act. It can be configured for various turn-taking schemes, such as
    round-robin, random, or sequential.
*   `scene_tracker.py`: For simulations with a strong narrative structure, the
    `SceneTracker` component manages the progression of scenes. This is useful
    for breaking a simulation into distinct parts, like "morning," "afternoon,"
    and "evening."
*   `terminate.py`: The `Terminate` component is responsible for ending the
    simulation when certain conditions are met, such as a maximum number of
    steps or a specific game state being reached.

### Environment and World State

These components maintain the state of the simulation world and create what
agents perceive.

*   `make_observation.py`: The `MakeObservation` component is crucial for
    communication. It takes events that happen in the simulation and formats
    them into observations for the agents. It also handles the queuing of
    observations, so they can be delivered to the correct agents at the correct
    time.
*   `world_state.py`: This component maintains a key-value store of the
    simulation's world state. It can be used to track things like the location
    of objects, the weather, or any other global variables that might be
    relevant to the simulation.

### Specialized Components

Concordia also provides a rich set of specialized GM components for common
simulation patterns.

*   `questionnaire.py` & `open_ended_questionnaire.py`: These components allow
    the GM to administer surveys and interviews to agents, which is invaluable
    for collecting data in social science simulations.
*   `inventory.py` & `payoff_matrix.py`: For economic and game-theoretic
    simulations, these components provide the mechanics for managing agent
    inventories and calculating payoffs based on their actions.
*   `formative_memories_initializer.py`: This powerful component runs at the
    beginning of a simulation to generate rich backstories and formative
    memories for the agents, giving them a history and personality from the
    outset.

## 4. The Component Lifecycle and Execution Sequence

The `EntityAgent` class orchestrates the interaction of components through a
series of phases. When an agent acts or observes, its components are called in
a specific sequence:

*   **`agent.act()` sequence:** The `act` method is initiated with an
    `ActionSpec`, which defines the expected format of the action (e.g.,
    free-form text, a choice from a list). This `ActionSpec` is passed to the
    components.
    1.  `pre_act(self, action_spec: ActionSpec)`: All components are called in
        parallel. Components can inspect the `ActionSpec` to tailor their
        output accordingly, providing the most relevant context for the
        required action.
    2.  `post_act(self, action_attempt: str)`: After an action is attempted,
        all components are called in parallel to process the action that was
        taken.
    3.  `update(self)`: Components are called to update their internal state.

*   **`agent.observe()` sequence:**
    1.  `pre_observe(self, observation: str)`: Components are called in
        parallel to process an incoming observation.
    2.  `post_observe(self)`: After the observation has been processed,
        components can perform additional updates.
    3.  `update(self)`: Components update their final state.

## 5. Inter-Component Communication Patterns

Beyond the standard lifecycle phases, components can communicate directly with
each other. This is a powerful pattern for creating complex interactions within
a single entity.

### Key-based Retrieval

Components are not aware of each other by default. Instead, they are given keys
(strings) that they can use to retrieve other components from their parent
entity. This is typically done during the initialization of a component.

*   **Case Study: `EventResolution` calling `MakeObservation`:** A great
    example of this is the `EventResolution` game master component. When it
    resolves an event, it needs to inform the relevant agents. To do this, it
    uses a key to get a reference to the `MakeObservation` component and then
    calls its `add_to_queue()` method to deliver the observation.
*   **The General Pattern:** This mechanism of `get_entity().get_component(key)`
    is the standard way for components to interact. It allows for a flexible
    and decoupled architecture where components can be wired together in
    different ways.

### Reasoning with `ActionSpecIgnored` Components

Some components exist not to drive the agent's action directly, but to provide
context for other components. These components inherit from
`ActionSpecIgnored`.

*   **The Role of `ActionSpecIgnored`:** These components are special because
    they do not use the `action_spec` provided by the `act` call to compute
    their state. This means they generate their context regardless of the
    specific action being requested, making them ideal for providing
    foundational context like goals, personality, or high-level reflections.
*   **Dependency Chaining:** This allows you to create chains of reasoning. For
    example, a `QuestionOfRecentMemories` component can be configured to depend
    on the output of another component. This creates a chain of thought where
    the agent first reflects on one aspect of its state, and then uses that
    reflection to inform the next step in its reasoning process.

## 6. Concurrency and Thread Safety

Concordia is designed to support parallel execution of simulations. To manage
this, the `EntityAgent` has a built-in concurrency model.

*   **`_parallel_call_`:** This internal method uses a `ThreadPoolExecutor` to
    call the methods of all components in parallel during the `act` and
    `observe` phases.
*   **`_control_lock`:** To prevent race conditions and ensure that the agent's
    internal state remains consistent, the `act` and `observe` methods are
    wrapped in a `_control_lock`. This means that only one of these methods can
    be executing at a time for a given agent.
*   **Implications for Custom Components:** When you create your own
    components, it's important to be aware of this concurrency model. If your
    component modifies shared state, you must ensure that it is thread-safe.
    For most components that only modify their own internal state, this is not
    an issue. However, if you have multiple components that need to access a
    shared resource, you will need to implement your own locking mechanism
    within those components.

## 7. Putting It All Together: A Practical Example

Let's walk through an example of how to build a custom agent and game master
using the components we've discussed. In this scenario, we'll create a simple
simulation where a detective interviews a witness.

```python
import dataclasses
from concordia.agents import entity_agent
from concordia.components.agent import (memory,
                                        observation,
                                        instructions,
                                        question_of_recent_memories,
                                        concat_act_component)
from concordia.components.game_master import (next_acting,
                                              make_observation,
                                              scene_tracker)
from concordia.typing import prefab as prefab_lib

# --- Custom Agent Prefab ---

@dataclasses.dataclass
class ReflectiveDetective(prefab_lib.Prefab):
    """A detective agent that reflects on the situation before acting."""

    def build(self, model, memory_bank):
        agent_name = self.params.get("name", "Detective")

        # Core Components
        mem = memory.AssociativeMemory(memory_bank=memory_bank)
        obs = observation.LastNObservations(history_length=50)
        inst = instructions.Instructions(agent_name=agent_name)

        # Reasoning Component (ActionSpecIgnored)
        reflection = question_of_recent_memories.QuestionOfRecentMemories(
            model=model,
            pre_act_label=f"{agent_name}'s reflection",
            question="What is the most important clue I've learned so far?",
            answer_prefix=f"{agent_name} thinks to themself: ",
            add_to_memory=False, # Don't clutter memory with thoughts
            components=[obs.name, inst.name] # Condition on observation and instructions
        )

        components = {
            mem.name: mem,
            obs.name: obs,
            inst.name: inst,
            "reflection": reflection,
        }

        # Action Component
        act_component = concat_act_component.ConcatActComponent(
            model=model,
            component_order=list(components.keys()),
        )

        return entity_agent.EntityAgent(
            agent_name=agent_name,
            act_component=act_component,
            context_components=components,
        )
```

In this example, we created a `ReflectiveDetective` agent that uses
`QuestionOfRecentMemories` to reflect on the situation before acting. This
reflection is then fed into the `ConcatActComponent` along with the agent's
other components to produce a more considered action. This example illustrates
how you can chain `ActionSpecIgnored` components to create sophisticated
reasoning and then use other components to translate that reasoning into
action.

## 8. Conclusion

This tutorial has covered the fundamentals of Concordia's component-based
architecture. You've learned about the different types of components for agents
and game masters, how they communicate with each other, and how to combine them
to create custom entities. With this knowledge, you are now equipped to build
your own sophisticated simulations in Concordia. For more in-depth information,
you can always refer to the source code of the components and the prefabs that
use them.
