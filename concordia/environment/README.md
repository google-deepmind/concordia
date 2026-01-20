# Concordia Environment and Engines

This directory contains the core logic that drives Concordia simulations. While
components constitute the "body" and "mind" of an agent, and the Game Master
provides the world model, the **Environment** (specifically the **Engine**)
provides the *time* and *causality* of the simulation.

## 1. The Engine Interface

The `Engine` is the main loop driver. It orchestrates the interaction between
the Game Master (GM) and the Agents. It is responsible for:

1.  **Observing**: Asking the GM what each agent sees.
2.  **Scheduling**: Deciding which agent(s) act next.
3.  **Resolving**: Taking agent actions and asking the GM to resolve their
    outcome into a new world state.
4.  **Terminating**: Deciding when the simulation ends.

The base class is defined in `engine.py`. Key methods include:

*   `run_loop`: The main entry point. It runs the simulation cycle until
    `terminate` returns `True`.
*   `next_acting`: Determines which entities get to act in the current step.
*   `make_observation`: Generates the string observation for a specific entity.
*   `resolve`: Takes a "putative event" (agent action) and finalizes it
    (updating GM state).

## 2. Standard Engines

Concordia provides two primary implementations of the `Engine` interface in the
`engines/` subdirectory.

### Sequential Engine (`engines/sequential.py`)

The `Sequential` engine implements a **turn-based** flow.

*   **Logic**: At each step, it asks the GM to select *one* agent to act.
*   **Best For**: Conversations, interviews, or any scenario where precise
    order matters (e.g., A speaks, then B answers).
*   **Flow**:
    1.  GM selects one agent.
    2.  That agent acts.
    3.  GM resolves the action immediately.

### Simultaneous Engine (`engines/simultaneous.py`)

The `Simultaneous` engine implements a **parallel** flow.

*   **Logic**: It can ask *multiple* agents to act at the same time.
*   **Best For**: Voting, auctions, movement in a grid, or efficient large-scale
    simulations where agents don't need to wait for each other.
*   **Flow**:
    1.  GM selects a group of agents (or all of them).
    2.  All selected agents generate their actions in parallel.
    3.  The engine collects all actions and passes them *together* to the GM for
        batched resolution.

## 3. Scenes and Execution

Complex simulations often require more than a single infinite loop. They need
structure—chapters, days, or distinct phases.

### Scene Runner (`scenes/runner.py`)

The `run_scenes` function allows you to string together multiple "Scenes".

*   **Scene**: A configuration object containing a specific `Engine` and
    `GameMaster` for a period of time.
*   **Transition**: Handles "premise" messages (intro text) and "conclusion"
    messages (outro text) when switching scenes.
*   **Clock**: Integrates with a `GameClock` to manage in-game time progression
    between scenes.

This allows for simulations like:

*   **Scene 1**: Morning briefing (Sequential engine, Meeting Room GM).
*   **Scene 2**: Work day (Simultaneous engine, Office GM).
*   **Scene 3**: Evening debrief (Sequential engine, Home GM).

## 4. Extending the Engine

You can create custom engines by inheriting from
`concordia.environment.engine.Engine`. However, this is rarely necessary.

*   **Logic belongs in components**: Termination criteria and specific
    turn-taking schedules (e.g., "A and B act, then C acts") should typically
    be implemented as Game Master components (specifically `Terminate` and
    `NextActing`).
*   **When to extend**: You should only write a new engine if the fundamental
    *pattern of interaction* is different and cannot be expressed by the
    standard `Sequential` or `Simultaneous` engines.

When implementing a custom engine,
`concordia.environment.engine.action_spec_parser` is a useful utility for
converting the GM's string output into structured `ActionSpec` objects for the
agents.
