# Concordia Typing

The `typing` module defines the core data structures, interfaces, and types used
throughout the Concordia library. It serves as the contract between different
parts of the system.

## Core Interfaces

### Entity (`entity.py`)
Defines the `Entity` abstract base class, which is the fundamental building
block of the simulation.

*   **`Entity`**: Abstract base class. Must implement `name`, `act`, and
    `observe`.
*   **`ActionSpec`**: Defines the expected format of an entity's action (e.g.,
    formatted text, multiple choice).
*   **`OutputType`**: Enum defining types of outputs (FREE, CHOICE, FLOAT, etc.).

### Entity Component (`entity_component.py`)
Defines the component system interfaces.

*   **`BaseComponent`**: The root of the component hierarchy.
*   **`ContextComponent`**: A component that provides context for acting and
    observing (`pre_act`, `post_act`, `pre_observe`, `post_observe`, `update`).
*   **`Phases`**: Enum defining the lifecycle phases (`PRE_ACT`, `POST_ACT`,
    etc.).

## Scene and Simulation Types

### Scene (`scene.py`)
Data structures for defining narrative scenes.

*   **`SceneSpec`**: Configuration for a concrete scene instance (participants,
    premise, etc.).
*   **`SceneTypeSpec`**: Configuration for a template/type of scene (e.g.,
    "Argue", "Vote").

### Simulation (`simulation.py`)

*   **`Simulation`**: Abstract base class for the top-level simulation runner.

### Prefab (`prefab.py`)

*   **`Prefab`**: Abstract base class for prefabs (pre-configured entities).
*   **`Config`**: Configuration object for defining a simulation setup.

## Logging (`logging.py`)

*   **`Metric`**: Data structure for defining evaluation metrics.
