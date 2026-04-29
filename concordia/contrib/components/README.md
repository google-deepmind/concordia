# Contributed Components

This directory contains components contributed by the Concordia community. These components extend the core functionality with specialized behaviors useful for various simulation scenarios.

## Agent Components

### choice_of_component
Select a component from a list and dynamically choose which to activate based on the current situation.

**Use cases**: Context-aware component selection, adaptive agent behavior

### emotional_state ⭐ NEW
Track and reflect on agent emotional states based on recent experiences.

**Components**:
- `EmotionalState`: Analyzes memories to determine current emotional state
- `EmotionalAppraisal`: Evaluates emotional consequences of situations

**Use cases**: 
- Social simulations with psychological realism
- Mental health scenarios
- Interpersonal dynamics
- Decision-making under emotional influence
- Therapy/counseling simulations

**See**: `examples/emotional_schedule_example.md`

## Game Master Components

### death
Handle agent death/removal from simulations.

**Use cases**: Survival scenarios, life cycle simulations

### day_in_the_life_initializer
Initialize agents with daily routine memories.

**Use cases**: Establishing baseline agent behavior patterns

### forum
Manage forum-style communication between agents.

**Use cases**: Online community simulations, social media

### marketplace
Economic transactions and market dynamics.

**Use cases**: Economic simulations, trading scenarios

### schedule_tracker ⭐ NEW
Comprehensive schedule and event management system.

**Features**:
- Event scheduling with time, location, participants, duration
- Priority system (1-10 scale)
- Recurring events (daily, weekly, monthly)
- Event queries (upcoming, current, overdue, by location, by participant)
- Event completion tracking
- State persistence

**Use cases**:
- Workplace simulations
- Academic/school environments
- Medical appointments
- Project management
- Social calendars
- Time-pressure scenarios

**See**: `examples/emotional_schedule_example.md`

### spaceship_system
Manage spaceship systems and states.

**Use cases**: Science fiction scenarios, space exploration

## Using Contributed Components

Import components from the contrib package:

```python
from concordia.contrib.components.agent import emotional_state
from concordia.contrib.components.game_master import schedule_tracker

# Use in your agents
agent_components = [
    # ... core components ...
    emotional_state.EmotionalState(
        model=model,
        num_memories_to_retrieve=20,
    ),
]

# Use in your game master
gm_components = [
    # ... core components ...
    schedule_tracker.ScheduleTracker(
        clock_now=clock_now,
    ),
]
```

## Contributing Your Own Components

See [CONTRIBUTING.md](../../../CONTRIBUTING.md) for guidelines on contributing new components.

**Requirements**:
- Must be broadly applicable (not project-specific)
- Include comprehensive tests
- Follow Google Python style guide
- Document with docstrings
- Provide usage examples

**Process**:
1. Open an issue tagged with 'contribution' to discuss
2. Implement component in appropriate contrib subdirectory
3. Add tests (e.g., `your_component_test.py`)
4. Update this README
5. Submit pull request

## Testing Contributed Components

```bash
# Test specific components
python -m pytest concordia/contrib/components/agent/emotional_state_test.py
python -m pytest concordia/contrib/components/game_master/schedule_tracker_test.py

# Test all contrib components
python -m pytest concordia/contrib/components/
```

## Component Maturity

| Component | Status | Tests | Documentation |
|-----------|--------|-------|---------------|
| choice_of_component | Stable | ✓ | ✓ |
| emotional_state | New | ✓ | ✓ |
| death | Stable | ✓ | ✓ |
| day_in_the_life_initializer | Stable | - | ✓ |
| forum | Stable | ✓ | ✓ |
| marketplace | Stable | - | ✓ |
| schedule_tracker | New | ✓ | ✓ |
| spaceship_system | Stable | - | ✓ |

## Support

For issues or questions about contributed components:
- Check component docstrings and examples
- Review test files for usage patterns
- Open an issue on GitHub with 'contrib' label

## License

All contributed components are licensed under Apache 2.0, matching the Concordia project license.
