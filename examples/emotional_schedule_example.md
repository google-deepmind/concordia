# Emotional State and Schedule Tracking Example

This example demonstrates how to use the new contributed components:
- `EmotionalState`: Tracks agent emotional states based on experiences
- `EmotionalAppraisal`: Evaluates emotional impact of situations
- `ScheduleTracker`: Manages time-based events and schedules for simulations

## Overview

This simulation models a day in the life of a workplace with:
- **Agents**: Employees with emotional states that change based on events
- **Game Master**: Tracks scheduled meetings, deadlines, and events
- **Emotional Dynamics**: Agents experience and respond to emotional states

## Key Features Demonstrated

1. **Emotional State Tracking**: Agents maintain and update emotional states
2. **Schedule Management**: Time-based events with recurring meetings
3. **Emotional Appraisal**: Agents consider emotional consequences of actions
4. **Integration**: Components work together naturally

## Usage

```python
import datetime
from concordia.contrib.components.agent import emotional_state
from concordia.contrib.components.game_master import schedule_tracker
from concordia.components.agent import memory, observation, instructions
from concordia.language_model import language_model

# Setup (requires an LLM):
# model = your_language_model()

# Create an agent with emotional state tracking
agent_components = [
    memory.AssociativeMemory(memory_bank=your_memory_bank),
    observation.LastNObservations(n=5),
    instructions.Instructions(
        agent_name='Alice',
        goal='Complete work tasks while maintaining emotional wellbeing'
    ),
    emotional_state.EmotionalState(
        model=model,
        num_memories_to_retrieve=20,
        include_intensity=True,
    ),
    emotional_state.EmotionalAppraisal(
        model=model,
    ),
]

# Create a game master with schedule tracking
def clock_now():
    return datetime.datetime(2024, 1, 15, 9, 0, 0)

schedule = schedule_tracker.ScheduleTracker(
    clock_now=clock_now,
    verbose=True,
)

# Add scheduled events
schedule.add_event(
    name='Team Standup',
    scheduled_time=datetime.datetime(2024, 1, 15, 9, 30, 0),
    participants=['Alice', 'Bob', 'Charlie'],
    location='Conference Room A',
    duration_minutes=15,
    recurring='daily',
    priority=7,
)

schedule.add_event(
    name='Project Deadline',
    scheduled_time=datetime.datetime(2024, 1, 15, 17, 0, 0),
    priority=10,
    details='Final submission for Q1 project',
)

schedule.add_event(
    name='Lunch Break',
    scheduled_time=datetime.datetime(2024, 1, 15, 12, 0, 0),
    duration_minutes=60,
    recurring='daily',
    priority=5,
)
```

## Example Scenario

The simulation demonstrates:

1. **Morning**: Alice starts work, check schedule shows upcoming standup
2. **Standup**: Alice participates in meeting, emotional state reflects engagement
3. **Mid-morning**: Works on tasks, schedule shows looming deadline
4. **Emotional Response**: Deadline pressure affects emotional state
5. **Lunch**: Break time provides emotional relief
6. **Afternoon**: Final push before deadline
7. **Completion**: Emotional state improves after task completion

## Emotional State Progression

```
9:00 AM  - Calm, ready to start day
9:30 AM  - Engaged during standup meeting
11:00 AM - Slightly anxious about deadline (3 hours warning)
12:00 PM - Relief during lunch break
2:00 PM  - Moderately stressed (deadline approaching)
4:30 PM  - Very anxious (30 minutes to deadline)
5:00 PM  - Relieved and accomplished (deadline met)
```

## Schedule View Example

```
Scheduled events:
Currently happening:
  - Team Standup at Conference Room A (ends at 09:45)
    Participants: Alice, Bob, Charlie

Upcoming events:
  - Lunch Break at 12:00 (in 2h 15m)
    Location: 
  - Project Deadline at 17:00 (in 7h 15m)
```

## Advanced Usage

### Custom Emotional Categories

```python
emotional_state.EmotionalState(
    model=model,
    emotion_categories=[
        'focused', 'overwhelmed', 'motivated', 
        'burned out', 'collaborative', 'isolated'
    ],
    include_intensity=True,
)
```

### Event Filtering

```python
# Get Alice's schedule only
alice_schedule = schedule.get_upcoming_events(
    participant='Alice',
    time_window_minutes=240  # Next 4 hours
)

# Get all events in a location
room_events = schedule.get_events_by_location('Conference Room A')

# Check for overdue items
overdue = schedule.get_overdue_events()
```

### Emotional-Schedule Integration

Agents can query both components together:

```python
# Check upcoming emotionally significant events
upcoming = schedule.get_upcoming_events(limit=3)
for event in upcoming:
    if event.priority >= 8:
        # High priority events may trigger anticipatory emotions
        agent.observe(f"Important event approaching: {event.name}")
```

## Benefits for Simulations

1. **Psychological Depth**: Agents exhibit more realistic emotional responses
2. **Temporal Awareness**: Events unfold with proper timing
3. **Social Dynamics**: Shared schedules create natural interaction points
4. **Stress & Wellbeing**: Model work-life balance and emotional regulation
5. **Causal Reasoning**: Events have emotional consequences

## See Also

- `concordia/components/agent/` - Core agent components
- `concordia/components/game_master/` - Time and world management
- `examples/tutorial.ipynb` - Basic Concordia tutorial

## Contributing

These components are community contributions! If you:
- Find bugs or have improvements
- Want to extend functionality
- Create interesting simulations using these components

Please open an issue or PR at the Concordia repository.
