# Pub/sub toolkit for multi-entity simulations

`concordia.components.agent.pubsub` lets one entity run a slow computation in
the background (`Publisher`) while it keeps acting on the latest completed value
(`Subscriber`). This package adds what a *world* of such entities needs.

| Module         | What it adds                          |
| -------------- | ------------------------------------- |
| `operators.py` | ReactiveX operators that build        |
:                : beliefs from topic streams\:          :
:                : `json_field`, `with_age` (re-renders  :
:                : a belief every tick with its age,     :
:                : e.g. "3 ticks old"),                  :
:                : `rolling_digest`, `deadband` (alarm   :
:                : hysteresis), `combine_labelled`, and  :
:                : `transport` (an imperfect channel per :
:                : subscription\: latency on a sim-time  :
:                : scheduler, seeded drops, noise and    :
:                : bias on numeric JSON fields). Use     :
:                : them as `transform=`/`pipeline=` of   :
:                : `TopicBus.make_subscriber`.           :
| `reflexes.py`  | Standing orders as data: `ReflexSpec` |
:                : validates a JSON-shaped reflex (input :
:                : topic, a pipeline of allowlisted      :
:                : operators such as `json_field`,       :
:                : `threshold`, `deadband`,              :
:                : `moving_average`, `change`, `scale`,  :
:                : `every`, an emit topic and template,  :
:                : a cooldown) and `Reflexes` installs   :
:                : it on a `TopicBus` with no language   :
:                : model call, counting fires and        :
:                : checkpointing the installed specs.    :
:                : `parse_specs` pulls reflexes out of a :
:                : model's prose; `allowed_ops` lets a   :
:                : game gate operators (e.g. behind      :
:                : research).                            :
| `sim_time.py`  | `SimTimeScheduler`: a ReactiveX       |
:                : virtual-time scheduler whose clock    :
:                : counts simulation ticks, with         :
:                : deterministic ordering (due tick,     :
:                : key, submission) even when entities   :
:                : trigger from parallel threads.        :
:                : `LatencyScheduler` views delay work   :
:                : by a configurable number of ticks\:   :
:                : pass one as `scheduler=` to           :
:                : `pubsub.Publisher` and a deliberation :
:                : lands exactly *k* ticks after its     :
:                : trigger.                              :
| `topics.py`    | `TopicBus`: named, hierarchical       |
:                : topics with retention (latest,        :
:                : replay, live-only), message envelopes :
:                : (sequence, sim time, source), prefix  :
:                : subscriptions compatible with         :
:                : `interrupt_scheduling.InterruptMask`, :
:                : factories for topic-bound             :
:                : `Subscriber`s and `Publisher`s,       :
:                : topology dumps (Mermaid, DOT), a      :
:                : `Measurements` tap and checkpointing. :

Related contributions built on this package:

*   `contrib/components/game_master/topic_observations.py`: deliver topic
    messages as observations, with per-entity access lists.
*   `contrib/prefabs/entity/basic_with_async_deliberation.py`: an entity prefab
    with all three cognition tiers on stock components: topic beliefs that show
    their age, the stock `Plan` running in the background inside a `Publisher`
    (threads, or a fixed number of ticks on a `SimTimeScheduler`), and a fast
    act that reads the latest finished plan. Like `basic` and `minimal`, its
    `build` takes a runtime acting policy (`act_component` or
    `act_component_factory`), so a person can play it through
    `HumanActComponent` while it still deliberates in the background.
*   `contrib/components/game_master/stock_flow_ledger.py`: exact, conserved
    stocks and flows (recipes, links with capacity and latency, per-tick
    production statistics) and `balance_chain`, a ratio planner that works back
    from a target recipe through every input's producer.
*   `contrib/components/game_master/marketplace.py`: a programmatic API for the
    `MarketPlace` clearing house (`place_order`, `clear`, `drain_outcomes`), so
    a game can run markets every tick without narrated actions.
*   `contrib/components/game_master/numeric_time.py`: a numeric clock whose
    ticks can be seconds or millions of years, for simulations that span eras.

## Quick start

```python
from concordia.contrib.pubsub import topics

bus = topics.TopicBus(clock=lambda: world.tick)
bus.declare('climate.global', description='Global mean temperature, JSON.')

# A Subscriber component: the entity's belief about the climate.
thermometer = bus.make_subscriber(
    'climate.global',
    subscriber_name='Daisies',
    initial_value='No reading yet.',
)

# A Publisher component whose completed values are published on a topic.
chronicler = bus.make_publisher(
    'chronicle', component=my_narrative_component, source='Chronicler',
    skip_if_busy=True,
)

bus.publish('climate.global', '{"temp_c": 21.5}', source='climate model')
print(bus.to_mermaid())  # publisher -> topic -> subscriber graph
```

Both factories return the standard components from
`concordia.components.agent.pubsub`, so they drop into any prefab's
`context_components`.

## Beliefs that age

```python
from concordia.contrib.pubsub import operators
from reactivex import operators as ops

ticks = bus.observable('world.tick').pipe(ops.map(lambda m: m.time))
thermometer = bus.make_subscriber(
    'climate.global',
    transform=operators.with_age(ticks, render=render_climate),
)
# "Global mean temperature 24.3 C [fresh]" ... "[3 ticks old]" if it stops.
```

## Deterministic latency

```python
from concordia.contrib.pubsub import sim_time

sim = sim_time.SimTimeScheduler()
adaptation = pubsub.Publisher(
    component=question_component,
    skip_if_busy=True,
    scheduler=sim.latency(3, key='White daisies'),
)
# Once per simulation step (e.g. in the game master's NEXT_ACTING handler):
sim.advance()
```

With a mock or cached model the whole run is then reproducible, and latency
becomes a design parameter (`view.set_latency(1)` when a faster technology is
unlocked). `sim_time.publisher_scheduler(None, ...)` returns `None`, so the same
code builds ordinary thread-pool publishers for live asynchronous play.

What is still in flight can be shown to players: `sim.due('White daisies')`
lists the ticks on which that mind's queued work will land (`sim.due()` for
all keys), and `sim.pending(key)` counts it.

## Imperfect channels

`operators.transport` carries messages to one subscriber the way an instrument,
a courier or a radio link would: late, sometimes lost, noisy. The spec may be a
callable, read for every message, so a channel can be upgraded mid-run (a better
instrument, the telegraph) or set per subscriber (a colony light-years away).

```python
link = operators.Transport(
    latency=2,                      # ticks on the sim-time scheduler
    drop_rate=0.1,                  # seeded per subscription
    noise_sigma=1.5, fields=('temp_c',),
)
thermometer = bus.make_subscriber(
    'climate.global',
    transform=reactivex.compose(
        operators.transport(lambda: link, seed=7, scheduler=sim),
        operators.with_age(ticks, render=render_climate),
    ),
)
```

A message that arrives after a newer one (possible when the latency shrinks) is
discarded, so a belief never goes backwards. A news feed that wants every item
sets `keep_late=True`.

With `per_message=True` the spec is a function of each message, so messages can
travel differently, e.g. news that takes longer to arrive the farther away it
happened:

```python
def by_distance(message):
  away = distance(home, json.loads(message.payload)['where'])
  return operators.Transport(latency=away / speed, keep_late=True)

local_news = bus.make_subscriber(
    ['local.r3', 'local.r4'],
    transform=operators.transport(by_distance, per_message=True, scheduler=sim),
)
```

## Balanced production chains

```python
from concordia.contrib.components.game_master import stock_flow_ledger as sfl

recipes = [
    sfl.Recipe('mine', {}, {'ore': 1.0}),
    sfl.Recipe('kiln', {'wood': 2.0}, {'charcoal': 1.0}),
    sfl.Recipe('bloomery', {'ore': 2.0, 'charcoal': 1.0}, {'iron': 1.0}),
]
plan = sfl.balance_chain(recipes, 'bloomery', batches=3.0)
# plan.batches: bloomery 3, mine 6, kiln 3; plan.external: wood 6 a tick.
```

Shared inputs are summed, a recipe that makes several needed resources runs at
the rate its most demanded output needs, producers can be chosen per resource,
and circular chains raise.

## Design notes

*   Payloads are strings (use JSON for structure), matching `Subscriber`.
*   Publishing is serialized with a re-entrant lock: sequence numbers and
    delivery order agree, and a subscriber callback may publish to another topic
    (useful for automation pipelines).
*   `set_state` replays restored messages into topic subjects only, not into
    `matching` streams or measurements, so restored values do not re-trigger
    automation.
*   `as_interrupt_event` converts a message into an `interrupt_scheduling.Event`
    whose tag is the topic name, so interrupt masks can select topics.
