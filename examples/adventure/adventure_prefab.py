# Copyright 2026 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Prefab builders and simulation container for the Adventure engine.

Assembles:
- `NpcEntity(prefab_lib.Prefab)` with `ReactiveMeasurements`, `TopicBus`
  beliefs, strong agency instructions, and optional background `Plan`
  deliberation.
- `PlayerEntity(prefab_lib.Prefab)` with pluggable `ActingComponent`
  (`HumanActComponent` for web UI or `PlaytesterActComponent` for headless
  evaluation).
- `AdventureGameMaster(prefab_lib.Prefab)` combining `WorldLedger`,
  `StoryDirector`, `RosterManager`, `EntityScheduler`, `FastSceneResolver`,
  `AdventureNextActing`, `TopicObservations`, `MakeObservation`,
  `FrontierArchitectPublisher`, and stock `SwitchAct`.
- `AdventureSimulation(generic.Simulation)` running on Concordia's stock
  `Asynchronous` engine with full checkpointing and dynamic state support.
"""

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import datetime
import functools
import threading
from typing import Any, ClassVar, cast, override

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as agent_components
from concordia.components import game_master as gm_components
from concordia.contrib.components.game_master import topic_observations
from concordia.contrib.pubsub import topics
from concordia.environment import step_controller as step_controller_lib
from concordia.environment.engines import asynchronous
from examples.adventure import adventure_specific_components
from examples.adventure import roster_manager
from examples.adventure import story_director
from examples.adventure import story_state
from examples.adventure import world_ledger
from concordia.language_model import language_model
from concordia.prefabs.simulation import generic
from concordia.typing import entity_component
from concordia.typing import prefab as prefab_lib
from concordia.utils import async_measurements
from concordia.utils import structured_logging
import numpy as np

DEFAULT_PREMISE = (
    'The tidal port of Glassharbor clings to basalt cliffs above a fog-bound '
    'inlet. Smugglers, harbor wardens, and guild archivists trade secrets '
    'around the lantern-lit piers while an ancient clockwork floodgate '
    'threatens to fail.'
)

DEFAULT_STARTING_LOCATION = 'The Salt-Glass Wharf'
DEFAULT_STARTING_DESCRIPTION = (
    'Rain slicks the basalt flagstones of the wharf. Iron cargo cranes creak '
    'overhead in the sea wind, and salt-crusted oil lanterns cast amber rings '
    'across moored skiffs and a bolted customs shed.'
)
DEFAULT_STARTING_AFFORDANCES = (
    'padlocked customs shed door with a corroded brass seal',
    'waterlogged manifest ledger chained to the bollard',
    'coil of tarred hemp rope beside the Winch House',
)
DEFAULT_STARTING_EXITS = {
    'north': 'The Amber Lantern Tavern',
    'east': 'The Floodgate Gearworks',
    'west': 'The Sunken Archive Vaults',
}


def default_embedder(text: str) -> np.ndarray:
  """Deterministic lightweight hash embedder for local associative memory."""
  vec = np.zeros(64, dtype=np.float32)
  for idx, token in enumerate(text.lower().split()):
    slot = hash(token) % 64
    vec[slot] += 1.0 / (1.0 + idx * 0.05)
  norm = float(np.linalg.norm(vec))
  if norm > 0:
    vec /= norm
  return vec


@dataclasses.dataclass
class NpcEntity(prefab_lib.Prefab):
  """Prefab for a fast-acting Adventure NPC with TopicBus beliefs and Plan."""

  description: ClassVar[str] = (
      'Fast-acting Adventure NPC entity with TopicBus beliefs and optional '
      'async Plan deliberation.'
  )
  default_role: ClassVar[prefab_lib.Role] = prefab_lib.Role.ENTITY
  supports_extra_components: ClassVar[bool] = True

  @override
  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: basic_associative_memory.AssociativeMemoryBank,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    self.check_extra_components()
    raw_record = self.params.get('record')
    if isinstance(raw_record, world_ledger.NpcRecord):
      record = raw_record
    else:
      record = world_ledger.NpcRecord(
          name=str(self.params.get('name', 'NPC')),
          role_summary=str(self.params.get('role_summary', 'Local inhabitant')),
          goal=str(self.params.get('goal', 'Pursue personal agenda')),
          traits=str(self.params.get('traits', 'Observant and guarded')),
          location_id=str(self.params.get('location_id', 'start')),
          faction=str(self.params.get('faction', '')),
          secret=str(self.params.get('secret', '')),
          wound=str(self.params.get('wound', '')),
      )

    raw_bus = self.params.get('bus')
    bus = raw_bus if isinstance(raw_bus, topics.TopicBus) else topics.TopicBus()
    raw_ledger = self.params.get('ledger')
    ledger = (
        raw_ledger if isinstance(raw_ledger, world_ledger.WorldLedger) else None
    )
    raw_tracker = self.params.get('latency_tracker')
    latency_tracker = (
        raw_tracker
        if isinstance(raw_tracker, adventure_specific_components.LatencyTracker)
        else None
    )
    enable_parallel_subminds = bool(
        self.params.get('enable_parallel_subminds', True)
    )

    measurements = async_measurements.ReactiveMeasurements()
    instructions = agent_components.instructions.Instructions(
        agent_name=record.name,
        pre_act_label='\nInstructions',
    )
    faction_tag = f' [{record.faction}]' if record.faction else ''
    identity_text = (
        f'{record.name}{faction_tag} — {record.role_summary}.\n'
        f'Speech Register & Manner: {record.traits}'
    )
    persona = agent_components.constant.Constant(
        state=identity_text,
        pre_act_label='\nIdentity & Voice Register',
    )
    rumor_sub = bus.make_subscriber(
        'world.rumors',
        subscriber_name=record.name,
        initial_value='No recent rumors heard.',
        pre_act_label='\nCirculating World Rumors',
    )
    obs_to_mem = agent_components.observation.ObservationToMemory()
    recent_obs = agent_components.observation.LastNObservations(
        history_length=12,
        pre_act_label='\nRecent Scene Observations',
    )
    obs_key = agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY
    mem_key = agent_components.memory.AssociativeMemory
    mem_comp_key = agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY

    components: dict[str, entity_component.ContextComponent] = {
        topics.DEFAULT_TOPIC_BUS_COMPONENT_KEY: topics.TopicBusComponent(bus),
        'Instructions': instructions,
        'Persona': persona,
        'Rumors': rumor_sub,
        'ObservationToMemory': obs_to_mem,
        obs_key: recent_obs,
        mem_comp_key: mem_key(memory_bank=memory_bank),
    }
    order = ['Instructions', 'Persona', 'Rumors', obs_key]

    if ledger is not None:
      npc_council = adventure_specific_components.NpcSubMindCouncilComponent(
          npc_name=record.name,
          ledger=ledger,
          model=model,
          latency_tracker=latency_tracker,
          parallel_deliberation_enabled=enable_parallel_subminds,
      )
      components[
          adventure_specific_components.DEFAULT_NPC_SUBMIND_COUNCIL_COMPONENT_KEY
      ] = npc_council
      for sub_comp in npc_council.submind_components.values():
        components[sub_comp.component_key] = sub_comp
      order.append(
          adventure_specific_components.DEFAULT_NPC_SUBMIND_COUNCIL_COMPONENT_KEY
      )

    self.validate_extra_components(components)
    extras = self.params.get('extra_components', {})
    if isinstance(extras, Mapping):
      for key, comp in extras.items():
        components[str(key)] = comp
        if str(key) not in order:
          order.append(str(key))

    act_comp = roster_manager.CleanNpcActComponent(
        model=model,
        component_order=order,
    )
    return entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=record.name,
        act_component=act_comp,
        context_components=components,
        measurements=measurements,
    )


@dataclasses.dataclass
class PlayerEntity(prefab_lib.Prefab):
  """Prefab for the Adventure player entity."""

  description: ClassVar[str] = (
      'Adventure player entity driven by a human or synthetic playtester '
      'ActingComponent.'
  )
  default_role: ClassVar[prefab_lib.Role] = prefab_lib.Role.ENTITY
  supports_extra_components: ClassVar[bool] = True

  @override
  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: basic_associative_memory.AssociativeMemoryBank,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    del model
    self.check_extra_components()
    player_name = str(self.params.get('name', 'Traveler'))
    raw_bus = self.params.get('bus')
    bus = raw_bus if isinstance(raw_bus, topics.TopicBus) else topics.TopicBus()
    act_component = cast(
        entity_component.ActingComponent, self.params['act_component']
    )

    measurements = async_measurements.ReactiveMeasurements()
    obs_key = agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY
    mem_key = agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY

    components: dict[str, entity_component.ContextComponent] = {
        topics.DEFAULT_TOPIC_BUS_COMPONENT_KEY: topics.TopicBusComponent(bus),
        'Instructions': agent_components.instructions.Instructions(
            agent_name=player_name,
            pre_act_label='\nInstructions',
        ),
        'Rumors': bus.make_subscriber(
            'world.rumors',
            subscriber_name=player_name,
            initial_value='None heard yet.',
            pre_act_label='\nWorld Rumors',
        ),
        'ObservationToMemory': (
            agent_components.observation.ObservationToMemory()
        ),
        obs_key: agent_components.observation.LastNObservations(
            history_length=15,
            pre_act_label='\nRecent Observations',
        ),
        mem_key: agent_components.memory.AssociativeMemory(
            memory_bank=memory_bank
        ),
    }
    raw_council = self.params.get('submind_council')
    if isinstance(
        raw_council, adventure_specific_components.SubMindCouncilCoordinator
    ):
      components[
          adventure_specific_components.DEFAULT_SUBMIND_COUNCIL_COMPONENT_KEY
      ] = raw_council
      for sub_comp in raw_council.submind_components.values():
        components[sub_comp.component_key] = sub_comp

    self.validate_extra_components(components)
    extras = self.params.get('extra_components', {})
    if isinstance(extras, Mapping):
      for key, comp in extras.items():
        components[str(key)] = comp

    return entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=player_name,
        act_component=act_component,
        context_components=components,
        measurements=measurements,
    )


@dataclasses.dataclass
class AdventureGameMaster(prefab_lib.Prefab):
  """Prefab for the split Adventure Game Master using stock `SwitchAct`."""

  description: ClassVar[str] = (
      'Split low-latency Adventure Game Master with WorldLedger, '
      'StoryDirector, RosterManager, EntityScheduler, FrontierArchitect, and '
      'FastSceneResolver.'
  )
  default_role: ClassVar[prefab_lib.Role] = prefab_lib.Role.GAME_MASTER
  supports_extra_components: ClassVar[bool] = True

  @override
  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: basic_associative_memory.AssociativeMemoryBank,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    self.check_extra_components()
    gm_name = str(self.params.get('name', 'AdventureGM'))
    player_name = str(self.params.get('player_name', 'Traveler'))
    ledger = cast(world_ledger.WorldLedger, self.params['ledger'])
    raw_director = self.params.get('director')
    director = (
        raw_director
        if isinstance(raw_director, story_director.StoryDirector)
        else story_director.StoryDirector(ledger)
    )
    roster = cast(roster_manager.RosterManager, self.params['roster'])
    raw_obs_queue = self.params.get('observation_queue')
    if isinstance(
        raw_obs_queue, gm_components.make_observation.ObservationQueue
    ):
      observation_queue = raw_obs_queue
    else:
      observation_queue = gm_components.make_observation.ObservationQueue()
    raw_topic_obs = self.params.get('topic_obs')
    topic_obs = (
        raw_topic_obs
        if isinstance(raw_topic_obs, topic_observations.TopicObservations)
        else topic_observations.TopicObservations(ledger.bus, observation_queue)
    )
    raw_tracker = self.params.get('latency_tracker')
    latency_tracker = (
        raw_tracker
        if isinstance(raw_tracker, adventure_specific_components.LatencyTracker)
        else adventure_specific_components.LatencyTracker()
    )
    player_ready_fn = self.params.get('player_ready_fn')
    hot_cooldown_seconds = float(self.params.get('hot_cooldown_seconds', 0.05))
    max_consecutive_npc_turns = int(
        self.params.get('max_consecutive_npc_turns', 1)
    )
    warm_tick_interval = int(self.params.get('warm_tick_interval', 3))

    time_model = gm_components.interrupt_time_model.DatetimeTimeModel(
        start_time=datetime.datetime(2026, 1, 1, 8, 0, 0)
    )
    scheduler = gm_components.interrupt_scheduling.EntityScheduler(
        player_names=[],
        time_model=time_model,
    )
    next_acting_component = adventure_specific_components.AdventureNextActing(
        player_ready_fn=player_ready_fn,
        observation_queue=observation_queue,
        hot_cooldown_seconds=hot_cooldown_seconds,
        max_consecutive_npc_turns=max_consecutive_npc_turns,
        warm_tick_interval=warm_tick_interval,
        time_model=time_model,
    )
    frontier_worker = adventure_specific_components.FrontierArchitectWorker(
        model,
        latency_tracker=latency_tracker,
    )
    frontier_publisher = (
        adventure_specific_components.FrontierArchitectPublisher(
            component=frontier_worker,
        )
    )
    scene_resolver = adventure_specific_components.FastSceneResolver(
        model,
        observation_queue=observation_queue,
        latency_tracker=latency_tracker,
    )
    terminate_component = adventure_specific_components.AdventureTerminate()

    gm_context_components: dict[str, entity_component.ContextComponent] = {
        world_ledger.DEFAULT_WORLD_LEDGER_COMPONENT_KEY: ledger,
        story_director.DEFAULT_STORY_DIRECTOR_COMPONENT_KEY: director,
        roster_manager.DEFAULT_ROSTER_COMPONENT_KEY: roster,
        gm_components.interrupt_scheduling.DEFAULT_SCHEDULER_COMPONENT_KEY: (
            scheduler
        ),
        gm_components.terminate.DEFAULT_TERMINATE_COMPONENT_KEY: (
            terminate_component
        ),
        gm_components.make_observation.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY: (
            gm_components.make_observation.MakeObservation(
                model=model,
                player_names=[player_name],
                external_queue=observation_queue,
                allow_llm_fallback=False,
            )
        ),
        topic_observations.DEFAULT_TOPIC_OBSERVATIONS_COMPONENT_KEY: topic_obs,
        gm_components.next_acting.DEFAULT_NEXT_ACTING_COMPONENT_KEY: (
            next_acting_component
        ),
        gm_components.next_acting.DEFAULT_NEXT_ACTION_SPEC_COMPONENT_KEY: (
            next_acting_component
        ),
        gm_components.next_game_master.DEFAULT_NEXT_GAME_MASTER_COMPONENT_KEY: (
            next_acting_component
        ),
        gm_components.switch_act.DEFAULT_RESOLUTION_COMPONENT_KEY: (
            scene_resolver
        ),
        adventure_specific_components.DEFAULT_FRONTIER_COMPONENT_KEY: (
            frontier_publisher
        ),
        agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY: (
            agent_components.memory.AssociativeMemory(memory_bank=memory_bank)
        ),
    }
    self.validate_extra_components(gm_context_components)
    extras = self.params.get('extra_components', {})
    if isinstance(extras, Mapping):
      for key, comp in extras.items():
        gm_context_components[str(key)] = comp

    entity_names = (
        [e.name for e in self.entities] if self.entities else [player_name]
    )
    gm_measurements = async_measurements.ReactiveMeasurements()
    return entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=gm_name,
        act_component=gm_components.switch_act.SwitchAct(
            model=model,
            entity_names=entity_names,
        ),
        context_components=gm_context_components,
        measurements=gm_measurements,
    )


def build_npc_entity(
    record: world_ledger.NpcRecord,
    *,
    model: language_model.LanguageModel,
    embedder: Callable[[str], np.ndarray],
    bus: topics.TopicBus,
    enable_async_plan: bool = True,
    ledger: world_ledger.WorldLedger | None = None,
    latency_tracker: adventure_specific_components.LatencyTracker | None = None,
    enable_parallel_subminds: bool = True,
) -> entity_agent_with_logging.EntityAgentWithLogging:
  """Builds a fast-acting NPC entity via `NpcEntity` prefab."""
  memory_bank = basic_associative_memory.AssociativeMemoryBank(
      sentence_embedder=embedder
  )
  return NpcEntity(
      params={
          'name': record.name,
          'record': record,
          'bus': bus,
          'enable_async_plan': enable_async_plan,
          'ledger': ledger,
          'latency_tracker': latency_tracker,
          'enable_parallel_subminds': enable_parallel_subminds,
      }
  ).build(model=model, memory_bank=memory_bank)


def build_player_entity(
    player_name: str,
    *,
    embedder: Callable[[str], np.ndarray],
    bus: topics.TopicBus,
    act_component: entity_component.ActingComponent,
) -> entity_agent_with_logging.EntityAgentWithLogging:
  """Builds the human or synthetic playtester entity via `PlayerEntity`."""
  memory_bank = basic_associative_memory.AssociativeMemoryBank(
      sentence_embedder=embedder
  )
  return PlayerEntity(
      params={
          'name': player_name,
          'bus': bus,
          'act_component': act_component,
      }
  ).build(
      model=cast(language_model.LanguageModel, None),
      memory_bank=memory_bank,
  )


class AdventureSimulation(generic.Simulation):
  """Idiomatic Concordia `Simulation` for interactive or headless Adventure."""

  def __init__(
      self,
      *,
      model: language_model.LanguageModel,
      embedder: Callable[[str], np.ndarray] = default_embedder,
      player_name: str = 'Traveler',
      premise: str = DEFAULT_PREMISE,
      starting_location_name: str = DEFAULT_STARTING_LOCATION,
      starting_location_description: str = DEFAULT_STARTING_DESCRIPTION,
      starting_affordances: Sequence[str] = DEFAULT_STARTING_AFFORDANCES,
      starting_exits: Mapping[str, str] | None = None,
      player_act_component: entity_component.ActingComponent,
      player_ready_fn: Callable[[], bool] | None = None,
      seed_starter_npcs: bool = True,
      enable_async_npc_plan: bool = True,
      max_hot_npcs: int = 2,
      max_warm_npcs: int = 2,
      warm_tick_interval: int = 3,
      hot_cooldown_seconds: float = 0.05,
      max_consecutive_npc_turns: int = 1,
      bible: story_state.WorldBible | None = None,
      enable_parallel_submind_prompts: bool = False,
  ):
    """Initializes the simulation, ledger, roster, GM, and engine.

    Args:
      model: Language model shared by the GM, NPCs, and frontier worker.
      embedder: Sentence embedder for associative memory.
      player_name: Name of the player entity.
      premise: World premise; overridden by `bible.premise` when a bible is
        supplied.
      starting_location_name: Starting room name; overridden by the bible.
      starting_location_description: Starting room description; overridden by
        the bible.
      starting_affordances: Starting room props; overridden by the bible.
      starting_exits: Starting room exits; overridden by the bible.
      player_act_component: Acting component driving the player entity.
      player_ready_fn: Optional readiness predicate for the player.
      seed_starter_npcs: Whether to seed starting NPCs (from the bible if
        supplied, otherwise the built-in Glassharbor trio).
      enable_async_npc_plan: Whether NPCs run an async Plan publisher.
      max_hot_npcs: Hot tier capacity.
      max_warm_npcs: Warm tier capacity.
      warm_tick_interval: Player steps between Warm ticks.
      hot_cooldown_seconds: Minimum wall time between Hot NPC turns.
      max_consecutive_npc_turns: NPC turns allowed between player turns.
      bible: Optional authored world bible. When supplied it provides the
        premise, starting room, factions, truths, clues, seed NPCs, and
        storylets, and the ledger's story layer becomes active.
      enable_parallel_submind_prompts: Whether to run parallel per-Sub-Mind LLM
        deliberation calls after player turns in addition to the inline Council
        synthesis inside `FastSceneResolver`.
    """
    self._player_name = player_name
    self._log_lock = threading.Lock()
    self._bible = bible

    story = None
    starting_clue_ids: Sequence[str] = ()
    if bible is not None:
      problems = story_state.validate_bible(bible)
      if problems:
        raise ValueError(
            'World bible failed validation:\n  ' + '\n  '.join(problems)
        )
      premise = bible.premise
      if bible.starting_location is not None:
        starting_location_name = bible.starting_location.name
        starting_location_description = bible.starting_location.description
        starting_affordances = tuple(bible.starting_location.affordances)
        starting_exits = dict(bible.starting_location.exits)
        starting_clue_ids = tuple(bible.starting_location.clue_ids)
      story = story_state.StoryState.from_bible(bible)

    exits = (
        dict(starting_exits)
        if starting_exits is not None
        else dict(DEFAULT_STARTING_EXITS)
    )
    self.latency_tracker = adventure_specific_components.LatencyTracker()
    if bible is not None and bible.starting_inventory:
      initial_inventory: Sequence[str] = tuple(bible.starting_inventory)
    else:
      initial_inventory = (
          'unsealed inquiry writ',
          "inspector's pocket ledger",
      )
    self.ledger = world_ledger.WorldLedger(
        premise=premise,
        player_name=player_name,
        starting_location_name=starting_location_name,
        starting_location_description=starting_location_description,
        starting_affordances=starting_affordances,
        starting_exits=exits,
        starting_inventory=initial_inventory,
        story=story,
    )
    if starting_clue_ids:
      self.ledger.expand_location(
          starting_location_name,
          description=starting_location_description,
          clue_ids=starting_clue_ids,
      )
    self.director = story_director.StoryDirector()
    self.submind_council = (
        adventure_specific_components.SubMindCouncilCoordinator(
            ledger=self.ledger,
            director=self.director,
            model=model,
            latency_tracker=self.latency_tracker,
            parallel_deliberation_enabled=enable_parallel_submind_prompts,
        )
    )
    self.observation_queue = gm_components.make_observation.ObservationQueue()
    self.topic_obs = topic_observations.TopicObservations(
        self.ledger.bus,
        self.observation_queue,
    )

    self.roster = roster_manager.RosterManager(
        self.ledger,
        entity_builder=lambda rec: build_npc_entity(
            rec,
            model=model,
            embedder=embedder,
            bus=self.ledger.bus,
            enable_async_plan=enable_async_npc_plan,
            ledger=self.ledger,
            latency_tracker=self.latency_tracker,
            enable_parallel_subminds=True,
        ),
        topic_obs=self.topic_obs,
        max_hot=max_hot_npcs,
        max_warm=max_warm_npcs,
    )

    config = prefab_lib.Config(
        prefabs={
            'player': PlayerEntity(),
            'npc': NpcEntity(),
            'adventure_gm': AdventureGameMaster(),
        },
        instances=[
            prefab_lib.InstanceConfig(
                prefab='player',
                role=prefab_lib.Role.ENTITY,
                params={
                    'name': player_name,
                    'bus': self.ledger.bus,
                    'act_component': player_act_component,
                    'submind_council': self.submind_council,
                },
            ),
            prefab_lib.InstanceConfig(
                prefab='adventure_gm',
                role=prefab_lib.Role.GAME_MASTER,
                params={
                    'name': 'AdventureGM',
                    'player_name': player_name,
                    'ledger': self.ledger,
                    'director': self.director,
                    'roster': self.roster,
                    'observation_queue': self.observation_queue,
                    'topic_obs': self.topic_obs,
                    'latency_tracker': self.latency_tracker,
                    'player_ready_fn': player_ready_fn,
                    'hot_cooldown_seconds': hot_cooldown_seconds,
                    'max_consecutive_npc_turns': max_consecutive_npc_turns,
                    'warm_tick_interval': warm_tick_interval,
                },
            ),
        ],
        default_premise=premise,
        default_max_steps=25,
    )
    super().__init__(
        config=config,
        model=model,
        embedder=embedder,
        engine=asynchronous.Asynchronous(sleep_time=0.03),
    )

    self.player_entity = cast(
        entity_agent_with_logging.EntityAgentWithLogging, self.entities[0]
    )
    self.game_master = cast(
        entity_agent_with_logging.EntityAgentWithLogging, self.game_masters[0]
    )
    self.gm_memory_bank = self.game_master_memory_bank
    self.engine = cast(asynchronous.Asynchronous, self._engine)
    self.next_acting_component = self.game_master.get_component(
        gm_components.next_acting.DEFAULT_NEXT_ACTING_COMPONENT_KEY,
        type_=adventure_specific_components.AdventureNextActing,
    )
    self.terminate_component = self.game_master.get_component(
        gm_components.terminate.DEFAULT_TERMINATE_COMPONENT_KEY,
        type_=adventure_specific_components.AdventureTerminate,
    )
    self.scene_resolver = self.game_master.get_component(
        gm_components.switch_act.DEFAULT_RESOLUTION_COMPONENT_KEY,
        type_=adventure_specific_components.FastSceneResolver,
    )
    self.scene_resolver.set_submind_council(self.submind_council)
    self.frontier_publisher = self.game_master.get_component(
        adventure_specific_components.DEFAULT_FRONTIER_COMPONENT_KEY,
        type_=adventure_specific_components.FrontierArchitectPublisher,
    )
    self.frontier_worker = self.frontier_publisher.worker
    self.scheduler = self.game_master.get_component(
        gm_components.interrupt_scheduling.DEFAULT_SCHEDULER_COMPONENT_KEY,
        type_=gm_components.interrupt_scheduling.EntityScheduler,
    )

    if seed_starter_npcs:
      if bible is not None and bible.seed_npcs:
        self._seed_npcs_from_bible(bible)
      else:
        self._seed_initial_npcs()

    self.entities = self.all_engine_entities()

    # Queue initial opening scene observation and seed the visible intro card.
    start_node = self.ledger.get_player_location()
    co_npcs = [
        f'{n.name} ({n.role_summary})'
        for n in self.ledger.npcs_at_location(start_node.location_id)
    ]
    exit_labels = []
    for d, tid in start_node.exits.items():
      target_loc = self.ledger.get_location(tid)
      exit_labels.append(
          f'{d} -> {target_loc.name if target_loc is not None else tid}'
      )
    if bible is not None and bible.opening.strip():
      lead = bible.opening.format(start_name=start_node.name)
    else:
      lead = (
          f'You have just stepped off a storm-battered coastal skiff onto '
          f'{start_node.name}, carrying an unsealed inquiry writ and your '
          f"inspector's pocket ledger to investigate rumors of sabotage along "
          f'the floodgate seawalls before the midnight king-tide.'
      )
    opening_card = f'{lead}\n\n{start_node.description}'
    opening_obs = (
        f'{premise}\n\n'
        f'{lead}\n\n'
        f'{start_node.description}\n\n'
        f'Around you, you spot: {", ".join(start_node.affordances)}. '
        f'Standing nearby: {", ".join(co_npcs) or "no one"}. '
        f'Passages lead: {", ".join(exit_labels)}.'
    )
    self.ledger.record_intro(opening_card, location_id=start_node.location_id)
    if self.director.active:
      self.director.refresh_voice_options(start_node.location_id)
      self.submind_council.sync_from_ledger()
    self.observation_queue.add(self._player_name, opening_obs, ())

  def configure_player_parliament(
      self,
      *,
      active_ranks: Mapping[str, int] | None = None,
      custom_voices: Sequence[
          Mapping[str, Any] | story_state.SubMindState
      ] = (),
      use_parallel_llm: bool = False,
  ) -> list[story_state.VoiceOption]:
    """Configures the player's Parliament of Sub-Minds and refreshes options."""
    with self.ledger.lock:
      self.ledger.story.configure_parliament(
          active_ranks=active_ranks,
          custom_voices=custom_voices,
      )
      self.ledger.story.active_voice_options.clear()
      if self.ledger.tick == 0:
        self.ledger.story.voice_option_log.clear()
    self.submind_council.sync_components()
    return self.submind_council.convene_council(
        location_id=self.ledger.player_location_id,
        use_parallel_llm=use_parallel_llm,
    )

  def mint_custom_sub_mind(
      self,
      *,
      concept: str,
      name_hint: str = '',
      domain_hint: str = '',
      rank: int = 2,
      origin: str = 'custom',
  ) -> story_state.SubMindState:
    """Procedurally mints a new Sub-Mind and mounts its component on PlayerEntity."""
    return self.submind_council.mint_sub_mind(
        concept=concept,
        name_hint=name_hint,
        domain_hint=domain_hint,
        rank=rank,
        origin=origin,
    )

  @property
  def bible(self) -> story_state.WorldBible | None:
    """Returns the authored world bible, if one was supplied."""
    return self._bible

  def _seed_npcs_from_bible(self, bible: story_state.WorldBible) -> None:
    """Seeds authored NPCs from the bible, creating stub rooms as needed."""
    tier_map = {
        'hot': world_ledger.NpcTier.HOT,
        'warm': world_ledger.NpcTier.WARM,
        'cold': world_ledger.NpcTier.COLD,
    }
    clue_owner: dict[str, list[str]] = {}
    for clue in bible.clues:
      if clue.carrier == 'npc' and clue.carrier_hint:
        clue_owner.setdefault(clue.carrier_hint, []).append(clue.id)

    for seed in bible.seed_npcs:
      loc_id = self.ledger.ensure_stub_location(seed.location_name)
      for slot in seed.routine:
        if slot.location_name:
          self.ledger.ensure_stub_location(slot.location_name)
      tier = tier_map.get(seed.tier, world_ledger.NpcTier.COLD)
      known = list(clue_owner.get(seed.name, []))
      with self.ledger.lock:
        for cid in known:
          clue = self.ledger.story.clues.get(cid)
          if clue is not None:
            clue.placed_at = seed.name
      record = world_ledger.NpcRecord(
          name=seed.name,
          role_summary=seed.role_summary,
          goal=seed.goal,
          traits=seed.traits,
          location_id=loc_id,
          tier=tier,
          relationships=dict(seed.relationships),
          formative_memories=list(seed.formative_memories),
          faction=seed.faction,
          secret=seed.secret,
          wound=seed.wound,
          routine=list(seed.routine),
          known_clue_ids=known,
          sub_minds=dict(seed.sub_minds),
      )
      self.roster.spawn_npc(record, initial_tier=tier, rebalance=False)
    self.roster.rebalance_tiers()

  def _seed_initial_npcs(self) -> None:
    """Seeds starter NPCs across starting and adjacent locations."""
    start_id = self.ledger.player_location_id
    start_node = self.ledger.get_location(start_id)
    adjacent_ids = list(start_node.exits.values()) if start_node else []

    harbor_warden = world_ledger.NpcRecord(
        name='Captain Mara Vance',
        role_summary='One-eyed Harbor Warden inspecting cargo seals',
        goal=(
            'Uncover who tampered with the customs shed seal and confiscate '
            'any unlicensed floodgate keys'
        ),
        traits=(
            'Clipped maritime cadence, suspicious of strangers, respects '
            'hard evidence over flattery'
        ),
        location_id=start_id,
        tier=world_ledger.NpcTier.HOT,
        formative_memories=[
            (
                'Someone pried at the customs shed padlock during the midnight'
                ' tide.'
            ),
            'Archivist Orin owes the harbor watch two crates of lamp oil.',
        ],
    )
    self.roster.spawn_npc(
        harbor_warden, initial_tier=world_ledger.NpcTier.HOT, rebalance=False
    )

    if adjacent_ids:
      first_adj = adjacent_ids[0]
      smuggler = world_ledger.NpcRecord(
          name='Soren Vale',
          role_summary='Disgraced clockwork engineer and contraband broker',
          goal=(
              'Retrieve the brass pressure valve from the customs shed before '
              'the evening high tide floods the lower vaults'
          ),
          traits=(
              'Fast-talking, ink-stained fingers, deflects direct questions '
              'with trade bargains'
          ),
          location_id=first_adj,
          tier=world_ledger.NpcTier.WARM,
          formative_memories=[
              (
                  'Soren hid a ciphered pressure chart inside the Amber Lantern'
                  ' Tavern.'
              ),
              (
                  'Captain Mara Vance will arrest Soren on sight if he returns'
                  ' to the wharf.'
              ),
          ],
      )
      self.roster.spawn_npc(
          smuggler, initial_tier=world_ledger.NpcTier.WARM, rebalance=False
      )

    if len(adjacent_ids) > 1:
      second_adj = adjacent_ids[1]
      archivist = world_ledger.NpcRecord(
          name='Archivist Orin',
          role_summary='Keeper of the Sunken Archive tide-ledgers',
          goal=(
              'Seal the lower gearworks sluice and recover the missing '
              'waterlogged manifest ledger'
          ),
          traits=(
              'Formal, archaic vocabulary, fiercely protective of guild '
              'records, refuses aid without a signed writ or rare artifact'
          ),
          location_id=second_adj,
          tier=world_ledger.NpcTier.WARM,
          formative_memories=[
              (
                  'The floodgate gearworks cannot hold another storm surge'
                  ' without the brass valve.'
              ),
          ],
      )
      self.roster.spawn_npc(
          archivist, initial_tier=world_ledger.NpcTier.WARM, rebalance=False
      )

    self.roster.rebalance_tiers()

  def all_engine_entities(self) -> list[Any]:
    """Returns the player plus stable NPC wrappers for `Asynchronous`."""
    all_entities = [self.player_entity, *self.roster.engine_entities()]
    self.entities = list(all_entities)
    return all_entities

  @property
  def player_steps_completed(self) -> int:
    """Returns the number of resolved player turns so far."""
    return self.scene_resolver.player_steps_completed

  def is_turn_busy(self) -> bool:
    """Returns True when an entity turn or resolution is in flight."""
    return self.scene_resolver.is_turn_busy()

  def stop(self) -> None:
    """Signals the running `Asynchronous` loop to exit cleanly."""
    self.terminate_component.request_stop()
    self.engine.play()

  @override
  def play(
      self,
      premise: str | None = None,
      max_steps: int | None = 25,
      raw_log: list[Mapping[str, Any]] | None = None,
      get_state_callback: Callable[[dict[str, Any]], None] | None = None,
      checkpoint_path: str | None = None,
      step_controller: step_controller_lib.StepController | None = None,
      step_callback: (
          Callable[[step_controller_lib.StepData], None] | None
      ) = None,
      *,
      verbose: bool = False,
  ) -> structured_logging.SimulationLog:
    """Runs the simulation for `max_steps` player turns and returns the log."""
    effective_premise = premise if premise is not None else self.ledger.premise
    effective_steps = max_steps if max_steps is not None else 25
    if raw_log is None:
      raw_log = []
    self._raw_log = raw_log
    self._get_state_callback = get_state_callback

    checkpoint_callback = None
    if checkpoint_path or get_state_callback:
      checkpoint_callback = functools.partial(
          self.save_checkpoint,
          checkpoint_path=checkpoint_path or '',
      )

    callback = step_callback
    if step_controller is not None:

      def completed_step(data: step_controller_lib.StepData) -> None:
        try:
          if step_callback is not None:
            step_callback(data)
        finally:
          step_controller.complete_step()

      callback = completed_step

    self.terminate_component.set_max_player_steps(effective_steps)
    self.engine.run_loop(
        game_masters=[self.game_master],
        entities=self.all_engine_entities(),
        premise=effective_premise,
        max_steps=max(100_000, effective_steps * 50_000),
        verbose=verbose,
        log=raw_log,
        checkpoint_callback=checkpoint_callback,
        step_controller=step_controller,
        step_callback=callback,
    )
    return self.build_structured_log()

  def build_structured_log(self) -> structured_logging.SimulationLog:
    """Builds a `SimulationLog` snapshot from current raw events and memories."""
    with self._log_lock:
      # Materialize any buffered async events into _raw_log.
      self.engine._collector.materialize(self._raw_log)  # pylint: disable=protected-access
      sim_log = structured_logging.SimulationLog.from_raw_log(self._raw_log)
      entity_memories = self.roster.collect_all_entity_memories(
          self.player_entity
      )
      gm_memories = list(self.gm_memory_bank.get_all_memories_as_text())
      sim_log.attach_memories(
          entity_memories=entity_memories,
          game_master_memories=gm_memories,
      )
      return sim_log
