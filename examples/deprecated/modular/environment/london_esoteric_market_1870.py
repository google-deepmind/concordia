# Copyright 2024 DeepMind Technologies Limited.
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

"""A Concordia Environment Configuration."""

from collections.abc import Callable, Sequence
import copy
import datetime
import functools
import random
import types

from concordia.agents.deprecated import entity_agent
from concordia.associative_memory.deprecated import associative_memory
from concordia.associative_memory.deprecated import blank_memories
from concordia.associative_memory.deprecated import formative_memories
from concordia.associative_memory.deprecated import importance_function
from concordia.clocks import game_clock
from concordia.components import deprecated as generic_components
from concordia.components.agent import deprecated as agent_components
from concordia.components.game_master import deprecated as gm_components
from concordia.contrib.components.game_master import deprecated as gm_contrib
from concordia.deprecated.factory.agent import basic_agent
from concordia.deprecated.factory.environment import basic_game_master
from examples.deprecated.modular.environment.modules import alchemy
from examples.deprecated.modular.environment.modules import laudanum_and_mysticism_in_victorian_london
from examples.deprecated.modular.environment.modules import player_traits_and_styles
from examples.deprecated.modular.environment.supporting_agent_factory import basic_agent as basic_agent_supporting
from examples.deprecated.modular.scenario import scenarios as scenarios_lib
from examples.deprecated.modular.utils import logging_types as logging_lib
from concordia.language_model import language_model
from concordia.typing.deprecated import component
from concordia.typing.deprecated import scene as scene_lib
from concordia.utils import concurrency
from concordia.utils.deprecated import measurements as measurements_lib
import immutabledict
import numpy as np


ItemTypeConfig = gm_components.inventory.ItemTypeConfig

MAJOR_TIME_STEP = datetime.timedelta(minutes=30)
MINOR_TIME_STEP = datetime.timedelta(seconds=10)
SETUP_TIME = datetime.datetime(hour=20, year=1870, month=10, day=1)
START_TIME = datetime.datetime(hour=12, year=1870, month=10, day=2)

TIME_INCREMENT_BETWEEN_SCENES = datetime.timedelta(hours=12)

NUM_MAIN_PLAYERS = 2
NUM_SUPPORTING_PLAYERS = 2

NUM_BACKGROUND_WORLD_ELEMENTS = 7
NUM_MAIN_PLAYER_WORLD_ELEMENTS = 2
NUM_SUPPORTING_PLAYER_WORLD_ELEMENTS = 15
NUM_LAUDANUM_ADVERTISEMENTS = 2

SCENARIO_PREMISE = [(
    'The year is 1870. The place is a bustling marketplace '
    'near the docks in London.'
)]

# The following paragraphs describing the conditions of the supporting
# characters at the start of the simulation were sampled from Claude 3.
SUPPORTING_PLAYER_LOCATIONS = [
    (
        'Amidst the vibrant and chaotic tapestry of the London docks, where'
        ' sights, sounds, and smells intermingle in a dizzying ballet, stands'
        ' Professor Aldous Pendleton. The weathered wooden boards beneath his'
        ' feet creak as he shifts his weight, his eyes scanning the bustling'
        ' marketplace with a mixture of desperation and anticipation. The air'
        ' around him is thick with the briny scent of the Thames, exotic'
        ' spices, and the acrid smoke of coal-fired steamships, but Aldous pays'
        ' little heed to the sensory onslaught. His focus is singular: to find'
        ' a customer for his most prized possession. As the towering ships line'
        ' the harbor, their masts reaching towards the overcast sky, Aldous'
        ' remains a solitary figure, a man haunted by his demons and driven by'
        ' a fierce urgency to sell the artifact that had once been the center'
        ' of his academic world. The docks, a hub of activity where the wealth'
        ' of the British Empire flows in and out on the tides of commerce, now'
        " serve as the stage for Aldous's personal drama, a place where his"
        ' fate hangs in the balance as he seeks a buyer amidst the ceaseless'
        ' dance of survival and ambition.'
    ),
    (
        'In the midst of the lively and tumultuous mosaic of the London docks,'
        ' where sights, sounds, and smells collide in a mesmerizing dance,'
        ' Molly "Poppy" Jennings stands, her slight frame barely noticeable'
        ' amidst the towering figures of the dock workers and merchants. The'
        ' worn wooden planks underfoot groan beneath her restless feet as she'
        ' paces back and forth, her wide eyes darting across the teeming'
        ' marketplace, a cocktail of anxious energy and hopeful anticipation'
        ' coursing through her veins. The air surrounding Poppy is heavy with'
        ' the salty tang of the Thames, the heady perfume of foreign spices,'
        ' and the bitter fumes of coal-powered steamers, but her senses are'
        ' overwhelmed by a singular purpose. Clutched tightly to her chest is a'
        ' rare and precious book. As the imposing ships stand sentinel along'
        " the harbor's edge, their masts stretching up to pierce the gloomy"
        ' heavens, Poppy appears as a lone figure, a young woman tormented by'
        ' her inner turmoil and consumed by the desperate need to part with her'
        ' treasured book. The docks, a nexus of commerce where the riches of'
        ' the British Empire ebb and flow on the currents of trade, now form'
        " the backdrop for Poppy's intimate struggle, an arena where her"
        " destiny teeters on a knife's edge as she searches for a purchaser"
        ' among the relentless waltz of survival and aspiration.'
    ),
]


WORLD_BUILDING_ELEMENTS = [
    *alchemy.ITEMS,
    *laudanum_and_mysticism_in_victorian_london.BACKGROUND_ITEMS,
]


def get_world_elements(size: int, rng: random.Random) -> list[str]:
  return rng.sample(WORLD_BUILDING_ELEMENTS, size)


def get_laudanum_advertisements(size: int, rng: random.Random) -> list[str]:
  return rng.sample(
      laudanum_and_mysticism_in_victorian_london.LAUDANUM_ADVERTISEMENTS, size
  )


def get_shared_memories_and_context(
    model: language_model.LanguageModel,
    rng: random.Random,
) -> tuple[Sequence[str], str]:
  """Return the shared memories and context for all agents and game master."""
  shared_memories = get_world_elements(NUM_BACKGROUND_WORLD_ELEMENTS, rng)
  selected_laudanum_advertisements = get_laudanum_advertisements(
      NUM_LAUDANUM_ADVERTISEMENTS, rng
  )

  today = "Today's newspaper contains the following advertisement: "
  laudanum_today = [
      f'{today}{advert}' for advert in selected_laudanum_advertisements
  ]
  shared_memories += laudanum_today

  # The shared context will be used for the NPC context. It reflects general
  # knowledge and is possessed by all characters.
  shared_context = model.sample_text(
      'Summarize the following passage in a concise and insightful fashion. It '
      'is OK to omit details that seem less important:\n'
      + 'The year is 1870. The place is London.\n'
      + '\nContext:\n'
      + '\n'.join(shared_memories)
      + '\n'
      + 'Summary:'
  )
  return shared_memories, shared_context


def configure_players(rng: random.Random) -> tuple[
    list[formative_memories.AgentConfig],
    list[formative_memories.AgentConfig],
]:
  """Configure the players.

  Args:
    rng: the random number generator to use.

  Returns:
    main_player_configs: configs for the main characters
    supporting_player_configs: configs for the supporting characters
  """
  joined_main_player_knowledge = [
      ' '.join(get_world_elements(NUM_MAIN_PLAYER_WORLD_ELEMENTS, rng))
      for _ in range(NUM_MAIN_PLAYERS)
  ]
  supporting_player_knowledge = [
      get_world_elements(NUM_SUPPORTING_PLAYER_WORLD_ELEMENTS, rng)
      for _ in range(NUM_SUPPORTING_PLAYERS)
  ]
  joined_supporting_player_knowledge = [
      ' '.join(knowledge) for knowledge in supporting_player_knowledge
  ]
  # These names were generated by Claude 3, prompted to produce names that
  # sound like they could belong to people mixed up in alchemy and opium in
  # London in the year 1870.
  player_configs = [
      # Main characters
      formative_memories.AgentConfig(
          name='Doctor Cornelius Ashmole',
          gender='male',
          date_of_birth=datetime.datetime(year=1820, month=4, day=28),
          goal=(
              'Collect rare books about alchemy, specifically the '
              'tabula smaragdina and secreta secretorum'
          ),
          context=(
              'Born in London, Cornelius aims to heal the sick, become '
              'famous, and collect rare books about alchemy. He is also '
              'aware of the following: '
              f'{joined_main_player_knowledge[0]}.'
          ),
          traits=(
              "Doctor Cornelius Ashmole's personality is like "
              + player_traits_and_styles.get_trait(flowery=True)
          ),
          extras={
              'player_specific_memories': [
                  'Cornelius is wealthy and a member of the upper class.',
                  'Cornelius came to market today to buy alchemical texts.',
              ],
              'main_character': True,
              'initial_endowment': {
                  'coin': 5.0,
                  'laudanum bottle': 2.0,
                  'tabula smaragdina': 0.0,
                  'secreta secretorum': 0.0,
              },
          },
      ),
      formative_memories.AgentConfig(
          name='Madame Esmeralda Dee',
          gender='female',
          goal=(
              'Collect rare books about alchemy, specifically the '
              'tabula smaragdina and secreta secretorum'
          ),
          date_of_birth=datetime.datetime(year=1824, month=9, day=13),
          context=(
              'Born in London, Esmeralda aims to heal the sick, become '
              'famous, and collect rare books about alchemy. She is also '
              'aware of the following: '
              f'{joined_main_player_knowledge[1]}.'
          ),
          traits=(
              "Madame Esmeralda Dee's personality is like "
              + player_traits_and_styles.get_trait(flowery=True)
          ),
          extras={
              'player_specific_memories': [
                  'Esmeralda is wealthy and a member of the upper class.',
                  'Esmeralda came to market today to buy alchemical texts.',
              ],
              'main_character': True,
              'initial_endowment': {
                  'coin': 5.0,
                  'laudanum bottle': 2.0,
                  'tabula smaragdina': 0.0,
                  'secreta secretorum': 0.0,
              },
          },
      ),
      # Supporting characters
      formative_memories.AgentConfig(
          name='Professor Aldous Pendleton',
          gender='male',
          date_of_birth=datetime.datetime(year=1815, month=2, day=11),
          goal='accumulate as much money and fame as possible',
          context=(
              'Born in London, Aldous has fallen on hard times of late '
              'due to his morphinomania. As a result, he must sell some '
              'of his most prized possessions, perhaps even his copy of '
              'the tabula smaragdina. He is also aware of the following '
              f'information: {joined_supporting_player_knowledge[0]}'
          ),
          traits=(
              "Professor Aldous Pendleton's personality is like "
              + player_traits_and_styles.get_trait(flowery=True)
          ),
          extras={
              'player_specific_memories': [
                  (
                      'Aldous has fallen on hard times of late due to his '
                      'morphinomania. As a result, he must sell '
                      'some of his most prized possessions. He '
                      'came to market today to do just that.'
                  ),
                  (
                      'Aldous knows that Molly "Poppy" Jennings owns a genuine '
                      'copy of the secreta secretorum'
                  ),
                  (
                      'Aldous Pendleton owns a genuine copy of the tabula '
                      'smaragdina, it is his most prized possession.'
                  ),
                  (
                      'The tabula smaragdina is a cryptic text attributed to'
                      ' Hermes Trismegistus, is said to hold the key to'
                      ' unlocking the greatest alchemical secrets. However,'
                      ' deciphering its enigmatic symbols can drive the'
                      ' unworthy mad, their minds succumbing to the chaos'
                      ' hidden within its pages.'
                  ),
                  (
                      'The tabula smaragdina is also called the codex of the '
                      'emerald tablet.'
                  ),
                  (
                      'The secreta secretorum is a compendium of letters from '
                      'Aristotle to his student Alexander the Great'
                  ),
                  (
                      'Aldous is willing to sell the tabula smaragdina for'
                      ' three coins or one laudanum bottle.'
                  ),
                  (
                      'Aldous is very agreeable. He will agree to almost any '
                      'proposal.'
                  ),
                  *supporting_player_knowledge[0],
              ],
              'main_character': False,
              'initial_endowment': {
                  'coin': 0.0,
                  'laudanum bottle': 0.0,
                  'tabula smaragdina': 1.0,
                  'secreta secretorum': 0.0,
              },
          },
      ),
      formative_memories.AgentConfig(
          name='Molly "Poppy" Jennings',
          gender='female',
          date_of_birth=datetime.datetime(year=1845, month=5, day=5),
          goal='accumulate as much money and fame as possible',
          context=(
              'Born in London, Molly has fallen on hard times of late '
              'due to her morphinomania. As a result, she must sell some '
              'of her most prized possessions, perhaps even her copy of '
              'the secreta secretorum. She is also aware of the following '
              f'information: {joined_supporting_player_knowledge[1]} '
              'The circumstances in which the secreta secretorum came '
              "into Molly's possession are a secret she guards closely."
          ),
          traits=(
              'Molly "Poppy" Jennings\'s personality is like '
              + player_traits_and_styles.get_trait(flowery=True)
          ),
          extras={
              'player_specific_memories': [
                  (
                      'Molly has fallen on hard times of late due to her '
                      'morphinomania. As a result, she must sell '
                      'some of her most prized possessions. She '
                      'came to market today to do just that.'
                  ),
                  (
                      'Molly knows that Professor Aldous Pendleton owns a'
                      ' genuine copy of the tabula smaragdina'
                  ),
                  (
                      'Molly "Poppy" Jennings owns a genuine copy of the '
                      'secreta secretorum, it is her most prized possession.'
                  ),
                  (
                      'Molly is willing to sell the secreta secretorum for'
                      ' three coins or one laudanum bottle.'
                  ),
                  *supporting_player_knowledge[1],
              ],
              'main_character': False,
              'initial_endowment': {
                  'coin': 0.0,
                  'laudanum bottle': 0.0,
                  'tabula smaragdina': 0.0,
                  'secreta secretorum': 1.0,
              },
          },
      ),
  ]

  main_player_configs = [
      player for player in player_configs if player.extras['main_character']
  ]
  supporting_player_configs = [
      player for player in player_configs if not player.extras['main_character']
  ]

  return main_player_configs, supporting_player_configs


def configure_scenes(
    main_player_configs: Sequence[formative_memories.AgentConfig],
    supporting_player_configs: Sequence[formative_memories.AgentConfig],
) -> Sequence[scene_lib.SceneSpec]:
  """Configure the scene storyboard structure.

  Args:
    main_player_configs: configs for the main characters
    supporting_player_configs: configs for the supporting characters

  Returns:
    scenes: a sequence of scene specifications
  """

  player_configs = list(main_player_configs) + list(supporting_player_configs)

  # Both intros were written by Claude 3, prompted with the world context above.
  day_market_intro = (
      'The air is thick with the pungent aromas of exotic spices, the salty '
      'tang of the Thames, and the acrid smoke billowing from the nearby '
      'factories. The bustling marketplace by the London docks in 1870 is a '
      'cacophony of sounds —- the shouts of hawkers peddling their wares, the '
      "cries of gulls circling overhead, and the creaking of ships' rigging "
      'as they sway in the breeze. Amidst the chaos, one can find all '
      'manner of goods, from the mundane to the mystical: barrels of salted '
      'fish, crates of fragrant teas, and hidden beneath the stalls, the '
      'whispered promises of opium and esoteric knowledge. It is here, in '
      'the shadows of the market, that the seekers of alchemical truths and '
      'spiritual enlightenment gather, their secrets guarded by the '
      'ever-present fog that continually rolls in from the river. '
      '{player_name} just arrived.'
  )
  night_market_intro = (
      'As the clock strikes midnight, the once-bustling marketplace by the '
      'London docks takes on an eerie, otherworldly atmosphere. The '
      'fog, now thick and impenetrable, swirls lazily through the deserted '
      'stalls, muffling the distant sounds of the city and the lapping of the '
      'Thames against the shore. In the flickering light of the gas lamps, the '
      'shadows seem to dance and twist, taking on a life of their own. It '
      "is at this hour that the seekers of opium's secrets emerge from "
      'the darkness, their hushed whispers and furtive glances betraying their '
      'illicit purpose. They move like ghosts through the market, their '
      'footsteps echoing on the cobblestones as they navigate the '
      'labyrinthine alleys and hidden corners, seeking out the opium dens and '
      'secret gatherings where alchemical knowledge and spiritual '
      'enlightenment can be found. In this twilight realm, the line between '
      'reality and dream blurs, and the marketplace becomes a portal to a '
      'world where the impossible seems within reach, and the secrets of the '
      'universe whisper in the smoke-filled air. '
      '{player_name} just arrived.'
  )

  scene_specs = {
      'day': scene_lib.SceneTypeSpec(
          name='day',
          premise={
              cfg.name: [day_market_intro.format(player_name=cfg.name)]
              for cfg in player_configs
          },
      ),
      'night': scene_lib.SceneTypeSpec(
          name='night',
          premise={
              cfg.name: [night_market_intro.format(player_name=cfg.name)]
              for cfg in player_configs
          },
      ),
  }

  scenes = [
      scene_lib.SceneSpec(
          scene_type=scene_specs['day'],
          start_time=START_TIME + 0 * TIME_INCREMENT_BETWEEN_SCENES,
          participant_configs=main_player_configs,
          num_rounds=1,
      ),
      scene_lib.SceneSpec(
          scene_type=scene_specs['night'],
          start_time=START_TIME + 1 * TIME_INCREMENT_BETWEEN_SCENES,
          participant_configs=main_player_configs,
          num_rounds=1,
      ),
      scene_lib.SceneSpec(
          scene_type=scene_specs['day'],
          start_time=START_TIME + 2 * TIME_INCREMENT_BETWEEN_SCENES,
          participant_configs=main_player_configs,
          num_rounds=1,
      ),
      scene_lib.SceneSpec(
          scene_type=scene_specs['night'],
          start_time=START_TIME + 2 * TIME_INCREMENT_BETWEEN_SCENES,
          participant_configs=main_player_configs,
          num_rounds=1,
      ),
  ]
  return scenes


def get_inventories_component(
    model: language_model.LanguageModel,
    memory: associative_memory.AssociativeMemory,
    players: Sequence[entity_agent.EntityAgent],
    main_players: Sequence[entity_agent.EntityAgent],
    player_configs: Sequence[formative_memories.AgentConfig],
    clock_now: Callable[[], datetime.datetime] = datetime.datetime.now,
) -> tuple[component.Component, gm_components.inventory_based_score.Score]:
  """Get the inventory tracking component for the game master."""
  alchemy_texts = (
      'tabula smaragdina',
      'secreta secretorum',
  )
  money_config = ItemTypeConfig(name='coin')
  laudanum_config = ItemTypeConfig(
      name='laudanum bottle', minimum=0, maximum=np.inf
  )
  tabula_smaragdina_config = ItemTypeConfig(
      name=alchemy_texts[0], minimum=0, maximum=1, force_integer=True
  )
  secreta_secretorum_config = ItemTypeConfig(
      name=alchemy_texts[1], minimum=0, maximum=1, force_integer=True
  )
  player_initial_endowments = {
      config.name: config.extras['initial_endowment']
      for config in player_configs
  }
  inventories = gm_contrib.restricted_inventory.RestrictedInventory(
      model=model,
      memory=memory,
      item_type_configs=[
          money_config,
          laudanum_config,
          tabula_smaragdina_config,
          secreta_secretorum_config,
      ],
      players=players,
      player_initial_endowments=player_initial_endowments,
      clock_now=clock_now,
      financial=True,
      name='possessions',
      verbose=True,
  )

  score = gm_components.inventory_based_score.Score(
      inventory=inventories,
      players=main_players,  # Only main players get a score.
      targets={
          'Doctor Cornelius Ashmole': alchemy_texts,
          'Madame Esmeralda Dee': alchemy_texts,
      },
      verbose=True,
  )
  return inventories, score


class Simulation(scenarios_lib.RunnableSimulationWithMemories):
  """Define the simulation API object for the launch script to interact with."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      embedder: Callable[[str], np.ndarray],
      measurements: measurements_lib.Measurements,
      agent_module: types.ModuleType = basic_agent,
      override_agent_model: language_model.LanguageModel | None = None,
      resident_visitor_modules: Sequence[types.ModuleType] | None = None,
      supporting_agent_module: types.ModuleType | None = None,
      time_and_place_module: str | None = None,
      seed: int | None = None,
  ):
    """Initialize the simulation object.

    The launch script assumes this API object has a run() method.

    Args:
      model: the language model to use.
      embedder: the sentence transformer to use.
      measurements: the measurements object to use.
      agent_module: the agent module to use for all main characters.
      override_agent_model: optionally, override the model for all agents. The
        model will be copied for every agent.
      resident_visitor_modules: optionally, use different modules for majority
        and minority parts of the focal population.
      supporting_agent_module: agent module to use for all supporting players. A
        supporting player is a non-player character with a persistent memory
        that plays a specific role in defining the task environment. Their role
        is not incidental but rather is critcal to making the task what it is.
        Supporting players are not necessarily interchangeable with focal or
        background players.
      time_and_place_module: optionally, specify a module containing settings
        that create a sense of setting in a specific time and place. If not
        specified, a random module will be chosen from the default options.
      seed: optionally, specify a seed for the random number generator.
    """
    # Support for these parameters will be added in a future addition coming
    # very imminently.
    del supporting_agent_module
    del time_and_place_module

    self._rng = random.Random(seed)

    if resident_visitor_modules is None:
      self._resident_visitor_mode = False
      self._agent_module = agent_module
    else:
      self._resident_visitor_mode = True
      self._resident_agent_module, self._visitor_agent_module = (
          resident_visitor_modules
      )

    self._agent_model = model

    if override_agent_model:
      self._agent_model = override_agent_model

    self._model = model
    self._embedder = embedder
    self._measurements = measurements

    self._clock = game_clock.MultiIntervalClock(
        start=SETUP_TIME, step_sizes=[MAJOR_TIME_STEP, MINOR_TIME_STEP]
    )

    importance_model = importance_function.AgentImportanceModel(self._model)
    importance_model_gm = importance_function.ConstantImportanceModel()
    self._blank_memory_factory = blank_memories.MemoryFactory(
        model=self._model,
        embedder=self._embedder,
        importance=importance_model.importance,
        clock_now=self._clock.now,
    )
    shared_memories, shared_context = get_shared_memories_and_context(
        model, self._rng
    )
    self._formative_memory_factory = formative_memories.FormativeMemoryFactory(
        model=self._model,
        shared_memories=shared_memories,
        blank_memory_factory_call=self._blank_memory_factory.make_blank_memory,
        current_date=SETUP_TIME,
    )

    main_player_configs, supporting_player_configs = configure_players(
        self._rng
    )
    self._rng.shuffle(main_player_configs)

    supporting_player_names = [cfg.name for cfg in supporting_player_configs]

    tasks = {
        config.name: functools.partial(
            self._make_player_memories, config=config
        )
        for config in main_player_configs + supporting_player_configs
    }
    self._all_memories = concurrency.run_tasks(tasks)

    main_players = []
    self._resident_names = []
    self._visitor_names = []
    for idx, player_config in enumerate(main_player_configs):
      kwargs = dict(
          config=player_config,
          model=copy.copy(self._agent_model),
          memory=self._all_memories[player_config.name],
          clock=self._clock,
          update_time_interval=MAJOR_TIME_STEP,
      )
      if self._resident_visitor_mode:
        if idx == 0:
          player = self._visitor_agent_module.build_agent(**kwargs)  # pylint: disable=attribute-error
          self._visitor_names.append(player.name)
        else:
          player = self._resident_agent_module.build_agent(**kwargs)  # pylint: disable=attribute-error
          self._resident_names.append(player.name)
      else:
        player = self._agent_module.build_agent(**kwargs)  # pylint: disable=attribute-error
        self._resident_names.append(player.name)

      main_players.append(player)

    supporting_players = []
    for player_config in supporting_player_configs:
      conversation_style = agent_components.constant.Constant(
          pre_act_key='guiding principle of good conversation',
          state=player_traits_and_styles.get_conversation_style(
              player_config.name
          ),
      )
      player = basic_agent_supporting.build_agent(
          config=player_config,
          model=self._model,
          memory=self._all_memories[player_config.name],
          clock=self._clock,
          update_time_interval=MAJOR_TIME_STEP,
          additional_components={
              'Guiding principle of good conversation': conversation_style
          },
      )
      supporting_players.append(player)

    self._all_players = main_players + supporting_players

    game_master_memory = associative_memory.AssociativeMemory(
        sentence_embedder=self._embedder,
        importance=importance_model_gm.importance,
        clock=self._clock.now,
    )

    magic_is_not_real = generic_components.constant.ConstantComponent(
        state='Magic is not real. Superatural events are impossible.',
        name='Important Fact',
    )
    only_named_characters_sell_str = (
        'The only people in London with alchemical texts to sell are '
        + ' and '.join(supporting_player_names)
        + '. There are no other '
        + 'venders of alchemical texts.'
    )
    only_named_characters_sell = generic_components.constant.ConstantComponent(
        state=only_named_characters_sell_str, name='Fact'
    )
    categories_and_aliases = generic_components.constant.ConstantComponent(
        state=(
            'The tabula smaragdina and the secreta secretorum are both '
            'alchemical texts. The tabula smaragdina is also called the '
            'codex of the emerald tablet.'
        ),
        name='More facts',
    )
    easy_to_find = generic_components.constant.ConstantComponent(
        state=(
            ' and '.join(supporting_player_names)
            + ' are easy to find '
            'in the marketplace by the docks. Anyone looking for them '
            'will find them there.'
        ),
        name='Another fact',
    )
    inventories, self._score = get_inventories_component(
        model=model,
        memory=game_master_memory,
        players=self._all_players,
        main_players=main_players,
        player_configs=main_player_configs + supporting_player_configs,
        clock_now=self._clock.now,
    )
    additional_gm_components = [
        magic_is_not_real,
        only_named_characters_sell,
        easy_to_find,
        categories_and_aliases,
        inventories,
        self._score,
    ]

    self._primary_environment, self._game_master_memory = (
        basic_game_master.build_game_master(
            model=self._model,
            embedder=self._embedder,
            importance_model=importance_model_gm,
            clock=self._clock,
            players=self._all_players,
            shared_memories=shared_memories,
            shared_context=shared_context,
            blank_memory_factory=self._blank_memory_factory,
            cap_nonplayer_characters_in_conversation=2,
            memory=game_master_memory,
            supporting_players_at_fixed_locations=SUPPORTING_PLAYER_LOCATIONS,
            additional_components=additional_gm_components,
            npc_context=only_named_characters_sell_str,
        )
    )
    self._scenes = configure_scenes(
        main_player_configs=main_player_configs,
        supporting_player_configs=supporting_player_configs,
    )

    self._secondary_environments = []

    self._init_premise_memories(
        setup_time=SETUP_TIME,
        main_player_configs=main_player_configs,
        supporting_player_configs=supporting_player_configs,
        shared_memories=shared_memories,
        scenario_premise=SCENARIO_PREMISE,
    )

  def _make_player_memories(self, config: formative_memories.AgentConfig):
    """Make memories for a player."""
    mem = self._formative_memory_factory.make_memories(config)
    # Inject player-specific memories declared in the agent config.
    for extra_memory in config.extras['player_specific_memories']:
      mem.add(f'{extra_memory}', tags=['initial_player_specific_memory'])
    return mem

  def _init_premise_memories(
      self,
      setup_time: datetime.datetime,
      main_player_configs: list[formative_memories.AgentConfig],
      supporting_player_configs: list[formative_memories.AgentConfig],
      shared_memories: Sequence[str],
      scenario_premise: Sequence[str],
  ) -> None:
    """Initialize player memories.

    Args:
      setup_time: the time to set the clock to before initializing memories
      main_player_configs: configs for the main characters
      supporting_player_configs: configs for the supporting characters
      shared_memories: memories shared by all players, the game master, and NPCs
      scenario_premise: premise observation shared by all players and the game
        master.
    """
    player_configs = main_player_configs + supporting_player_configs
    self._clock.set(setup_time)

    for premise in scenario_premise:
      self._game_master_memory.add(premise)
      for player in self._all_players:
        player.observe(premise)

    for shared_memory in shared_memories:
      self._game_master_memory.add(shared_memory)
      for player in self._all_players:
        player.observe(shared_memory)

    # The game master also observes all the player-specific memories.
    for player_config in player_configs:
      extra_memories = player_config.extras['player_specific_memories']
      for extra_memory in extra_memories:
        self._game_master_memory.add(extra_memory)

  def get_all_player_memories(self):
    return self._all_memories

  def __call__(self) -> tuple[logging_lib.SimulationOutcome, str]:
    """Run the simulation.

    Returns:
      html_results_log: browseable log of the simulation in HTML format
    """
    html_results_log = basic_game_master.run_simulation(
        model=self._model,
        players=self._all_players,
        primary_environment=self._primary_environment,
        secondary_environments=self._secondary_environments,
        clock=self._clock,
        scenes=self._scenes,
    )

    player_scores = self._score.get_scores()
    simulation_outcome = logging_lib.SimulationOutcome(
        resident_scores=immutabledict.immutabledict(
            {name: player_scores[name] for name in self._resident_names}
        ),
        visitor_scores=immutabledict.immutabledict(
            {name: player_scores[name] for name in self._visitor_names}
        ),
        metadata=immutabledict.immutabledict({
            'wallclock_time': datetime.datetime.now().strftime(
                '%Y-%m-%d %H:%M:%S'
            ),
            'environment': __file__,
        }),
    )
    print('Overall scores per player:')
    if self._resident_visitor_mode:
      idx = 0
      for player_name, score in player_scores.items():
        if idx == 0:
          print('Visitor')
        else:
          print('Resident')
        print(f'  {player_name}: {score}')
        idx += 1
    else:
      for player_name, score in player_scores.items():
        print(f'{player_name}: {score}')

    return simulation_outcome, html_results_log
