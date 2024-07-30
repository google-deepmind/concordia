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

from collections.abc import Callable, Mapping, Sequence
import datetime
import types

from concordia.associative_memory import blank_memories
from concordia.associative_memory import formative_memories
from concordia.associative_memory import importance_function
from concordia.clocks import game_clock
from concordia.components import game_master as gm_components
from concordia.components.agent import v2 as agent_components
from concordia.environment import game_master
from examples.modular.environment.modules import player_traits_and_styles
from concordia.factory.agent import basic_entity_agent__main_role
from concordia.factory.agent import basic_entity_agent__supporting_role
from concordia.factory.environment import basic_game_master
from concordia.language_model import language_model
from concordia.typing import agent as agent_lib
from concordia.typing import scene as scene_lib
from concordia.utils import concurrency
from concordia.utils import measurements as measurements_lib
import numpy as np
import sentence_transformers

Runnable = Callable[[], str]

MAJOR_TIME_STEP = datetime.timedelta(minutes=20)
MINOR_TIME_STEP = datetime.timedelta(seconds=10)

SETUP_TIME = datetime.datetime(hour=20, year=1750, month=10, day=1)
START_TIME = datetime.datetime(hour=18, year=1750, month=10, day=2)
HILL_TIME = datetime.datetime(hour=18, year=1750, month=10, day=3)
RETURN_HOME_SCENE_TIME = datetime.datetime(hour=20, year=1750, month=10, day=4)
DECISION_TIME = datetime.datetime(hour=20, year=1750, month=12, day=6)

RESOLUTION_SCENE_TYPE = 'decision'
SCENARIO_PREMISE = [
    (
        'Alice and Charlie meet periodically at the hill of accord to '
        'discuss current events and conduct diplomacy on behalf of the '
        'villages they represent.'
    ),
]
REPRESENTATIVE_BY_VILLAGE = {
    'Auroria': 'Alice',
    'Havenwood': 'Charlie',
}
SUPPORTING_PLAYER_LOCATIONS = [
    'Diana waits for Alice in the center of Auroria.',
    'George waits for Charlie in the center of Havenwood.',
]
DECISION_ACTION_SPEC = agent_lib.choice_action_spec(
    call_to_action=(
        'Would {name} follow through with their obligation under '
        'the agreement?'),
    options=('no', 'yes'),
    tag='decision',
)


def get_shared_memories_and_context(
    model: language_model.LanguageModel) -> tuple[Sequence[str], str]:
  """Return the shared memories and context for all agents and game master."""
  shared_memories = [
      'There are two villages: Auroria and Havenwood.',
      ('Elder representatives of the two villages meet one another at the ' +
       'hill of accord to discuss current events and conduct diplomacy.'),
      'Alice represents Auroria.',
      'Charlie represents Havenwood.',
      ('Havenwood, which is on the coast, is threatened by barbarian ' +
       'raiders who have been attacking from the sea more often lately.'),
      'Auroria is up in the mountains, far from the sea.',
      'Havenwood is on the coast.',
      ('Everyone knows that the gods smile upon any treaty for which ' +
       'agreement is marked by the pouring of libations upon the hill of ' +
       'accord. This ritual involves pouring precious wines upon the hill ' +
       "of accord's sacred ground."),
      ('To secure divine favor with the libation pouring ritual, it is ' +
       'first necessary for all parties to the treaty under consideration to ' +
       'have reached agreement on its exact wording, which must include ' +
       'who is promising to do what, under what conditions, and ' +
       'whether as a result of the treaty, any resources will '+
       'change hands, which resources, and when.'),
  ]
  # The shared context will be used for the NPC context. It reflects general
  # knowledge and is possessed by all characters.
  shared_context = model.sample_text(
      'Summarize the following passage in a concise and insightful fashion:\n'
      + '\n'.join(shared_memories)
      + '\n'
      + 'Summary:'
  )
  return shared_memories, shared_context


village_facts = {}
village_facts['auroria'] = [
    (
        'In Auroria, leadership is earned through skill and knowledge, without'
        ' regard to familial lineage.'
    ),
    (
        'The people of Auroria worship multiple gods, each representing'
        ' different natural elements, which are central to their celebrations'
        ' and daily lives.'
    ),
    (
        'Auroria follows a matrilineal lineage system, where family and'
        ' inheritance are traced through the mother\'s side, shaping social and'
        ' family structures.'
    ),
    (
        'Aurorian art is known for its geometric patterns and symbols,'
        ' reflecting the community\'s connection with nature.'
    ),
    (
        'Auroria has a gift-giving economy, where status is gained through'
        ' generosity and the ability to host elaborate feasts.'
    ),
    (
        'The political structure in Auroria is upheld by a council of elders,'
        ' who make decisions through consensus, embodying the village\'s'
        ' democratic values.'
    ),
    (
        'Auroria has a specialization of labor, with skilled craftspeople such'
        ' as potters, blacksmiths, and weavers. Their goods are traded within'
        ' the village and with neighboring communities.'
    ),
    (
        'Artistic expression is highly valued in Auroria. This could manifest'
        ' in decorated pottery, intricate carvings, or textiles with symbolic'
        ' designs.'
    ),
    (
        'The solstice festival, Lumina, is a significant event in Auroria,'
        ' featuring communal feasting, dancing, and bonfires to celebrate'
        ' seasonal cycles and community prosperity.'
    ),
    (
        'The elite of Auroria distinguish themselves through their stewardship'
        ' of knowledge and culture, hosting gatherings where wisdom, poetry,'
        ' and ancestral stories are shared.'
    ),
    (
        'The elite in Auroria may be expected to perform conspicuous displays'
        ' of generosity, such as sponsoring large feasts or communal projects.'
        ' This reinforces their social standing and positions them as'
        ' benefactors of the community.'
    ),
    (
        'In Auroria, the homes of the elite are recognized not by their size'
        ' but by the presence of rare artifacts and artworks that symbolize'
        ' their role as custodians of the community\'s heritage.'
    ),
    (
        'The elite in Auroria play a crucial role in the village\'s external'
        ' relations, representing their people in negotiations with neighboring'
        ' communities to ensure peace and mutual prosperity.'
    ),
    (
        'Rituals and ceremonies in Auroria often see the elite taking on'
        ' significant roles, wearing ceremonial attire that features intricate'
        ' designs symbolizing their responsibilities to the community and the'
        ' gods they venerate.'
    ),
    (
        'Many members of the elite in Auroria are very xenophobic, moreso than'
        ' the non-elite villagers.'
    ),
]

village_facts['havenwood'] = [
    (
        'Legend says Havenwood\'s founders are survivors of a shipwreck. They'
        ' found shelter in a hidden cove and built a new life, vigilant against'
        ' the dangers of the sea.'
    ),
    (
        'A tall, weathered tower built on the highest cliff overlooking the sea'
        ' serves as Havenwood\'s primary watchpoint. A signal fire is'
        ' ever-ready to warn of approaching raiders.'
    ),
    (
        'Havenwood is partially encircled by a sturdy wooden palisade,'
        ' reinforced with driftwood and scavenged debris from wrecked ships, a'
        ' testament to past raids.'
    ),
    (
        'A network of sea caves carved into the cliffs below Havenwood provides'
        ' hiding places for people and supplies when the barbarians attack.'
    ),
    (
        'The villagers of Havenwood repurposed their large fishing nets,'
        ' rigging them to ensnare smaller raiding boats attempting to land on'
        ' their shores.'
    ),
    (
        'Every Havenwood villager possesses both fishing and basic combat'
        ' skills. Survival depends on their ability to defend themselves and'
        ' reap the bounty of the sea.'
    ),
    (
        'Havenwood\'s diet relies heavily on fish, shellfish, and edible'
        ' seaweed. Their skills in navigating the coastal waters and tides are'
        ' unmatched.)'
    ),
    (
        'The barbarian raids have severely disrupted Havenwood\'s once-active'
        ' trade with inland settlements, limiting supplies and the exchange of'
        ' goods.'
    ),
    (
        'After raids, Havenwood villagers comb beaches and cliffsides for'
        ' usable items left behind by the barbarians, some of which find their'
        ' way into homes and tools.'
    ),
    (
        'Children in Havenwood learn cautionary songs and stories about the'
        ' raiders, weaving a mixture of fear and defiance into the tradition'
        ' and folklore of Havenwood.'
    ),
    (
        'Havenwood lacks a dedicated blacksmith, so villagers craft makeshift'
        ' weapons from fishing tools, boat parts, and sharpened flotsam.'
    ),
    (
        'Sacred relics and charms which protect Havenwood from harm are kept by'
        ' the elders, used in rituals to seek the favor of ancestral spirits.'
    ),
    (
        'The elite of Havenwood lead by example, joining the front lines in'
        " defense and sharing their own resources to strengthen the village's"
        ' protection.'
    ),
    (
        'The villagers of Havenwood engage in regular evacuation drills,'
        ' ensuring that everyone, including children, knows where to go and'
        ' what to do in the event of a barbarian attack.'
    ),
    (
        'Havenwood\'s most respected families hold knowledge of the tides and'
        ' secret paths through the coastal waters, guiding allies and evading'
        ' foes.'
    ),
]


def get_village_factoids(village: str, size: int) -> list[str]:
  """Get a random selection of village factoids from a preset list."""
  return np.random.choice(village_facts[village], size=size, replace=False)


def configure_players(
    shared_context: str) -> tuple[list[formative_memories.AgentConfig],
                                  list[formative_memories.AgentConfig],
                                  dict[str, formative_memories.AgentConfig]]:
  """Configure the players.

  Args:
    shared_context: context known to all players and the game master.
  Returns:
    main_player_configs: configs for the main characters
    supporting_player_configs: configs for the supporting characters
    player_configs_dict: dict mapping player name to corresponding config
  """
  auroria_factoids = get_village_factoids('auroria', 5)
  havenwood_factoids = get_village_factoids('havenwood', 5)

  player_configs = [
      # Main characters:
      formative_memories.AgentConfig(
          name='Alice',
          gender='female',
          date_of_birth=datetime.datetime(year=1700, month=9, day=13),
          goal=('Alice wants to do what is best for Auroria, especially when ' +
                'it is also best for Alice herself.'),
          context=shared_context,
          traits=(f'Personality: {player_traits_and_styles.get_trait()} and '
                  f'{player_traits_and_styles.get_trait()}'),
          extras={
              'player_specific_memories': [
                  'Alice is an elder of Auroria.',
                  'Alice represents Auroria at the meeting.',
                  'Auroria has 100 warriors.',
                  'Auroria is a prosperous place.',
                  *auroria_factoids,
              ],
              'home_village': 'Auroria',
              'main_character': True,
          },
      ),
      formative_memories.AgentConfig(
          name='Charlie',
          gender='male',
          date_of_birth=datetime.datetime(year=1700, month=2, day=12),
          goal=(
              'Charlie wants to do what is best for Havenwood, especially ' +
              'when it is also best for Charlie himself.'),
          context=shared_context,
          traits=(f'Personality: {player_traits_and_styles.get_trait()} and '
                  f'{player_traits_and_styles.get_trait()}'),
          extras={
              'player_specific_memories': [
                  'Charlie is an elder of Havenwood.',
                  'Charlie represents Havenwood at the meeting.',
                  'Havenwood has 5 warriors.',
                  ('The barbarians are destroying so much of Havenwood\'s ' +
                   'nearby farmland that the people may soon starve if ' +
                   'the status quo persists much longer.'),
                  'Havenwood\'s granaries are almost empty.',
                  *havenwood_factoids,
              ],
              'home_village': 'Havenwood',
              'main_character': True,
          },
      ),

      # Supporting characters
      formative_memories.AgentConfig(
          name='Diana',
          gender='female',
          date_of_birth=datetime.datetime(year=1725, month=4, day=28),
          goal=(
              'Diana manages the Auroria granary, she wants to do a good job ' +
              'and become a respected leader herself in the future.'),
          context=shared_context,
          traits=(f'Personality: {player_traits_and_styles.get_trait()} and '
                  f'{player_traits_and_styles.get_trait()}'),
          extras={
              'player_specific_memories': [
                  'Diana is from Auroria.',
                  'Diana manages the Auroria granary.',
                  ('Diana knows that Alice will represent Auroria in the ' +
                   'meeting at the hill of accord.'),
                  'Auroria is a prosperous place.',
                  ('Diana believes that Auroria is superior to Havenwood and '
                   'that Havenwood should be forced to pay tribute and '
                   'acknowledge Auroria\'s higher status.'),
                  *auroria_factoids,
              ],
              'home_village': 'Auroria',
              'main_character': False,
          },
      ),
      formative_memories.AgentConfig(
          name='George',
          gender='male',
          date_of_birth=datetime.datetime(year=1725, month=2, day=3),
          goal=(
              'George wants to be able to feed his family, and he wants ' +
              'compensation for his farm (which the barbarians destroyed), ' +
              'and he wants revenge, and security in the future.'
          ),
          context=shared_context,
          traits=(f'Personality: {player_traits_and_styles.get_trait()} and '
                  f'{player_traits_and_styles.get_trait()}'),
          extras={
              'player_specific_memories': [
                  'George is from Havenwood.',
                  ('George was a farmer, and could be one again if it were ' +
                   'safe to return to his farm and he had help to rebuild.'),
                  ('George\'s farm near the sea was razed to the ground by ' +
                   'barbarians. He lost everything.'),
                  ('George knows that Charlie will represent Havenwood in ' +
                   'the meeting at the hill of accord.'),
                  'George was driven from his land by marauding barbarians.',
                  'George is in danger of starvation.',
                  ('George and his family are camped out in the center of ' +
                   'Havenwood, having lost the farm to barbarians. The ' +
                   'village was the only safe place to go.'),
                  ("The barbarians are destroying a lot Havenwood's " +
                   'nearby farmland.'),
                  *havenwood_factoids,
              ],
              'home_village': 'Havenwood',
              'main_character': False,
          },
      ),
  ]

  main_player_configs = [
      player for player in player_configs if player.extras['main_character']]
  supporting_player_configs = [
      player for player in player_configs
      if not player.extras['main_character']]

  player_configs_dict = {
      player.name: player for player in player_configs}

  return main_player_configs, supporting_player_configs, player_configs_dict


def configure_scenes(
    main_player_configs: Sequence[formative_memories.AgentConfig],
    supporting_player_configs: Sequence[formative_memories.AgentConfig],
    player_configs_dict: dict[str, formative_memories.AgentConfig],
    decision_env: game_master.GameMaster,
) -> Sequence[scene_lib.SceneSpec]:
  """Configure the scene storyboard structure.

  Args:
    main_player_configs: configs for the main characters
    supporting_player_configs: configs for the supporting characters
    player_configs_dict: dict mapping player name to corresponding config
    decision_env: the decision environment to use
  Returns:
    scenes: the scenes to use
  """
  year_increment = datetime.timedelta(days=365)

  home_phase_premise = (
      'Elder {player_name} is home in {village_name}. They know that it will '
      'be critical to gain the support of influential stakeholders in the '
      'village if any agreement is to last. They should start now. There is '
      'no time to rest. Everyone in {village_name} knows {player_name} has '
      'been negotiating on their behalf at the hill of accord. Many will want '
      'to seek them out to convey their views, hopes, fears, and plans, and to '
      'try to influence {player_name} to align with them and their incentives.')

  supporting_character_home_phase_premise = (
      '{player_name} is currently in {village_name} and has no intention of '
      'leaving today.'
  )

  negotiation_phase_premise = (
      'Elder {player_name} left {village_name} early in the morning and '
      'arrived just now at the hill of accord. The reason for this meeting '
      'of the two elder representatives of their respective villages '
      '(Alice representing Auroria and Charlie representing Havenwood) is as '
      'follows: barbarian raiders have been pillaging and burning the land, '
      'and menacing both villages. It has been suggested that an alliance for '
      'the mutual defense of both villages against the barbarian threat would '
      'be beneficial. The elders are meeting to discuss this possibility.')

  scene_specs = {
      'home': scene_lib.SceneTypeSpec(
          name='home',
          premise={
              'Alice': [
                  home_phase_premise.format(
                      player_name=player_configs_dict['Alice'].name,
                      village_name=player_configs_dict['Alice'].extras[
                          'home_village'])
              ],
              'Charlie': [
                  home_phase_premise.format(
                      player_name=player_configs_dict['Charlie'].name,
                      village_name=player_configs_dict['Charlie'].extras[
                          'home_village'])
              ],
              'Diana': [
                  supporting_character_home_phase_premise.format(
                      player_name=player_configs_dict['Diana'].name,
                      village_name=player_configs_dict['Diana'].extras[
                          'home_village'])
              ],
              'George': [
                  supporting_character_home_phase_premise.format(
                      player_name=player_configs_dict['George'].name,
                      village_name=player_configs_dict['George'].extras[
                          'home_village'])
              ],
          },
      ),
      'negotiation': scene_lib.SceneTypeSpec(
          name='negotiation',
          premise={
              'Alice': [
                  negotiation_phase_premise.format(
                      player_name=player_configs_dict['Alice'].name,
                      village_name=player_configs_dict['Alice'].extras[
                          'home_village'])
              ],
              'Charlie': [
                  negotiation_phase_premise.format(
                      player_name=player_configs_dict['Charlie'].name,
                      village_name=player_configs_dict['Charlie'].extras[
                          'home_village'])
              ],
          },
      ),
      RESOLUTION_SCENE_TYPE: scene_lib.SceneTypeSpec(
          name=RESOLUTION_SCENE_TYPE,
          premise={},
          action_spec=DECISION_ACTION_SPEC,
          override_game_master=decision_env,
      ),
  }

  scenes = [
      # Year 1
      scene_lib.SceneSpec(
          scene_type=scene_specs['home'],
          start_time=START_TIME,
          participant_configs=main_player_configs,
          num_rounds=1,
      ),
      scene_lib.SceneSpec(
          scene_type=scene_specs['negotiation'],
          start_time=HILL_TIME,
          participant_configs=main_player_configs,
          num_rounds=1,
      ),
      scene_lib.SceneSpec(
          scene_type=scene_specs['home'],
          start_time=RETURN_HOME_SCENE_TIME,
          participant_configs=main_player_configs,
          num_rounds=1,
      ),
      scene_lib.SceneSpec(
          scene_type=scene_specs[RESOLUTION_SCENE_TYPE],
          start_time=DECISION_TIME,
          participant_configs=supporting_player_configs,
          num_rounds=1,
      ),

      # Year 2
      scene_lib.SceneSpec(
          scene_type=scene_specs['home'],
          start_time=START_TIME + year_increment,
          participant_configs=main_player_configs,
          num_rounds=2,
      ),
      scene_lib.SceneSpec(
          scene_type=scene_specs['negotiation'],
          start_time=HILL_TIME + year_increment,
          participant_configs=main_player_configs,
          num_rounds=2,
      ),
      scene_lib.SceneSpec(
          scene_type=scene_specs['home'],
          start_time=RETURN_HOME_SCENE_TIME + year_increment,
          participant_configs=main_player_configs,
          num_rounds=2,
      ),
      scene_lib.SceneSpec(
          scene_type=scene_specs[RESOLUTION_SCENE_TYPE],
          start_time=DECISION_TIME + year_increment,
          participant_configs=supporting_player_configs,
          num_rounds=1,
      ),

      # Year 3
      scene_lib.SceneSpec(
          scene_type=scene_specs['home'],
          start_time=START_TIME + (2 * year_increment),
          participant_configs=main_player_configs,
          num_rounds=1,
      ),
  ]
  return scenes


def outcome_summary_fn(
    binary_joint_action: Mapping[str, int],
    unused_rewards: Mapping[str, float]
) -> Mapping[str, str]:
  """Summarize outcome of decision scene (used by Schelling payoffs component).

  Args:
    binary_joint_action: map each player name to whether they cooperated or
      defected (0 indicates defection and 1 indicates cooperation).
    unused_rewards: map each player name to the reward they received
  Returns:
    result: dict mapping player name to outcome summary
  """
  result = {name: '' for name in binary_joint_action}
  num_cooperators = np.sum(list(binary_joint_action.values()))
  success = num_cooperators > 2
  common_part = ''
  if success:
    common_part += 'The barbarian invasion was successfully repulsed. '
  else:
    common_part += (
        'The barbarian invasion was not stopped. Barbarians ' +
        'overrun the region, taking whatever they please. After a season of ' +
        'terror they finally leave the region, not because they were driven ' +
        'out, but because precious little worth plundering remained. ')
  for player_name, action in binary_joint_action.items():
    result[player_name] += common_part
    # action == 1 indicates cooperation while action == 0 indicates defection
    if success and action == 1:
      result[player_name] += (f'{player_name} did their duty and helped ' +
                              'achieve this great victory.')
    elif success and action == 0:
      result[player_name] += (f'{player_name} chose not to do their duty, ' +
                              'but victory was obtained nonetheless.')
    elif not success and action == 1:
      result[player_name] += (f'{player_name} did their duty. However, too ' +
                              'few others joined. The wanton cruelty of the ' +
                              'barbarians caused much suffering throughout ' +
                              'the region.')
    elif not success and action == 0:
      result[player_name] += (f'{player_name} did not do their duty. ' +
                              'The wanton cruelty of the barbarians caused ' +
                              'much suffering throughout the region.')
  return result


class Simulation(Runnable):
  """Define the simulation API object for the launch script to interact with."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      embedder: sentence_transformers.SentenceTransformer,
      measurements: measurements_lib.Measurements,
      agent_module: types.ModuleType = basic_entity_agent__main_role,
  ):
    """Initialize the simulation object.

    Args:
      model: the language model to use.
      embedder: the sentence transformer to use.
      measurements: the measurements object to use.
      agent_module: the agent module to use for all main characters.
    """
    self._agent_module = agent_module
    self._model = model
    self._embedder = embedder
    self._measurements = measurements

    self._clock = game_clock.MultiIntervalClock(
        start=SETUP_TIME,
        step_sizes=[MAJOR_TIME_STEP, MINOR_TIME_STEP])

    importance_model = importance_function.AgentImportanceModel(self._model)
    importance_model_gm = importance_function.ConstantImportanceModel()
    self._blank_memory_factory = blank_memories.MemoryFactory(
        model=self._model,
        embedder=self._embedder,
        importance=importance_model.importance,
        clock_now=self._clock.now,
    )
    shared_memories, shared_context = get_shared_memories_and_context(model)
    self._formative_memory_factory = formative_memories.FormativeMemoryFactory(
        model=self._model,
        shared_memories=shared_memories,
        blank_memory_factory_call=self._blank_memory_factory.make_blank_memory,
    )

    main_player_configs, supporting_player_configs, player_configs_dict = (
        configure_players(shared_context=shared_context)
    )

    num_main_players = len(main_player_configs)
    num_supporting_players = len(supporting_player_configs)

    self._all_memories = {}

    main_player_memory_futures = []
    with concurrency.executor(max_workers=num_main_players) as pool:
      for player_config in main_player_configs:
        future = pool.submit(self._make_player_memories,
                             config=player_config)
        main_player_memory_futures.append(future)
      for player_config, future in zip(main_player_configs,
                                       main_player_memory_futures):
        self._all_memories[player_config.name] = future.result()

    if num_supporting_players > 0:
      supporting_player_memory_futures = []
      with concurrency.executor(max_workers=num_supporting_players) as pool:
        for player_config in supporting_player_configs:
          future = pool.submit(self._make_player_memories,
                               config=player_config)
          supporting_player_memory_futures.append(future)
        for player_config, future in zip(supporting_player_configs,
                                         supporting_player_memory_futures):
          self._all_memories[player_config.name] = future.result()

    main_players = []
    for player_config in main_player_configs:
      player = self._agent_module.build_agent(
          config=player_config,
          model=self._model,
          memory=self._all_memories[player_config.name],
          clock=self._clock,
          update_time_interval=MAJOR_TIME_STEP,
      )
      main_players.append(player)

    supporter_extra_components = {}
    for player_config in supporting_player_configs:
      village_name = player_config.extras['home_village']
      representative = REPRESENTATIVE_BY_VILLAGE[village_name]
      supporting_character_plan = agent_components.constant.Constant(
          pre_act_key='plan',
          state=(
              f'{player_config.name}\'s plan is to find {representative} to '
              'discuss weighty matters.'
          ))
      conversation_style = agent_components.constant.Constant(
          pre_act_key='guiding principle of good conversation',
          state=player_traits_and_styles.get_conversation_style(
              player_config.name
          ),
      )
      supporter_extra_components[player_config.name] = {
          'Plan': supporting_character_plan,
          'Guiding principle of good conversation': conversation_style,
      }

    supporting_players = []
    for player_config in supporting_player_configs:
      player = basic_entity_agent__supporting_role.build_agent(
          config=player_config,
          model=self._model,
          memory=self._all_memories[player_config.name],
          clock=self._clock,
          update_time_interval=MAJOR_TIME_STEP,
          additional_components=supporter_extra_components[player_config.name],
      )
      supporting_players.append(player)

    self._all_players = main_players + supporting_players

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
            supporting_players_at_fixed_locations=SUPPORTING_PLAYER_LOCATIONS,
        )
    )
    payoffs = gm_components.schelling_diagram_payoffs.SchellingPayoffs(
        model=self._model,
        memory=self._game_master_memory,
        cooperative_option='yes',
        resolution_scene=RESOLUTION_SCENE_TYPE,
        cooperator_reward_fn=lambda x: x,
        defector_reward_fn=lambda x: x + 1.0,
        players=self._all_players,
        acting_player_names=[cfg.name for cfg in supporting_player_configs],
        outcome_summarization_fn=outcome_summary_fn,
        clock_now=self._clock.now,
        name='scoring function',
    )
    decision_env = basic_game_master.build_decision_scene_game_master(
        model=self._model,
        memory=self._game_master_memory,
        clock=self._clock,
        players=self._all_players,
        decision_action_spec=DECISION_ACTION_SPEC,
        payoffs=payoffs,
    )
    self._scenes = configure_scenes(
        main_player_configs=main_player_configs,
        supporting_player_configs=supporting_player_configs,
        player_configs_dict=player_configs_dict,
        decision_env=decision_env,
    )
    self._secondary_environments = [decision_env]

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
    main_players = [
        player for player in self._all_players if player.name in [
            player_config.name for player_config in main_player_configs]]
    supporting_players = [
        player for player in self._all_players if player.name in [
            player_config.name for player_config in supporting_player_configs]]

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

    # Generate memory of each elder being at home in their home village.
    for idx, player in enumerate(main_players):
      village = player_configs[idx].extras['home_village']
      scene_premise = (
          f'Elder {player.name} is home in {village}. It is one day before '
          'they are to depart their village to travel to the hill of accord to '
          f'meet the representative of the other village.')
      # Add memory to both player and GM.
      player.observe(scene_premise)
      self._game_master_memory.add(scene_premise)

    for idx, player in enumerate(supporting_players):
      village = player_configs[idx].extras['home_village']
      teleport = (f'{player.name} is currently in {village} and has no ' +
                  'intention of leaving today.')
      player.observe(teleport)
      self._game_master_memory.add(teleport)

  def __call__(self) -> str:
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
    return html_results_log
