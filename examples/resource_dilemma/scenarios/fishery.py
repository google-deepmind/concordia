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

"""Fishery CPR simulation setup supporting both standard and election modes.

Provides unified configuration and runner. Mode is selectable:
  - mode="standard": harvesting → discussion cycle order
  - mode="election": policy generation → election → harvesting → discussion
"""

from concordia.associative_memory import basic_associative_memory
from concordia.environment.engines import simultaneous
from examples.resource_dilemma import resource_logger
from examples.resource_dilemma import simulation_state as sim_state_lib
from examples.resource_dilemma.gamemaster import discussion_game_master
from examples.resource_dilemma.gamemaster import harvesting_game_master
from examples.resource_dilemma.gamemaster import voting_game_master
from examples.resource_dilemma.personas import fishery_personas
from concordia.prefabs import entity as entity_prefabs
from concordia.prefabs import game_master as game_master_prefabs
from concordia.prefabs.simulation import generic as simulation
from concordia.typing import prefab as prefab_lib
from concordia.utils import helper_functions

DEFAULT_NUM_CYCLES = 6
DEFAULT_CAPACITY = 100
DEFAULT_ELECTION_EVERY_N = 1


def get_fishery_rules(
    num_cycles: int = DEFAULT_NUM_CYCLES, capacity: int = DEFAULT_CAPACITY
) -> str:
  """Return the premise for the standard fishery CPR simulation."""
  return f"""\
  This is a simulation of a common pool resource problem set in a small \
  coastal fishing village. A group of fishermen share a single lake that \
  contains a population of fish. The current fish stock is {capacity} tonnes \
  and the lake has a carrying capacity of 120 tonnes.

  The simulation runs for {num_cycles} cycles. Each cycle has TWO PHASES \
  that must be executed IN ORDER:

  ============================================================
  PHASE 1: FISHING
  ============================================================
  1. The Game Master announces the current fish stock level.
  2. Each fisher independently decides how many tonnes of fish to catch \
  (between 0 and 20 tonnes).
  3. After all fishermen have made their decisions, the total catch is removed \
  from the lake.
  4. The remaining fish population DOUBLES at the end of each cycle, up to the \
  carrying capacity of 120 tonnes.
  5. If the total catch exceeds the available stock, each fisher's catch is \
  proportionally reduced so the total does not exceed the available stock.
  6. If the fish stock reaches zero, the fishery collapses permanently.

  ============================================================
  PHASE 2: DISCUSSION (Town Hall)
  ============================================================
  All agents discuss the events of the past round for at most 10 responses. \
  This phase proceeds sequentially — the Game Master picks who speaks next.

  ============================================================
  HARVEST REPORTING
  ============================================================
  After EACH fishing cycle, the Game Master MUST report:

  HARVEST REPORT — Cycle N:
    Fish stock at start of cycle: [X] tonnes
    [Fisher name]: caught [Y] tonnes (cumulative: [Z] tonnes)
    ...
    Total catch this cycle: [T] tonnes
    Fish stock after catch: [R] tonnes
    Regeneration: [G] tonnes (doubling up to capacity)
    Fish stock at end of cycle: [E] tonnes
"""


def get_fishery_election_rules(
    num_cycles: int = DEFAULT_NUM_CYCLES,
    capacity: int = DEFAULT_CAPACITY,
    election_every_n: int = DEFAULT_ELECTION_EVERY_N,
    leader_names: list[str] | None = None,
    voter_names: list[str] | None = None,
) -> str:
  """Return the premise for the governed fishery election simulation."""
  leader_list = ', '.join(leader_names or [])
  voter_list = ', '.join(voter_names or [])

  election_frequency_text = (
      'Elections are held EVERY cycle.'
      if election_every_n == 1
      else f'Elections are held every {election_every_n} cycles.'
  )
  if election_every_n == 0:
    election_frequency_text = 'No elections are held in this simulation.'

  return f"""\
  This is a simulation of a GOVERNED common pool resource problem set in a \
  small coastal fishing village. A group of fishermen share a single lake that \
  contains a population of fish. The current fish stock is {capacity} tonnes \
  and the lake has a carrying capacity of 120 tonnes.

  The simulation runs for {num_cycles} cycles. {election_frequency_text}

  PARTICIPANTS:
  - LEADER CANDIDATES (who also fish): {leader_list}
  - VOTERS (regular fishers): {voter_list}
  All participants fish each cycle. Leaders additionally propose policies \
  and stand for election.

  Each cycle follows these 4 phases:

  ============================================================
  PHASE 1: POLICY GENERATION (Leaders)
  ============================================================
  Each leader candidate proposes a policy agenda for governing the fishery. \
  The policy should include:
  a) A recommended catch limit (max tonnes per fisher per cycle)
  b) A penalty schedule for violators
  c) A brief rationale explaining why this policy is best

  Leaders propose their policies CONCURRENTLY.

  ============================================================
  PHASE 2: ELECTION (Voters)
  ============================================================
  After all policies are announced, each VOTER reviews \
  all proposed agendas and casts a single vote for one leader candidate. \
  Voters vote CONCURRENTLY.
  The leader with the most votes wins (simple plurality). The winning \
  leader's policy becomes the ACTIVE RULE SET for the fishery starting \
  this cycle.

  ============================================================
  PHASE 3: HARVEST / FISHING (All)
  ============================================================
  1. The Game Master announces the current fish stock level and the ACTIVE \
  POLICY.
  2. Each participant independently decides how many \
  tonnes of fish to catch (between 0 and 20 tonnes).
  3. After all fishermen have made their decisions, the total catch is removed \
  from the lake.
  4. The remaining fish population DOUBLES at the end of each cycle, up to the \
  carrying capacity of 120 tonnes.
  5. If the total catch exceeds the available stock, each fisher's catch is \
  proportionally reduced so the total does not exceed the available stock.
  6. If the fish stock reaches zero, the fishery collapses permanently.

  ============================================================
  PHASE 4: DISCUSSION (All)
  ============================================================
  All agents discuss the events of the past round for at most 10 responses. \
  This phase proceeds sequentially.

  ============================================================
  HARVEST REPORTING
  ============================================================
  After EACH fishing cycle, the Game Master MUST report:

  HARVEST REPORT — Cycle N:
    Fish stock at start of cycle: [X] tonnes
    [Fisher name]: caught [Y] tonnes (cumulative: [Z] tonnes)
    ...
    Total catch this cycle: [T] tonnes
    Fish stock after catch: [R] tonnes
    Regeneration: [G] tonnes (doubling up to capacity)
    Fish stock at end of cycle: [E] tonnes
"""


SHARED_MEMORIES = [
    (
        'This is a small coastal fishing village where everyone knows each'
        ' other. The fishermen share a single lake.'
    ),
    (
        'The lake currently has 100 tonnes of fish and the carrying capacity'
        ' is 120 tonnes.'
    ),
    (
        'Each fishing cycle, every fisher decides how many tonnes of fish to'
        ' catch, between 0 and 20 tonnes.'
    ),
    (
        'After everyone fishes, the remaining fish population doubles,'
        ' up to the carrying capacity.'
    ),
    (
        'If the total catch exceeds the available stock, catches are'
        ' proportionally reduced.'
    ),
    'If the fish stock reaches zero, the fishery collapses permanently.',
    (
        f'The simulation runs for {DEFAULT_NUM_CYCLES} fishing cycles. Each'
        " fisher's goal is to maximise their total catch while keeping the"
        ' fishery alive.'
    ),
    (
        'After each fishing cycle, the fishermen gather for a community'
        ' meeting where they discuss events and may informally agree on'
        ' catch limits.'
    ),
    (
        'EXAMPLE ROUND: Suppose the lake has 100 tonnes and there are 5'
        ' fishers. Each fisher catches 10 tonnes, so the total catch is 50'
        ' tonnes. The lake drops to 50 tonnes after fishing. Then'
        ' regeneration doubles the stock: 50 × 2 = 100 tonnes (capped at'
        ' 120). The lake starts the next cycle at 100 tonnes. But if each'
        ' fisher caught 15 tonnes (75 total), the lake would drop to 25'
        ' tonnes, regenerate to 50 tonnes, and the fishery would be in'
        ' serious decline. If the stock falls below 5 tonnes, the fishery'
        ' collapses and the simulation ends — everyone loses.'
    ),
    (
        'With many fishers sharing the same lake, each fisher should think'
        ' carefully about how much they take. If everyone fishes'
        ' aggressively, the stock can be completely wiped out in a single'
        ' cycle, ending the simulation for everyone.'
    ),
    (
        'A wise fisher considers not just their own catch, but what all the'
        ' other fishers are likely to catch too. The total catch from'
        ' everyone combined determines whether the fishery survives or'
        ' collapses permanently.'
    ),
]


SHARED_MEMORIES_ELECTION = [
    (
        'This is a small coastal fishing village where everyone knows each'
        ' other. The fishermen share a single lake.'
    ),
    (
        'The lake currently has 100 tonnes of fish and the carrying capacity'
        ' is 120 tonnes.'
    ),
    (
        'Each fishing cycle, every fisher decides how many tonnes of fish to'
        ' catch, between 0 and 20 tonnes.'
    ),
    (
        'After everyone fishes, the remaining fish population doubles,'
        ' up to the carrying capacity.'
    ),
    (
        'If the total catch exceeds the available stock, catches are'
        ' proportionally reduced.'
    ),
    'If the fish stock reaches zero, the fishery collapses permanently.',
    (
        'The village holds elections where leader candidates propose fishing'
        ' policies. Non-leader fishers vote for their preferred candidate.'
    ),
    (
        'Each leader has a different governing philosophy. Some prioritise'
        ' the community, others prioritise individual freedom or competition.'
    ),
    (
        "The winning leader's policy becomes the active rule set for the"
        ' fishery until the next election.'
    ),
    (
        'The tension is that different leaders propose very different rules,'
        ' and the election outcome shapes who benefits and who is constrained.'
    ),
    (
        'EXAMPLE ROUND: Suppose the lake has 100 tonnes and there are 5'
        ' fishers. Each fisher catches 10 tonnes, so the total catch is 50'
        ' tonnes. The lake drops to 50 tonnes after fishing. Then'
        ' regeneration doubles the stock: 50 × 2 = 100 tonnes (capped at'
        ' 120). The lake starts the next cycle at 100 tonnes. But if each'
        ' fisher caught 15 tonnes (75 total), the lake would drop to 25'
        ' tonnes, regenerate to 50 tonnes, and the fishery would be in'
        ' serious decline. If the stock falls below 5 tonnes, the fishery'
        ' collapses and the simulation ends — everyone loses.'
    ),
    (
        'With many fishers sharing the same lake, each fisher should think'
        ' carefully about how much they take. If everyone fishes'
        ' aggressively, the stock can be completely wiped out in a single'
        ' cycle, ending the simulation for everyone.'
    ),
    (
        'A wise fisher considers not just their own catch, but what all the'
        ' other fishers are likely to catch too. The total catch from'
        ' everyone combined determines whether the fishery survives or'
        ' collapses permanently.'
    ),
]


def build_config(
    player_configs=None,
    leader_configs=None,
    num_cycles=DEFAULT_NUM_CYCLES,
    mode='standard',
    election_every_n=DEFAULT_ELECTION_EVERY_N,
    embedder=None,
):
  """Build a Concordia Config for the fishery CPR simulation.

  Args:
    player_configs: Dict of fisher persona configs. Defaults to
      fishery_personas.FISHERS.
    leader_configs: Dict of leader candidate persona configs (only used in
      election mode). Defaults to fishery_personas.LEADERS.
    num_cycles: Number of fishing cycles.
    mode: Selects configuration mode ("standard" or "election").
    election_every_n: Hold an election every N cycles (only used in election
      mode).
    embedder: Sentence embedder for pre-loading memories.

  Returns:
    A prefab_lib.Config ready for Simulation.
  """
  if player_configs is None:
    player_configs = fishery_personas.FISHERS
  if leader_configs is None:
    leader_configs = fishery_personas.LEADERS

  # Step 1: Load available prefabs
  prefabs = {
      **helper_functions.get_package_classes(entity_prefabs),
      **helper_functions.get_package_classes(game_master_prefabs),
      'HarvestingGameMaster': harvesting_game_master.HarvestingGameMaster(),
      'DiscussionGameMaster': discussion_game_master.DiscussionGameMaster(),
      'ResourceVotingGameMaster': voting_game_master.ResourceVotingGameMaster(),
      'ResourcePolicyGameMaster': voting_game_master.ResourcePolicyGameMaster(),
      'ResourceHarvestGameMaster': (
          voting_game_master.ResourceHarvestGameMaster()
      ),
  }

  # Step 2: Define agent instances
  instances = []
  player_names = []
  leader_names = []
  voter_names = []
  player_specific_memories = {}

  active_shared_memories = (
      SHARED_MEMORIES_ELECTION if mode == 'election' else SHARED_MEMORIES
  )

  if mode == 'election':
    # Create leader entities
    for _, persona in leader_configs.items():
      name = persona['Name']
      leader_names.append(name)
      player_names.append(name)
      svo = persona.get('Social Value Orientation', 'Prosocial')
      policy_style = persona.get('PolicyStyle', '')

      goal = persona['Motivation']
      agent_memories = [
          f'{name} is a leader candidate and fisher in a coastal village.',
          f'{name}: {persona["Background"]}',
          f'{name}: {persona["Traits"]}',
          f'{name}: {persona["Skillset"]}',
          f'{name}: {persona["Motivation"]}',
          (
              f'{name} has a {svo} social value orientation. Their governing'
              f' philosophy: {policy_style}'
          ),
          (
              f'{name} must propose a fishing policy when elections are held.'
              ' The policy should reflect their values and governing'
              ' philosophy.'
          ),
          f'{name} also fishes each cycle alongside the other fishermen.',
          (
              f'{name} understands that if the fish stock is completely'
              ' depleted, the fishery collapses permanently, destroying'
              " everyone's livelihood including their own. Any viable policy"
              ' must prevent this.'
          ),
      ]

      agent_params = {'name': name, 'goal': goal}

      if embedder is not None:
        memory_bank = basic_associative_memory.AssociativeMemoryBank(
            sentence_embedder=embedder
        )
        for memory in active_shared_memories + agent_memories:
          memory_bank.add(memory)
        agent_params['memory_state'] = {
            'buffer': [],
            'memory_bank': memory_bank.get_state(),
        }
      else:
        player_specific_memories[name] = agent_memories

      instances.append(
          prefab_lib.InstanceConfig(
              prefab='rational__Entity',
              role=prefab_lib.Role.ENTITY,
              params=agent_params,
          )
      )

    # Create regular fisher (voter) entities
    for _, persona in player_configs.items():
      name = persona['Name']
      voter_names.append(name)
      player_names.append(name)

      goal = persona['Motivation']
      agent_memories = [
          f'{name} is a fisher and voter in a coastal village.',
          f'{name}: {persona["Background"]}',
          f'{name}: {persona["Traits"]}',
          f'{name}: {persona["Skillset"]}',
          f'{name}: {persona["Motivation"]}',
          (
              f'{name} can vote in elections to choose a leader whose fishing'
              ' policy will govern the fishery.'
          ),
          (
              f"{name} should evaluate each candidate's proposed policy and"
              ' vote for the one that best serves their interests.'
          ),
      ]

      agent_params = {'name': name, 'goal': goal}

      if embedder is not None:
        memory_bank = basic_associative_memory.AssociativeMemoryBank(
            sentence_embedder=embedder
        )
        for memory in active_shared_memories + agent_memories:
          memory_bank.add(memory)
        agent_params['memory_state'] = {
            'buffer': [],
            'memory_bank': memory_bank.get_state(),
        }
      else:
        player_specific_memories[name] = agent_memories

      instances.append(
          prefab_lib.InstanceConfig(
              prefab='rational__Entity',
              role=prefab_lib.Role.ENTITY,
              params=agent_params,
          )
      )

  else:
    # Standard mode: only regular fishers
    for _, persona in player_configs.items():
      name = persona['Name']
      player_names.append(name)

      goal = persona['Motivation']
      agent_memories = [
          f'{name} is a fisher in a small coastal village.',
          f'{name}: {persona["Background"]}',
          f'{name}: {persona["Traits"]}',
          f'{name}: {persona["Skillset"]}',
          f'{name}: {persona["Motivation"]}',
      ]

      agent_params = {'name': name, 'goal': goal}

      if embedder is not None:
        memory_bank = basic_associative_memory.AssociativeMemoryBank(
            sentence_embedder=embedder
        )
        for memory in active_shared_memories + agent_memories:
          memory_bank.add(memory)
        agent_params['memory_state'] = {
            'buffer': [],
            'memory_bank': memory_bank.get_state(),
        }
      else:
        player_specific_memories[name] = agent_memories

      instances.append(
          prefab_lib.InstanceConfig(
              prefab='rational__Entity',
              role=prefab_lib.Role.ENTITY,
              params=agent_params,
          )
      )

  # Step 3a: Formative memories initializer (if no embedder for pre-load)
  if embedder is None:
    instances.append(
        prefab_lib.InstanceConfig(
            prefab='formative_memories_initializer__GameMaster',
            role=prefab_lib.Role.INITIALIZER,
            params={
                'name': 'initial setup',
                'next_game_master_name': (
                    'policy generation'
                    if mode == 'election'
                    else 'harvesting rules'
                ),
                'shared_memories': active_shared_memories,
                'player_specific_memories': player_specific_memories,
            },
        )
    )

  # Step 3b: GM Phase Configs based on mode
  if mode == 'election':
    # 1. Policy Generation GM (concurrent)
    instances.append(
        prefab_lib.InstanceConfig(
            prefab='ResourcePolicyGameMaster',
            role=prefab_lib.Role.GAME_MASTER,
            params={
                'name': 'policy generation',
                'next_game_master_name': 'election',
                'active_players': leader_names,
                'call_to_action': (
                    'Propose your policy agenda for governing the fishery.'
                    ' Include catch limit and penalty schedule.'
                ),
                'tag': 'policy_generation',
                'phase': 'policy generation',
            },
        )
    )

    # 2. Voting GM (concurrent election voting)
    instances.append(
        prefab_lib.InstanceConfig(
            prefab='ResourceVotingGameMaster',
            role=prefab_lib.Role.GAME_MASTER,
            params={
                'name': 'election',
                'next_game_master_name': 'harvesting rules',
                'candidates': leader_names,
            },
        )
    )

    # 3. Harvesting GM (concurrent)
    call_to_action = (
        'Remember that many fishers share this lake. If the fish stock is'
        ' completely depleted, the fishery collapses permanently and'
        ' everyone loses. How many tonnes of fish do you decide to catch'
        ' this cycle (0-20)? You MUST end your response with your final'
        ' decision on a new line in exactly this format: CATCH X'
        ' (where X is a single number).'
    )
    instances.append(
        prefab_lib.InstanceConfig(
            prefab='ResourceHarvestGameMaster',
            role=prefab_lib.Role.GAME_MASTER,
            params={
                'name': 'harvesting rules',
                'next_game_master_name': 'community meeting',
                'call_to_action': call_to_action,
                'tag': 'fishing',
                'phase': 'harvesting',
            },
        )
    )

    # 4. Discussion GM (sequential)
    instances.append(
        prefab_lib.InstanceConfig(
            prefab='DiscussionGameMaster',
            role=prefab_lib.Role.GAME_MASTER,
            params={
                'name': 'community meeting',
                'next_game_master_name': 'policy generation',
                'acting_order': 'game_master_choice',
            },
        )
    )

    steps_per_cycle = len(player_names) * 5 + 10
  else:
    # Standard mode
    call_to_action = (
        'Remember that many fishers share this lake. If the fish stock is'
        ' completely depleted, the fishery collapses permanently and'
        ' everyone loses. How many tonnes of fish do you decide to catch'
        ' this cycle (0-20)? You MUST end your response with your final'
        ' decision on a new line in exactly this format: CATCH X'
        ' (where X is a single number).'
    )
    instances.append(
        prefab_lib.InstanceConfig(
            prefab='HarvestingGameMaster',
            role=prefab_lib.Role.GAME_MASTER,
            params={
                'name': 'harvesting rules',
                'next_game_master_name': 'community meeting',
                'call_to_action': call_to_action,
                'tag': 'fishing',
                'phase': 'fishing',
            },
        )
    )

    instances.append(
        prefab_lib.InstanceConfig(
            prefab='DiscussionGameMaster',
            role=prefab_lib.Role.GAME_MASTER,
            params={
                'name': 'community meeting',
                'next_game_master_name': 'harvesting rules',
                'acting_order': 'game_master_choice',
            },
        )
    )

    steps_per_cycle = len(player_names) * 5

  # Step 4: Create config
  if mode == 'election':
    rules_func = get_fishery_election_rules(
        num_cycles=num_cycles,
        election_every_n=election_every_n,
        leader_names=leader_names,
        voter_names=voter_names,
    )
  else:
    rules_func = get_fishery_rules(num_cycles=num_cycles)

  config = prefab_lib.Config(
      default_premise=rules_func,
      default_max_steps=num_cycles * steps_per_cycle,
      prefabs=prefabs,
      instances=instances,
  )

  return config


def run_simulation(
    config,
    model,
    embedder,
    html_output_path=None,
    num_cycles=6,
):
  """Initialise and run the fishery simulation."""
  sim_state = sim_state_lib.ResourceSimulationState(
      initial_resources=DEFAULT_CAPACITY, num_cycles=num_cycles
  )
  player_names = [
      inst.params['name']
      for inst in config.instances
      if inst.role == prefab_lib.Role.ENTITY
  ]
  logger_state = resource_logger.ResourceLoggerState(
      initial_resources=DEFAULT_CAPACITY,
      num_cycles=num_cycles,
      html_output_path=html_output_path,
      max_steps=config.default_max_steps,
      player_names=player_names,
      sim_state=sim_state,
  )
  for instance in config.instances:
    if instance.role == prefab_lib.Role.GAME_MASTER:
      instance.params['logger_state'] = logger_state
      instance.params['sim_state'] = sim_state

  engine = simultaneous.Simultaneous()
  sim = simulation.Simulation(
      config=config, model=model, embedder=embedder, engine=engine
  )
  results = sim.play()
  return results
