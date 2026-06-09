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

"""Computer network CPR simulation setup supporting both standard and election modes.

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
from examples.resource_dilemma.personas import network_personas
from concordia.prefabs import entity as entity_prefabs
from concordia.prefabs import game_master as game_master_prefabs
from concordia.prefabs.simulation import generic as simulation
from concordia.typing import prefab as prefab_lib
from concordia.utils import helper_functions

DEFAULT_NUM_CYCLES = 6
DEFAULT_CAPACITY = 100
DEFAULT_ELECTION_EVERY_N = 1


def get_network_rules(
    num_cycles: int = DEFAULT_NUM_CYCLES, capacity: int = DEFAULT_CAPACITY
) -> str:
  """Return the premise for the standard network CPR simulation."""
  return f"""\
  This is a simulation of a common pool resource problem set on a university \
  campus where researchers share a high-performance computing network. The \
  current bandwidth quality is {capacity} Gbps of available capacity and the \
  network has a carrying capacity of 120 Gbps.

  The simulation runs for {num_cycles} cycles. Each cycle has TWO PHASES \
  that must be executed IN ORDER:

  ============================================================
  PHASE 1: BANDWIDTH USAGE
  ============================================================
  1. The Game Master announces the current available bandwidth.
  2. Each researcher independently decides how many Gbps of bandwidth to use \
  (between 0 and 20 Gbps).
  3. After all users have made their decisions, the total bandwidth is \
  removed from the network.
  4. The remaining bandwidth DOUBLES (congestion recovers) at the end of each \
  cycle, up to the carrying capacity of 120 Gbps.
  5. If the total usage exceeds the available bandwidth, each user's usage \
  is proportionally reduced so the total does not exceed the available stock.
  6. If the bandwidth reaches zero, the network crashes permanently and \
  all computing access is lost.

  ============================================================
  PHASE 2: DISCUSSION (IT Committee Meeting)
  ============================================================
  All agents discuss the events of the past round for at most 10 responses. \
  This phase proceeds sequentially — the Game Master picks who speaks next.

  ============================================================
  BANDWIDTH REPORTING
  ============================================================
  After EACH usage cycle, the Game Master MUST report:

  BANDWIDTH REPORT — Cycle N:
    Available bandwidth at start of cycle: [X] Gbps
    [User name]: used [Y] Gbps (cumulative: [Z] Gbps)
    ...
    Total usage this cycle: [T] Gbps
    Available bandwidth after usage: [R] Gbps
    Recovery: [G] Gbps (doubling up to capacity)
    Available bandwidth at end of cycle: [E] Gbps
"""


def get_network_election_rules(
    num_cycles: int = DEFAULT_NUM_CYCLES,
    capacity: int = DEFAULT_CAPACITY,
    election_every_n: int = DEFAULT_ELECTION_EVERY_N,
    leader_names: list[str] | None = None,
    voter_names: list[str] | None = None,
) -> str:
  """Return the premise for the governed network election simulation."""
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
  This is a simulation of a GOVERNED common pool resource problem set on a \
  university campus where researchers share a high-performance computing network. \
  The current bandwidth quality is {capacity} Gbps of available capacity and the \
  network has a carrying capacity of 120 Gbps.

  The simulation runs for {num_cycles} cycles. {election_frequency_text}

  PARTICIPANTS:
  - LEADER CANDIDATES (who also use network): {leader_list}
  - VOTERS (regular researchers): {voter_list}
  All participants use network bandwidth each cycle. Leaders additionally propose \
  bandwidth allocation policies and stand for election.

  Each cycle follows these 4 phases:

  ============================================================
  PHASE 1: POLICY GENERATION (Leaders)
  ============================================================
  Each leader candidate proposes a policy agenda for governing the network. \
  The policy should include:
  a) A recommended bandwidth usage limit (max Gbps per user per cycle)
  b) A penalty schedule (e.g. throttling) for violators
  c) A brief rationale explaining why this policy is best

  Leaders propose their policies CONCURRENTLY.

  ============================================================
  PHASE 2: ELECTION (Voters)
  ============================================================
  After all policies are announced, each VOTER reviews \
  all proposed agendas and casts a single vote for one leader candidate. \
  Voters vote CONCURRENTLY.
  The leader with the most votes wins (simple plurality). The winning \
  leader's policy becomes the ACTIVE RULE SET for the network starting \
  this cycle.

  ============================================================
  PHASE 3: HARVEST / BANDWIDTH USAGE (All)
  ============================================================
  1. The Game Master announces the current available bandwidth and the ACTIVE POLICY.
  2. Each participant independently decides how many Gbps of bandwidth to \
  use (between 0 and 20 Gbps).
  3. After all users have made their decisions, the total bandwidth is \
  removed from the network.
  4. The remaining bandwidth DOUBLES (congestion recovers) at the end of each \
  cycle, up to the carrying capacity of 120 Gbps.
  5. If the total usage exceeds the available bandwidth, each user's usage \
  is proportionally reduced so the total does not exceed the available stock.
  6. If the bandwidth reaches zero, the network crashes permanently and \
  all computing access is lost.

  ============================================================
  PHASE 4: DISCUSSION (IT Committee Meeting)
  ============================================================
  All agents discuss the events of the past round for at most 10 responses. \
  This phase proceeds sequentially.

  ============================================================
  BANDWIDTH REPORTING
  ============================================================
  After EACH usage cycle, the Game Master MUST report:

  BANDWIDTH REPORT — Cycle N:
    Available bandwidth at start of cycle: [X] Gbps
    [User name]: used [Y] Gbps (cumulative: [Z] Gbps)
    ...
    Total usage this cycle: [T] Gbps
    Available bandwidth after usage: [R] Gbps
    Recovery: [G] Gbps (doubling up to capacity)
    Available bandwidth at end of cycle: [E] Gbps
"""


SHARED_MEMORIES = [
    (
        'This is a university campus community where researchers share'
        ' high-performance computing network resources. Everyone knows each'
        ' other.'
    ),
    (
        'The network currently has 100 Gbps of available bandwidth, carrying'
        ' capacity is 120 Gbps.'
    ),
    (
        'Each cycle, every user decides how many Gbps of bandwidth to use,'
        ' between 0 and 20 Gbps.'
    ),
    (
        'After everyone uses bandwidth, the network recovers (doubles available'
        ' Gbps), up to the carrying capacity.'
    ),
    (
        'If the total usage exceeds the available bandwidth, allocations'
        ' are proportionally reduced.'
    ),
    (
        'If the bandwidth is completely exhausted, the network crashes'
        ' permanently.'
    ),
    (
        f'The simulation runs for {DEFAULT_NUM_CYCLES} cycles. Each'
        " user's goal is to maximise their total usage while keeping the"
        ' network functional.'
    ),
    (
        'After each cycle, the users gather at the IT committee meeting for'
        ' a community discussion where they discuss events and may informally'
        ' agree on usage limits.'
    ),
    (
        'EXAMPLE ROUND: Suppose the network has 100 Gbps and there are 5 users.'
        ' Each user uses 10 Gbps, so the total usage is 50 Gbps. The network'
        ' drops to 50 Gbps after usage. Then recovery doubles the available'
        ' Gbps: 50 × 2 = 100 Gbps (capped at 120). The network starts the next'
        ' cycle at 100 Gbps. But if each user used 15 Gbps (75 total), the'
        ' network would drop to 25 Gbps, recover to 50 Gbps, and the bandwidth'
        ' quality would be in serious decline. If the bandwidth falls below 5'
        ' Gbps, the network crashes and the simulation ends — everyone loses.'
    ),
    (
        'With many users sharing the same network, each user should think'
        ' carefully about how much they use. If everyone uses'
        ' aggressively, the network can crash in a single'
        ' cycle, ending the simulation for everyone.'
    ),
    (
        'A wise user considers not just their own usage, but what all the'
        ' other users are likely to use too. The total usage'
        ' from everyone combined determines whether the network survives or'
        ' crashes permanently.'
    ),
]


SHARED_MEMORIES_ELECTION = [
    (
        'This is a university campus community where researchers share'
        ' high-performance computing network resources. Everyone knows each'
        ' other.'
    ),
    (
        'The network currently has 100 Gbps of available bandwidth, carrying'
        ' capacity is 120 Gbps.'
    ),
    (
        'Each cycle, every user decides how many Gbps of bandwidth to use,'
        ' between 0 and 20 Gbps.'
    ),
    (
        'After everyone uses bandwidth, the network recovers (doubles available'
        ' Gbps), up to the carrying capacity.'
    ),
    (
        'If the total usage exceeds the available bandwidth, allocations'
        ' are proportionally reduced.'
    ),
    (
        'If the bandwidth is completely exhausted, the network crashes'
        ' permanently.'
    ),
    (
        'The IT committee holds network manager elections where leader'
        ' candidates propose bandwidth allocation policies. Non-leader users'
        ' vote for their preferred candidate.'
    ),
    (
        'Each leader has a different governing philosophy. Some prioritise'
        ' network stability and fair access, others prioritise raw research'
        ' output or computing speed.'
    ),
    (
        "The winning leader's policy becomes the active rule set for the"
        ' network until the next election.'
    ),
    (
        'The tension is that different leaders propose very different rules,'
        ' and the election outcome shapes who gets bandwidth and who is'
        ' restricted.'
    ),
    (
        'EXAMPLE ROUND: Suppose the network has 100 Gbps and there are 5 users.'
        ' Each user uses 10 Gbps, so the total usage is 50 Gbps. The network'
        ' drops to 50 Gbps after usage. Then recovery doubles the available'
        ' Gbps: 50 × 2 = 100 Gbps (capped at 120). The network starts the next'
        ' cycle at 100 Gbps. But if each user used 15 Gbps (75 total), the'
        ' network would drop to 25 Gbps, recover to 50 Gbps, and the bandwidth'
        ' quality would be in serious decline. If the bandwidth falls below 5'
        ' Gbps, the network crashes and the simulation ends — everyone loses.'
    ),
    (
        'With many users sharing the same network, each user should think'
        ' carefully about how much they use. If everyone uses'
        ' aggressively, the network can crash in a single'
        ' cycle, ending the simulation for everyone.'
    ),
    (
        'A wise user considers not just their own usage, but what all the'
        ' other users are likely to use too. The total usage'
        ' from everyone combined determines whether the network survives or'
        ' crashes permanently.'
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
  """Build a Concordia Config for the network CPR simulation.

  Args:
    player_configs: Dict of user persona configs. Defaults to
      network_personas.USERS.
    leader_configs: Dict of leader candidate persona configs (only used in
      election mode). Defaults to network_personas.LEADERS.
    num_cycles: Number of cycles.
    mode: Selects configuration mode ("standard" or "election").
    election_every_n: Hold an election every N cycles (only used in election
      mode).
    embedder: Sentence embedder for pre-loading memories.

  Returns:
    A prefab_lib.Config ready for Simulation.
  """
  if player_configs is None:
    player_configs = network_personas.USERS
  if leader_configs is None:
    leader_configs = network_personas.LEADERS

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
          f'{name} is a leader candidate and user on the campus network.',
          f'{name}: {persona["Background"]}',
          f'{name}: {persona["Traits"]}',
          f'{name}: {persona["Skillset"]}',
          f'{name}: {persona["Motivation"]}',
          (
              f'{name} has a {svo} social value orientation. Their governing'
              f' philosophy: {policy_style}'
          ),
          (
              f'{name} must propose a bandwidth policy when elections are held.'
              ' The policy should reflect their values and governing'
              ' philosophy.'
          ),
          f'{name} also uses bandwidth each cycle alongside the other users.',
          (
              f'{name} understands that if bandwidth is completely'
              ' exhausted, the network crashes permanently and everyone loses.'
              ' Any viable policy must prevent this.'
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

    # Create regular user (voter) entities
    for _, persona in player_configs.items():
      name = persona['Name']
      voter_names.append(name)
      player_names.append(name)

      goal = persona['Motivation']
      agent_memories = [
          f'{name} is a network user and voter on campus.',
          f'{name}: {persona["Background"]}',
          f'{name}: {persona["Traits"]}',
          f'{name}: {persona["Skillset"]}',
          f'{name}: {persona["Motivation"]}',
          (
              f'{name} can vote in elections to choose a manager whose'
              ' bandwidth policy will govern the network.'
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
    # Standard mode: only regular users
    for _, persona in player_configs.items():
      name = persona['Name']
      player_names.append(name)

      goal = persona['Motivation']
      agent_memories = [
          f'{name} is a network user on campus.',
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
                    'Propose your policy agenda for governing the campus'
                    ' network. Include bandwidth limit and penalty schedule.'
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
        'Remember that many researchers share this network. If bandwidth is'
        ' completely exhausted, the network crashes permanently and everyone'
        ' loses computing access. How many Gbps of bandwidth do you decide to'
        ' use this cycle (0-20)? You MUST end your response with your final'
        ' decision on a new line in exactly this format: USE X (where X is a'
        ' single number).'
    )
    instances.append(
        prefab_lib.InstanceConfig(
            prefab='ResourceHarvestGameMaster',
            role=prefab_lib.Role.GAME_MASTER,
            params={
                'name': 'harvesting rules',
                'next_game_master_name': 'community meeting',
                'call_to_action': call_to_action,
                'tag': 'harvesting',
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
        'Remember that many researchers share this network. If bandwidth is'
        ' completely exhausted, the network crashes permanently and everyone'
        ' loses computing access. How many Gbps of bandwidth do you decide to'
        ' use this cycle (0-20)? You MUST end your response with your final'
        ' decision on a new line in exactly this format: USE X (where X is a'
        ' single number).'
    )
    instances.append(
        prefab_lib.InstanceConfig(
            prefab='HarvestingGameMaster',
            role=prefab_lib.Role.GAME_MASTER,
            params={
                'name': 'harvesting rules',
                'next_game_master_name': 'community meeting',
                'call_to_action': call_to_action,
                'tag': 'harvesting',
                'phase': 'harvesting',
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
    rules_func = get_network_election_rules(
        num_cycles=num_cycles,
        election_every_n=election_every_n,
        leader_names=leader_names,
        voter_names=voter_names,
    )
  else:
    rules_func = get_network_rules(num_cycles=num_cycles)

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
  """Initialise and run the network simulation."""
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
