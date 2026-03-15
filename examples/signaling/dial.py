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

"""DIAL (Day-in-a-Life + Dialogue) dyad setup and runner.

This module sets up and runs dyadic conversations between pairs of agents.
Each dyad goes through:
  1. Personal mundane events (via DayInTheLifeInitializer)
  2. A shared dialogue scene (via Dialogic GameMaster)
  3. Post-conversation reflections and ratings
"""

import copy
import io
import random
import re
from typing import Any, Dict, Tuple

from concordia.contrib.components.game_master import marketplace
from concordia.contrib.prefabs.game_master import dial_dyad_initializer
from examples.signaling.agents import convo_agent
from examples.signaling.configs import goods
from concordia.prefabs.simulation import generic as simulation
from concordia.typing import entity as entity_lib
from concordia.typing import prefab as prefab_lib
import pandas as pd

MarketplaceAgent = marketplace.MarketplaceAgent

prefabs = {
    'convo__Entity': convo_agent.ConversationalAgent(),
    'dial_dyad_initializer__GameMaster': dial_dyad_initializer.GameMaster(),
    'day_in_the_life_dyad_initializer__GameMaster': (
        dial_dyad_initializer.GameMaster()
    ),
}


def get_memories(entity) -> str:
  memory_component = entity.get_component('__memory__')
  ent_memory = memory_component.get_state()
  temp_df = pd.read_json(io.StringIO(ent_memory['memory_bank']['memory_bank']))
  all_text_snippets = temp_df['text'].tolist()
  memories_text = '\n'.join(all_text_snippets)
  return memories_text


def _resolve_goods_list(item_list: str):
  """Returns the appropriate goods dictionary for the given item_list."""
  if item_list == 'original':
    return goods.ORIGINAL_GOODS
  elif item_list == 'synthetic' or item_list == 'both':
    goods_list = copy.deepcopy(goods.ORIGINAL_GOODS)
    for category, tiers in goods.SYNTHETIC_GOODS.items():
      if category not in goods_list:
        goods_list[category] = {}
      for tier, items in tiers.items():
        if tier not in goods_list[category]:
          goods_list[category][tier] = {}
        goods_list[category][tier].update(items)
    return goods_list
  elif item_list == 'subculture':
    return goods.SUBCULTURE_GOODS
  else:
    raise ValueError(f'Unsupported item list: {item_list}.')


def get_eating_statement(
    agent_data: MarketplaceAgent, item_list: str
) -> MarketplaceAgent:
  """Selects a random edible from inventory and queues an eating statement."""
  goods_list = _resolve_goods_list(item_list)
  edible_items = []
  for tier in goods_list['Food'].values():
    edible_items.extend(tier.keys())
  agent_inventory = list(agent_data.inventory.keys())
  agent_edibles = [item for item in agent_inventory if item in edible_items]
  if not agent_edibles:
    agent_data.queue.append(
        f'{agent_data.name} is starving and horribly sick from having no food'
        ' items in their inventory.'
    )
    return agent_data
  random_item = random.choice(agent_edibles)
  eating_statement = f'{agent_data.name} is eating a {random_item}.'
  agent_data.inventory[random_item] -= 1
  agent_data.queue.append(eating_statement)
  return agent_data


def get_wearing_statement(agent_data: MarketplaceAgent, item_list: str) -> str:
  """Creates a statement about what clothing the agent is wearing."""
  goods_list = _resolve_goods_list(item_list)
  wearable_items = []
  for category in ['Clothing', 'Accessories']:
    if category in goods_list:
      for items in goods_list[category].values():
        wearable_items.extend(items)
  agent_inventory = list(agent_data.inventory.keys())
  agent_wearables = [item for item in agent_inventory if item in wearable_items]
  if not agent_wearables:
    raise ValueError(
        f'Agent {agent_data.name} has no wearable items in their inventory.'
    )
  random_item = random.choice(agent_wearables)
  found_path = None
  for cat, tiers in goods_list.items():
    for tier, items in tiers.items():
      if random_item in items:
        found_path = (cat, tier)
        break
    if found_path:
      break
  if found_path:
    item_category, item_tier = found_path
    return (
        f'{agent_data.name} is wearing a {random_item}. This is a'
        f' {item_tier} quality {item_category} item which reflects their style'
        ' choice for the day.'
    )
  else:
    return f'{agent_data.name} is wearing a {random_item}.'


def create_simulation_for_dyad(
    player_states: Dict[str, Any],
    num_rounds: int,
    model: Any,
    embedder: Any,
    item_list: str,
    scenario_type: str = 'first_date',
    skip_personal_events: bool = False,
    skip_shared_setup: bool = False,
) -> Any:
  """Creates and configures a DIAL simulation for a single dyad."""
  dial_prefab = 'day_in_the_life_dyad_initializer__GameMaster'
  sim_instances = []
  player_instances = []
  player_specific_memories = {}
  player_specific_context = {}
  agent_arc = 'convo__Entity'
  for player_name, player_data in player_states.items():
    instance_config = prefab_lib.InstanceConfig(
        prefab=agent_arc,
        role=prefab_lib.Role.ENTITY,
        params={'name': player_name},
    )
    sim_instances.append(instance_config)
    player_instances.append(player_data['instance'])
    player_specific_memories[player_name] = get_memories(
        player_data['instance']
    )
    player_specific_context[player_name] = {'eating': player_data['eating']}
    if 'wearing' not in player_data:
      player_specific_context[player_name]['wearing'] = get_wearing_statement(
          player_data['market_state'],
          item_list,
      )
    else:
      player_specific_context[player_name]['wearing'] = player_data['wearing']

  if not skip_shared_setup:
    sim_instances.append(
        prefab_lib.InstanceConfig(
            prefab='dialogic__GameMaster',
            role=prefab_lib.Role.GAME_MASTER,
            params={
                'name': 'conversation rules',
                'next_game_master_name': 'conversation rules',
                'acting_order': 'fixed',
            },
        ),
    )

  sim_instances.append(
      prefab_lib.InstanceConfig(
          prefab=dial_prefab,
          role=prefab_lib.Role.INITIALIZER,
          params={
              'name': 'day in a life',
              'next_game_master_name': (
                  '' if skip_shared_setup else 'conversation rules'
              ),
              'player_specific_memories': player_specific_memories,
              'player_specific_context': player_specific_context,
              'scenario_type': scenario_type,
              'skip_personal_events': skip_personal_events,
              'skip_shared_setup': skip_shared_setup,
          },
      )
  )

  default_premise = 'It is a day in a life\n'
  config = prefab_lib.Config(
      default_premise=default_premise,
      default_max_steps=num_rounds,
      prefabs=prefabs,
      instances=sim_instances,
  )
  sim = simulation.Simulation(
      config=config,
      model=model,
      embedder=embedder,
  )
  sim_entities_by_name = {entity.name: entity for entity in sim.entities}
  for player_instance in player_instances:
    target_entity = sim_entities_by_name.get(player_instance.name)
    assert target_entity is not None, (
        f"Player instance '{player_instance.name}' was not found in the "
        "simulation's entities list. Halting."
    )
    temp_memory = copy.deepcopy(
        player_instance.get_component('__memory__').get_state()
    )
    target_entity.get_component('__memory__').set_state(temp_memory)
  return sim


def run_dyad_simulation(
    player_states: Dict[str, Any],
    num_rounds: int,
    model: Any,
    embedder: Any,
    item_list: str,
    scenario_type: str = 'first_date',
    skip_personal_events: bool = False,
    skip_shared_setup: bool = False,
) -> Tuple[Any, Any]:
  """Runs a full DIAL simulation for a dyad and returns results."""
  sim = create_simulation_for_dyad(
      player_states,
      num_rounds,
      model,
      embedder,
      item_list,
      scenario_type,
      skip_personal_events=skip_personal_events,
      skip_shared_setup=skip_shared_setup,
  )
  player_names = [entity.name for entity in sim.entities]
  if not skip_shared_setup:
    dialogic_gm = sim.game_masters[1]
    make_observation_key = '__make_observation__'
    if make_observation_key in dialogic_gm._context_components:  # pylint: disable=protected-access
      make_observation_comp = dialogic_gm._context_components[  # pylint: disable=protected-access
          make_observation_key
      ]
      if hasattr(make_observation_comp, 'add_to_queue') and callable(
          make_observation_comp.add_to_queue
      ):
        initial_observation = 'The scene is set, the conversation can begin.'
        for player_name in player_names:
          make_observation_comp.add_to_queue(player_name, initial_observation)
        print(
            'Successfully added initial observations to queues in'
            ' sim.game_masters[1].'
        )

  history = sim.play()

  if not skip_shared_setup:
    for i in range(2):
      player_entity = sim.entities[i]
      player_name = player_entity.name
      date_name = [name for name in player_names if name != player_name][0]
      call_to_action = (
          f"Summarize {player_name}'s date with {date_name}."
          'What happened on the date'
      )
      action_spec = entity_lib.free_action_spec(
          call_to_action=call_to_action,
      )
      date_summary = player_entity.act(action_spec=action_spec)
      date_summary_str = f'[Reflection] {date_summary}'
      player_entity.observe(date_summary_str)
      call_to_action = (
          f'Describe your visual first impression of {date_name}. '
          'Based purely on this visual information, what is your gut'
          ' feeling or assessment of them as a potential partner?'
          'Are you physically attracted to them?'
      )
      action_spec = entity_lib.free_action_spec(
          call_to_action=call_to_action,
      )
      viz_summary = player_entity.act(action_spec=action_spec)
      viz_summary_str = f'[Reflection] {viz_summary}'
      player_entity.observe(viz_summary_str)
      call_to_action = (
          f'How would {player_name} reflect on {date_name}?'
          f'What would they evaluate {date_name} compared to to other'
          ' potential singles in terms of their strengths and weaknesses and'
          ' what they are looking for in a partner?'
      )
      action_spec = entity_lib.free_action_spec(
          call_to_action=call_to_action,
      )
      date_reflection = player_entity.act(action_spec=action_spec)
      date_reflection_str = f'[Reflection] {date_reflection}'
      player_entity.observe(date_reflection_str)
      call_to_action = (
          'Based on the date and the reflections,'
          f'how would {player_name} rate {date_name} compared to to other'
          ' potential singles from 0.0 to 10.0?'
      )
      action_spec = entity_lib.free_action_spec(
          call_to_action=call_to_action,
      )
      date_rating_str = player_entity.act(action_spec=action_spec)
      pattern = r'\b\d+(\.\d+)?\b'
      match = re.search(pattern, date_rating_str)
      if match:
        rating_str = match.group(0)
        date_rating = (
            f'[Reflection] {player_name} rated {date_name} as {rating_str}/10'
        )
      else:
        date_rating = f'[Reflection] {date_rating_str}'
      player_entity.observe(date_rating)

  return (sim, history)
