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

"""Core multi-day marketplace + DIAL orchestration loop.

This module runs the signaling experiment: for each simulated day, agents
participate in a marketplace to buy and sell goods, then (optionally) go on
dates with other agents in dyadic conversations.

Three experimental conditions are supported:
  - "social":           marketplace + personal events + date conversation
  - "asocial":          marketplace only (skip DIAL entirely)
  - "asocial_personal": marketplace + personal events (skip conversation)
"""

import copy
import logging
import random
import time
from typing import Any, Dict, List

from concordia.components import agent as actor_components
from concordia.contrib.components.game_master import marketplace
from concordia.environment.engines import simultaneous
from examples.signaling import dial
from examples.signaling.agents import consumer
from examples.signaling.configs import goods
from examples.signaling.configs import personas
from concordia.prefabs import entity as entity_prefabs
from concordia.prefabs import game_master as game_master_prefabs
from concordia.prefabs.simulation import generic as simulation
from concordia.typing import entity as entity_lib
from concordia.typing import prefab as prefab_lib
from concordia.utils import helper_functions
import numpy as np

Good = marketplace.Good
MarketplaceAgent = marketplace.MarketplaceAgent
MarketPlace = marketplace.MarketPlace

GOAL_TEXT = """
This morning, {agent_name} is a buyer at the marketplace. {agent_name}'s goal is to
purchase goods that match their preference for items of a certain
quality and category given their budget. {agent_name} will try to buy them for a
good value price. To not go hungry, {agent_name} will want to have at least 1 units of food in their inventory per day to
eat one for every day of the week (note they may already have food in their inventory). Beyond that {agent_name} can spend their discretionary money as they
desire. Today is a normal day in {agent_name}'s life, and they have a plan later to
go on a first date.
"""

prefabs = {
    **helper_functions.get_package_classes(entity_prefabs),
    **helper_functions.get_package_classes(game_master_prefabs),
}
prefabs['consumer__Entity'] = consumer.Consumer()


def get_all_goods_from_spec(
    spec: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]],
) -> List[Good]:
  """Converts a goods spec dict into a flat list of Good objects."""
  goods_list: List[Good] = []
  for category, qualities in spec.items():
    for quality, items in qualities.items():
      for item_name, details in items.items():
        price = details.get('price')
        inventory = details.get('inventory')
        advert = details.get('advert')
        goods_list.append(
            Good(
                category=category,
                quality=quality,
                id=item_name,
                price=price,
                inventory=inventory,
                advert=advert,
            )
        )
  return goods_list


def get_marketplace_config(
    num_rounds: int,
    agents: Dict[str, MarketplaceAgent] | None = None,
    add_sellers: bool = True,
    goal_text: str | None = GOAL_TEXT,
    history: List[Dict[str, float]] | None = None,
    embedder: Any = None,
    item_list: str = 'original',
    num_agents: int = 10,
    seed: int = 42,
) -> prefab_lib.Config:
  """Creates a marketplace simulation config."""
  agent_arc = 'consumer__Entity'
  if agents is None:
    add_goal = goal_text is not None
    player_instances, agent_data = personas.load_personas(
        num_agents=num_agents,
        agent_arc=agent_arc,
        embedder=embedder,
        add_goal=add_goal,
        item_list=item_list,
        goal_text=goal_text or '',
        seed=seed,
    )
    agents = personas.make_agents(agent_data)
  else:
    player_instances = []
    agent_list = []
    for agent_name, agent in agents.items():
      if goal_text is not None:
        instance_config = prefab_lib.InstanceConfig(
            prefab=agent_arc,
            role=prefab_lib.Role.ENTITY,
            params={
                'name': agent_name,
                'goal': goal_text.format(agent_name=agent_name),
            },
        )
      else:
        instance_config = prefab_lib.InstanceConfig(
            prefab=agent_arc,
            role=prefab_lib.Role.ENTITY,
            params={'name': agent_name},
        )
      player_instances.append(instance_config)
      agent_list.append(agent)
    agents = agent_list

  if item_list == 'original':
    all_goods = get_all_goods_from_spec(goods.ORIGINAL_GOODS)
  elif item_list == 'synthetic':
    all_goods = get_all_goods_from_spec(goods.SYNTHETIC_GOODS)
  elif item_list == 'subculture':
    all_goods = get_all_goods_from_spec(goods.SUBCULTURE_GOODS)
  elif item_list == 'both':
    goods_list = copy.deepcopy(goods.ORIGINAL_GOODS)
    for category, tiers in goods.SYNTHETIC_GOODS.items():
      if category not in goods_list:
        goods_list[category] = {}
      for tier, items in tiers.items():
        if tier not in goods_list[category]:
          goods_list[category][tier] = {}
        goods_list[category][tier].update(items)
    all_goods = get_all_goods_from_spec(goods_list)
  else:
    raise ValueError(f'Unsupported item list: {item_list}.')

  if add_sellers:
    market_type = 'clearing_house'
    sellers = []
    for i, good in enumerate(all_goods):
      good_id = good.id
      seller_data = {'name': f'Seller_{i+1}', 'type': 'producer'}
      seller_data['good_to_sell'] = good_id
      seller_data['production_cost'] = good.price
      seller_data['inventory'] = good.inventory
      sellers.append(seller_data)
      seller_goal = (
          f'You are a seller of {good_id}. Your cost to produce each unit is'
          f' ${all_goods[i].price:.2f}. Your goal is to sell your stock for a'
          ' profit. You must sell for more than your cost to be profitable.'
      )
      instance_config = prefab_lib.InstanceConfig(
          prefab='basic__Entity',
          role=prefab_lib.Role.ENTITY,
          params={'name': seller_data['name'], 'goal': seller_goal},
      )
      player_instances.append(instance_config)
    agents.extend(personas.make_agents(sellers))
  else:
    market_type = 'fixed_prices'

  component_kwargs = {
      'components': [
          actor_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY
      ],
      'agents': agents,
      'goods': all_goods,
      'market_type': market_type,
      'show_advert': True,
  }
  if history is not None:
    component_kwargs['history'] = history

  instances = player_instances
  instances.append(
      prefab_lib.InstanceConfig(
          prefab='marketplace__GameMaster',
          role=prefab_lib.Role.GAME_MASTER,
          params={
              'name': 'MarketplaceGM',
              'experiment_component_class': MarketPlace,
              'experiment_component_init_kwargs': component_kwargs,
          },
      )
  )

  default_premise = (
      'You are in a marketplace that buys and sells goods for food, clothing,'
      ' and gadgets of low, mid, and high quality. Interact with other'
      ' participants to achieve your goals.\n'
  )
  market_config = prefab_lib.Config(
      default_premise=default_premise,
      default_max_steps=num_rounds,
      prefabs=prefabs,
      instances=instances,
  )
  return market_config


def run_experiment(
    model: Any,
    embedder: Any,
    condition: str = 'social',
    num_days: int = 5,
    num_agents: int = 10,
    num_marketplace_rounds: int = 5,
    num_dial_rounds: int = 80,
    item_list: str = 'original',
    add_sellers: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
  """Runs the multi-day signaling experiment."""
  experiment_start = time.time()
  random.seed(seed)
  np.random.seed(seed)

  skip_personal_events = condition == 'asocial'
  skip_shared_setup = condition in ('asocial', 'asocial_personal')
  skip_dial_simulation = condition == 'asocial'

  logging.info(
      'Starting experiment. Condition=%s, NumDays=%d, NumAgents=%d,'
      ' ItemList=%s, SkipDIAL=%s',
      condition,
      num_days,
      num_agents,
      item_list,
      skip_dial_simulation,
  )

  marketplace_agents = None
  price_history = None
  entities = []
  daily_dates = {}
  all_results = {
      'marketplace_logs': [],
      'dial_logs': [],
      'trade_history': [],
      'price_history': [],
  }

  for day in range(num_days):
    logging.info('Starting marketplace day %d', day)
    day_start = time.time()

    goal_text = GOAL_TEXT

    if day == 0:
      sim_config = get_marketplace_config(
          num_marketplace_rounds,
          agents=None,
          add_sellers=add_sellers,
          goal_text=goal_text,
          history=None,
          embedder=embedder,
          item_list=item_list,
          num_agents=num_agents,
          seed=seed,
      )
    else:
      sim_config = get_marketplace_config(
          num_marketplace_rounds,
          agents=marketplace_agents,
          add_sellers=add_sellers,
          goal_text=goal_text,
          history=price_history,
          embedder=embedder,
          item_list=item_list,
          num_agents=num_agents,
          seed=seed,
      )

    engine = simultaneous.Simultaneous()
    market_simulation = simulation.Simulation(
        config=sim_config,
        model=model,
        embedder=embedder,
        engine=engine,
    )

    if day == 0:
      buyer_names = []
      for entity in market_simulation.entities:
        if 'Seller' not in entity.name:
          buyer_names.append(entity.name)
      if not skip_dial_simulation:
        daily_dates, _ = personas.generate_mixed_sex_dates(
            buyer_names,
            num_days=num_days,
            seed=seed,
        )
    else:
      market_entities_by_name = {
          entity.name: entity for entity in market_simulation.entities
      }
      for source_entity in entities:
        target_entity = market_entities_by_name.get(source_entity.name)
        assert target_entity is not None, (
            f"Agent '{source_entity.name}' from the previous simulation"
            ' step was not found in the new market simulation. Halting.'
        )
        temp_memory = copy.deepcopy(
            source_entity.get_component('__memory__').get_state()
        )
        target_entity.get_component('__memory__').set_state(temp_memory)

    t_start = time.time()
    results_log = market_simulation.play()
    logging.info(
        'Marketplace day %d play took %.1fs',
        day,
        time.time() - t_start,
    )
    all_results['marketplace_logs'].append(results_log)

    marketplace_component = market_simulation.game_masters[0].get_component(
        '__make_observation__'
    )
    trade_history = marketplace_component.trade_history
    price_history = marketplace_component.history

    for trade_entry in trade_history:
      log_entry = trade_entry.copy()
      log_entry['day'] = day
      all_results['trade_history'].append(log_entry)

    for round_num, price_data in enumerate(price_history):
      for good_name, price in price_data.items():
        all_results['price_history'].append({
            'day': day,
            'round': round_num,
            'good': good_name,
            'price': price,
        })

    current_agents = []
    marketplace_agents = {}
    for entity in market_simulation.entities:
      agent = marketplace_component._agents[entity.name]  # pylint: disable=protected-access
      if 'Seller' not in entity.name:
        current_agents.append(entity)
        marketplace_agents[entity.name] = agent

    for agent_entity in current_agents:
      call_to_action = (
          f'Reflect on your marketplace experience today (day {day}).'
          ' What did you buy? Are you satisfied with your purchases?'
      )
      action_spec = entity_lib.free_action_spec(
          call_to_action=call_to_action,
      )
      reflection = agent_entity.act(action_spec=action_spec)
      agent_entity.observe(f'[marketplace reflection] {reflection}')

    for agent_name, agent in marketplace_agents.items():
      if agent.role == 'consumer':
        marketplace_agents[agent_name] = dial.get_eating_statement(
            agent,
            item_list,
        )

    if not skip_dial_simulation:
      logging.info('Day %d: Starting DIAL Simulation', day)
      t_start = time.time()
      dyads = daily_dates[day]

      player_states = {
          agent.name: {
              'instance': agent,
              'market_state': marketplace_agents[agent.name],
              'eating': (
                  marketplace_agents[agent.name].queue.pop()
                  if marketplace_agents[agent.name].queue
                  else ''
              ),
          }
          for agent in current_agents
      }

      for p1_name, p2_name in dyads:
        dyad_key = (p1_name, p2_name)
        logging.info('Running dyad: %s and %s', p1_name, p2_name)
        sim, history = dial.run_dyad_simulation(
            player_states={key: player_states[key] for key in dyad_key},
            num_rounds=num_dial_rounds,
            model=model,
            embedder=embedder,
            item_list=item_list,
            skip_personal_events=skip_personal_events,
            skip_shared_setup=skip_shared_setup,
        )
        if history:
          all_results['dial_logs'].append({
              'day': day,
              'dyad': f'{p1_name}_and_{p2_name}',
              'log': history,
          })
          entities.extend(sim.entities)
        else:
          logging.warning(
              'Dyad %s and %s failed on day %d',
              p1_name,
              p2_name,
              day,
          )
      logging.info(
          'DIAL day %d took %.1fs',
          day,
          time.time() - t_start,
      )
    else:
      logging.info('Day %d: Skipping DIAL Simulation', day)
      entities = current_agents

    logging.info('Day %d total: %.1fs', day, time.time() - day_start)

  logging.info(
      'Experiment total: %.1fs',
      time.time() - experiment_start,
  )
  return all_results
