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

from collections.abc import Collection, Mapping, Sequence
import json
import threading
import time
from typing import Any, override
import urllib.request

from absl.testing import absltest
from concordia.components.agent import human_act_component
from concordia.components.game_master import event_resolution
from concordia.components.game_master import interrupt_scheduling
from concordia.components.game_master import switch_act
from concordia.environment.engines import asynchronous
from examples.adventure import adventure_cli
from examples.adventure import adventure_prefab
from examples.adventure import adventure_specific_components
from examples.adventure import evaluate_run
from examples.adventure import roster_manager
from examples.adventure import server
from examples.adventure import story_director
from examples.adventure import world_ledger
from concordia.language_model import language_model
from concordia.prefabs.simulation import generic
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.typing import prefab as prefab_lib
from concordia.typing import simulation as simulation_lib


class CountingAdventureMockModel(language_model.LanguageModel):
  """Deterministic mock LLM that counts calls by prompt type."""

  def __init__(self):
    """Initializes the instance."""
    super().__init__()
    self.total_calls = 0
    self.resolve_calls = 0
    self.frontier_calls = 0
    self.npc_calls = 0
    self._room_counter = 0

  @override
  def sample_text(
      self,
      prompt: str,
      *,
      max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
      terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
      temperature: float = language_model.DEFAULT_TEMPERATURE,
      top_p: float = language_model.DEFAULT_TOP_P,
      top_k: int = language_model.DEFAULT_TOP_K,
      timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
      seed: int | None = None,
  ) -> str:
    del max_tokens, terminators, temperature, top_p, top_k, timeout, seed
    self.total_calls += 1
    if 'Frontier Architect' in prompt:
      self.frontier_calls += 1
      self._room_counter += 1
      idx = self._room_counter
      return json.dumps({
          'name': f'Expanded Chamber {idx}',
          'description': (
              f'Vaulted stone chamber {idx} with dripping conduits and '
              'brass pressure gauges.'
          ),
          'affordances': [
              f'rusted valve wheel {idx}',
              f'ciphered logbook {idx}',
          ],
          'exits': {
              'onward': f'Deep Gallery {idx}',
          },
          'npc': {
              'name': f'Warden {idx}',
              'role_summary': f'Watchkeeper of Chamber {idx}',
              'goal': f'Maintain pressure in Chamber {idx}',
              'traits': 'Terse, pragmatic, distrusts unsealed papers',
              'formative_memories': [
                  f'Warden {idx} heard strange knocking in the pipes.'
              ],
          },
      })

    if 'Putative Action by' in prompt:
      self.resolve_calls += 1
      move_to = ''
      for line in prompt.splitlines():
        if line.startswith('Putative Action by'):
          lower = line.lower()
          if 'go north' in lower:
            move_to = 'north'
          elif 'go onward' in lower:
            move_to = 'onward'
          elif 'go east' in lower:
            move_to = 'east'
          elif 'go west' in lower:
            move_to = 'west'
          elif 'go back' in lower:
            move_to = 'back'
      return json.dumps({
          'narrative_event': (
              'The brass mechanisms clatter in response as the action unfolds '
              'under the amber lantern light.'
          ),
          'delta': {
              'move_to': move_to,
              'advance_minutes': 4,
              'add_affordances': ['freshly scratched tally mark'],
              'remove_affordances': [],
              'new_exits': {},
              'add_inventory': [],
              'remove_inventory': [],
              'player_status': 'Focused and alert',
              'npcs_addressed': ['Captain Mara Vance'],
              'relationship_updates': {
                  'Captain Mara Vance': 'Observing closely'
              },
          },
      })

    self.npc_calls += 1
    return 'checks the brass pressure gauge and mutters a warning.'

  @override
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, Mapping[str, Any]]:
    del prompt, seed
    self.total_calls += 1
    return 0, responses[0], {}


class ScriptedPlayerActComponent(
    entity_component.ActingComponent, entity_component.ComponentWithLogging
):
  """Plays a scripted sequence of actions for deterministic testing."""

  def __init__(self, actions: Sequence[str]):
    """Initializes the instance."""
    super().__init__()
    self._actions = list(actions)
    self._idx = 0

  @override
  def get_action_attempt(
      self,
      contexts: entity_component.ComponentContextMapping,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    del contexts, action_spec
    if self._idx < len(self._actions):
      act = self._actions[self._idx]
      self._idx += 1
    else:
      act = 'inspect the surroundings carefully'
    self._logging_channel({'Key': '__act__', 'Value': act})
    return act

  @override
  def get_state(self) -> entity_component.ComponentState:
    return {'idx': self._idx}

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    raw_idx = state.get('idx', 0)
    self._idx = int(raw_idx) if isinstance(raw_idx, (int, float, str)) else 0


class AdventureTest(absltest.TestCase):

  def test_world_ledger_graph_and_delta(self):
    ledger = world_ledger.WorldLedger(
        premise='Test world',
        player_name='Hero',
        starting_location_name='Start Room',
        starting_location_description='A stone hall.',
        starting_affordances=['iron lever'],
        starting_exits={'north': 'North Tower'},
    )
    self.assertEqual(ledger.player_location_id, 'start_room')
    stubs = ledger.unexpanded_adjacent_locations()
    self.assertLen(stubs, 1)
    self.assertEqual(stubs[0].location_id, 'north_tower')

    ledger.expand_location(
        'north_tower',
        name='North Tower',
        description='High windy tower.',
        affordances=['spyglass'],
        exits={'up': 'Bell Loft'},
        backlink_from_id='start_room',
    )
    self.assertEmpty(ledger.unexpanded_adjacent_locations('start_room'))

    ledger.apply_delta(
        {
            'move_to': 'north',
            'advance_minutes': 10,
            'add_inventory': ['brass key'],
        },
        acting_entity='Hero',
        narrative_event='Hero climbs the stairs into the North Tower.',
    )
    self.assertEqual(ledger.player_location_id, 'north_tower')
    self.assertEqual(ledger.player_inventory, ['brass key'])
    telemetry = ledger.telemetry_summary()
    self.assertEqual(telemetry['frontier_traversals'], 1)
    self.assertEqual(telemetry['frontier_cache_hits'], 1)
    self.assertEqual(telemetry['frontier_cache_hit_rate'], 1.0)

    # Verify round-trip serialization.
    saved = ledger.to_dict()
    restored = world_ledger.WorldLedger(premise='Empty')
    restored.load_dict(saved)
    self.assertEqual(restored.player_location_id, 'north_tower')
    self.assertEqual(restored.player_inventory, ['brass key'])

  def test_roster_tier_lifecycle_and_rehydration_preserves_memories_and_bus(
      self,
  ):
    mock_model = CountingAdventureMockModel()
    ledger = world_ledger.WorldLedger(
        premise='Test world',
        player_name='Hero',
        starting_location_name=' Plaza',
        starting_location_description='Central plaza.',
        starting_exits={'east': 'Market', 'west': 'Docks'},
    )
    roster = roster_manager.RosterManager(
        ledger,
        entity_builder=lambda rec: adventure_prefab.build_npc_entity(
            rec,
            model=mock_model,
            embedder=adventure_prefab.default_embedder,
            bus=ledger.bus,
            enable_async_plan=False,
        ),
        max_hot=1,
        max_warm=1,
    )

    npc_record = world_ledger.NpcRecord(
        name='Elena',
        role_summary='Apothecary',
        goal='Find rare nightshade',
        traits='Precise and skeptical',
        location_id='plaza',
        formative_memories=['Elena hid the silver vial under the floorboards.'],
    )
    entity = roster.spawn_npc(npc_record, initial_tier=world_ledger.NpcTier.HOT)
    self.assertIsNotNone(entity)
    assert entity is not None
    entity.observe('Hero asked Elena about the silver vial.')

    # Advance bus clock and hibernate Elena to Cold.
    ledger.advance_clock(minutes=15)
    bus_seq_before_hibernate = ledger.bus.get_state()['seq']
    cold_state = roster.hibernate_to_cold('Elena')
    self.assertIsNotNone(cold_state)
    self.assertIsNone(roster.get_live_entity('Elena'))
    elena_rec = ledger.get_npc('Elena')
    self.assertIsNotNone(elena_rec)
    assert elena_rec is not None
    self.assertEqual(elena_rec.tier, world_ledger.NpcTier.COLD)

    # Advance bus further while Elena is Cold.
    ledger.advance_clock(minutes=30)
    ledger.record_rumor('A storm is approaching.', is_warm=False)
    bus_seq_before_rehydrate = ledger.bus.get_state()['seq']
    self.assertGreater(bus_seq_before_rehydrate, bus_seq_before_hibernate)

    # Rehydrate Elena and verify memories, catch-up observation, and bus seq.
    rehydrated = roster.rehydrate_from_cold(
        'Elena', target_tier=world_ledger.NpcTier.HOT
    )
    self.assertIsNotNone(rehydrated)
    assert rehydrated is not None
    self.assertGreaterEqual(
        ledger.bus.get_state()['seq'], bus_seq_before_rehydrate
    )

    memories = roster_manager.extract_entity_memories(rehydrated)
    joined = ' '.join(memories)
    self.assertIn('silver vial', joined)
    self.assertIn('steps elapsed since Elena last acted', joined)
    self.assertEqual(ledger.telemetry_summary()['hibernations_count'], 1)
    self.assertEqual(ledger.telemetry_summary()['rehydrations_count'], 1)

  def test_end_to_end_simulation_invariants_and_zero_control_calls(self):
    mock_model = CountingAdventureMockModel()
    scripted_actions = [
        'inspect the padlocked customs shed door',
        'go north',
        'inspect rusted valve wheel 1',
        'go onward',
        'go back',
        'go back',
    ]
    sim = adventure_prefab.AdventureSimulation(
        model=mock_model,
        player_act_component=ScriptedPlayerActComponent(scripted_actions),
        enable_async_npc_plan=False,
        max_hot_npcs=1,
        max_warm_npcs=1,
        warm_tick_interval=2,
    )

    # Test control hooks directly to verify 0 LLM calls.
    calls_before = mock_model.total_calls
    _ = sim.engine.terminate(sim.game_master)
    _ = sim.engine.make_observation(sim.game_master, sim.player_entity)
    _ = sim.engine.next_acting(sim.game_master, [sim.player_entity])
    self.assertEqual(mock_model.total_calls, calls_before)

    # Run 6 player turns end-to-end.
    sim_log = sim.play(max_steps=6, verbose=False)
    telemetry = sim.ledger.telemetry_summary()
    latency = sim.latency_tracker.summary()

    self.assertEqual(latency['control_hook_llm_calls'], 0)
    self.assertEqual(latency['max_llm_calls_per_player_resolve'], 1)
    self.assertGreaterEqual(telemetry['frontier_cache_hit_rate'], 0.8)
    self.assertGreaterEqual(telemetry['hibernations_count'], 1)
    self.assertGreaterEqual(telemetry['rehydrations_count'], 1)

    # Verify zero JSON contamination in entity memories.
    log_dict = sim_log.to_dict()
    for _, mem_list in log_dict.get('entity_memories', {}).items():
      for mem in mem_list:
        self.assertNotIn('"narrative_event":', str(mem))
        self.assertNotIn('"output_type":', str(mem))

  def test_evaluator_and_http_server_on_localhost(self):
    mock_model = CountingAdventureMockModel()
    bridge = server.BrowserInputBridge(timeout_seconds=1.0)
    sim = adventure_prefab.AdventureSimulation(
        model=mock_model,
        player_act_component=human_act_component.HumanActComponent(
            input_reader=bridge
        ),
        player_ready_fn=bridge.has_pending_action,
        enable_async_npc_plan=False,
    )
    adv_server = server.AdventureServer(sim, bridge, host='127.0.0.1', port=0)
    port = adv_server.start_background()
    try:
      url = f'http://127.0.0.1:{port}/api/state'
      with urllib.request.urlopen(url, timeout=3.0) as resp:
        self.assertEqual(resp.status, 200)
        self.assertEqual(resp.headers.get('X-Content-Type-Options'), 'nosniff')
        self.assertEqual(resp.headers.get('X-Frame-Options'), 'DENY')
        payload = json.loads(resp.read().decode('utf-8'))
        self.assertIn('ledger', payload)
        init_log = payload['ledger'].get('narrative_log', [])
        self.assertNotEmpty(init_log)
        self.assertEqual(init_log[0]['kind'], 'intro')
        self.assertIn('You have just stepped off', init_log[0]['text'])

      with urllib.request.urlopen(
          f'http://127.0.0.1:{port}/app.js', timeout=3.0
      ) as js_resp:
        self.assertEqual(js_resp.status, 200)
        self.assertEqual(
            js_resp.headers.get('Content-Type'),
            'application/javascript; charset=utf-8',
        )
        self.assertNotEmpty(js_resp.read())

      cur_loc_id = payload['ledger']['player_location_id']
      loc_image_url = payload['ledger']['locations'][cur_loc_id].get(
          'image_url', ''
      )
      self.assertTrue(loc_image_url.startswith('/api/image/loc_'))
      with urllib.request.urlopen(
          f'http://127.0.0.1:{port}{loc_image_url}', timeout=3.0
      ) as img_resp:
        self.assertEqual(img_resp.status, 200)
        self.assertEqual(
            img_resp.headers.get('X-Content-Type-Options'), 'nosniff'
        )
        self.assertNotEmpty(img_resp.read())

      mara_portrait_url = payload['ledger']['npcs']['Captain Mara Vance'].get(
          'portrait_url', ''
      )
      self.assertTrue(mara_portrait_url.startswith('/api/image/npc_'))
      with urllib.request.urlopen(
          f'http://127.0.0.1:{port}{mara_portrait_url}', timeout=3.0
      ) as portrait_resp:
        self.assertEqual(portrait_resp.status, 200)
        self.assertNotEmpty(portrait_resp.read())

      # Submit an action via POST /api/action
      req = urllib.request.Request(
          f'http://127.0.0.1:{port}/api/action',
          data=json.dumps({'action': 'inspect the brass seal'}).encode('utf-8'),
          headers={'Content-Type': 'application/json'},
          method='POST',
      )
      with urllib.request.urlopen(req, timeout=3.0) as post_resp:
        self.assertEqual(post_resp.status, 200)
        post_payload = json.loads(post_resp.read().decode('utf-8'))
        updated_log = post_payload['state']['ledger']['narrative_log']
        self.assertEqual(updated_log[-1]['kind'], 'player_action')
        self.assertEqual(updated_log[-1]['text'], 'inspect the brass seal')
        ids = [entry['id'] for entry in updated_log]
        self.assertEqual(len(ids), len(set(ids)))
      self.assertTrue(bridge.has_pending_action())
      # Verify AdventureCliClient formatting and endpoints against localhost.
      cli = adventure_cli.AdventureCliClient(f'http://127.0.0.1:{port}')
      state_via_cli = cli.get_state()
      status_text = adventure_cli.format_status(state_via_cli)
      self.assertIn('The Salt-Glass Wharf', status_text)
      stream_text = adventure_cli.format_stream(state_via_cli)
      self.assertIn('unique_ids=True', stream_text)
      npcs_text = adventure_cli.format_npcs(state_via_cli)
      self.assertIn('Captain Mara Vance', npcs_text)
      map_text = adventure_cli.format_map(state_via_cli)
      self.assertIn('[PLAYER HERE]', map_text)
      logs_dict = cli.get_logs()
      self.assertIn('entries', logs_dict)
      debug_text = adventure_cli.format_debug(state_via_cli, logs_dict)
      self.assertIn('ENGINE TELEMETRY & LATENCY TRACKER', debug_text)
      self.assertIn('THREE-TIER NPC ROSTER', debug_text)
      self.assertIn('Captain Mara Vance', debug_text)
    finally:
      adv_server.shutdown()

    # Verify evaluate_hard_invariants on clean state.
    invariants = evaluate_run.evaluate_hard_invariants(
        telemetry={
            'frontier_cache_hit_rate': 1.0,
            'hibernations_count': 1,
            'rehydrations_count': 1,
            'warm_events_count': 2,
        },
        latency={
            'control_hook_llm_calls': 0,
            'max_llm_calls_per_player_resolve': 1,
        },
        entity_memories={'Traveler': ['Observed the harbour wharf.']},
    )
    self.assertTrue(invariants['all_passed'])

  def test_human_ux_sanitizers_and_starting_inventory(self):
    self.assertEqual(
        adventure_specific_components.dedup_leading_actor_name(
            'Ilsa Vance', 'Ilsa Vance Ilsa’s chalk-stained fingers freeze.'
        ),
        'Ilsa’s chalk-stained fingers freeze.',
    )
    self.assertEqual(
        adventure_specific_components.dedup_leading_actor_name(
            'Soren Vale', "Soren Vale Soren Vale's ink-stained hand moves."
        ),
        "Soren Vale's ink-stained hand moves.",
    )
    self.assertEqual(
        world_ledger.clean_affordance_text(
            'chalk tally board behind the bar displaying [c_ilsa_debt]'
        ),
        'chalk tally board behind the bar',
    )
    mock_model = CountingAdventureMockModel()
    sim = adventure_prefab.AdventureSimulation(
        model=mock_model,
        player_act_component=ScriptedPlayerActComponent([]),
        enable_async_npc_plan=False,
    )
    self.assertIn('unsealed inquiry writ', sim.ledger.player_inventory)
    self.assertIn("inspector's pocket ledger", sim.ledger.player_inventory)

  def test_stock_asynchronous_engine_type_and_dynamic_slots(self):
    mock_model = CountingAdventureMockModel()
    scripted_actions = [
        'inspect the padlocked customs shed door',
        'go north',
        'inspect rusted valve wheel 1',
    ]
    sim = adventure_prefab.AdventureSimulation(
        model=mock_model,
        player_act_component=ScriptedPlayerActComponent(scripted_actions),
        enable_async_npc_plan=False,
        max_hot_npcs=1,
        max_warm_npcs=1,
        warm_tick_interval=2,
    )
    self.assertIs(type(sim.engine), asynchronous.Asynchronous)
    entities = sim.all_engine_entities()
    self.assertIs(entities[0], sim.player_entity)
    unbound_before = [
        e.name for e in entities if e.name.startswith('__npc_slot_')
    ]
    self.assertNotEmpty(unbound_before)

    _ = sim.play(max_steps=3, verbose=False)
    self.assertEqual(sim.player_steps_completed, 3)

    # Procedural NPCs spawned by FrontierArchitectWorker bind to dynamic slots.
    bound_names = {e.name for e in sim.roster.engine_entities() if e.is_bound}
    self.assertIn('Warden 1', bound_names)

  def test_concurrent_putative_events_isolate_by_actor(self):
    mock_model = CountingAdventureMockModel()
    sim = adventure_prefab.AdventureSimulation(
        model=mock_model,
        player_act_component=ScriptedPlayerActComponent([]),
        enable_async_npc_plan=False,
    )
    tag = event_resolution.PUTATIVE_EVENT_TAG
    sim.game_master.observe(
        f'{tag} Captain Mara Vance: checks the brass seal carefully.'
    )
    sim.game_master.observe(f'{tag} Traveler: inspect the customs shed door')

    results: dict[str, str] = {}

    def _resolve_as(entity_name: str) -> None:
      tid = threading.current_thread().ident
      assert tid is not None
      sim.game_master.set_capture_key_for_thread(tid, entity_name)
      try:
        results[entity_name] = sim.game_master.act(
            action_spec=entity_lib.ActionSpec(
                call_to_action='Resolve',
                output_type=entity_lib.OutputType.RESOLVE,
            )
        )
      finally:
        sim.game_master.clear_capture_key_for_thread(tid)

    t1 = threading.Thread(target=_resolve_as, args=('Traveler',))
    t1.start()
    t1.join(timeout=5.0)
    t2 = threading.Thread(target=_resolve_as, args=('Captain Mara Vance',))
    t2.start()
    t2.join(timeout=5.0)

    self.assertEqual(sim.player_steps_completed, 1)
    self.assertIn('amber lantern light', results['Traveler'])
    self.assertIn('Captain Mara Vance', results['Captain Mara Vance'])
    self.assertIn('brass seal', results['Captain Mara Vance'])

  def test_warm_off_screen_turns_and_stop_under_asynchronous(self):
    mock_model = CountingAdventureMockModel()
    scripted_actions = [
        'inspect the padlocked customs shed door',
        'inspect the waterlogged manifest ledger',
        'inspect the coil of tarred hemp rope',
        'inspect the iron cargo cranes',
    ]
    sim = adventure_prefab.AdventureSimulation(
        model=mock_model,
        player_act_component=ScriptedPlayerActComponent(scripted_actions),
        enable_async_npc_plan=False,
        max_hot_npcs=1,
        max_warm_npcs=2,
        warm_tick_interval=2,
    )
    _ = sim.play(max_steps=4, verbose=False)
    self.assertEqual(sim.player_steps_completed, 4)
    self.assertTrue(sim.terminate_component.should_terminate())
    telemetry = sim.ledger.telemetry_summary()
    self.assertGreaterEqual(telemetry['warm_events_count'], 1)
    scheduler = sim.next_acting_component.warm_scheduler
    warm_actors = [
        str(e['actor'])
        for e in sim.ledger.get_narrative_log()
        if e.get('kind') == 'rumor' and e.get('actor')
    ]
    self.assertNotEmpty(warm_actors)
    self.assertIsNotNone(scheduler.get_timer(warm_actors[0]))

    # Verify stop() cleanly exits an idle interactive simulation loop.
    bridge = server.BrowserInputBridge(timeout_seconds=5.0)
    live_sim = adventure_prefab.AdventureSimulation(
        model=mock_model,
        player_act_component=human_act_component.HumanActComponent(
            input_reader=bridge
        ),
        player_ready_fn=bridge.has_pending_action,
        enable_async_npc_plan=False,
    )
    live_sim.terminate_component.set_max_player_steps(100)
    worker = threading.Thread(
        target=live_sim.engine.run_loop,
        kwargs={
            'game_masters': [live_sim.game_master],
            'entities': live_sim.all_engine_entities(),
            'premise': live_sim.ledger.premise,
            'max_steps': 500_000,
            'verbose': False,
        },
        daemon=True,
    )
    worker.start()
    time.sleep(0.1)
    self.assertTrue(worker.is_alive())
    live_sim.stop()
    bridge.clear(wake_blocked_reader=True)
    worker.join(timeout=3.0)
    self.assertFalse(worker.is_alive())

  def test_idiomatic_prefabs_components_checkpoint_and_dynamic_state(self):
    self.assertTrue(issubclass(adventure_prefab.NpcEntity, prefab_lib.Prefab))
    self.assertTrue(
        issubclass(adventure_prefab.PlayerEntity, prefab_lib.Prefab)
    )
    self.assertTrue(
        issubclass(adventure_prefab.AdventureGameMaster, prefab_lib.Prefab)
    )

    mock_model = CountingAdventureMockModel()
    scripted_actions = [
        'inspect the padlocked customs shed door',
        'go north',
    ]
    sim = adventure_prefab.AdventureSimulation(
        model=mock_model,
        player_act_component=ScriptedPlayerActComponent(scripted_actions),
        enable_async_npc_plan=False,
        max_hot_npcs=1,
        max_warm_npcs=1,
        warm_tick_interval=2,
    )
    self.assertIsInstance(sim, generic.Simulation)
    self.assertIsInstance(sim, simulation_lib.Simulation)
    self.assertIsInstance(
        sim.game_master.get_act_component(), switch_act.SwitchAct
    )

    # Verify WorldLedger, StoryDirector, RosterManager, EntityScheduler, and
    # FrontierArchitectPublisher are mounted on AdventureGM as components.
    gm = sim.game_master
    self.assertIs(
        gm.get_component(world_ledger.DEFAULT_WORLD_LEDGER_COMPONENT_KEY),
        sim.ledger,
    )
    self.assertIs(
        gm.get_component(story_director.DEFAULT_STORY_DIRECTOR_COMPONENT_KEY),
        sim.director,
    )
    self.assertIs(
        gm.get_component(roster_manager.DEFAULT_ROSTER_COMPONENT_KEY),
        sim.roster,
    )
    self.assertIsInstance(
        gm.get_component(interrupt_scheduling.DEFAULT_SCHEDULER_COMPONENT_KEY),
        interrupt_scheduling.EntityScheduler,
    )
    self.assertIsInstance(
        gm.get_component(
            adventure_specific_components.DEFAULT_FRONTIER_COMPONENT_KEY
        ),
        adventure_specific_components.FrontierArchitectPublisher,
    )

    # Run 2 steps so the player moves and state mutates.
    _ = sim.play(max_steps=2, verbose=False)
    self.assertEqual(sim.player_steps_completed, 2)
    loc_at_step_2 = sim.ledger.player_location_id
    clock_at_step_2 = sim.ledger.clock_minutes
    checkpoint = sim.make_checkpoint_data()
    self.assertIn('AdventureGM', checkpoint['game_masters'])
    self.assertIn('Traveler', checkpoint['entities'])

    # Test set_component_dynamic_state on WorldLedger mounted on AdventureGM.
    sim.set_component_dynamic_state(
        'AdventureGM',
        world_ledger.DEFAULT_WORLD_LEDGER_COMPONENT_KEY,
        'player_status',
        'Wounded',
    )
    self.assertEqual(sim.ledger.player_status, 'Wounded')
    sim.ledger.advance_clock(minutes=120)
    self.assertNotEqual(sim.ledger.clock_minutes, clock_at_step_2)

    # Restore from checkpoint and verify state reverts cleanly.
    sim.load_from_checkpoint(checkpoint)
    self.assertEqual(sim.ledger.player_location_id, loc_at_step_2)
    self.assertEqual(sim.ledger.clock_minutes, clock_at_step_2)
    self.assertEqual(sim.ledger.player_status, ' Focused and alert'.strip())

  def test_player_entity_mounts_submind_components_and_council_coordinator(
      self,
  ):
    mock_model = CountingAdventureMockModel()
    sim = adventure_prefab.AdventureSimulation(
        model=mock_model,
        player_act_component=ScriptedPlayerActComponent([]),
        enable_async_npc_plan=False,
    )
    council = sim.player_entity.get_component(
        adventure_specific_components.DEFAULT_SUBMIND_COUNCIL_COMPONENT_KEY
    )
    self.assertIsInstance(
        council, adventure_specific_components.SubMindCouncilCoordinator
    )
    self.assertIs(council, sim.submind_council)

    # Mint a custom Sub-Mind and verify its SubMindComponent is registered.
    minted = sim.mint_custom_sub_mind(
        concept=(
            'An ex-boxing promoter who treats every standoff like a fixed bout.'
        ),
        rank=3,
    )
    self.assertEqual(minted.origin, 'custom')
    self.assertEqual(minted.rank, 3)
    self.assertIn(minted.name, sim.ledger.story.sub_minds)
    self.assertIn(minted.name, council.submind_components)

    # Configure Parliament weights and verify character_initialized + options.
    sim.configure_player_parliament(
        active_ranks={minted.name: 4},
        custom_voices=[],
    )
    self.assertTrue(sim.ledger.story.character_initialized)
    self.assertTrue(sim.ledger.story.sub_minds[minted.name].intrusive)
    self.assertNotEmpty(sim.ledger.story.active_voice_options)
    self.assertEqual(
        sim.ledger.story.active_voice_options[0].voice, minted.name
    )

    # Parallel deliberation across active SubMindComponents produces proposals.
    proposals = council.convene_council(
        latest_event='A dock guard reaches for the brass whistle.',
        use_parallel_llm=True,
    )
    self.assertNotEmpty(proposals)
    self.assertEqual(proposals[0].voice, minted.name)

  def test_npc_submind_council_parallel_deliberation_and_social_combat(self):
    mock_model = CountingAdventureMockModel()
    sim = adventure_prefab.AdventureSimulation(
        model=mock_model,
        player_act_component=ScriptedPlayerActComponent([]),
        enable_async_npc_plan=False,
    )
    # 1. Verify every seeded NPC has 3 Sub-Minds and round-trips via to_dict.
    all_npcs = sim.ledger.all_npcs()
    self.assertNotEmpty(all_npcs)
    hot_npcs = sim.ledger.npcs_in_tier(world_ledger.NpcTier.HOT)
    self.assertNotEmpty(hot_npcs)
    npc = hot_npcs[0]
    self.assertIn('The Mask', npc.sub_minds)
    self.assertIn('The Angle', npc.sub_minds)
    self.assertIn('The Wound', npc.sub_minds)
    self.assertEqual(npc.dominant_sub_mind().domain, 'stonewalling')
    self.assertEqual(npc.dominant_sub_mind().name, "The Warden's Seal")
    initial_mask_rank = npc.sub_minds['The Mask'].rank

    restored = world_ledger.NpcRecord.from_dict(npc.to_dict())
    self.assertEqual(restored.last_winning_voice, npc.last_winning_voice)
    self.assertEqual(set(restored.sub_minds.keys()), set(npc.sub_minds.keys()))

    # 2. Verify live Hot NPC mounts NpcSubMindCouncilComponent and deliberates
    # in parallel across all 3 NpcSubMindComponents.
    live_npc = sim.roster.get_live_entity(npc.name)
    self.assertIsNotNone(live_npc)
    npc_council = live_npc.get_component(
        adventure_specific_components.DEFAULT_NPC_SUBMIND_COUNCIL_COMPONENT_KEY
    )
    self.assertIsInstance(
        npc_council, adventure_specific_components.NpcSubMindCouncilComponent
    )
    opts, winning = npc_council.convene_council(use_parallel_llm=True)
    self.assertLen(opts, 3)
    self.assertIsNotNone(winning)
    self.assertGreaterEqual(
        sim.latency_tracker.summary()['submind_parallel_calls_total'], 3
    )

    # 3. Verify deduplication of WorldLedger.record_player_action at same tick.
    e1 = sim.ledger.record_player_action(
        "[Copper's Nose] Confront " + npc.name + ' with the forged manifest'
    )
    e2 = sim.ledger.record_player_action(
        "[Copper's Nose] Confront " + npc.name + ' with the forged manifest'
    )
    self.assertEqual(e1.get('id'), e2.get('id'))

    # 4. Verify Voice-vs-Voice social combat in StoryDirector.on_delta cracks
    # The Mask and spikes The Wound when leverage/clues hit.
    sim.ledger.set_npc_addressed(npc.name, True)
    summary = sim.director.on_delta(
        {'clues_revealed': ['c_test'], 'npcs_addressed': [npc.name]},
        acting_entity=sim.ledger.player_name,
        narrative_event=(
            f'You slap the forged manifest down in front of {npc.name}, '
            'exposing their hidden secret.'
        ),
        active_location_id=sim.ledger.player_location_id,
    )
    self.assertIn('npc_voice_combat', summary)
    updated_npc = sim.ledger.get_npc(npc.name)
    assert updated_npc is not None
    self.assertLess(updated_npc.sub_minds['The Mask'].rank, initial_mask_rank)
    self.assertGreater(updated_npc.sub_minds['The Wound'].rank, 1)

    # 5. Verify Warm NPC off-screen move is driven by dominant Sub-Mind.
    warm_npcs = sim.ledger.npcs_in_tier(world_ledger.NpcTier.WARM)
    if warm_npcs:
      warm_npc = warm_npcs[0]
      move = sim.director.choose_warm_move(warm_npc)
      self.assertIn('Your dominant inner voice is', move.call_to_action)
      sim.director.apply_warm_outcome(
          warm_npc,
          move,
          'stashes the ledger in a dry locker',
          warm_npc.location_id,
          'Dockside Shed',
      )
      after_warm = sim.ledger.get_npc(warm_npc.name)
      assert after_warm is not None
      dom_after = after_warm.dominant_sub_mind()
      self.assertNotEmpty(dom_after.scars)

  def test_npc_dialogic_responsiveness_and_evaluator_caps(self):
    """Verifies NPC deliberation receives player prompts and evaluator catches loops."""
    model = CountingAdventureMockModel()
    sim = adventure_prefab.AdventureSimulation(
        model=model,
        player_act_component=ScriptedPlayerActComponent([]),
        enable_async_npc_plan=False,
    )
    hot_npcs = sim.ledger.npcs_in_tier(world_ledger.NpcTier.HOT)
    self.assertNotEmpty(hot_npcs)
    npc = hot_npcs[0]
    live_npc = sim.roster.get_live_entity(npc.name)
    assert live_npc is not None

    # 1. Individual NpcSubMindComponent.pre_act returns '' so ConcatActComponent
    # does not append duplicate/stale Sub-Mind lines outside component_order.
    npc_council = live_npc.get_component(
        adventure_specific_components.DEFAULT_NPC_SUBMIND_COUNCIL_COMPONENT_KEY
    )
    assert isinstance(
        npc_council, adventure_specific_components.NpcSubMindCouncilComponent
    )
    for sub_comp in npc_council.submind_components.values():
      self.assertEqual(
          sub_comp.pre_act(
              entity_lib.free_action_spec(call_to_action='What next?')
          ),
          '',
      )

    # 2. Empathy probing a wound/secret spikes The Wound rather than The Angle.
    first_voice = sim.mint_custom_sub_mind(
        concept='An empathetic listener attuned to exhaustion and hidden fear.',
        rank=3,
    )
    first_voice.domain = 'empathy'
    first_voice.role_archetype = 'social'
    sim.ledger.record_player_action(
        f"[{first_voice.name}] Ask why he looks like he hasn't slept in days."
    )
    sim.ledger.set_npc_addressed(npc.name, True)
    summary = sim.director.on_delta(
        {'clues_revealed': ['c_wound'], 'npcs_addressed': [npc.name]},
        acting_entity=sim.ledger.player_name,
        narrative_event=(
            f'You press {npc.name} about the dark hollows under his eyes and '
            'the missing three-day blank in his memory, exposing his secret.'
        ),
        active_location_id=sim.ledger.player_location_id,
    )
    combat_events = summary.get('npc_voice_combat', [])
    self.assertNotEmpty(combat_events)
    self.assertEqual(combat_events[0].get('outcome'), 'wound_spiked')
    updated_npc = sim.ledger.get_npc(npc.name)
    assert updated_npc is not None
    self.assertGreater(updated_npc.sub_minds['The Wound'].rank, 1)

    # 3. Evaluator pairs engine_step != tick and flags bargain_push / monopoly.
    fake_transcript = {
        'resolutions': (
            [
                {
                    'step': t * 2 - 1,
                    'engine_step': t * 100 - 10,
                    'actor': sim.ledger.player_name,
                    'location': sim.ledger.player_location_id,
                    'text': (
                        f'Traveler asks question {t} about the missing crew.'
                    ),
                    'tiers': {'hot': [npc.name], 'warm': [], 'cold': []},
                    'story_delta': {},
                }
                for t in range(1, 6)
            ]
            + [
                {
                    'step': t * 2,
                    'engine_step': t * 100,
                    'actor': npc.name,
                    'location': sim.ledger.player_location_id,
                    'text': (
                        f'{npc.name} slides the transfer voucher across the'
                        ' table and demands you clear his manifest in exchange'
                        ' for the key.'
                    ),
                    'tiers': {'hot': [npc.name], 'warm': [], 'cold': []},
                    'story_delta': {},
                }
                for t in range(1, 6)
            ]
        ),
        'npc_council_turns': [
            {
                'step': t * 100,
                'engine_step': t * 100,
                'npc': npc.name,
                'player_action': (
                    f'[Cabin Fever] Ask question {t} about the missing crew.'
                ),
                'winning_voice': 'The Angle',
                'options': [{
                    'voice': 'The Angle',
                    'domain': 'leverage',
                    'rank': 4,
                    'intrusive': True,
                    'action': (
                        'Slide the transfer voucher across the table and '
                        'demand manifest clearance.'
                    ),
                }],
            }
            for t in range(1, 6)
        ],
    }
    det = evaluate_run.compute_deterministic_metrics(
        ledger_dict=sim.ledger.to_dict(),
        transcript=fake_transcript,
    )
    self.assertIn(npc.name, det['beat_repetition']['flagged_npcs'])
    npc_parl = det['parliament']['npc']
    self.assertIsNotNone(npc_parl['winning_action_support_mean'])
    self.assertIn(npc.name, npc_parl['npcs_with_voice_monopoly'])
    self.assertGreater(npc_parl['stale_directive_rate'], 0.5)
    self.assertGreater(npc_parl['unresponsive_turn_rate'], 0.5)
    caps = evaluate_run.metric_caps(det, {'info_policy': 'player_visible_only'})
    self.assertIn('npc_parliament', caps)
    self.assertLessEqual(caps['npc_parliament'][0], 55)
    self.assertIn('npc_agency', caps)
    self.assertLessEqual(caps['npc_agency'][0], 55)

  def test_narrative_and_dialog_coherence_invariants(self):
    """Verifies fixes for ghost rooms, spliced prefixes, and truncated props."""
    model = CountingAdventureMockModel()
    sim = adventure_prefab.AdventureSimulation(
        model=model,
        player_act_component=ScriptedPlayerActComponent([]),
        enable_async_npc_plan=False,
    )
    # 1. Duplicate/ghost location deduplication on shorthand & compound names.
    sim.ledger.expand_location(
        'the_cargo_hangar_cold_locker',
        name='The Cargo Hangar & Cold Locker',
        description='Unheated hangar bay and cold storage.',
        affordances=['open supply crate'],
        exits={'back': sim.ledger.player_location_id},
    )
    self.assertEqual(
        sim.ledger.ensure_stub_location('The Cargo Hangar'),
        'the_cargo_hangar_cold_locker',
    )
    self.assertEqual(
        sim.ledger.ensure_stub_location('The Unheated Cargo Hangar'),
        'the_cargo_hangar_cold_locker',
    )

    # 2. Clean prop noun phrase extraction on verb and NPC-targeted affordances.
    extract = sim.director._extract_clean_prop_noun_phrase  # pylint: disable=protected-access
    self.assertEqual(extract('turn the brass valve'), 'the brass valve')
    self.assertEqual(
        extract('confront Cole Mercer over the red rubber coating'),
        'the red rubber coating',
    )
    self.assertEqual(
        extract('demand Dr. Lin produce the stopped watch'),
        'the stopped watch',
    )
    self.assertEqual(
        extract('the shorting auxiliary terminal spitting sulfurous sparks'),
        'the shorting auxiliary terminal',
    )
    self.assertEqual(
        extract("the humming bio-monitor displaying Jonas Reed's vitals"),
        'the humming bio-monitor',
    )
    self.assertEqual(
        extract('knock a rhythmic cadence on the iron bulkhead'),
        'the iron bulkhead',
    )
    self.assertEqual(
        extract('haul the third exposed lithium tank'),
        'the third exposed lithium tank',
    )

    # 2b. ManagedNpcEntity._clean_npc_act_output strips first-name duplicates,
    # 'Answer:' prefixes, stuttered verbs, and bare 2-word outputs.
    managed_slot = roster_manager.ManagedNpcEntity(
        sim.roster, 'Jonas Reed', is_dynamic_slot=False
    )
    self.assertEqual(
        managed_slot._clean_npc_act_output(  # pylint: disable=protected-access
            'Jonas Reed Jonas flinches as the south corridor hatch bangs open.'
        ),
        'Jonas Reed flinches as the south corridor hatch bangs open.',
    )
    self.assertEqual(
        managed_slot._clean_npc_act_output(  # pylint: disable=protected-access
            'Jonas Reed Answer: Jonas Reed inspects the cut lead marks.'
        ),
        'Jonas Reed inspects the cut lead marks.',
    )
    self.assertEqual(
        managed_slot._clean_npc_act_output(  # pylint: disable=protected-access
            'Jonas Reed slam slammed his heel against the alternator.'
        ),
        'Jonas Reed slammed his heel against the alternator.',
    )
    self.assertIn(
        'checks the nearest hatch seal',
        managed_slot._clean_npc_act_output('Jonas Reed'),  # pylint: disable=protected-access
    )

    # 3. _soften_repeated_npc_beats does not splice second-person imperative
    # prefixes onto non-repeating NPC turns.
    sim.ledger.record_player_action(
        '[Hard Line] Drive your forearm into his collarbone and demand what '
        'he hid in the locker.'
    )
    sim.scene_resolver._soften_repeated_npc_beats(  # pylint: disable=protected-access
        'Jonas Reed',
        'Jonas Reed coughs and points at the workbench. "Check the drawer."',
    )
    second_turn = sim.scene_resolver._soften_repeated_npc_beats(  # pylint: disable=protected-access
        'Jonas Reed',
        ' Jonas Reed steps back toward the stove. "The snowcat keys are gone."',
    )
    self.assertNotIn('Bristling as', second_turn)
    self.assertNotIn('drive your forearm', second_turn)

    # 4. Warm outcome scars never slice mid-word, and spillover affordances
    # replace prior spillovers rather than stacking indefinitely.
    hot_npc = sim.ledger.npcs_in_tier(world_ledger.NpcTier.HOT)[0]
    move = story_director.WarmMove(
        kind='confront',
        target_npc='Traveler',
        call_to_action='Act off-screen.',
    )
    sim.director.apply_warm_outcome(
        hot_npc,
        move,
        'drives the heavy bolt cutters into the mess hall pneumatic manifold',
        sim.ledger.player_location_id,
        'The Outpost 7 Mess Hall',
    )
    updated = sim.ledger.get_npc(hot_npc.name)
    assert updated is not None
    latest_scar = updated.dominant_sub_mind().scars[-1]
    self.assertNotIn('manif]', latest_scar)
    self.assertFalse(latest_scar.endswith('manif'))

    loc_id = sim.ledger.player_location_id
    sim.ledger.add_location_affordance(
        loc_id, 'physical spillover from Shack (Jonas Reed cuts the wire)'
    )
    sim.ledger.add_location_affordance(
        loc_id, 'physical spillover from Hangar (Cole Mercer bolts the hatch)'
    )
    node = sim.ledger.get_location(loc_id)
    assert node is not None
    spillovers = [
        a
        for a in node.affordances
        if a.lower().startswith('physical spillover from ')
    ]
    self.assertLen(spillovers, 1)

  def test_server_tracks_resolving_player_and_responding_actors(self):
    """Verifies state payload exposes resolving_player and responding_actors."""
    model = CountingAdventureMockModel()
    bridge = server.BrowserInputBridge(timeout_seconds=1.0)
    sim = adventure_prefab.AdventureSimulation(
        model=model,
        player_act_component=ScriptedPlayerActComponent([]),
        player_ready_fn=bridge.has_pending_action,
        enable_async_npc_plan=False,
    )
    srv = server.AdventureServer(sim, bridge, host='127.0.0.1', port=0)
    try:
      hot_npcs = sim.ledger.npcs_in_tier(world_ledger.NpcTier.HOT)
      self.assertNotEmpty(hot_npcs)
      hot_name = hot_npcs[0].name

      init_state = srv.build_state_payload()
      self.assertFalse(init_state['resolving_player'])
      self.assertEqual(init_state['responding_actors'], [])

      bridge.submit_action('Ask about the ledger')
      queued_state = srv.build_state_payload()
      self.assertTrue(queued_state['resolving_player'])
      self.assertEqual(queued_state['responding_actors'], [])
      self.assertTrue(queued_state['turn_busy'])

      bridge.clear()
      sim.ledger.apply_delta(
          {'advance_minutes': 2},
          acting_entity=sim.ledger.player_name,
          narrative_event='You question the room.',
      )
      sim.next_acting_component.notify_player_acted()
      mid_turn_state = srv.build_state_payload()
      self.assertFalse(mid_turn_state['resolving_player'])
      self.assertEqual(mid_turn_state['responding_actors'], [hot_name])
      self.assertTrue(mid_turn_state['turn_busy'])
    finally:
      srv.shutdown()

  def test_concise_prose_mode_headline_and_delta_chips(self):
    """Verifies concise prose mode toggle, outcome headlines, and delta chips."""
    model = CountingAdventureMockModel()
    bridge = server.BrowserInputBridge(timeout_seconds=1.0)
    sim = adventure_prefab.AdventureSimulation(
        model=model,
        player_act_component=ScriptedPlayerActComponent([]),
        player_ready_fn=bridge.has_pending_action,
        enable_async_npc_plan=False,
    )
    srv = server.AdventureServer(sim, bridge, host='127.0.0.1', port=0)
    try:
      # Server defaults to concise mode and strips redundant intro lists.
      self.assertEqual(sim.ledger.prose_mode, 'concise')
      init_log = sim.ledger.get_narrative_log()
      self.assertNotEmpty(init_log)
      self.assertNotIn('Around you, you spot:', init_log[0]['text'])

      # Toggling to literary and back works via set_prose_mode.
      lit_state = srv.set_prose_mode('literary')
      self.assertEqual(lit_state['ledger']['prose_mode'], 'literary')
      con_state = srv.set_prose_mode('concise')
      self.assertEqual(con_state['ledger']['prose_mode'], 'concise')

      # Applying a delta attaches headline + chips to the resolution entry.
      first_clue_id = next(iter(sim.ledger.story.clues.keys()), '')
      sim.ledger.apply_delta(
          {
              'outcome_headline': 'Padlock forced — layoff ledger uncovered',
              'add_inventory': ['Brass Skeleton Key'],
              'add_affordances': ['pried-open iron lockbox'],
              'clues_revealed': [first_clue_id] if first_clue_id else [],
          },
          acting_entity=sim.ledger.player_name,
          narrative_event=(
              'You snap the rusted hasp with the tyre lever and pull out the'
              ' hidden carbon ledger.'
          ),
      )
      log_after = sim.ledger.get_narrative_log()
      res_entries = [
          e for e in log_after if e.get('kind') == 'scene_resolution'
      ]
      self.assertNotEmpty(res_entries)
      latest_res = res_entries[-1]
      self.assertEqual(
          latest_res.get('headline'),
          'Padlock forced — layoff ledger uncovered',
      )
      chips = latest_res.get('chips', [])
      self.assertNotEmpty(chips)
      chip_kinds = {c.get('kind') for c in chips}
      self.assertIn('item', chip_kinds)
      self.assertIn('prop', chip_kinds)

      # Synthetic 'Key specifics noted' entries are suppressed in concise mode.
      for entry in log_after:
        self.assertFalse(
            str(entry.get('text', '')).startswith('Key specifics noted (')
        )
    finally:
      srv.shutdown()

  def test_browser_input_bridge_isolates_sessions_across_resets(self):
    """Verifies old session readers cannot steal actions after session reset."""
    bridge = server.BrowserInputBridge(timeout_seconds=2.0)
    reader_1, ready_1 = bridge.bind_new_session()
    dummy_req = server.human_input.HumanInputRequest(
        request_id='req-1',
        entity_name='Traveler',
        action_spec=entity_lib.DEFAULT_ACTION_SPEC,
        contexts={},
    )
    old_reader_out: list[str] = []
    old_thread = threading.Thread(
        target=lambda: old_reader_out.append(reader_1(dummy_req)),
        daemon=True,
    )
    old_thread.start()
    time.sleep(0.05)

    # Binding a new session wakes the blocked old reader with '' and isolates
    # the new session queue from any late calls on reader_1.
    reader_2, ready_2 = bridge.bind_new_session()
    old_thread.join(timeout=1.0)
    self.assertFalse(old_thread.is_alive())
    self.assertEqual(old_reader_out, [''])

    bridge.submit_action('Inspect the iron valve')
    self.assertFalse(ready_1())
    self.assertTrue(ready_2())
    # Even a late call to reader_1 must return '' without touching reader_2's
    # queued action.
    self.assertEqual(reader_1(dummy_req), '')
    self.assertTrue(ready_2())
    self.assertEqual(reader_2(dummy_req), 'Inspect the iron valve')
    self.assertFalse(ready_2())


if __name__ == '__main__':
  absltest.main()
