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

from collections.abc import Collection
import dataclasses
import json
import tempfile
from typing import override

from absl.testing import absltest
from concordia.components.agent import human_act_component
from examples.adventure import adventure_prefab
from examples.adventure import author_world
from examples.adventure import evaluate_run
from examples.adventure import runner
from examples.adventure import server
from examples.adventure import story_director
from examples.adventure import story_state
from examples.adventure import world_ledger
from concordia.language_model import language_model


def _load_glassharbor() -> story_state.WorldBible:
  return story_state.load_shipped_bible('glassharbor')


def _load_bermondsey() -> story_state.WorldBible:
  return story_state.load_shipped_bible('bermondsey')


class _SilentModel(language_model.LanguageModel):
  """Model that records calls; used to verify zero-LLM director operations."""

  def __init__(self):
    super().__init__()
    self.calls = 0

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
    del prompt, max_tokens, terminators, temperature, top_p, top_k, timeout
    del seed
    self.calls += 1
    return '{}'

  @override
  def sample_choice(self, prompt, responses, *, seed=None):
    del prompt, seed
    return 0, responses[0], {}


def _make_sim_with_bible(
    model: language_model.LanguageModel | None = None,
    bible: story_state.WorldBible | None = None,
) -> adventure_prefab.AdventureSimulation:
  return adventure_prefab.AdventureSimulation(
      model=model or _SilentModel(),
      player_act_component=human_act_component.HumanActComponent(
          input_reader=server.BrowserInputBridge(timeout_seconds=0.1)
      ),
      enable_async_npc_plan=False,
      bible=bible or _load_glassharbor(),
  )


class StoryDirectorUnitTest(absltest.TestCase):

  def test_selector_is_deterministic_and_respects_once(self):
    sim1 = _make_sim_with_bible()
    sim2 = _make_sim_with_bible()
    d1 = story_director.StoryDirector(sim1.ledger, rng_seed=42)
    d2 = story_director.StoryDirector(sim2.ledger, rng_seed=42)

    picks1 = [s.id for s in d1.pick('scene', k=2)]
    picks2 = [s.id for s in d2.pick('scene', k=2)]
    self.assertEqual(picks1, picks2)
    self.assertIn('s_warden_tests_writ', picks1)

    # Firing a `once=True` storylet removes it from future eligibility.
    writ_storylet = sim1.ledger.story.storylets['s_warden_tests_writ']
    d1.fire_storylet(writ_storylet)
    remaining_ids = [s.id for s in d1.eligible('scene')]
    self.assertNotIn('s_warden_tests_writ', remaining_ids)

  def test_notes_for_scene_includes_tone_clues_secrets_and_voices(self):
    sim = _make_sim_with_bible()
    ctx = sim.ledger.build_scene_context()
    self.assertIn('Director Notes', ctx)
    self.assertIn('Tone (noir)', ctx)
    self.assertIn('c_manifest', ctx)
    self.assertIn('c_seal_replaced', ctx)
    self.assertIn('Captain Mara Vance [wardens]: secret=', ctx)
    self.assertIn('Inner Voices', ctx)
    self.assertIn('Faction Dispositions toward Player', ctx)
    self.assertIn('Wary neutrality', ctx)

  def test_on_delta_reveals_clues_completes_truth_and_records_consequence(self):
    sim = _make_sim_with_bible()
    applied = sim.ledger.apply_delta(
        {
            'advance_minutes': 5,
            'beat_used': 's_warden_tests_writ',
            'clues_revealed': ['c_manifest', 'c_ilsa_debt'],
            'faction_shifts': {'wardens': -1},
            'tension_delta': 1.5,
            'inner_voice': 'Ledger: Nobody signs out lamp fittings at 2 AM.',
        },
        acting_entity=sim.ledger.player_name,
        narrative_event='I read the chained manifest ledger on the bollard.',
    )
    story_summary = applied['story']
    self.assertEqual(story_summary['beat_used'], 's_warden_tests_writ')
    self.assertCountEqual(
        story_summary['clues_revealed'], ['c_manifest', 'c_ilsa_debt']
    )
    self.assertIn('t_manifest', story_summary['truths_completed'])
    self.assertTrue(sim.ledger.story.truths['t_manifest'].discovered)
    self.assertEqual(sim.ledger.story.act, 2)
    self.assertEqual(
        sim.ledger.story.factions['wardens'].standing_with_player, -1
    )

    # Legible consequence added to the room's recent_events.
    wharf = sim.ledger.get_player_location()
    self.assertTrue(
        any(
            'Harbor Wardens standing cooled' in ev for ev in wharf.recent_events
        )
    )

    # Inner voice and revelation entries recorded in the narrative log.
    kinds = [e['kind'] for e in sim.ledger.to_dict()['narrative_log']]
    self.assertIn('inner_voice', kinds)
    self.assertIn('revelation', kinds)

    # Natural-language faction_shifts work cleanly alongside an invoked voice.
    sim.ledger.record_player_action(
        '[Ledger] Cross-check the compact tally board against the manifest.'
    )
    sim.ledger.apply_delta(
        {
            'advance_minutes': 5,
            'faction_shifts': {
                'compact': (
                    'Rattled and exposed by your audit of the tally board.'
                )
            },
        },
        acting_entity=sim.ledger.player_name,
        narrative_event='You expose the discrepancy on the tally board.',
    )
    self.assertIn(
        'Rattled and exposed',
        sim.ledger.story.factions['compact'].current_disposition(),
    )

  def test_warm_moves_cover_storylet_confront_grudge_and_mutates_both_npcs(
      self,
  ):
    sim = _make_sim_with_bible()
    director = sim.director

    # 1. Eligible `warm` storylet fires first for Soren Vale, producing a
    #    distorted rumor and adding drag marks to the boathouse.
    soren = sim.ledger.get_npc('Soren Vale')
    assert soren is not None
    move1 = director.choose_warm_move(soren)
    self.assertEqual(move1.kind, 'storylet')
    self.assertIsNotNone(move1.storylet)
    rumor1 = director.apply_warm_outcome(
        soren,
        move1,
        'hauls the valve under oilcloth',
        soren.location_id,
        'The Amber Lantern Tavern',
    )
    self.assertIn('oilcloth', rumor1)
    boathouse = sim.ledger.get_location('the_compact_boathouse')
    assert boathouse is not None
    self.assertTrue(any('drag marks' in a for a in boathouse.affordances))
    # Check rumor reliability tag in ledger.rumors.
    self.assertTrue(
        any('distorted rumor' in r for r in sim.ledger.to_dict()['rumors'])
    )

    # 2. Drain remaining warm storylets so we hit `confront` (NPC-vs-NPC).
    for ws in list(director.eligible('warm', npc_name='Soren Vale')):
      director.fire_storylet(ws)
    soren = sim.ledger.get_npc('Soren Vale')
    assert soren is not None
    move2 = director.choose_warm_move(soren)
    self.assertEqual(move2.kind, 'confront')
    target_name = move2.target_npc
    self.assertIn(target_name, ('Captain Mara Vance', 'Archivist Orin'))
    director.apply_warm_outcome(
        soren,
        move2,
        'bribes a clerk to undercut the watch',
        soren.location_id,
        'The Amber Lantern Tavern',
    )
    updated_soren = sim.ledger.get_npc('Soren Vale')
    updated_target = sim.ledger.get_npc(target_name)
    assert updated_soren is not None and updated_target is not None
    self.assertIn(
        'Moved against', updated_soren.relationships.get(target_name, '')
    )
    self.assertIn(
        'Heard that Soren Vale acted',
        updated_target.relationships.get('Soren Vale', ''),
    )

    # 3. Make the player an enemy of the Compact -> `grudge` move lays a trap.
    sim.ledger.story.shift_faction('compact', -3)
    soren = sim.ledger.get_npc('Soren Vale')
    assert soren is not None
    move3 = director.choose_warm_move(soren)
    self.assertEqual(move3.kind, 'grudge')
    director.apply_warm_outcome(
        soren,
        move3,
        'sets a tripwire by the back stairs',
        soren.location_id,
        'The Amber Lantern Tavern',
    )
    tavern = sim.ledger.get_location(soren.location_id)
    assert tavern is not None
    self.assertTrue(
        any('reprisal prepared by Soren Vale' in a for a in tavern.affordances)
    )

  def test_cold_routines_move_cold_npcs_by_hour_without_touching_hot(self):
    sim = _make_sim_with_bible()
    # At 08:00 AM (start), Ilsa Vance (Cold) is at The Amber Lantern Tavern.
    ilsa = sim.ledger.get_npc('Ilsa Vance')
    assert ilsa is not None
    self.assertEqual(ilsa.tier, world_ledger.NpcTier.COLD)
    self.assertEqual(ilsa.location_id, 'the_amber_lantern_tavern')

    # Advance clock by 9 hours -> 17:00 (5 PM). Ilsa's 16-24 routine is
    # The Compact Boathouse. Captain Mara Vance is Hot at the wharf and must
    # NOT be auto-moved even though her 14-22 routine is the Customs House.
    sim.ledger.advance_clock(minutes=9 * 60, increment_tick=True)
    moved = sim.director.step_cold_routines()
    moved_names = [m[0] for m in moved]
    self.assertIn('Ilsa Vance', moved_names)
    self.assertNotIn('Captain Mara Vance', moved_names)
    ilsa_after = sim.ledger.get_npc('Ilsa Vance')
    assert ilsa_after is not None
    self.assertEqual(ilsa_after.location_id, 'the_compact_boathouse')

  def test_rehydration_injects_relationship_updates_and_rehydrate_beat(self):
    sim = _make_sim_with_bible()
    # Record a relationship change on Cold NPC Ilsa Vance, then rehydrate her.
    sim.ledger.set_npc_relationship(
        'Ilsa Vance', 'Soren Vale', 'Caught Soren skimming from the tally.'
    )
    entity = sim.roster.rehydrate_from_cold('Ilsa Vance')
    self.assertIsNotNone(entity)
    mems = sim.roster.collect_all_entity_memories(sim.player_entity)[
        'Ilsa Vance'
    ]
    joined = ' '.join(mems)
    self.assertIn('Caught Soren skimming', joined)

    # Also test a rehydrate storylet firing for Captain Mara Vance after she
    # hibernates and rehydrates.
    sim.roster.hibernate_to_cold('Captain Mara Vance')
    sim.roster.rehydrate_from_cold('Captain Mara Vance')
    self.assertGreaterEqual(sim.ledger.story.fired.get('r_vance_recalls', 0), 1)

  def test_frontier_expansion_places_clues_and_enriches_npc(self):
    mock_model = runner.RichLocalMockModel()
    sim = _make_sim_with_bible(model=mock_model)
    expanded = sim.frontier_worker.expand_all_adjacent()
    self.assertNotEmpty(expanded)
    gearworks = sim.ledger.get_location('the_floodgate_gearworks')
    assert gearworks is not None
    self.assertTrue(gearworks.expanded)
    # Clues hinted for The Floodgate Gearworks (`c_valve_scratches`,
    # `c_bypass_wheel`) should be placed on the node.
    self.assertContainsSubset(
        ['c_valve_scratches', 'c_bypass_wheel'], gearworks.clue_ids
    )
    self.assertNotEmpty(gearworks.faction_lean)

  def test_doom_clock_stages_and_hard_vs_soft_ending(self):
    sim = _make_sim_with_bible()
    self.assertEqual(sim.ledger.story.doom_stage, 0)
    # Advance 25% of 960m = 240m -> stage 1 ("the gate groans"),
    # fires d_stage1_bells.
    sim.ledger.advance_clock(minutes=240, increment_tick=True)
    self.assertEqual(sim.ledger.story.doom_stage, 1)
    self.assertEqual(sim.ledger.story.fired.get('d_stage1_bells'), 1)

    # Advance to 100% -> stage 4 ("the king-tide"); with doom_hard=True,
    # run ends.
    sim.ledger.advance_clock(minutes=720, increment_tick=True)
    self.assertEqual(sim.ledger.story.doom_stage, 4)
    self.assertTrue(sim.ledger.story.ended)
    self.assertIn('FAILURE', sim.ledger.story.ending_summary)

    # With doom_hard=False and no ending storylet, stage progression does not
    # force-end the run by itself if ending storylets are removed.
    bible = _load_glassharbor()
    soft_bible = dataclasses.replace(
        bible,
        tone=dataclasses.replace(bible.tone, doom_hard=False),
        storylets=[s for s in bible.storylets if s.id != 'd_stage4_kingtide'],
    )
    soft_sim = _make_sim_with_bible(bible=soft_bible)
    soft_sim.ledger.advance_clock(minutes=960, increment_tick=True)
    self.assertEqual(soft_sim.ledger.story.doom_stage, 4)
    self.assertFalse(soft_sim.ledger.story.ended)

  def test_bermondsey_warm_moves_relocate_offscreen_clues_and_ignite_turf_wars(
      self,
  ):
    bible = _load_bermondsey()
    sim = _make_sim_with_bible(bible=bible)
    director = sim.director

    # Expand two off-screen rooms and place movable clue c_chiller_manifest in
    # Spitalfields Meat Lockup 9.
    sim.ledger.expand_location(
        'Spitalfields Meat Lockup 9',
        description='Frost-coated meat hooks and Bedford chiller van.',
        clue_ids=['c_chiller_manifest'],
        faction_lean='flying_squad',
    )
    sim.ledger.expand_location(
        'Rotherhithe Scrap & Breakers',
        description='Hydraulic baler and locked shipping container.',
        faction_lean='yardies',
    )

    # 1. Storylet `w_desmond_relocates_manifest` fires for Desmond, relocating
    #    `c_chiller_manifest` via `move_clue` and shifting inter-faction
    #    standing via `inter_faction_shift`.
    desmond = sim.ledger.get_npc("Desmond 'Nine-Pint' Baptiste")
    assert desmond is not None
    lift_storylet = sim.ledger.story.storylets['w_desmond_relocates_manifest']
    self.assertIn(
        lift_storylet, director.eligible('warm', npc_name=desmond.name)
    )
    move = story_director.WarmMove(
        kind='storylet',
        call_to_action=lift_storylet.beat,
        storylet=lift_storylet,
    )
    rumor = director.apply_warm_outcome(
        desmond,
        move,
        'lifts the chiller-van delivery docket and bolts toward Rotherhithe',
        'spitalfields_meat_lockup_9',
        'Rotherhithe Scrap & Breakers',
    )
    self.assertIn('chiller-van delivery docket', rumor)
    self.assertEqual(
        sim.ledger.story.clues['c_chiller_manifest'].placed_at,
        'rotherhithe_scrap_breakers',
    )
    self.assertEqual(
        sim.ledger.story.inter_faction_standing('yardies', 'flying_squad'), -3
    )

    # 2. A subsequent cross-faction `confront` move steals the relocated clue
    #    off-screen and pushes yardies<->flying_squad into an active turf war
    #    (<= -4).
    finch = sim.ledger.get_npc('Det. Insp. Ronald Finch')
    assert finch is not None
    confront_move = story_director.WarmMove(
        kind='confront',
        call_to_action=(
            'Raid Rotherhithe Scrap & Breakers and seize the chiller manifest.'
        ),
        target_npc=desmond.name,
    )
    rumor2 = director.apply_warm_outcome(
        finch,
        confront_move,
        'kicks in the portakabin door',
        'spitalfields_meat_lockup_9',
        'Spitalfields Meat Lockup 9',
    )
    self.assertIn('[Lifted:', rumor2)
    self.assertIn('[Turf war:', rumor2)
    self.assertEqual(
        sim.ledger.story.clues['c_chiller_manifest'].placed_at,
        'spitalfields_meat_lockup_9',
    )
    self.assertNotEmpty(sim.ledger.story.active_turf_wars())
    scene_notes = director.notes_for_scene()
    self.assertIn('Active Inter-Faction Turf Wars', scene_notes)

  def test_frontier_ask_includes_setting_style_room_palette_and_archetypes(
      self,
  ):
    sim = _make_sim_with_bible(bible=_load_bermondsey())
    stub = sim.ledger.get_location('Rotherhithe Scrap & Breakers')
    assert stub is not None
    ask = sim.director.notes_for_frontier(stub, 'the_half_moon_snooker_social')
    self.assertIn('Setting & Aesthetic:', ask.prompt_block)
    self.assertIn('Architectural / Set-Piece Motif:', ask.prompt_block)
    self.assertIn('Resident NPC Archetype:', ask.prompt_block)
    self.assertIn('South London underworld caper', ask.prompt_block)

  def test_hrl_submind_voice_options_affinity_and_multi_voice_chorus(self):
    sim = _make_sim_with_bible(bible=_load_bermondsey())
    self.assertLen(sim.ledger.story.player_voices, 7)
    self.assertIn("Copper's Nose", sim.ledger.story.player_voices)
    self.assertIn("The Guv'nor", sim.ledger.story.player_voices)
    # Seeded at Tick 0 with 4 room-grounded HRL options.
    self.assertLen(sim.ledger.story.active_voice_options, 4)

    # Acting via a sub-mind option bumps its dominance/affinity level.
    sim.ledger.record_player_action(
        "[The Book] Pull the chained layoff pad out from under Frankie's elbow."
    )
    self.assertEqual(sim.ledger.story.voice_affinity['The Book'], 2)

    # Resolving a turn with multi-voice chorus and fresh voice_options updates
    # both the narrative log and active_voice_options.
    applied = sim.ledger.apply_delta(
        {
            'advance_minutes': 4,
            'inner_voices': [
                "The Book: Sixty grand shy on Tuesday's traps.",
                'Proper Naughty: His eyes just flicked toward the cue case.',
            ],
            'voice_options': [
                {
                    'voice': 'The Book',
                    'perception': (
                        'The carbon sheet proves Frankie is £60k down.'
                    ),
                    'action': 'Pin Frankie to the bar with the carbon figures.',
                },
                {
                    'voice': 'Proper Naughty',
                    'perception': (
                        'The cracked cue case under Table 3 is heavy.'
                    ),
                    'action': 'Unlatch the cracked leather cue case now.',
                },
            ],
        },
        acting_entity=sim.ledger.player_name,
        narrative_event='You slide the blue carbon sheet into the light.',
    )
    self.assertLen(applied['story']['inner_voices'], 2)
    self.assertLen(sim.ledger.story.active_voice_options, 4)
    self.assertEqual(sim.ledger.story.active_voice_options[0].voice, 'The Book')
    self.assertEqual(sim.ledger.story.active_voice_options[0].affinity, 2)
    self.assertEqual(
        sim.ledger.story.active_voice_options[0].action,
        'Pin Frankie to the bar with the carbon figures.',
    )

    # An immediately following Hot NPC turn without voice_options must NOT
    # overwrite the player's fresh options back to static Tick-0 fallbacks.
    sim.ledger.apply_delta(
        {'advance_minutes': 5},
        acting_entity="Frankie 'The Moat' Slater",
        narrative_event='Frankie slams his palm over the layoff pad.',
    )
    self.assertEqual(
        sim.ledger.story.active_voice_options[0].action,
        'Pin Frankie to the bar with the carbon figures.',
    )

    # A subsequent player turn that omits LLM voice_options still regenerates
    # new event-reactive options rather than repeating the previous turn's.
    prev_actions = [opt.action for opt in sim.ledger.story.active_voice_options]
    sim.ledger.apply_delta(
        {'advance_minutes': 5},
        acting_entity=sim.ledger.player_name,
        narrative_event=(
            'Frankie rips the carbon sheet free and backs toward the yard.'
        ),
    )
    next_actions = [opt.action for opt in sim.ledger.story.active_voice_options]
    self.assertNotEqual(prev_actions, next_actions)

  def test_submind_mechanical_leverage_scars_intrusive_and_emergent_spawning(
      self,
  ):
    sim = _make_sim_with_bible(bible=_load_bermondsey())
    story = sim.ledger.story

    # 1. Configure Brass Neck to Rank 4 (Intrusive) and act with it.
    story.configure_parliament(
        active_ranks={
            'Brass Neck': 4,
            'The Book': 2,
            "Copper's Nose": 1,
            'Proper Naughty': 1,
        }
    )
    sim.ledger.record_player_action(
        '[Brass Neck] Slam the cue across the baize and demand the keys.'
    )
    scene_notes = sim.director.notes_for_scene()
    self.assertIn('Active Player Sub-Mind Channel (Brass Neck', scene_notes)
    self.assertIn('HIGH MECHANICAL LEVERAGE', scene_notes)
    self.assertIn('OVER-DOMINANCE BLIND SPOT', scene_notes)

    # 2. Intrusive voice crowds the council down to 3 voices and leads the list.
    selected = sim.director.select_active_voices()
    self.assertLen(selected, 3)
    self.assertEqual(selected[0][0], 'Brass Neck')
    refreshed_opts = sim.configure_player_parliament(
        active_ranks={
            'Brass Neck': 4,
            'The Book': 2,
            "Copper's Nose": 1,
            'Proper Naughty': 1,
        }
    )
    for opt in refreshed_opts:
      self.assertFalse(
          opt.action.lower().startswith('go '),
          f'Voice option should never be a movement command: {opt.action}',
      )

    # 3. Backfiring delta records a formative scar + obsession on Brass Neck.
    sim.ledger.apply_delta(
        {
            'advance_minutes': 5,
            'tension_shift': 2.0,
            'faction_shifts': {'corrigan_firm': -1},
        },
        acting_entity=sim.ledger.player_name,
        narrative_event="Corrigan's minders cock the hammer of a revolver.",
    )
    brass = story.get_sub_mind('Brass Neck')
    assert brass is not None
    self.assertNotEmpty(brass.scars)
    self.assertIn('Burned:', brass.scars[-1])
    self.assertIn('Settle the score at', brass.obsession)

    # 4. Acting off-piste with a manual prompt spawns an Emergent Sub-Mind.
    sim.ledger.record_player_action(
        'Hold an occult séance to summon the ghost of Shergar at Table 3.'
    )
    sim.ledger.apply_delta(
        {'advance_minutes': 5},
        acting_entity=sim.ledger.player_name,
        narrative_event=(
            'You chalk a pentagram across the baize as Frankie stares.'
        ),
    )
    emergent = [
        sm for sm in story.sub_minds.values() if sm.origin == 'emergent'
    ]
    self.assertNotEmpty(emergent)
    self.assertEqual(emergent[0].name, 'The Third Eye')
    self.assertTrue(emergent[0].active)
    self.assertTrue(any('New Sub-Mind awakened:' in j for j in story.journal))

  def test_emergent_act3_endings_and_modular_epilogue_slides(self):
    bible = _load_bermondsey()
    self.assertEmpty([s for s in bible.storylets if s.trigger == 'ending'])

    def _make_act3_sim():
      sim = _make_sim_with_bible(bible=bible)
      for cid in (
          'c_layoff_carbon',
          'c_frankie_pager',
          'c_van_key_wax',
          'c_cue_case_shells',
          'c_chiller_manifest',
          'c_gun_oil_rag',
      ):
        sim.ledger.story.mark_clue_discovered(cid, tick=10)
      sim.ledger.story.act = 3
      return sim

    # 1. Syndicate Triumph: high governor standing + recovered contraband.
    sim_syn = _make_act3_sim()
    sim_syn.ledger.story.factions['corrigan_firm'].standing_with_player = 3
    sim_syn.ledger.player_inventory.extend([
        'Engraved presentation pistols',
        'Prize greyhound Duchess',
    ])
    self.assertTrue(sim_syn.director.is_climax_ready('corrigans_snug'))
    sim_syn.ledger.record_player_action(
        "[The Guv'nor's Shadow] Deliver the presentation pistols and Duchess"
        ' to settle the debt.'
    )
    sim_syn.ledger.apply_delta(
        {'advance_minutes': 5},
        acting_entity=sim_syn.ledger.player_name,
        narrative_event=(
            "You drop the pigskin holdall onto Corrigan's mahogany desk."
        ),
    )
    self.assertTrue(sim_syn.ledger.story.ended)
    self.assertEqual(sim_syn.ledger.story.ending_archetype, 'syndicate_triumph')
    self.assertLen(sim_syn.ledger.story.ending_slides, 3)
    self.assertEqual(
        [s['kind'] for s in sim_syn.ledger.story.ending_slides],
        ['factions', 'npc_fates', 'parliament_coda'],
    )
    self.assertNotEmpty(sim_syn.ledger.story.ending_slides[1]['entries'])

    # 2. Double-Cross Sting: high rival standing + exposed truths.
    sim_sting = _make_act3_sim()
    for cid in ('c_jersey_sort_code', 'c_rover_radio_log', 'c_scorched_folio'):
      sim_sting.ledger.story.mark_clue_discovered(cid, tick=15)
    sim_sting.ledger.story.factions['hatton_brokers'].standing_with_player = 3
    arch, title, beat = sim_sting.director.classify_emergent_ending(
        'corrigans_snug'
    )
    self.assertEqual(arch, 'double_cross_sting')
    self.assertIn('Hatton Garden', title)
    self.assertNotEmpty(beat)

    # 3. Whistleblower Reckoning: Copper's Nose dominant + 3+ truths + page 14.
    sim_whistle = _make_act3_sim()
    for cid in ('c_jersey_sort_code', 'c_rover_radio_log', 'c_scorched_folio'):
      sim_whistle.ledger.story.mark_clue_discovered(cid, tick=15)
    sim_whistle.ledger.story.configure_parliament(
        active_ranks={"Copper's Nose": 4, 'The Book': 3}
    )
    sim_whistle.ledger.player_inventory.append("Page 14 of the Butcher's Bill")
    arch_w, _, _ = sim_whistle.director.classify_emergent_ending(
        'the_blind_beggar_snooker_hall'
    )
    self.assertEqual(arch_w, 'whistleblower_reckoning')

    # 4. Scorched-Earth Coup: cracked NPC wounds + active turf war +
    # Proper Naughty.
    sim_coup = _make_act3_sim()
    sim_coup.ledger.story.configure_parliament(
        active_ranks={'Proper Naughty': 4, 'Brass Neck': 3}
    )
    sim_coup.ledger.story.shift_inter_faction('corrigan_firm', 'yardies', -5)
    wound_sm = sim_coup.ledger.apply_npc_voice_clash(
        "Frankie 'The Moat' Slater",
        wound_delta=2,
        scar_target='The Wound',
        scar_note='Cracked at Tick 18 (Proper Naughty)',
    )
    assert wound_sm is not None
    sim_coup.director.resolve_emergent_climax(
        'the_blind_beggar_snooker_hall',
        trigger_reason='player_climax_action',
    )
    self.assertTrue(sim_coup.ledger.story.ended)
    self.assertEqual(
        sim_coup.ledger.story.ending_archetype, 'scorched_earth_coup'
    )
    npc_slide = sim_coup.ledger.story.ending_slides[1]
    frankie_entry = next(
        e
        for e in npc_slide['entries']
        if e['npc'] == "Frankie 'The Moat' Slater"
    )
    self.assertEqual(frankie_entry['dominant_voice'], 'The Shortfall')
    self.assertIn('Cracked at Tick 18', frankie_entry['scar'])
    coda_slide = sim_coup.ledger.story.ending_slides[2]
    self.assertEqual(coda_slide['voice'], 'Proper Naughty')


class AuthorWorldTest(absltest.TestCase):

  def test_draft_and_write_world_bible_for_all_tone_presets(self):
    for tone_name in (
        'noir',
        'heroic',
        'grim',
        'caper',
        'vampire_romance',
        'dystopian_scifi',
    ):
      bible = author_world.draft_world_bible(
          title=f'Test {tone_name}',
          premise='A fog-bound canal city where the lock-weights are missing.',
          tone_name=tone_name,
      )
      problems = story_state.validate_bible(bible)
      self.assertEmpty(problems, '\n'.join(problems))
      self.assertEqual(bible.tone.name, tone_name)
      self.assertNotEmpty(bible.tone.setting_style)
      self.assertNotEmpty(bible.tone.room_palette)
      self.assertNotEmpty(bible.tone.npc_archetypes)

      with tempfile.NamedTemporaryFile(suffix='.json', mode='r+') as tmp:
        author_world.write_world_bible(bible, tmp.name)
        reloaded = story_state.WorldBible.from_json_path(tmp.name)
        self.assertEmpty(story_state.validate_bible(reloaded))
        self.assertEqual(reloaded.title, f'Test {tone_name}')

  def test_multi_genre_vampire_romance_and_dystopian_scifi_parliaments_combat_and_endings(
      self,
  ):
    # 1. Grounded Vampire Love Drama (The Night Shift at St. Jude's)
    vamp_bible = author_world.draft_world_bible(
        title="The Night Shift at St. Jude's",
        premise='',
        tone_name='vampire_romance',
    )
    sim_vamp = _make_sim_with_bible(bible=vamp_bible)
    story_vamp = sim_vamp.ledger.story
    self.assertIn('The Hunger', story_vamp.sub_minds)
    self.assertIn('Cover Story', story_vamp.sub_minds)
    self.assertIn('Unspoken Vow', story_vamp.sub_minds)
    self.assertEqual(story_vamp.sub_minds['The Hunger'].domain, 'predation')
    self.assertIn('decorum', story_vamp.sub_minds['The Hunger'].strong_against)
    self.assertNotEmpty(story_vamp.get_character_presets())

    # Generalized narrative arcs & catalysts
    self.assertEqual(story_vamp.truths['t_forbidden_bond'].arc_kind, 'bond')
    self.assertEqual(story_vamp.clues['c_ribbon_vow'].catalyst_kind, 'vow')
    self.assertEqual(story_vamp.clues['c_ribbon_vow'].carrier, 'bond')

    # Bespoke NPC Sub-Minds for Claire Moreau
    claire = sim_vamp.ledger.get_npc('Claire Moreau')
    self.assertIsNotNone(claire)
    assert claire is not None
    self.assertEqual(claire.sub_minds['The Mask'].name, 'Night-Shift Calm')
    self.assertEqual(
        claire.sub_minds['The Angle'].name, 'Save David Before Dawn'
    )
    self.assertEqual(
        claire.sub_minds['The Wound'].name, 'Forty Years of Hiding'
    )

    # Dynamic Voice-vs-Voice clash (Unspoken Vow [passion] vs Claire)
    story_vamp.configure_parliament(
        active_ranks={'Unspoken Vow': 4, 'The Hunger': 3}
    )
    sim_vamp.ledger.record_player_action(
        '[Unspoken Vow] Reach for Claire’s hand by the centrifuge and ask what'
        ' happened in the ambulance bay.'
    )
    sim_vamp.ledger.apply_delta(
        {
            'advance_minutes': 5,
            'npcs_addressed': ['Claire Moreau'],
            'clues_revealed': ['c_ribbon_vow', 'c_seraphine_confession'],
        },
        acting_entity=sim_vamp.ledger.player_name,
        narrative_event=(
            'Claire touches the silver watch on her wrist and admits she gave'
            ' David her blood.'
        ),
    )
    self.assertTrue(story_vamp.truths['t_forbidden_bond'].discovered)
    self.assertNotEmpty(story_vamp.voice_choice_log)
    self.assertEqual(story_vamp.voice_choice_log[-1]['voice'], 'Unspoken Vow')
    self.assertTrue(
        any(sm.scars for sm in claire.sub_minds.values())
        or any(sm.rank >= 3 for sm in claire.sub_minds.values())
    )

    # Genre-adaptive emergent climax & epilogue (no Bermondsey hardcoding)
    story_vamp.mark_clue_discovered('c_chalice_ash', tick=4)
    story_vamp.mark_clue_discovered('c_reliquary_ledger', tick=5)
    story_vamp.act = 3
    story_vamp.factions['house_valerius'].standing_with_player = 3
    sim_vamp.director.resolve_emergent_climax(
        sim_vamp.ledger.player_location_id,
        trigger_reason='player_climax_action',
    )
    self.assertTrue(story_vamp.ended)
    self.assertIn('The Obsessive Heart', story_vamp.sub_minds)
    self.assertNotEmpty(story_vamp.ending_title)
    self.assertIn('Forty-Year Pact', story_vamp.ending_summary)
    self.assertNotIn('Bermondsey', story_vamp.ending_summary)
    self.assertLen(story_vamp.ending_slides, 3)

    # 2. Dystopian Sci-Fi + live HTTP /api/generate_world endpoint
    bridge = server.BrowserInputBridge()
    adv_server = server.AdventureServer(
        sim_vamp,
        bridge,
        host='127.0.0.1',
        port=0,
    )
    state_scifi = adv_server.generate_world(preset='dystopian_scifi')
    scifi_story = state_scifi['ledger']['story']
    self.assertEqual(scifi_story['world_title'], 'The Sabotage at Outpost 7')
    self.assertIn('Biometric Lens', scifi_story['sub_minds'])
    self.assertIn('Hard Line', scifi_story['sub_minds'])
    self.assertEqual(
        scifi_story['truths']['t_oxygen_cull']['arc_kind'], 'system_shift'
    )
    self.assertEqual(
        scifi_story['clues']['c_airlock_telemetry']['catalyst_kind'],
        'transmission',
    )
    jonas_minds = state_scifi['ledger']['npcs']['Jonas Reed']['sub_minds']
    self.assertIn('Wrench & Coffee', jonas_minds)
    adv_server.shutdown()

  def test_repair_bible_dict_fixes_dangling_ids(self):
    broken = author_world.build_template_bible_dict(
        title='Broken', premise='Premise', tone_name='noir'
    )
    broken['clues'].append({
        'id': 'c_orphan',
        'truth_id': 't_nonexistent',
        'carrier': 'weird_carrier',
        'text': 'Orphan clue',
    })
    broken['storylets'].append({
        'id': 's_bad',
        'trigger': 'bad_trigger',
        'beat': '',
        'preconditions': [{'unknown_cond': 1}],
        'effects': [{'unknown_effect': 2}],
    })
    repaired = author_world.repair_bible_dict(broken)
    bible = story_state.WorldBible.from_dict(repaired)
    self.assertEmpty(story_state.validate_bible(bible))


class EndToEndBiblePlaytestTest(absltest.TestCase):

  def test_playtest_with_glassharbor_bible_fires_storylets_and_passes_invariants(
      self,
  ):
    model = runner.RichLocalMockModel()
    bible = _load_glassharbor()
    with tempfile.TemporaryDirectory() as tmpdir:
      report = runner.run_playtest_mode(
          model,
          bot_name='DetectiveInvestigator',
          steps=12,
          output_dir=tmpdir,
          run_eval=True,
          bible=bible,
      )
      invariants = report['invariants']
      self.assertTrue(
          invariants['architectural_passed'],
          f'Invariants failed: {json.dumps(invariants, indent=2)}',
      )
      self.assertIn('dormant_npc_addressability_ok', invariants['checks'])
      self.assertTrue(invariants['checks']['dormant_npc_addressability_ok'])
      self.assertEqual(report['header']['archetype'], 'DetectiveInvestigator')
      self.assertEqual(report['header']['info_policy'], 'player_visible_only')
      self.assertFalse(
          report['applicability']['consequence_and_resistance']['applicable']
      )
      self.assertIsNone(
          report['autorater']['dimensions']['consequence_and_resistance'][
              'score'
          ]
      )
      self.assertTrue(
          report['deterministic']['fidelity']['fired_log_available']
      )
      telemetry = invariants['telemetry']
      self.assertEqual(telemetry['director_llm_calls'], 0)
      self.assertGreater(telemetry['storylets_fired'], 0)
      self.assertGreater(telemetry['clues_discovered'], 0)

      with open(f'{tmpdir}/world_ledger.json', 'r', encoding='utf-8') as f:
        saved_ledger = json.load(f)
      self.assertIn('story', saved_ledger)
      self.assertNotEmpty(saved_ledger['story']['journal'])
      self.assertEmpty(
          evaluate_run.check_memory_contamination(
              invariants['memory_audit']
              if 'samples' not in invariants['memory_audit']
              else {}
          )['samples']
      )

  def test_playtest_with_bermondsey_bible_fires_storylets_and_passes_invariants(
      self,
  ):
    model = runner.RichLocalMockModel()
    bible = runner.load_bible('bermondsey')
    assert bible is not None
    with tempfile.TemporaryDirectory() as tmpdir:
      report = runner.run_playtest_mode(
          model,
          bot_name='DetectiveInvestigator',
          steps=10,
          output_dir=tmpdir,
          run_eval=True,
          bible=bible,
      )
      invariants = report['invariants']
      self.assertTrue(
          invariants['architectural_passed'],
          f'Invariants failed: {json.dumps(invariants, indent=2)}',
      )
      telemetry = invariants['telemetry']
      self.assertEqual(telemetry['director_llm_calls'], 0)
      self.assertGreater(telemetry['storylets_fired'], 0)
      self.assertGreater(telemetry['clues_discovered'], 0)


if __name__ == '__main__':
  absltest.main()
