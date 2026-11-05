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
from typing import override

from absl.testing import absltest
from concordia.components.agent import human_act_component
from examples.adventure import adventure_prefab
from examples.adventure import author_world
from examples.adventure import server
from examples.adventure import story_state
from examples.adventure import world_ledger
from concordia.language_model import language_model


def _load_glassharbor() -> story_state.WorldBible:
  return story_state.load_shipped_bible('glassharbor')


def _load_bermondsey() -> story_state.WorldBible:
  return story_state.load_shipped_bible('bermondsey')


def _player_act_component() -> human_act_component.HumanActComponent:
  return human_act_component.HumanActComponent(
      input_reader=server.BrowserInputBridge(timeout_seconds=0.1)
  )


class _SilentModel(language_model.LanguageModel):
  """Model that returns empty JSON; the story layer must never need it."""

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


class BibleLoadingTest(absltest.TestCase):

  def test_glassharbor_bible_loads_and_validates(self):
    bible = _load_glassharbor()
    problems = story_state.validate_bible(bible)
    self.assertEmpty(problems, '\n'.join(problems))
    self.assertEqual(bible.title, 'Glassharbor')
    self.assertLen(bible.factions, 4)
    self.assertLen(bible.truths, 4)
    self.assertGreaterEqual(len(bible.storylets), 40)
    self.assertTrue(bible.tone.doom_hard)
    self.assertTrue(bible.tone.inner_voices)
    self.assertIn('Suspicion', bible.player_voices)

  def test_bermondsey_bible_loads_and_validates(self):
    bible = _load_bermondsey()
    problems = story_state.validate_bible(bible)
    self.assertEmpty(problems, '\n'.join(problems))
    self.assertEqual(bible.title, 'The Bermondsey Drop')
    self.assertEqual(bible.tone.name, 'caper')
    self.assertNotEmpty(bible.tone.setting_style)
    self.assertNotEmpty(bible.tone.room_palette)
    self.assertNotEmpty(bible.tone.npc_archetypes)
    self.assertLen(bible.factions, 4)
    self.assertLen(bible.truths, 4)
    self.assertLen(bible.clues, 17)
    self.assertGreaterEqual(len(bible.storylets), 24)
    self.assertNotEmpty(bible.starting_inventory)
    self.assertIn('Brass Neck', bible.player_voices)
    self.assertIn('The Book', bible.player_voices)

  def test_every_truth_is_reachable_from_clues(self):
    bible = _load_glassharbor()
    clue_ids = {c.id for c in bible.clues}
    for truth in bible.truths:
      self.assertGreaterEqual(len(truth.clue_ids), truth.min_clues_to_discover)
      for cid in truth.clue_ids:
        self.assertIn(cid, clue_ids)

  def test_storylet_triggers_cover_all_phases(self):
    bible = _load_glassharbor()
    triggers = {s.trigger for s in bible.storylets}
    self.assertContainsSubset(
        {'scene', 'arrival', 'warm', 'doom', 'rehydrate'}, triggers
    )
    self.assertNotIn('ending', triggers)
    bermondsey_triggers = {s.trigger for s in _load_bermondsey().storylets}
    self.assertNotIn('ending', bermondsey_triggers)

  def test_validator_catches_dangling_references(self):
    bible = _load_glassharbor()
    bad = dataclasses.replace(
        bible,
        storylets=bible.storylets
        + [
            story_state.Storylet(
                id='bad_one',
                trigger='scene',
                beat='x',
                preconditions=[{'clue_discovered': 'c_does_not_exist'}],
                effects=[{'faction_shift': ['nobody', 1]}],
            ),
            story_state.Storylet(
                id='bad_two',
                trigger='nope',
                beat='',
                preconditions=[{'what': 1}],
            ),
        ],
    )
    problems = story_state.validate_bible(bad)
    joined = '\n'.join(problems)
    self.assertIn('c_does_not_exist', joined)
    self.assertIn('nobody', joined)
    self.assertIn("unknown trigger 'nope'", joined)
    self.assertIn("unknown condition 'what'", joined)
    self.assertIn('empty beat', joined)

  def test_routine_slot_wraps_midnight(self):
    slot = story_state.RoutineSlot(22, 6, 'Wharf')
    self.assertTrue(slot.covers_hour(23))
    self.assertTrue(slot.covers_hour(3))
    self.assertFalse(slot.covers_hour(12))
    day = story_state.RoutineSlot(6, 14, 'Wharf')
    self.assertTrue(day.covers_hour(6))
    self.assertFalse(day.covers_hour(14))

  def test_author_world_drafts_valid_bible_with_empty_premise(self):
    for tone in (
        'london_caper',
        'noir_mystery',
        'vampire_romance',
        'dystopian_scifi',
    ):
      bible = author_world.draft_world_bible(
          title='',
          premise='',
          tone_name=tone,
          model=None,
      )
      self.assertTrue(bible.title)
      self.assertTrue(bible.premise)
      self.assertEmpty(story_state.validate_bible(bible))
      self.assertEmpty(story_state.find_bible_leaks(bible))


class StoryStateTest(absltest.TestCase):

  def test_from_bible_and_round_trip(self):
    bible = _load_glassharbor()
    state = story_state.StoryState.from_bible(bible, start_clock_minutes=480)
    self.assertEqual(state.doom_start_minutes, 480)
    self.assertEqual(state.act, 1)
    restored = story_state.StoryState.from_dict(
        json.loads(json.dumps(state.to_dict()))
    )
    self.assertEqual(restored.to_dict(), state.to_dict())
    # Mutating the state must not touch the bible.
    state.clues['c_manifest'].discovered = True
    self.assertFalse(
        next(c for c in bible.clues if c.id == 'c_manifest').discovered
    )

  def test_clue_discovery_completes_truth_and_advances_act(self):
    state = story_state.StoryState.from_bible(_load_glassharbor())
    self.assertIsNone(state.mark_clue_discovered('c_manifest', tick=3))
    self.assertEqual(state.truths_discovered(), 0)
    truth = state.mark_clue_discovered('c_ilsa_debt', tick=5)
    self.assertIsNotNone(truth)
    self.assertEqual(truth.id, 't_manifest')
    self.assertTrue(state.truths['t_manifest'].discovered)
    self.assertEqual(state.truths_discovered(), 1)
    self.assertLen(state.journal, 3)
    # Re-marking is a no-op.
    self.assertIsNone(state.mark_clue_discovered('c_manifest', tick=9))
    self.assertEqual(state.recompute_act(), 2)
    self.assertIn('Act 2', state.current_act_note())

  def test_doom_fraction_and_stages(self):
    state = story_state.StoryState.from_bible(
        _load_glassharbor(), start_clock_minutes=480
    )
    self.assertEqual(state.doom.total_minutes, 960)
    self.assertEqual(
        state.stage_for_fraction(state.doom_fraction(480)).index, 0
    )
    self.assertEqual(
        state.stage_for_fraction(state.doom_fraction(480 + 240)).index, 1
    )
    self.assertEqual(
        state.stage_for_fraction(state.doom_fraction(480 + 959)).index, 3
    )
    self.assertEqual(
        state.stage_for_fraction(state.doom_fraction(480 + 960)).index, 4
    )
    self.assertEqual(state.doom_minutes_remaining(480 + 100), 860)

  def test_faction_and_tension_clamping(self):
    state = story_state.StoryState.from_bible(_load_glassharbor())
    self.assertEqual(state.shift_faction('wardens', -9), -5)
    self.assertEqual(state.shift_faction('wardens', 20), 5)
    self.assertIsNone(state.shift_faction('nobody', 1))
    self.assertEqual(state.shift_tension(100), 10.0)
    self.assertEqual(state.shift_tension(-100), 0.0)

  def test_inter_faction_standing_and_turf_war_ignition(self):
    state = story_state.StoryState.from_bible(_load_bermondsey())
    self.assertEqual(
        state.inter_faction_standing('corrigan_firm', 'yardies'), -3
    )
    self.assertEmpty(state.active_turf_wars())
    val = state.shift_inter_faction('corrigan_firm', 'yardies', -1)
    self.assertEqual(val, -4)
    self.assertIn(('corrigan_firm', 'yardies', -4), state.active_turf_wars())
    restored = story_state.StoryState.from_dict(
        json.loads(json.dumps(state.to_dict()))
    )
    self.assertEqual(
        restored.inter_faction_standing('corrigan_firm', 'yardies'), -4
    )
    self.assertIn(('corrigan_firm', 'yardies', -4), restored.active_turf_wars())
    self.assertNotEmpty(restored.describe_turf_wars())
    self.assertIn('Open turf war', restored.describe_turf_wars()[0])
    state.shift_faction(
        'corrigan_firm',
        'Cut into a reciprocal side-angle over the layoff pad.',
        tick=12,
    )
    self.assertIn(
        'reciprocal side-angle',
        state.factions['corrigan_firm'].current_disposition(),
    )
    self.assertNotEmpty(state.factions['corrigan_firm'].recent_actions)

  def test_submind_parliament_configuration_and_scars(self):
    state = story_state.StoryState.from_bible(_load_bermondsey())
    self.assertIn('Brass Neck', state.sub_minds)
    self.assertFalse(state.character_initialized)
    self.assertEqual(state.sub_minds['Brass Neck'].domain, 'audacity')

    # Configure starting Parliament: Brass Neck at rank 4, The Book at 2.
    state.configure_parliament(
        active_ranks={'Brass Neck': 4, 'The Book': 2},
        custom_sub_minds=[
            story_state.SubMindState(
                name='Ex-Boxing Promoter',
                domain='deception',
                worldview='Treats every standoff like a rigged undercard.',
                blind_spot='Assumes everyone has a dive price.',
                rank=2,
                origin='custom',
            )
        ],
    )
    self.assertTrue(state.character_initialized)
    self.assertTrue(state.sub_minds['Brass Neck'].intrusive)
    self.assertTrue(state.sub_minds['Brass Neck'].active)
    self.assertFalse(state.sub_minds["Copper's Nose"].active)
    self.assertIn('Ex-Boxing Promoter', state.sub_minds)
    self.assertEqual(state.voice_affinity['Ex-Boxing Promoter'], 2)

    # Record a backfired voice outcome and check scar + obsession formation.
    updated = state.record_voice_outcome(
        'Brass Neck',
        backfired=True,
        scar='Corrigan minder pulled a revolver when bluffed',
        obsession='Settle the score with the minder',
    )
    assert updated is not None
    self.assertIn(
        'Corrigan minder pulled a revolver when bluffed', updated.scars
    )
    self.assertEqual(updated.obsession, 'Settle the score with the minder')

    restored = story_state.StoryState.from_dict(
        json.loads(json.dumps(state.to_dict()))
    )
    self.assertTrue(restored.character_initialized)
    self.assertEqual(
        restored.sub_minds['Brass Neck'].obsession,
        'Settle the score with the minder',
    )
    self.assertIn(
        'Corrigan minder pulled a revolver when bluffed',
        restored.sub_minds['Brass Neck'].scars,
    )


class TruthLeakDetectionTest(absltest.TestCase):

  def _bible(self, opening: str, question: str = '') -> story_state.WorldBible:
    return story_state.WorldBible.from_dict({
        'title': 'Outpost 7',
        'premise': (
            'A four-person weather station. A supply crate has arrived with '
            'a frozen corpse wearing the winter parka of chief mechanic Jonas '
            'Reed, who is drinking coffee in the mess hall.'
        ),
        'opening': opening,
        'starting_location': {
            'name': 'The Outpost 7 Mess Hall',
            'description': 'Steel tables, a coffee urn, a dead radio.',
            'affordances': ['bolt cutters by the airlock door'],
            'exits': {},
        },
        'seed_npcs': [{
            'name': 'Jonas Reed',
            'role_summary': 'Station chief mechanic with a mug of coffee.',
            'goal': 'x',
            'traits': 'quiet',
            'location_name': 'The Outpost 7 Mess Hall',
        }],
        'truths': [{
            'id': 't_cull',
            'question': question,
            'statement': (
                'The corpse is undercover safety auditor Elias Ward. Dr. Mara '
                'Lin sedated him in the medical bay on company orders three '
                'nights ago and he froze to death in the outgoing crate.'
            ),
            'clue_ids': [],
        }],
        'clues': [],
        'storylets': [],
    })

  def test_public_vocabulary_excludes_truth_private_tokens(self):
    bible = self._bible('You arrive.')
    vocab = story_state.public_vocabulary(bible)
    self.assertIn('corpse', vocab)
    self.assertIn('coffee', vocab)
    self.assertNotIn('auditor', vocab)
    self.assertNotIn('sedated', vocab)

  def test_opening_that_restates_the_truth_is_flagged(self):
    bible = self._bible(
        'Find out how Dr. Lin sedated undercover safety auditor Elias Ward '
        'in the medical bay.'
    )
    findings = story_state.find_bible_leaks(bible)
    self.assertLen(findings, 1)
    self.assertEqual(findings[0]['text'], 'opening')
    self.assertEqual(findings[0]['severity'], 'leak')
    self.assertIn('sedated', findings[0]['hits'])
    problems = story_state.validate_bible(bible)
    self.assertTrue(any('restates truth t_cull' in p for p in problems))

  def test_opening_that_names_only_inciting_facts_passes(self):
    bible = self._bible(
        "The crate held a frozen corpse in Jonas Reed's parka; Jonas is "
        'drinking coffee in the mess hall. Work out who the dead man is '
        'before the transport lands.',
        question='Who is the dead man in Jonas Reed’s parka?',
    )
    self.assertEmpty(story_state.find_bible_leaks(bible))
    self.assertEmpty(story_state.validate_bible(bible))

  def test_question_field_round_trips(self):
    bible = self._bible(
        'You arrive.', question='Who is the dead man in Jonas Reed’s parka?'
    )
    truth = bible.truths[0]
    self.assertEqual(
        truth.question, 'Who is the dead man in Jonas Reed’s parka?'
    )
    self.assertEqual(
        story_state.Truth.from_dict(truth.to_dict()).question,
        'Who is the dead man in Jonas Reed’s parka?',
    )

  def test_shipped_bibles_and_role_briefings_do_not_leak(self):
    for bible in (_load_glassharbor(), _load_bermondsey()):
      state = story_state.StoryState.from_bible(bible)
      role = state.get_player_role_info()
      findings = story_state.find_bible_leaks(
          bible, extra_texts={'role_briefing': role['summary']}
      )
      leaks = [f for f in findings if f['severity'] == 'leak']
      self.assertEmpty(leaks, f'{bible.title}: {leaks}')
      for truth in bible.truths:
        self.assertTrue(truth.question, f'{truth.id} has no question')


class LedgerIntegrationTest(absltest.TestCase):

  def test_npc_record_round_trip_with_story_fields(self):
    rec = world_ledger.NpcRecord(
        name='X',
        role_summary='r',
        goal='g',
        traits='t',
        location_id='loc',
        faction='wardens',
        secret='s',
        wound='w',
        routine=[story_state.RoutineSlot(6, 14, 'The Wharf', 'watch')],
        known_clue_ids=['c_warden_count'],
    )
    restored = world_ledger.NpcRecord.from_dict(
        json.loads(json.dumps(rec.to_dict()))
    )
    self.assertEqual(restored.secret, 's')
    self.assertEqual(restored.routine_location_for_hour(7), 'The Wharf')
    self.assertIsNone(restored.routine_location_for_hour(20))
    self.assertEqual(restored.known_clue_ids, ['c_warden_count'])

  def test_legacy_payloads_still_load(self):
    legacy_npc = {
        'name': 'Old',
        'role_summary': 'r',
        'goal': 'g',
        'traits': 't',
        'location_id': 'loc',
        'tier': 'cold',
    }
    rec = world_ledger.NpcRecord.from_dict(legacy_npc)
    self.assertEqual(rec.faction, '')
    self.assertEqual(rec.routine, [])
    legacy_loc = {'location_id': 'loc', 'name': 'Loc'}
    node = world_ledger.LocationNode.from_dict(legacy_loc)
    self.assertEqual(node.clue_ids, [])
    ledger = world_ledger.WorldLedger(premise='p', starting_location_name='S')
    state = ledger.to_dict()
    del state['story']
    ledger.load_dict(state)
    self.assertEqual(ledger.story.truths, {})

  def test_ledger_serializes_story_and_tracks_visits(self):
    bible = _load_glassharbor()
    ledger = world_ledger.WorldLedger(
        premise=bible.premise,
        starting_location_name='The Salt-Glass Wharf',
        starting_exits={'north': 'The Amber Lantern Tavern'},
        story=story_state.StoryState.from_bible(bible),
    )
    self.assertEqual(ledger.story.doom_start_minutes, 8 * 60)
    self.assertEqual(
        ledger.story.visited_location_ids, ['the_salt_glass_wharf']
    )
    ledger.move_entity(ledger.player_name, 'north')
    self.assertIn('the_amber_lantern_tavern', ledger.story.visited_location_ids)

    ledger.expand_location(
        'The Floodgate Gearworks',
        description='d',
        clue_ids=['c_valve_scratches'],
        faction_lean='floodgate',
    )
    node = ledger.get_location('the_floodgate_gearworks')
    assert node is not None
    self.assertEqual(node.clue_ids, ['c_valve_scratches'])
    self.assertEqual(node.faction_lean, 'floodgate')
    self.assertEqual(
        ledger.story.clues['c_valve_scratches'].placed_at,
        'the_floodgate_gearworks',
    )

    fresh = world_ledger.WorldLedger(premise='x', starting_location_name='S')
    fresh.load_dict(json.loads(json.dumps(ledger.to_dict())))
    self.assertEqual(
        fresh.story.clues['c_valve_scratches'].placed_at,
        'the_floodgate_gearworks',
    )
    telemetry = fresh.telemetry_summary()
    self.assertEqual(telemetry['director_llm_calls'], 0)
    self.assertEqual(telemetry['truths_discovered'], 0)

  def test_relocate_clue_moves_item_and_leaves_affordance_traces(self):
    bible = _load_bermondsey()
    sim = adventure_prefab.AdventureSimulation(
        model=_SilentModel(),
        player_act_component=_player_act_component(),
        enable_async_npc_plan=False,
        bible=bible,
    )
    ledger = sim.ledger
    # Expand two off-screen rooms and place c_chiller_manifest in Lockup 9.
    ledger.expand_location(
        'Spitalfields Meat Lockup 9',
        description='Frost-coated meat hooks.',
        clue_ids=['c_chiller_manifest'],
        faction_lean='flying_squad',
    )
    ledger.expand_location(
        'Rotherhithe Scrap & Breakers',
        description='Hydraulic baler and shipping container.',
        faction_lean='yardies',
    )
    moved_desc = ledger.relocate_clue(
        'c_chiller_manifest',
        'Rotherhithe Scrap & Breakers',
        by_npc="Desmond 'Nine-Pint' Baptiste",
    )
    self.assertIsNotNone(moved_desc)
    lockup = ledger.get_location('spitalfields_meat_lockup_9')
    scrap = ledger.get_location('rotherhithe_scrap_breakers')
    assert lockup is not None and scrap is not None
    self.assertNotIn('c_chiller_manifest', lockup.clue_ids)
    self.assertIn('c_chiller_manifest', scrap.clue_ids)
    self.assertEqual(
        ledger.story.clues['c_chiller_manifest'].placed_at,
        'rotherhithe_scrap_breakers',
    )
    self.assertTrue(any('fresh jimmy marks' in a for a in lockup.affordances))
    self.assertTrue(any('newly stashed' in a for a in scrap.affordances))
    self.assertEqual(ledger.telemetry_summary()['clues_relocated'], 1)


class PrefabWithBibleTest(absltest.TestCase):

  def test_simulation_seeds_from_bible_without_llm_calls(self):
    bible = _load_glassharbor()
    model = _SilentModel()
    sim = adventure_prefab.AdventureSimulation(
        model=model,
        player_act_component=_player_act_component(),
        enable_async_npc_plan=False,
        bible=bible,
    )
    self.assertEqual(model.calls, 0)
    self.assertIs(sim.bible, bible)
    self.assertEqual(sim.ledger.premise, bible.premise)
    npcs = sim.ledger.all_npcs()
    self.assertLen(npcs, len(bible.seed_npcs))
    vance = npcs['Captain Mara Vance']
    self.assertEqual(vance.faction, 'wardens')
    self.assertIn('manifest', vance.secret)
    self.assertLen(vance.routine, 3)
    self.assertEqual(vance.location_id, 'the_salt_glass_wharf')
    self.assertEqual(vance.tier, world_ledger.NpcTier.HOT)
    self.assertIn('c_warden_count', vance.known_clue_ids)
    self.assertEqual(
        sim.ledger.story.clues['c_warden_count'].placed_at, 'Captain Mara Vance'
    )
    # Starting room clues are placed; routine rooms exist as stubs.
    start = sim.ledger.get_player_location()
    self.assertCountEqual(start.clue_ids, ['c_manifest', 'c_seal_replaced'])
    self.assertIsNotNone(sim.ledger.get_location('The Warden Customs House'))
    self.assertIsNotNone(sim.ledger.get_location('The Guild Dormitory'))
    intro = sim.ledger.to_dict()['narrative_log'][0]['text']
    self.assertIn('unsealed inquiry writ', intro)
    assert bible.starting_location is not None
    self.assertIn(bible.starting_location.name, intro)

  def test_simulation_rejects_invalid_bible(self):
    bible = _load_glassharbor()
    bad = dataclasses.replace(bible, clues=bible.clues[:-1])
    with self.assertRaisesRegex(ValueError, 'failed validation'):
      adventure_prefab.AdventureSimulation(
          model=_SilentModel(),
          player_act_component=_player_act_component(),
          enable_async_npc_plan=False,
          bible=bad,
      )

  def test_simulation_without_bible_is_unchanged(self):
    sim = adventure_prefab.AdventureSimulation(
        model=_SilentModel(),
        player_act_component=_player_act_component(),
        enable_async_npc_plan=False,
    )
    self.assertIsNone(sim.bible)
    self.assertEqual(sim.ledger.story.truths, {})
    self.assertLen(sim.ledger.all_npcs(), 3)

  def test_simulation_seeds_bermondsey_with_custom_starting_inventory(self):
    bible = _load_bermondsey()
    sim = adventure_prefab.AdventureSimulation(
        model=_SilentModel(),
        player_act_component=_player_act_component(),
        enable_async_npc_plan=False,
        bible=bible,
    )
    self.assertEqual(
        sim.ledger.player_inventory, list(bible.starting_inventory)
    )
    self.assertIn(
        'monogrammed empty pigskin holdall', sim.ledger.player_inventory
    )
    start = sim.ledger.get_player_location()
    self.assertEqual(start.name, 'The Half-Moon Snooker & Social')
    self.assertCountEqual(
        start.clue_ids, ['c_layoff_carbon', 'c_cue_case_shells']
    )

  def test_glassharbor_voices_and_presets_have_no_scifi_cross_contamination(
      self,
  ):
    bible = _load_glassharbor()
    state = story_state.StoryState.from_bible(bible)
    self.assertIn('Empathy', state.sub_minds)
    self.assertEqual(state.sub_minds['Empathy'].domain, 'empathy')
    for v_name, sm in state.sub_minds.items():
      self.assertNotEqual(
          sm.domain,
          'cybernetics',
          f'Glassharbor voice {v_name} got sci-fi cybernetics domain!',
      )
      self.assertNotIn(
          'ICE',
          sm.blind_spot,
          f'Glassharbor voice {v_name} got sci-fi ICE blind spot!',
      )
    presets = state.get_character_presets()
    self.assertGreaterEqual(len(presets), 4)
    for preset in presets:
      self.assertTrue(preset.get('name'))
      self.assertTrue(preset.get('id'))
      self.assertTrue(preset.get('tagline'))
      self.assertIsInstance(preset.get('ranks'), dict)
    role_info = state.get_player_role_info()
    self.assertEqual(role_info['title'], 'Maritime Inquiry Inspector')
    self.assertIn('unsealed inquiry writ', role_info['summary'])

  def test_scifi_and_vampire_preset_worlds_have_role_briefing_and_clean_intro(
      self,
  ):
    for preset_name, expected_role in (
        (
            'dystopian_scifi',
            'Outpost Relief Marshal',
        ),
        ('vampire_romance', 'Night-Shift Blood Bank Tech'),
    ):
      bible = author_world.draft_world_bible(
          title=preset_name,
          premise=preset_name,
          tone_name=preset_name,
          model=None,
      )
      self.assertEmpty(story_state.validate_bible(bible))
      self.assertTrue(bible.opening)
      preset_state = story_state.StoryState.from_bible(bible)
      briefing = preset_state.get_player_role_info()['summary']
      briefing_leaks = [
          f
          for f in story_state.find_bible_leaks(
              bible, extra_texts={'role_briefing': briefing}
          )
          if f['severity'] == 'leak'
      ]
      self.assertEmpty(briefing_leaks, f'{preset_name}: {briefing_leaks}')
      assert bible.starting_location is not None
      self.assertNotIn(bible.premise, bible.starting_location.description)
      sim = adventure_prefab.AdventureSimulation(
          model=_SilentModel(),
          player_act_component=_player_act_component(),
          enable_async_npc_plan=False,
          bible=bible,
      )
      intro = sim.ledger.to_dict()['narrative_log'][0]['text']
      # Premise lives in the world briefing and is not duplicated in the intro.
      self.assertNotIn(bible.premise, intro)
      self.assertIn(sim.ledger.story.world_opening, intro)
      self.assertNotIn('{start_name}', sim.ledger.story.world_opening)
      story_dict = sim.ledger.story.to_dict()
      self.assertEqual(story_dict['player_role']['title'], expected_role)
      for preset in story_dict['character_presets']:
        self.assertTrue(preset.get('name'))
        self.assertTrue(preset.get('id'))
      # Verify fallback voice options are clean and free of concatenation bugs.
      opts = sim.director.synthesize_fallback_voice_options()
      self.assertNotEmpty(opts)
      for opt in opts:
        self.assertTrue(opt.perception, f'Empty perception for {opt.voice}')
        self.assertTrue(opt.action, f'Empty action for {opt.voice}')
        self.assertNotIn('through your', opt.perception)


if __name__ == '__main__':
  absltest.main()
