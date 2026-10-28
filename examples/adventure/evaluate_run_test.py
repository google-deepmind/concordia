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

from typing import Any

from absl.testing import absltest
from examples.adventure import evaluate_run
from examples.adventure import playtester_bot

_PLAYER = 'Traveler'
_VANCE = 'Captain Mara Vance'
_PELL = 'Pell'


def _resolution(step, actor, text, hot, warm=(), cold=()):
  return {
      'step': step,
      'actor': actor,
      'location': 'wharf',
      'text': text,
      'tiers': {'hot': list(hot), 'warm': list(warm), 'cold': list(cold)},
  }


def _ledger(*, clues=None, truths=None, fired_log=None, narrative=None):
  return {
      'player_name': _PLAYER,
      'npcs': [
          {'name': _VANCE, 'location_id': 'wharf'},
          {'name': _PELL, 'location_id': 'wharf'},
      ],
      'telemetry': {'tick': 10},
      'story': {
          'clues': clues or {},
          'truths': truths or {},
          'storylets': {
              's_deal': {
                  'beat': (
                      'Vance offers to deputise the player and hands over the '
                      'key-ring count on condition that Ilsa stays off every '
                      'page.'
                  )
              }
          },
          'fired_log': fired_log or [],
          'doom_stage': 0,
          'act': 1,
          'tension': 2.0,
          'ended': False,
      },
      'narrative_log': narrative or [],
  }


class FidelityMetricsTest(absltest.TestCase):

  def test_unsupported_clue_is_flagged_and_supported_clue_is_not(self):
    ledger = _ledger(
        clues={
            'c_confession': {
                'text': (
                    'Captain Vance admits she signed the manifest. She says '
                    'it the way people say the name of a dead relative.'
                ),
                'discovered': True,
                'discovered_tick': 8,
            },
            'c_scratches': {
                'text': (
                    'The pressure valve housing shows bright tool scratches '
                    'under the grease, not corrosion.'
                ),
                'discovered': True,
                'discovered_tick': 13,
            },
        },
        narrative=[
            {
                'kind': 'npc_resolution',
                'tick': 10,
                'text': 'Every official iron bit is accounted for on my hip.',
            },
            {
                'kind': 'scene_resolution',
                'tick': 13,
                'text': (
                    'Beneath the grease, bright steel scratches bite across '
                    'the valve housing threads, unpitted by corrosion.'
                ),
            },
        ],
    )
    metrics = evaluate_run.compute_fidelity_metrics(ledger)
    self.assertEqual(metrics['clues_checked'], 2)
    self.assertEqual(
        [c['id'] for c in metrics['unsupported_clues']], ['c_confession']
    )
    self.assertEqual(metrics['clue_support_rate'], 0.5)

  def test_verbatim_clue_text_in_prose_is_a_leak(self):
    clue_text = (
        "A torn index card reads 'Folio 9 — the Lesser Sluice — manual "
        "bypass, west wall'. The folio itself is not where the card says."
    )
    ledger = _ledger(
        clues={
            'c_stub': {
                'text': clue_text,
                'discovered': True,
                'discovered_tick': 37,
            }
        },
        narrative=[{
            'kind': 'scene_resolution',
            'tick': 37,
            'text': (
                'The stub bears faded ink reading Folio 9 — the Lesser Sluice '
                '— manual bypass, west wall, but the sleeve behind it is empty.'
            ),
        }],
    )
    metrics = evaluate_run.compute_fidelity_metrics(ledger)
    self.assertEqual([l['id'] for l in metrics['clue_text_leaks']], ['c_stub'])

  def test_storylet_adoption_uses_fired_log_ticks(self):
    ledger = _ledger(
        fired_log=[
            {'storylet_id': 's_deal', 'tick': 8, 'source': _VANCE},
            {'storylet_id': 's_deal', 'tick': 56, 'source': _VANCE},
        ],
        narrative=[
            {
                'kind': 'npc_resolution',
                'tick': 10,
                'text': 'Hand over the bypass key or I clap you in irons.',
            },
            {
                'kind': 'npc_resolution',
                'tick': 56,
                'text': (
                    'She hauls a spindle-key from her pocket: take the '
                    'wheelhouse bypass, and wipe her name from that pocket '
                    'ledger. I will deputise you and open the key-ring count '
                    'on that one condition.'
                ),
            },
        ],
    )
    metrics = evaluate_run.compute_fidelity_metrics(ledger)
    self.assertTrue(metrics['fired_log_available'])
    self.assertEqual(metrics['storylets_checked'], 2)
    self.assertEqual([b['tick'] for b in metrics['unadopted_storylets']], [8])

  def test_no_fired_log_reports_unavailable(self):
    metrics = evaluate_run.compute_fidelity_metrics(_ledger())
    self.assertFalse(metrics['fired_log_available'])
    self.assertIsNone(metrics['storylet_adoption_rate'])


class PacingMetricsTest(absltest.TestCase):

  def test_truth_before_first_departure_is_detected(self):
    ledger = _ledger(
        truths={
            't_manifest': {'discovered': True, 'discovered_tick': 3},
            't_tide': {'discovered': False, 'discovered_tick': -1},
        },
        narrative=[{'kind': 'arrival', 'tick': 11, 'text': 'Gearworks.'}],
    )
    pacing = evaluate_run.compute_pacing_metrics(ledger)
    self.assertEqual(pacing['first_truth_tick'], 3)
    self.assertEqual(pacing['first_departure_tick'], 11)
    self.assertTrue(pacing['truth_before_first_departure'])
    self.assertEqual(pacing['truths_discovered'], 1)
    self.assertEqual(pacing['truths_total'], 2)

  def test_truth_after_departure_is_fine(self):
    ledger = _ledger(
        truths={'t_manifest': {'discovered': True, 'discovered_tick': 20}},
        narrative=[{'kind': 'arrival', 'tick': 11, 'text': 'Gearworks.'}],
    )
    self.assertFalse(
        evaluate_run.compute_pacing_metrics(ledger)[
            'truth_before_first_departure'
        ]
    )


class DormantNpcTest(absltest.TestCase):

  def test_addressed_cold_npc_who_never_wakes_is_a_violation(self):
    ledger = _ledger(
        narrative=[{
            'kind': 'player_action',
            'tick': 5,
            'location_id': 'wharf',
            'text': 'Step over to Pell and offer him a silver bit.',
        }]
    )
    resolutions = [
        _resolution(
            6, _PLAYER, 'No runner named Pell here.', [_VANCE], cold=[_PELL]
        ),
        _resolution(
            7, _VANCE, 'Broke, chasing ghosts.', [_VANCE], cold=[_PELL]
        ),
    ]
    result = evaluate_run.compute_dormant_npc_addressability(
        ledger, resolutions
    )
    self.assertFalse(result['ok'])
    self.assertEqual(result['violations'][0]['npc'], _PELL)

  def test_addressed_npc_who_wakes_next_step_passes(self):
    ledger = _ledger(
        narrative=[{
            'kind': 'player_action',
            'tick': 5,
            'location_id': 'wharf',
            'text': 'Step over to Pell and offer him a silver bit.',
        }]
    )
    resolutions = [
        _resolution(6, _PLAYER, 'Pell looks up.', [_VANCE, _PELL]),
    ]
    result = evaluate_run.compute_dormant_npc_addressability(
        ledger, resolutions
    )
    self.assertTrue(result['ok'])
    self.assertEqual(result['addressed_count'], 1)


class BeatRepetitionTest(absltest.TestCase):

  def test_paraphrased_threats_are_flagged_despite_low_lexical_overlap(self):
    threats = [
        'Turn them over or explain yourself to the harbor brig.',
        'Empty your coat pockets, or I haul you to the cells myself.',
        'Talk now, or I clap the irons on you for loitering.',
        'Spit out where Vale took it, or I clap these irons on your wrists.',
        'Crawling back empty-handed? Step off my pier.',
    ]
    resolutions = [
        _resolution(i, _VANCE, text, [_VANCE])
        for i, text in enumerate(threats, start=1)
    ]
    result = evaluate_run.compute_beat_repetition(resolutions, _PLAYER)
    self.assertEqual(result['flagged_npcs'], [_VANCE])
    vance = result['per_npc'][_VANCE]
    self.assertGreaterEqual(vance['beat_fractions']['threat'], 0.6)
    self.assertLess(vance['max_lexical_overlap'], 0.3)

  def test_player_turns_are_excluded(self):
    resolutions = [
        _resolution(1, _PLAYER, 'I clap the irons on him.', [_VANCE])
    ]
    result = evaluate_run.compute_beat_repetition(resolutions, _PLAYER)
    self.assertEqual(result['per_npc'], {})


class ReentryPairsTest(absltest.TestCase):

  def test_warm_reentry_pair_is_built_from_before_and_after_lines(self):
    resolutions = [
        _resolution(2, _VANCE, 'Hands off the shed.', [_VANCE]),
        _resolution(11, _PLAYER, 'Gearworks.', ['Kaelen Voss'], warm=[_VANCE]),
        _resolution(21, _PLAYER, 'Back to the wharf.', [_VANCE]),
        _resolution(22, _VANCE, 'Crawling back empty-handed?', [_VANCE]),
    ]
    pairs = evaluate_run.compute_reentry_pairs(resolutions, _PLAYER)
    self.assertLen(pairs, 1)
    self.assertEqual(pairs[0]['npc'], _VANCE)
    self.assertEqual(pairs[0]['kind'], 'warm')
    self.assertEqual(pairs[0]['before'], 'Hands off the shed.')
    self.assertEqual(pairs[0]['after'], 'Crawling back empty-handed?')

  def test_cold_reentry_is_labelled_cold(self):
    resolutions = [
        _resolution(2, _VANCE, 'Hands off.', [_VANCE]),
        _resolution(11, _PLAYER, 'Away.', ['Kaelen Voss'], cold=[_VANCE]),
        _resolution(21, _PLAYER, 'Back.', [_VANCE]),
        _resolution(22, _VANCE, 'You again.', [_VANCE]),
    ]
    pairs = evaluate_run.compute_reentry_pairs(resolutions, _PLAYER)
    self.assertEqual([p['kind'] for p in pairs], ['cold'])

  def test_first_appearance_is_not_a_reentry(self):
    resolutions = [
        _resolution(1, _PLAYER, 'Arrive.', [_VANCE]),
        _resolution(2, _VANCE, 'Hands off.', [_VANCE]),
    ]
    self.assertEmpty(evaluate_run.compute_reentry_pairs(resolutions, _PLAYER))


class ApplicabilityTest(absltest.TestCase):

  def test_detective_run_without_reentry_disables_two_dimensions(self):
    applicability = evaluate_run.compute_applicability(
        'DetectiveInvestigator', []
    )
    self.assertFalse(applicability['consequence_and_resistance']['applicable'])
    self.assertFalse(applicability['persistence_and_rehydration']['applicable'])
    self.assertTrue(applicability['npc_agency']['applicable'])

  def test_chaos_run_with_everything_logged_enables_everything(self):
    applicability = evaluate_run.compute_applicability(
        'ChaosBoundaryTester',
        [{'kind': 'warm'}],
        {'enabled': True, 'choices_exercised': True},
        {'npc': {'available': True}, 'combat': {'available': True}},
    )
    self.assertTrue(all(info['applicable'] for info in applicability.values()))

  def test_parliament_dimensions_off_without_their_logs(self):
    applicability = evaluate_run.compute_applicability(
        'ChaosBoundaryTester', [{'kind': 'warm'}]
    )
    self.assertFalse(applicability['player_parliament']['applicable'])
    self.assertFalse(applicability['npc_parliament']['applicable'])
    self.assertFalse(applicability['voice_combat']['applicable'])


class ParliamentMetricsTest(absltest.TestCase):

  def _npc_ledger(self):
    ledger = _ledger()
    ledger['npcs'] = {
        _VANCE: {
            'name': _VANCE,
            'location_id': 'wharf',
            'sub_minds': {
                'stonewalling': {'name': 'The Warden', 'rank': 3},
                'leverage': {'name': 'The Bargain', 'rank': 1},
                'self_preservation': {'name': 'The Daughter', 'rank': 2},
            },
        }
    }
    return ledger

  def _turn(self, step, winner, action='Hand over the key.'):
    return {
        'step': step,
        'npc': _VANCE,
        'winning_voice': winner,
        'options': [
            {
                'voice': 'The Warden',
                'domain': 'stonewalling',
                'rank': 3,
                'intrusive': False,
                'action': action,
            },
            {
                'voice': 'The Bargain',
                'domain': 'leverage',
                'rank': 1,
                'intrusive': False,
                'action': 'Offer a trade for the key.',
            },
        ],
    }

  def test_npc_mask_share_and_directive_support(self):
    ledger = self._npc_ledger()
    transcript = {
        'npc_council_turns': [
            self._turn(2, 'The Warden'),
            self._turn(5, 'The Warden'),
            self._turn(7, 'The Bargain'),
            self._turn(10, 'The Bargain'),
        ],
        'resolutions': [
            _resolution(2, _VANCE, 'Hand it over. The key. Now.', [_VANCE]),
            _resolution(5, _VANCE, 'Hands off the shed.', [_VANCE]),
            _resolution(
                7, _VANCE, 'A trade, then. The key for her name.', [_VANCE]
            ),
            _resolution(
                10, _VANCE, 'Offer stands: trade me the key.', [_VANCE]
            ),
        ],
    }
    metrics = evaluate_run.compute_parliament_metrics(ledger, transcript)
    npc = metrics['npc']
    self.assertTrue(npc['available'])
    self.assertEqual(npc['council_turns'], 4)
    self.assertEqual(npc['overall_mask_share'], 0.5)
    self.assertEqual(npc['mask_share_first_half'], 1.0)
    self.assertEqual(npc['mask_share_second_half'], 0.0)
    self.assertEmpty(npc['npcs_never_leaving_mask'])
    self.assertEqual(
        npc['per_npc'][_VANCE]['winning_domains'],
        {'stonewalling': 2, 'leverage': 2},
    )
    self.assertGreater(npc['winning_action_support_mean'], 0.0)

  def test_npc_never_leaving_mask_is_listed(self):
    ledger = self._npc_ledger()
    transcript = {
        'npc_council_turns': [self._turn(s, 'The Warden') for s in (2, 4, 6)],
        'resolutions': [],
    }
    npc = evaluate_run.compute_parliament_metrics(ledger, transcript)['npc']
    self.assertEqual(npc['npcs_never_leaving_mask'], [_VANCE])
    self.assertIsNone(npc['winning_action_support_mean'])

  def test_combat_outcomes_from_story_delta(self):
    def _res(step, outcome):
      record = _resolution(step, _PLAYER, 'x', [_VANCE])
      record['story_delta'] = {
          'npc_voice_combat': [{
              'npc': _VANCE,
              'player_voice': 'Brass Neck',
              'outcome': outcome,
              'roll': '12v9',
              'dominant_voice': 'The Warden',
              'dominant_rank': 3,
              'intrusive': False,
          }]
      }
      return record

    transcript = {
        'npc_council_turns': [],
        'resolutions': [
            _res(1, 'mask_vindicated'),
            _res(3, 'grazing_shift'),
            _res(5, 'wound_spiked'),
            _res(7, 'mask_vindicated'),
        ],
    }
    combat = evaluate_run.compute_parliament_metrics(_ledger(), transcript)[
        'combat'
    ]
    self.assertTrue(combat['available'])
    self.assertEqual(combat['events'], 4)
    self.assertEqual(combat['decisive_rate'], 0.25)
    self.assertEqual(combat['stonewalled_rate'], 0.5)
    self.assertFalse(combat['single_outcome_only'])
    self.assertEqual(combat['player_voices_used'], {'Brass Neck': 4})

  def test_player_sheet_metrics(self):
    ledger = _ledger()
    ledger['story']['sub_minds'] = {
        'The Book': {
            'name': 'The Book',
            'rank': 4,
            'intrusive': True,
            'scars': ['a', 'b'],
            'obsession': 'audit',
            'origin': 'setting',
        },
        'Nerve': {'name': 'Nerve', 'rank': 1, 'scars': [], 'origin': 'custom'},
    }
    player = evaluate_run.compute_parliament_metrics(
        ledger, {'npc_council_turns': [], 'resolutions': []}
    )['player']
    self.assertTrue(player['available'])
    self.assertEqual(player['rank_spread'], 3)
    self.assertEqual(player['intrusive_voices'], ['The Book'])
    self.assertEqual(player['total_scars'], 2)
    self.assertEqual(player['origins'], {'setting': 1, 'custom': 1})


class MetricCapsTest(absltest.TestCase):

  def test_inert_npc_parliament_and_uniform_combat_are_capped(self):
    caps = evaluate_run.metric_caps(
        {
            'parliament': {
                'npc': {
                    'available': True,
                    'overall_mask_share': 0.95,
                    'winning_action_support_mean': 0.4,
                },
                'combat': {
                    'available': True,
                    'single_outcome_only': True,
                    'outcomes': {'mask_vindicated': 5},
                },
            },
            'voices': {'enabled': False},
        },
        {'info_policy': 'player_visible_only'},
    )
    self.assertEqual(caps['npc_parliament'][0], 50)
    self.assertEqual(caps['voice_combat'][0], 50)
    self.assertNotIn('npc_agency', caps)

  def test_leaky_playtester_caps_consequence_and_agency(self):
    caps = evaluate_run.metric_caps({}, {'info_policy': 'director_omniscient'})
    self.assertEqual(caps['consequence_and_resistance'][0], 60)
    self.assertEqual(caps['npc_agency'][0], 60)

  def test_truth_leak_cap_beats_stale_cap(self):
    caps = evaluate_run.metric_caps(
        {
            'voices': {
                'enabled': True,
                'premature_leaks': [{'kind': 'truth'}],
                'unknown_faculty_lines': 0,
                'options': {'stale_option_rate': 0.5},
            }
        },
        {},
    )
    self.assertEqual(caps['player_parliament'][0], 40)

  def test_narrative_coherence_defects_and_ghost_locations_are_capped(self):
    ledger = _ledger(
        narrative=[{
            'kind': 'player_action',
            'tick': 1,
            'location_id': 'wharf',
            'text': "Inspect the room's across turn 22.",
        }]
    )
    ledger['locations'] = {
        'comms_and_telemetry_annex': {},
        'comms_telemetry_annex': {},
    }
    transcript = {
        'resolutions': [
            _resolution(
                2,
                _VANCE,
                'Captain Mara Vance Mara Vance slam slammed the desk.',
                [_VANCE],
            ),
            _resolution(
                3,
                _VANCE,
                "Bristling as Traveler presses to ask, Answer: I won't say.",
                [_VANCE],
            ),
        ],
        'action_lines': [
            "Step 1 [Traveler]: Inspect the room's across turn 22.",
            'Step 2 [Captain Mara Vance]: Answer: Hands off.',
        ],
        'warm_lines': ['Step 4 [Warm]: Pell.'],
    }
    coherence = evaluate_run.compute_narrative_coherence_metrics(
        ledger, transcript
    )
    self.assertFalse(coherence['defect_free'])
    self.assertNotEmpty(coherence['duplicated_actor_prefixes'])
    self.assertNotEmpty(coherence['raw_prompt_tags'])
    self.assertNotEmpty(coherence['spliced_gerund_prefixes'])
    self.assertNotEmpty(coherence['broken_action_syntax'])
    self.assertNotEmpty(coherence['truncated_or_bare_lines'])
    self.assertNotEmpty(coherence['ghost_locations'])
    caps = evaluate_run.metric_caps({'narrative_coherence': coherence}, {})
    self.assertEqual(caps['narrative_and_dialogue_coherence'][0], 55)


class HardInvariantsTest(absltest.TestCase):

  def test_content_invariants_are_skipped_without_deterministic_metrics(self):
    result = evaluate_run.evaluate_hard_invariants(
        telemetry={
            'hibernations_count': 1,
            'rehydrations_count': 1,
            'warm_events_count': 1,
        },
        latency={},
        entity_memories={},
    )
    self.assertTrue(result['all_passed'])
    self.assertNotIn('dormant_npc_addressability_ok', result['checks'])

  def test_content_invariants_fail_the_gate(self):
    result = evaluate_run.evaluate_hard_invariants(
        telemetry={
            'hibernations_count': 1,
            'rehydrations_count': 1,
            'warm_events_count': 1,
        },
        latency={},
        entity_memories={},
        deterministic={
            'pacing': {'truth_before_first_departure': True},
            'dormant_npc_addressability': {'ok': False},
        },
    )
    self.assertFalse(result['all_passed'])
    self.assertTrue(result['architectural_passed'])
    self.assertFalse(result['content_passed'])
    self.assertFalse(result['checks']['dormant_npc_addressability_ok'])
    # Truth pacing is reported to the grader (Dim 7) but is not a gate.
    self.assertNotIn('no_truth_before_first_departure', result['checks'])


class ParseDimensionTest(absltest.TestCase):

  def test_missing_evidence_raises(self):
    with self.assertRaisesRegex(
        evaluate_run.AutoraterParseError, 'evidence_for and evidence_against'
    ):
      evaluate_run._parse_dimension(  # pylint: disable=protected-access
          'npc_agency', {'score': 70, 'evidence_for': 'x'}, True
      )

  def test_not_applicable_dimension_accepts_null_score(self):
    parsed = evaluate_run._parse_dimension(  # pylint: disable=protected-access
        'consequence_and_resistance',
        {'score': None, 'evidence_against': 'archetype never tested it'},
        False,
    )
    self.assertIsNone(parsed['score'])

  def test_applicable_dimension_with_null_score_raises(self):
    with self.assertRaisesRegex(
        evaluate_run.AutoraterParseError, 'gave no score'
    ):
      evaluate_run._parse_dimension(  # pylint: disable=protected-access
          'npc_agency', {'score': None}, True
      )


class VoiceMetricsTest(absltest.TestCase):

  def _voice_ledger(self, **overrides):
    ledger = _ledger(**overrides)
    ledger['story']['player_voices'] = {
        'The Book': 'Runs the vig and the odds.',
        'The Hustle': 'Spots the swap and the leverage.',
        'Brass Neck': 'Audacity first.',
    }
    ledger['story']['tone'] = {'inner_voices': True}
    ledger['story']['voice_affinity'] = {
        'The Book': 3,
        'The Hustle': 1,
        'Brass Neck': 1,
    }
    return ledger

  def test_disabled_when_bible_has_no_voices(self):
    metrics = evaluate_run.compute_voice_metrics(_ledger())
    self.assertFalse(metrics['enabled'])
    applicability = evaluate_run.compute_applicability('', [], metrics)
    self.assertFalse(applicability['player_parliament']['applicable'])

  def test_premature_truth_leak_through_voice_is_detected(self):
    ledger = self._voice_ledger(
        truths={
            't_ringer': {
                'statement': (
                    "The prize greyhound Duchess hasn't left London—Carla "
                    'Kovacs dyed her white star with black boot polish so '
                    'Desmond and Frankie can run her as a 12-to-1 ringer named '
                    "'Biscuit' in Trap 4 at Catford."
                ),
                'discovered': False,
                'discovered_tick': -1,
            }
        },
        narrative=[
            {
                'kind': 'inner_voice',
                'tick': 51,
                'actor': 'The Hustle',
                'text': (
                    'The Hustle: Duchess dyed black and entered in Trap 4 as '
                    "'Biscuit' at 12-to-1. That's audacious, even for "
                    'Bermondsey.'
                ),
            },
            {
                'kind': 'inner_voice',
                'tick': 3,
                'actor': 'The Book',
                'text': (
                    'The Book: Look at him clutching that layoff pad like a '
                    "lifebelt—he's terrified of an audit."
                ),
            },
        ],
    )
    metrics = evaluate_run.compute_voice_metrics(ledger)
    self.assertTrue(metrics['enabled'])
    self.assertLen(metrics['premature_leaks'], 1)
    leak = metrics['premature_leaks'][0]
    self.assertEqual(
        (leak['kind'], leak['id'], leak['voice']),
        ('truth', 't_ringer', 'The Hustle'),
    )
    invariants = evaluate_run.evaluate_hard_invariants(
        telemetry={
            'hibernations_count': 1,
            'rehydrations_count': 1,
            'warm_events_count': 1,
        },
        latency={},
        entity_memories={},
        deterministic={
            'pacing': {},
            'dormant_npc_addressability': {'ok': True},
            'voices': metrics,
        },
    )
    self.assertFalse(invariants['checks']['no_inner_voice_truth_leaks'])

  def test_coverage_and_duplicates(self):
    ledger = self._voice_ledger(
        narrative=[
            {
                'kind': 'inner_voice',
                'tick': t,
                'actor': 'The Book',
                'text': 'The Book: Sixty grand shy on the traps.',
            }
            for t in (1, 3, 6)
        ]
        + [{
            'kind': 'inner_voice',
            'tick': 8,
            'actor': 'Nerve',
            'text': 'Nerve: Steady.',
        }],
    )
    metrics = evaluate_run.compute_voice_metrics(ledger)
    self.assertEqual(metrics['faculty_line_counts']['The Book'], 3)
    self.assertEqual(metrics['max_faculty_share'], 1.0)
    self.assertCountEqual(
        metrics['silent_faculties'], ['The Hustle', 'Brass Neck']
    )
    self.assertEqual(metrics['unknown_faculty_lines'], 1)
    self.assertEqual(metrics['duplicate_line_rate'], 0.5)

  def test_option_log_freshness_and_choices(self):
    ledger = self._voice_ledger()
    ledger['story']['voice_option_log'] = [
        {
            'tick': 1,
            'location_id': 'wharf',
            'options': [
                {
                    'voice': 'The Book',
                    'action': 'Audit the layoff pad.',
                    'perception': 'p',
                    'source': 'llm',
                },
                {
                    'voice': 'The Hustle',
                    'action': 'Offer Frankie a swap.',
                    'perception': 'p',
                    'source': 'fallback',
                },
            ],
        },
        {
            'tick': 2,
            'location_id': 'wharf',
            'options': [
                {
                    'voice': 'The Book',
                    'action': 'Audit the layoff pad.',
                    'perception': 'p',
                    'source': 'fallback',
                },
                {
                    'voice': 'Brass Neck',
                    'action': 'Front out the doorman.',
                    'perception': 'p',
                    'source': 'llm',
                },
            ],
        },
    ]
    ledger['story']['voice_choice_log'] = [
        {'tick': 2, 'voice': 'The Book', 'via': 'prefix'}
    ]
    metrics = evaluate_run.compute_voice_metrics(ledger)
    options = metrics['options']
    self.assertTrue(options['available'])
    self.assertEqual(options['total_options'], 4)
    self.assertEqual(options['llm_option_rate'], 0.5)
    self.assertEqual(options['stale_option_rate'], 0.25)
    self.assertTrue(metrics['choices_exercised'])
    self.assertEqual(metrics['choice_counts'], {'The Book': 1})
    self.assertEqual(metrics['dominant_voice'], 'The Book')
    applicability = evaluate_run.compute_applicability('', [], metrics)
    self.assertTrue(applicability['player_parliament']['applicable'])

  def test_no_choices_keeps_dimension_applicable_with_caveat(self):
    metrics = evaluate_run.compute_voice_metrics(self._voice_ledger())
    self.assertFalse(metrics['choices_exercised'])
    applicability = evaluate_run.compute_applicability('', [], metrics)
    self.assertTrue(applicability['player_parliament']['applicable'])
    self.assertIn('rank/scar', applicability['player_parliament']['reason'])


class SpoilerMetricsTest(absltest.TestCase):

  def _ledger_with_story(self, opening: str) -> dict[str, Any]:
    ledger = _ledger()
    ledger['player_location_id'] = 'wharf'
    ledger['locations'] = {
        'wharf': {
            'name': 'The Mess Hall',
            'description': 'Steel tables and a coffee urn.',
            'affordances': ['bolt cutters by the door'],
        }
    }
    ledger['npcs'] = {
        'Jonas Reed': {
            'name': 'Jonas Reed',
            'role_summary': 'Chief mechanic with a mug of coffee.',
            'traits': 'quiet',
        }
    }
    ledger['story'].update({
        'world_premise': 'A frozen corpse in the parka of Jonas Reed.',
        'world_opening': opening,
        'visited_location_ids': ['wharf'],
        'truths': {
            't_cull': {
                'id': 't_cull',
                'statement': (
                    'Dr. Mara Lin sedated undercover safety auditor Elias Ward '
                    'in the medical bay three nights ago.'
                ),
                'question': 'Who is the dead man in Jonas Reed’s parka?',
            }
        },
    })
    return ledger

  def test_leaky_opening_fails_the_content_gate(self):
    ledger = self._ledger_with_story(
        'Find out how Dr. Lin sedated undercover safety auditor Elias Ward.'
    )
    spoilers = evaluate_run.compute_spoiler_metrics(ledger)
    self.assertTrue(spoilers['has_leak'])
    # The generic role briefing is derived from the opening, so both leak.
    self.assertIn('opening', {f['text'] for f in spoilers['findings']})
    invariants = evaluate_run.evaluate_hard_invariants(
        telemetry={
            'hibernations_count': 1,
            'rehydrations_count': 1,
            'warm_events_count': 1,
        },
        latency={},
        entity_memories={},
        deterministic={
            'dormant_npc_addressability': {'ok': True},
            'voices': {'enabled': False},
            'spoilers': spoilers,
        },
    )
    self.assertFalse(invariants['checks']['no_briefing_truth_leaks'])

  def test_clean_opening_passes(self):
    ledger = self._ledger_with_story(
        'Work out who the dead man in Jonas Reed’s parka is before the '
        'transport lands.'
    )
    spoilers = evaluate_run.compute_spoiler_metrics(ledger)
    self.assertFalse(spoilers['has_leak'])
    self.assertIn('role_briefing', spoilers['checked_texts'])
    self.assertIn('truth[t_cull].question', spoilers['checked_texts'])


class PlaytesterVisibilityTest(absltest.TestCase):

  def test_only_learned_entries_are_visible(self):
    journal = [
        (
            '[tick 1] Learned: The manifest lists lamp fittings at two in the '
            'morning.'
        ),
        '[tick 3] Understood: Captain Vance signed the forged manifest.',
    ]
    self.assertEqual(
        playtester_bot.player_visible_evidence(journal),
        'The manifest lists lamp fittings at two in the morning.',
    )

  def test_empty_when_nothing_learned(self):
    self.assertEqual(
        playtester_bot.player_visible_evidence(['[tick 3] Understood: secret']),
        '',
    )


if __name__ == '__main__':
  absltest.main()
