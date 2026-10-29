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

"""Shared execution helpers and model setup for the Adventure example."""

from collections.abc import Collection, Mapping, Sequence
import importlib
import json
import os
import threading
import time
from typing import Any, override

from absl import logging
from concordia.components.agent import human_act_component
from concordia.environment import step_controller as step_controller_lib
from examples.adventure import adventure_prefab
from examples.adventure import author_world
from examples.adventure import evaluate_run
from examples.adventure import playtester_bot
from examples.adventure import server
from examples.adventure import story_state
from concordia.language_model import call_limit_wrapper
from concordia.language_model import language_model
from concordia.language_model import retry_wrapper
from concordia.utils import sampling
from concordia.utils import text

DEFAULT_MODEL_NAME = 'gemini-flash-latest'


class RichLocalMockModel(language_model.LanguageModel):
  """Fallback deterministic model for offline testing."""

  def __init__(self):
    """Initializes the instance."""
    super().__init__()
    self._room_seq = 0
    self._turn_seq = 0

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
    if 'Frontier Architect' in prompt:
      self._room_seq += 1
      idx = self._room_seq
      room_name = f'Glassharbor Sector {idx}'
      for line in prompt.splitlines():
        if line.startswith('Target Location to Expand:'):
          raw_target = line.split(':', 1)[1].split(' (id:', 1)[0].strip()
          if raw_target:
            room_name = raw_target
          break
      return json.dumps({
          'name': room_name,
          'description': (
              f'Vaulted basalt passage {idx} where pressurized steam pipes '
              'hiss above tidewater sluices and locked guild lockers.'
          ),
          'affordances': [
              f'sealed guild locker #{idx} with a notched iron dial',
              f'oil-stained maintenance log #{idx}',
              f'brass sluice lever #{idx}',
          ],
          'exits': {
              'deeper': f'Lower Conduit {idx}',
          },
          'npc': {
              'name': f'Inspector Kaelen {idx}',
              'role_summary': f'Guild Sluice Inspector of Sector {idx}',
              'goal': (
                  f'Prevent unauthorized tampering with sluice lever #{idx}'
              ),
              'traits': 'Clipped, skeptical, demands guild credentials',
              'formative_memories': [
                  f'Inspector Kaelen {idx} noticed missing pressure seals.'
              ],
          },
      })

    if 'Sub-Mind Council Update' in prompt:
      return json.dumps({
          'inner_voices': [],
          'voice_options': [],
      })

    if 'Putative Action by' in prompt:
      self._turn_seq += 1
      move_to = ''
      beat_used = ''
      clues_revealed: list[str] = []
      for line in prompt.splitlines():
        if line.startswith('Putative Action by'):
          lower = line.lower()
          for d in ('north', 'east', 'west', 'back', 'deeper'):
            if f'go {d}' in lower or f'walk {d}' in lower:
              move_to = d
              break
        stripped = line.strip()
        if not beat_used and stripped.startswith('* [') and ']' in stripped:
          beat_used = stripped.split('[', 1)[1].split(']', 1)[0].strip()
        if (
            not clues_revealed
            and stripped.startswith('* c_')
            and ' ' in stripped
        ):
          clues_revealed.append(stripped[2:].split(' ', 1)[0].strip())
      return json.dumps({
          'narrative_event': (
              f'[Turn {self._turn_seq}] The corroded brass fittings resist at '
              'first, drawing a sharp look from the watch before revealing a '
              'stamped harbor cipher.'
          ),
          'delta': {
              'move_to': move_to,
              'advance_minutes': 5,
              'add_affordances': [f'stamped harbor cipher #{self._turn_seq}'],
              'remove_affordances': [],
              'new_exits': {},
              'add_inventory': [],
              'remove_inventory': [],
              'player_status': 'Watchful',
              'npcs_addressed': [],
              'relationship_updates': {},
              'beat_used': beat_used,
              'clues_revealed': clues_revealed,
              'faction_shifts': (
                  {'wardens': -1} if self._turn_seq % 4 == 0 else {}
              ),
              'tension_delta': 0.5,
              'inner_voice': (
                  'Ledger: The dates on the margin do not match the tide table.'
                  if clues_revealed
                  else ''
              ),
          },
      })

    if 'strict, non-sycophantic simulation evaluator' in prompt:
      return json.dumps({
          'dimensions': {
              'consequence_and_resistance': {
                  'score': 68,
                  'evidence_for': (
                      'The corroded brass fittings resist at first.'
                  ),
                  'evidence_against': 'Mock run uses deterministic templates.',
              },
              'npc_agency': {
                  'score': 70,
                  'evidence_for': 'NPCs pursue distinct inspection goals.',
                  'evidence_against': 'Mock responses are concise.',
              },
              'persistence_and_rehydration': {
                  'score': 72,
                  'evidence_for': 'Cold NPCs rehydrate with intact memories.',
                  'evidence_against': 'Single backtrack cycle tested.',
              },
              'off_screen_emergence': {
                  'score': 66,
                  'evidence_for': 'Warm chronicler publishes rumors to bus.',
                  'evidence_against': 'Could tie rumors more tightly to locks.',
              },
              'environmental_affordances': {
                  'score': 74,
                  'evidence_for': 'Each room adds 3 concrete affordances.',
                  'evidence_against': 'Template naming in mock mode.',
              },
              'ledger_prose_fidelity': {
                  'score': 70,
                  'evidence_for': 'Mock beats echo their storylet keywords.',
                  'evidence_against': 'Clue text is templated.',
              },
              'mystery_pacing': {
                  'score': 60,
                  'evidence_for': 'Truths land after room changes.',
                  'evidence_against': 'Doom clock barely advances in mock.',
              },
              'player_parliament': {
                  'score': 62,
                  'evidence_for': 'Mock voices answer in their own register.',
                  'evidence_against': 'No [Faculty] option is ever chosen.',
              },
              'npc_parliament': {
                  'score': 58,
                  'evidence_for': 'NPC councils log a winning voice per turn.',
                  'evidence_against': 'Mock NPC lines are templated.',
              },
              'narrative_and_dialogue_coherence': {
                  'score': 72,
                  'evidence_for': (
                      'Mock actions and resolutions are grammatically complete '
                      'with zero structural defects.'
                  ),
                  'evidence_against': 'Mock templates repeat phrasing.',
              },
              'voice_combat': {
                  'score': 55,
                  'evidence_for': 'Clashes record rolls and outcomes.',
                  'evidence_against': 'Mock rolls are not legible in prose.',
              },
          },
          'lowest_dimension': 'off_screen_emergence',
          'top_improvement_target': (
              'Cross-reference Warm rumors in scene context.'
          ),
      })

    return 'examines the pressure gauge and demands to see a harbor writ.'

  @override
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, Mapping[str, Any]]:
    del prompt, seed
    return 0, responses[0], {}


class GenaiLanguageModel(language_model.LanguageModel):
  """Low-latency Gemini LanguageModel using google.genai SDK directly."""

  def __init__(
      self,
      *,
      api_key: str,
      model_name: str = DEFAULT_MODEL_NAME,
      temperature: float = 0.4,
      fallback_model: language_model.LanguageModel | None = None,
  ):
    """Initializes the GenaiLanguageModel."""
    super().__init__()
    genai = importlib.import_module('google.genai')
    self._genai_types = importlib.import_module('google.genai.types')
    self._client = genai.Client(api_key=api_key)
    self._model_name = model_name
    self._default_temperature = temperature
    self._fallback_model = fallback_model

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
    del top_p, top_k, timeout, seed
    temp = (
        self._default_temperature
        if temperature == language_model.DEFAULT_TEMPERATURE
        else temperature
    )
    effective_max_tokens = max(max_tokens * 4, 2048)
    try:
      config = self._genai_types.GenerateContentConfig(
          max_output_tokens=effective_max_tokens,
          temperature=temp,
          thinking_config=self._genai_types.ThinkingConfig(thinking_budget=0),
      )
      response = self._client.models.generate_content(
          model=self._model_name,
          contents=prompt,
          config=config,
      )
      sample = (response.text or '') if response else ''
      return text.truncate(sample, delimiters=terminators)
    except Exception as err:  # pylint: disable=broad-exception-caught
      try:
        err_text = str(err)
        if any(
            code in err_text
            for code in (
                '503',
                'UNAVAILABLE',
                '429',
                'RESOURCE_EXHAUSTED',
            )
        ):
          time.sleep(1.5)
          try:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=self._genai_types.GenerateContentConfig(
                    max_output_tokens=effective_max_tokens,
                    temperature=temp,
                    thinking_config=self._genai_types.ThinkingConfig(
                        thinking_budget=0
                    ),
                ),
            )
            sample = (response.text or '') if response else ''
            if sample:
              return text.truncate(sample, delimiters=terminators)
          except Exception:  # pylint: disable=broad-exception-caught
            pass
        retry_model = self._model_name
        if '404' in err_text or 'NOT_FOUND' in err_text:
          retry_model = DEFAULT_MODEL_NAME
        logging.warning(
            'Primary call to %s failed (%s); retrying with %s',
            self._model_name,
            err,
            retry_model,
        )
        config_no_thinking = self._genai_types.GenerateContentConfig(
            max_output_tokens=effective_max_tokens,
            temperature=temp,
        )
        response = self._client.models.generate_content(
            model=retry_model,
            contents=prompt,
            config=config_no_thinking,
        )
        if retry_model != self._model_name:
          self._model_name = retry_model
        sample = (response.text or '') if response else ''
        return text.truncate(sample, delimiters=terminators)
      except Exception as backup_err:  # pylint: disable=broad-exception-caught
        if self._fallback_model is not None:
          return self._fallback_model.sample_text(
              prompt,
              max_tokens=max_tokens,
              terminators=terminators,
              temperature=temp,
          )
        raise err from backup_err

  @override
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, Mapping[str, Any]]:
    del seed
    choice_labels = [chr(ord('a') + i) for i in range(len(responses))]
    formatted_choices = '\n'.join(
        f'({label}) {response}'
        for label, response in zip(choice_labels, responses)
    )
    question = (
        'Select one of the choices below, responding ONLY with the letter in '
        'parentheses such as (a) or (b).\n\n'
        f'{prompt}\n\nChoices:\n{formatted_choices}\n\nAnswer:'
    )
    sample = self.sample_text(question, max_tokens=16, terminators=())
    answer = sampling.extract_choice_response(sample)
    if answer is not None and answer in choice_labels:
      idx = choice_labels.index(answer)
      return idx, responses[idx], {}
    return 0, responses[0], {}


def setup_language_model(
    *,
    use_mock: bool = False,
    disable_language_model: bool = False,
    api_type: str = 'google_aistudio',
    model_name: str = DEFAULT_MODEL_NAME,
    api_key: str = '',
    max_calls: int = 2000,
) -> language_model.LanguageModel:
  """Initializes an external Concordia language model or the offline mock."""
  if use_mock or disable_language_model:
    logging.info('Using RichLocalMockModel.')
    return RichLocalMockModel()

  clean_api_type = api_type.strip().lower()
  if clean_api_type in (
      'genai',
      'google_aistudio',
      'google_aistudio_model',
      'aistudio',
      'gemini',
  ):
    effective_key = (
        api_key
        or os.environ.get('GEMINI_API_KEY', '')
        or os.environ.get('GOOGLE_API_KEY', '')
    )
    if not effective_key:
      raise ValueError(
          'Provide --api_key, set GEMINI_API_KEY / GOOGLE_API_KEY in the '
          'environment, or pass --use_mock for offline deterministic runs.'
      )
    try:
      base_model = GenaiLanguageModel(
          api_key=effective_key,
          model_name=model_name,
          temperature=0.4,
      )
    except ModuleNotFoundError:
      lm_setup = importlib.import_module('concordia.contrib.language_models')
      base_model = lm_setup.language_model_setup(
          api_type='google_aistudio',
          model_name=model_name,
          api_key=effective_key,
      )
  else:
    lm_setup = importlib.import_module('concordia.contrib.language_models')
    base_model = lm_setup.language_model_setup(
        api_type=clean_api_type,
        model_name=model_name,
        api_key=api_key or None,
        disable_language_model=disable_language_model,
    )

  wrapped = retry_wrapper.RetryLanguageModel(
      base_model,
      retry_on_exceptions=(TimeoutError, ConnectionError),
      retry_tries=2,
  )
  if max_calls > 0:
    wrapped = call_limit_wrapper.CallLimitLanguageModel(
        wrapped, max_calls=max_calls
    )
  return wrapped


def load_bible(spec: str) -> story_state.WorldBible | None:
  """Resolves a bible name or path to a validated WorldBible, or None."""
  if not spec:
    return None
  clean_spec = spec.strip().lower()
  if clean_spec in ('vampire_romance', 'crimson_court', 'st_judes'):
    bible = author_world.draft_world_bible(
        title="The Night Shift at St. Jude's",
        premise='',
        tone_name='vampire_romance',
    )
  elif clean_spec in ('dystopian_scifi', 'sector_9', 'neon_hab', 'outpost_7'):
    bible = author_world.draft_world_bible(
        title='The Sabotage at Outpost 7',
        premise='',
        tone_name='dystopian_scifi',
    )
  elif clean_spec in ('glassharbor', 'bermondsey', 'bermondsey_drop'):
    bible = story_state.load_shipped_bible(clean_spec)
  else:
    path = story_state.resolve_bible_path(spec)
    bible = story_state.WorldBible.from_json_path(path)
  problems = story_state.validate_bible(bible)
  if problems:
    raise ValueError(
        'World bible failed validation:\n  ' + '\n  '.join(problems)
    )
  logging.info(
      'Loaded world bible %r: %d factions, %d truths, %d clues, %d storylets',
      bible.title,
      len(bible.factions),
      len(bible.truths),
      len(bible.clues),
      len(bible.storylets),
  )
  return bible


def run_playtest_mode(
    model: language_model.LanguageModel,
    *,
    bot_name: str,
    steps: int,
    output_dir: str,
    run_eval: bool,
    bible: story_state.WorldBible | None = None,
    council_follow_rate: float = playtester_bot.DEFAULT_COUNCIL_FOLLOW_RATE,
    target_voice: str = '',
    require_full_steps: bool = False,
) -> dict[str, Any]:
  """Executes a headless playtest and writes all artifacts and eval reports.

  Args:
    model: Language model for GM, NPCs, playtester phrasing, and the grader.
    bot_name: `PlaytesterArchetype` value.
    steps: Engine steps to run (player turns are roughly half of these).
    output_dir: Where artifacts and the evaluation report are written.
    run_eval: Whether to run the evaluator after the playtest.
    bible: World bible; None uses the built-in premise.
    council_follow_rate: See `PlaytesterActComponent`.
    target_voice: Optional single Sub-Mind voice name to follow exclusively.
    require_full_steps: Whether to continue past `story.ended` until `steps`.

  Returns:
    The evaluation report, or {} when `run_eval` is False.
  """
  os.makedirs(output_dir, exist_ok=True)
  archetype = playtester_bot.PlaytesterArchetype(bot_name)

  class _DeferredPlaytester(playtester_bot.PlaytesterActComponent):
    """Binds the playtester policy once the simulation ledger is created."""

  sim = adventure_prefab.AdventureSimulation(
      model=model,
      player_act_component=human_act_component.HumanActComponent(
          input_reader=server.BrowserInputBridge()
      ),
      enable_async_npc_plan=False,
      max_hot_npcs=2,
      max_warm_npcs=1,
      warm_tick_interval=2,
      bible=bible,
  )
  clean_target_voice = target_voice.strip()
  sim.terminate_component.set_require_full_steps(require_full_steps)
  if clean_target_voice:
    sim.configure_player_parliament(
        active_ranks={
            name: 3 if name.lower() == clean_target_voice.lower() else 1
            for name in sim.ledger.story.sub_minds
        }
    )
    minutes_per_step_budget = 14 if require_full_steps else 10
    with sim.ledger.lock:
      sim.ledger.story.doom.total_minutes = max(
          sim.ledger.story.doom.total_minutes, steps * minutes_per_step_budget
      )
  bot_policy = _DeferredPlaytester(
      model=model,
      ledger=sim.ledger,
      archetype=archetype,
      total_steps=steps,
      council_follow_rate=council_follow_rate,
      target_voice=clean_target_voice,
  )
  sim.player_entity._act_component = bot_policy  # pylint: disable=protected-access
  bot_policy.set_entity(sim.player_entity)
  bot_policy.set_logging_channel(
      lambda datum: sim.player_entity.measurements.publish_datum(
          '__act__', datum, capture_key=sim.player_entity.name
      )
  )

  ledger_path = os.path.join(output_dir, 'world_ledger.json')

  def _on_step(step_data: step_controller_lib.StepData) -> None:
    logging.info(
        '[Step %d | PlayerSteps %d/%d] %s -> %s',
        step_data.step,
        sim.player_steps_completed,
        steps,
        step_data.acting_entity,
        step_data.action[:140],
    )
    if step_data.acting_entity == sim.ledger.player_name:
      try:
        with open(ledger_path + '.tmp', 'w', encoding='utf-8') as f:
          json.dump(sim.ledger.to_dict(), f, indent=2)
        os.replace(ledger_path + '.tmp', ledger_path)
      except OSError:
        pass

  logging.info(
      'Starting headless playtest: bot=%s, target_voice=%s, steps=%d,'
      ' output_dir=%s',
      bot_name,
      clean_target_voice or 'none',
      steps,
      output_dir,
  )
  sim_log = sim.play(max_steps=steps, verbose=False, step_callback=_on_step)

  structured_path = os.path.join(output_dir, 'simulation_structured.json')
  with open(structured_path, 'w', encoding='utf-8') as f:
    f.write(sim_log.to_json())

  html_path = os.path.join(output_dir, 'simulation_log.html')
  with open(html_path, 'w', encoding='utf-8') as f:
    f.write(sim_log.to_html())

  with open(ledger_path, 'w', encoding='utf-8') as f:
    json.dump(sim.ledger.to_dict(), f, indent=2)

  latency_path = os.path.join(output_dir, 'latency.json')
  with open(latency_path, 'w', encoding='utf-8') as f:
    json.dump(sim.latency_tracker.summary(), f, indent=2)

  effective_follow_rate = council_follow_rate
  if (
      archetype == playtester_bot.PlaytesterArchetype.COUNCIL_FOLLOWER
      or clean_target_voice
  ):
    effective_follow_rate = 1.0
  meta_path = os.path.join(output_dir, evaluate_run.PLAYTEST_META_FILENAME)
  with open(meta_path, 'w', encoding='utf-8') as f:
    json.dump(
        {
            'archetype': archetype.value,
            'target_voice': clean_target_voice,
            'info_policy': playtester_bot.INFO_POLICY,
            'planned_steps': steps,
            'council_follow_rate': effective_follow_rate,
            'council_follows': bot_policy.council_follows,
            'bible': bible.title if bible is not None else '',
        },
        f,
        indent=2,
    )

  logging.info('Saved simulation artifacts to %s', output_dir)

  if run_eval:
    report = evaluate_run.evaluate_artifacts_directory(output_dir, model=model)
    logging.info(
        'Evaluation complete: all_invariants_passed=%s, weighted_score=%.2f, '
        'lowest_dimension=%s',
        report['invariants']['all_passed'],
        report['autorater']['weighted_total'],
        report['autorater']['lowest_dimension'],
    )
    return report
  return {}


def run_server_mode(
    model: language_model.LanguageModel,
    *,
    host: str,
    port: int,
    max_steps: int,
    output_dir: str,
    bible: story_state.WorldBible | None = None,
    extra_server_banner: str = '',
) -> None:
  """Runs the interactive HTTP + SSE server for human browser and CLI play."""
  os.makedirs(output_dir, exist_ok=True)
  bridge = server.BrowserInputBridge()
  active_sim_holder: list[adventure_prefab.AdventureSimulation] = []
  loop_thread_holder: list[threading.Thread] = []
  active_bible_holder: list[story_state.WorldBible | None] = [bible]

  def _make_and_start_sim(
      new_bible: story_state.WorldBible | None = None,
  ) -> adventure_prefab.AdventureSimulation:
    if new_bible is not None:
      active_bible_holder[0] = new_bible
    if active_sim_holder:
      active_sim_holder[0].stop()
      bridge.clear(wake_blocked_reader=True)
      if loop_thread_holder and loop_thread_holder[0].is_alive():
        loop_thread_holder[0].join(timeout=2.0)

    session_reader, session_ready = bridge.bind_new_session()
    human_act = human_act_component.HumanActComponent(
        input_reader=session_reader
    )
    new_sim = adventure_prefab.AdventureSimulation(
        model=model,
        player_act_component=human_act,
        player_ready_fn=session_ready,
        enable_async_npc_plan=True,
        max_hot_npcs=2,
        max_warm_npcs=2,
        warm_tick_interval=2,
        hot_cooldown_seconds=0.1,
        bible=active_bible_holder[0],
    )
    new_sim.terminate_component.set_max_player_steps(max_steps)
    active_sim_holder[:] = [new_sim]

    def _checkpoint_cb(player_step: int) -> None:
      del player_step
      ledger_path = os.path.join(output_dir, 'world_ledger.json')
      with open(ledger_path, 'w', encoding='utf-8') as f:
        json.dump(new_sim.ledger.to_dict(), f, indent=2)

    def _loop_runner() -> None:
      new_sim.engine.run_loop(
          game_masters=[new_sim.game_master],
          entities=new_sim.all_engine_entities(),
          premise=new_sim.ledger.premise,
          max_steps=max(100_000, max_steps * 50_000),
          verbose=False,
          log=new_sim._raw_log,  # pylint: disable=protected-access
          checkpoint_callback=_checkpoint_cb,
      )

    worker = threading.Thread(
        target=_loop_runner,
        name='adventure-sim-loop',
        daemon=True,
    )
    loop_thread_holder[:] = [worker]
    worker.start()
    return new_sim

  sim = _make_and_start_sim()
  adv_server = server.AdventureServer(
      sim,
      bridge,
      host=host,
      port=port,
      reset_fn=_make_and_start_sim,
  )
  bound_port = adv_server.start_background()
  banner_suffix = (
      f' {extra_server_banner.format(port=bound_port)}'
      if extra_server_banner
      else ''
  )
  logging.info(
      'Adventure Server listening on http://%s:%d%s',
      host,
      bound_port,
      banner_suffix,
  )

  try:
    while True:
      time.sleep(1.0)
  except KeyboardInterrupt:
    logging.info('Shutting down Adventure Server...')
  finally:
    if active_sim_holder:
      active_sim_holder[0].stop()
    adv_server.shutdown()


def run_author_world_mode(
    *,
    title: str,
    premise: str,
    tone_name: str,
    output_path: str,
) -> story_state.WorldBible:
  """Drafts, validates, and saves a WorldBible JSON file."""
  bible = author_world.draft_world_bible(
      title=title,
      premise=premise,
      tone_name=tone_name,
  )
  problems = story_state.validate_bible(bible)
  if problems:
    raise ValueError(
        'World bible failed validation:\n  ' + '\n  '.join(problems)
    )
  author_world.write_world_bible(bible, output_path)
  logging.info(
      'Wrote validated WorldBible (%d storylets, %d truths, %d clues) to %s',
      len(bible.storylets),
      len(bible.truths),
      len(bible.clues),
      output_path,
  )
  return bible
