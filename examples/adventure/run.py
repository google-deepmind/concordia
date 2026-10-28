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

r"""CLI entry point for interactive HTTP play and autonomous playtesting.

Offline playtest (deterministic mock model):

  python -m examples.adventure.run --mode=playtest --use_mock

Interactive browser server (live Gemini via AI Studio):

  GEMINI_API_KEY=... python -m examples.adventure.run --mode=server \
      --bible=bermondsey --port=8765

Live playtest with another Concordia provider (e.g. Together AI, OpenAI, Ollama):

  python -m examples.adventure.run --mode=playtest --api_type=together_ai \
      --model_name=deepseek-ai/DeepSeek-V3 --bible=glassharbor
"""

from collections.abc import Sequence

from absl import app
from absl import flags
from absl import logging
from concordia.command_line_interface import concordia_log
from examples.adventure import evaluate_run
from examples.adventure import playtester_bot
from examples.adventure import runner


def define_flags() -> None:
  """Defines command-line flags when invoked as an entry point."""
  flags.DEFINE_enum(
      'mode',
      'playtest',
      ['playtest', 'server', 'log_cli', 'evaluate_dir', 'author_world'],
      'Execution mode: headless playtest, HTTP server, log_cli, evaluate_dir, '
      'or author_world.',
  )
  flags.DEFINE_enum(
      'bot',
      'CuriousExplorer',
      [
          'CuriousExplorer',
          'ChaosBoundaryTester',
          'SocialInvestigator',
          'DetectiveInvestigator',
          'CouncilFollower',
      ],
      'Synthetic playtester archetype for --mode=playtest.',
  )
  flags.DEFINE_float(
      'council_follow_rate',
      playtester_bot.DEFAULT_COUNCIL_FOLLOW_RATE,
      'Probability that the playtester follows an inner-voice council option '
      'on a non-movement turn (CouncilFollower always follows).',
  )
  flags.DEFINE_string(
      'target_voice',
      '',
      'Optional single Sub-Mind voice name to follow exclusively across the '
      'playtest (e.g. "Brass Neck", "The Book", "Proper Naughty").',
  )
  flags.DEFINE_integer(
      'steps', 16, 'Number of player turns to execute in playtest mode.'
  )
  flags.DEFINE_bool(
      'require_full_steps',
      False,
      'If True, continue running until all --steps complete even after '
      'ledger.story.ended becomes True.',
  )
  flags.DEFINE_string(
      'output_dir',
      '/tmp/adventure_run',
      'Directory for structured logs, world_ledger.json, and eval reports.',
  )
  flags.DEFINE_string(
      'host', '0.0.0.0', 'Bind address when running in --mode=server.'
  )
  flags.DEFINE_integer('port', 8765, 'TCP port when running in --mode=server.')
  flags.DEFINE_bool(
      'use_mock', False, 'Use fast deterministic mock LLM for offline testing.'
  )
  flags.DEFINE_bool(
      'disable_language_model',
      False,
      'Alias for --use_mock to run with the deterministic offline model.',
  )
  flags.DEFINE_string(
      'api_type',
      'google_aistudio_model',
      'Concordia language model API type (e.g. google_aistudio_model, '
      'together_ai, openai, ollama, pytorch_gemma).',
  )
  flags.DEFINE_string(
      'model_name',
      runner.DEFAULT_MODEL_NAME,
      'Language model name for the selected --api_type.',
  )
  flags.DEFINE_string(
      'api_key',
      '',
      'API key for the language model provider (or set GEMINI_API_KEY / '
      'GOOGLE_API_KEY in the environment).',
  )
  flags.DEFINE_integer(
      'max_model_calls',
      2000,
      'Maximum number of LLM calls before CallLimitLanguageModel degrades.',
  )
  flags.DEFINE_bool(
      'run_eval',
      True,
      'Run the two-layer evaluator after completing a playtest.',
  )
  flags.DEFINE_string(
      'bible',
      '',
      'World bible name ("bermondsey", "glassharbor", "vampire_romance", '
      '"dystopian_scifi") or path to a WorldBible JSON file.',
  )
  flags.DEFINE_string(
      'author_title',
      'The Brass-bound Ledger of Deptford Creek',
      'World title when --mode=author_world.',
  )
  flags.DEFINE_string(
      'author_premise',
      '',
      'Optional custom opening premise when --mode=author_world.',
  )
  flags.DEFINE_enum(
      'author_tone',
      'london_caper',
      [
          'london_caper',
          'noir_mystery',
          'vampire_romance',
          'dystopian_scifi',
      ],
      'Tone template when --mode=author_world.',
  )
  flags.DEFINE_string(
      'author_output',
      '/tmp/authored_bible.json',
      'Output JSON path when --mode=author_world.',
  )


def main(argv: Sequence[str]) -> None:
  values = flags.FLAGS
  if values.mode == 'log_cli':
    concordia_log.main(list(argv[1:]))
    return

  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if values.mode == 'author_world':
    runner.run_author_world_mode(
        title=values.author_title,
        premise=values.author_premise,
        tone_name=values.author_tone,
        output_path=values.author_output,
    )
    return

  model = runner.setup_language_model(
      use_mock=values.use_mock,
      disable_language_model=values.disable_language_model,
      api_type=values.api_type,
      model_name=values.model_name,
      api_key=values.api_key,
      max_calls=values.max_model_calls,
  )

  bible_spec = values.bible or ('bermondsey' if values.mode == 'server' else '')
  try:
    bible = runner.load_bible(bible_spec)
  except ValueError as err:
    raise app.UsageError(str(err)) from err

  if values.mode == 'evaluate_dir':
    report = evaluate_run.evaluate_artifacts_directory(
        values.output_dir, model=model
    )
    logging.info(
        'Evaluation complete: all_invariants_passed=%s, weighted_score=%.2f, '
        'lowest_dimension=%s',
        report['invariants']['all_passed'],
        report['autorater']['weighted_total'],
        report['autorater']['lowest_dimension'],
    )
    return

  if values.mode == 'server':
    runner.run_server_mode(
        model,
        host=values.host,
        port=values.port,
        max_steps=max(500, values.steps),
        output_dir=values.output_dir,
        bible=bible,
    )
  else:
    runner.run_playtest_mode(
        model,
        bot_name=values.bot,
        steps=values.steps,
        output_dir=values.output_dir,
        run_eval=values.run_eval,
        bible=bible,
        council_follow_rate=values.council_follow_rate,
        target_voice=values.target_voice,
        require_full_steps=values.require_full_steps,
    )


if __name__ == '__main__':
  define_flags()
  app.run(main)
