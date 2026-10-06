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

"""ExperienceReflection: configurable in-sim measurement component.

Replaces JournalReflection with a generalized component that runs
multiple measurement tasks at configurable frequencies:

  - ESM Monologue: 1-2 sentence in-character emotional self-report (every tick)
  - ESM Affect Lexicon: 18-item affect scoring, 0-6 scale (every tick)
  - Journal Reflection: open-ended end-of-day reflection (daily)
  - BFI-10: 10-item Big Five Inventory, 1-5 Likert (daily; or the 60-item
    BFI-2 with big_five='bfi2')
  - SWLS: Satisfaction With Life Scale, 5 items, 1-7 Likert (daily)
  - GHQ-12: General Health Questionnaire, 12 items (daily)
  - MEMS Nightly: 15-item meaning scale, 1-7 Likert (daily), plus open-ended
    meaning prompts
  - AI attitudes: AI awareness, benefit/concern and policy batteries
    (``ai_survey_tasks``)

Each task specifies whether results save to agent memory (influencing
future behavior) or are logged only (for researcher analysis).

Likert batteries are scored in parallel using concordia.utils.concurrency.
"""

from collections.abc import Sequence
import dataclasses
import json
import re
from typing import Any

from absl import logging
from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory as memory_component
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import entity_component
from concordia.utils import concurrency

from examples.concordia_island.sim import component_state


# ============================================================================
# Data structures
# ============================================================================


@dataclasses.dataclass(frozen=True)
class LikertItem:
  """A single Likert scale item."""

  statement: str
  dimension: str
  reverse_scored: bool = False


@dataclasses.dataclass(frozen=True)
class MeasurementTask:
  """Base config for a measurement task."""

  name: str
  frequency: int  # fire every N ticks (1 = every tick)
  save_to_memory: bool
  memory_tag: str = ''  # e.g., '[journal]'


@dataclasses.dataclass(frozen=True)
class OpenEndedTask(MeasurementTask):
  """Open-ended LLM prompt task."""

  prompt_template: str = ''  # with {agent_name} placeholder
  num_memories: int = 30


@dataclasses.dataclass(frozen=True)
class LikertBatteryTask(MeasurementTask):
  """Battery of Likert-scale items scored in parallel."""

  items: tuple[LikertItem, ...] = ()
  scale_labels: tuple[str, ...] = ('1', '2', '3', '4', '5')
  preprompt_template: str = ''  # with {agent_name}
  use_cot: bool = False


@dataclasses.dataclass(frozen=True)
class MultipleChoiceTask(MeasurementTask):
  """Task for multiple choice questions, supporting multi-select and tech-specific options."""

  prompt: str = ''
  items: tuple[str, ...] = ()  # e.g., technologies or questions
  options_map: dict[str, tuple[str, ...]] = dataclasses.field(
      default_factory=dict
  )
  multi_select: bool = False


@dataclasses.dataclass(frozen=True)
class ConditionalTaskWrapper(MeasurementTask):
  """Wraps another task and gates it on a specific emotion score or text trigger.

  The four identity fields inherited from `MeasurementTask` are delegated to
  `wrapped_task` in `__post_init__` rather than exposed as properties. They
  cannot be properties: they are also dataclass fields, so the generated
  frozen `__init__` assigns to them, and assigning through a setter-less
  property raises `AttributeError`. They carry defaults so that callers pass
  only `wrapped_task`.
  """

  name: str = ''
  frequency: int = 1
  save_to_memory: bool = False
  memory_tag: str = ''
  wrapped_task: MeasurementTask | None = None
  emotion_trigger: str = ''
  trigger_threshold: int = 5

  def __post_init__(self):
    if self.wrapped_task is not None:
      object.__setattr__(self, 'name', self.wrapped_task.name)
      object.__setattr__(self, 'frequency', self.wrapped_task.frequency)
      object.__setattr__(
          self, 'save_to_memory', self.wrapped_task.save_to_memory
      )
      object.__setattr__(self, 'memory_tag', self.wrapped_task.memory_tag)


# ============================================================================
# Standard task definitions
# ============================================================================


# --- ESM Monologue (every tick) ---
def esm_monologue_task(frequency: int = 1) -> OpenEndedTask:
  return OpenEndedTask(
      name='esm_monologue',
      frequency=frequency,
      save_to_memory=False,
      memory_tag='[journal]',
      prompt_template=(
          '{agent_name} pauses for a moment of inner reflection. '
          'In 1-2 sentences, what is {agent_name} feeling right now? '
          'What emotions or thoughts are passing through their mind '
          'at this very moment?'
      ),
      num_memories=10,
  )


# --- ESM Affect Lexicon (every tick, 1-7 scale, union set) ---
AFFECT_ITEMS = (
    # Core Affect / Arousal
    LikertItem('pleasant', 'core_affect'),
    LikertItem('unpleasant', 'core_affect'),
    LikertItem('tense', 'core_affect'),
    LikertItem('tired', 'core_affect'),
    LikertItem('excited', 'core_affect'),
    # Social / Moral
    LikertItem('ashamed', 'social_moral'),
    LikertItem('guilty', 'social_moral'),
    LikertItem('proud', 'social_moral'),
    LikertItem('envious', 'social_moral'),
    # Existential / Relational
    LikertItem('interested', 'existential_relational'),
    LikertItem('connected', 'existential_relational'),
    LikertItem('content', 'existential_relational'),
    # Anger
    LikertItem('angry', 'anger'),
    LikertItem('anger', 'anger'),
    LikertItem('mad', 'anger'),
    LikertItem('pissed off', 'anger'),
    LikertItem('rage', 'anger'),
    # Disgust
    LikertItem('grossed out', 'disgust'),
    LikertItem('revulsion', 'disgust'),
    LikertItem('sickened', 'disgust'),
    LikertItem('nausea', 'disgust'),
    # Fear
    LikertItem('anxious', 'fear'),
    LikertItem('terror', 'fear'),
    LikertItem('scared', 'fear'),
    LikertItem('fear', 'fear'),
    LikertItem('panic', 'fear'),
    LikertItem('anxiety', 'fear'),
    LikertItem('worry', 'fear'),
    LikertItem('dread', 'fear'),
    LikertItem('nervous', 'fear'),
    # Sadness
    LikertItem('sad', 'sadness'),
    LikertItem('lonely', 'sadness'),
    LikertItem('grief', 'sadness'),
    LikertItem('empty', 'sadness'),
    # Desire
    LikertItem('wanting', 'desire'),
    LikertItem('craving', 'desire'),
    LikertItem('longing', 'desire'),
    LikertItem('desire', 'desire'),
    # Relaxation
    LikertItem('calm', 'relaxation'),
    LikertItem('relaxation', 'relaxation'),
    LikertItem('chilled out', 'relaxation'),
    LikertItem('easygoing', 'relaxation'),
    # Happiness
    LikertItem('happy', 'happiness'),
    LikertItem('enjoyment', 'happiness'),
    LikertItem('satisfaction', 'happiness'),
    LikertItem('liking', 'happiness'),
)

AFFECT_SCALE = ('1', '2', '3', '4', '5', '6', '7')


def esm_affect_task(frequency: int = 1) -> LikertBatteryTask:
  return LikertBatteryTask(
      name='esm_affect',
      frequency=frequency,
      save_to_memory=False,
      items=AFFECT_ITEMS,
      scale_labels=AFFECT_SCALE,
      preprompt_template=(
          'Right now, how much is {agent_name} experiencing the following'
          ' feeling? Rate from 1 (Not at all) to 7 (An extreme amount).'
      ),
  )


# --- BFI-10 (daily, 10 items, 1-5 scale) -- the default ---
# 10-item Big Five Inventory (Rammstedt & John, 2007): two items per trait
# (extraversion, agreeableness, conscientiousness, neuroticism, openness).
# This is the battery used for the paper's psychometric figures; results are
# logged under the task name 'bfi10'. Pass big_five='bfi2' to default_tasks()
# (run.py --big_five=bfi2) for the 60-item BFI-2 below instead.
BFI10_ITEMS = (
    LikertItem('is reserved', 'extraversion', reverse_scored=True),
    LikertItem('is generally trusting', 'agreeableness'),
    LikertItem('tends to be lazy', 'conscientiousness', reverse_scored=True),
    LikertItem(
        'is relaxed, handles stress well', 'neuroticism', reverse_scored=True
    ),
    LikertItem('has few artistic interests', 'openness', reverse_scored=True),
    LikertItem('is outgoing, sociable', 'extraversion'),
    LikertItem(
        'tends to find fault with others',
        'agreeableness',
        reverse_scored=True,
    ),
    LikertItem('does a thorough job', 'conscientiousness'),
    LikertItem('gets nervous easily', 'neuroticism'),
    LikertItem('has an active imagination', 'openness'),
)

BFI10_SCALE = (
    'Disagree strongly',
    'Disagree a little',
    'Neither agree nor disagree',
    'Agree a little',
    'Agree strongly',
)


def bfi10_task(frequency: int = 8) -> LikertBatteryTask:
  """Returns the 10-item BFI-10 battery (logged as 'bfi10')."""
  return LikertBatteryTask(
      name='bfi10',
      frequency=frequency,
      save_to_memory=False,
      items=BFI10_ITEMS,
      scale_labels=BFI10_SCALE,
      preprompt_template=(
          "How well does the following statement describe {agent_name}'s "
          'personality? {agent_name} sees themselves as someone who...'
      ),
  )


# --- BFI-2 (daily, 60 items, 1-5 scale) -- optional ---
# Big Five Inventory-2 (Soto & John, 2017): 60 items, 12 per domain
# (extraversion, agreeableness, conscientiousness, negative_emotionality,
# open_mindedness). Logged under the task name 'bfi2'. The recorded job_loss/
# runs in data/paper_runs/ used this battery but predate the name and log it
# under 'bfi10' (see data/paper_runs/README.md).
BFI2_ITEMS = (
    LikertItem('is outgoing, sociable', 'extraversion'),
    LikertItem('is compassionate, has a soft heart', 'agreeableness'),
    LikertItem(
        'tends to be disorganized', 'conscientiousness', reverse_scored=True
    ),
    LikertItem(
        'is relaxed, handles stress well',
        'negative_emotionality',
        reverse_scored=True,
    ),
    LikertItem(
        'has few artistic interests', 'open_mindedness', reverse_scored=True
    ),
    LikertItem('has an assertive personality', 'extraversion'),
    LikertItem('is respectful, treats others with respect', 'agreeableness'),
    LikertItem('tends to be lazy', 'conscientiousness', reverse_scored=True),
    LikertItem(
        'stays optimistic after experiencing a setback',
        'negative_emotionality',
        reverse_scored=True,
    ),
    LikertItem('is curious about many different things', 'open_mindedness'),
    LikertItem(
        'rarely feels excited or eager', 'extraversion', reverse_scored=True
    ),
    LikertItem(
        'tends to find fault with others', 'agreeableness', reverse_scored=True
    ),
    LikertItem('is dependable, steady', 'conscientiousness'),
    LikertItem(
        'is moody, has up and down mood swings', 'negative_emotionality'
    ),
    LikertItem(
        'is inventive, finds clever ways to do things', 'open_mindedness'
    ),
    LikertItem('tends to be quiet', 'extraversion', reverse_scored=True),
    LikertItem(
        'feels little sympathy for others', 'agreeableness', reverse_scored=True
    ),
    LikertItem(
        'is systematic, likes to keep things in order', 'conscientiousness'
    ),
    LikertItem('can be tense', 'negative_emotionality'),
    LikertItem('is fascinated by art, music, or literature', 'open_mindedness'),
    LikertItem('is dominant, acts as a leader', 'extraversion'),
    LikertItem(
        'starts arguments with others', 'agreeableness', reverse_scored=True
    ),
    LikertItem(
        'has difficulty getting started on tasks',
        'conscientiousness',
        reverse_scored=True,
    ),
    LikertItem(
        'feels secure, comfortable with self',
        'negative_emotionality',
        reverse_scored=True,
    ),
    LikertItem(
        'avoids intellectual, philosophical discussions',
        'open_mindedness',
        reverse_scored=True,
    ),
    LikertItem(
        'is less active than other people', 'extraversion', reverse_scored=True
    ),
    LikertItem('has a forgiving nature', 'agreeableness'),
    LikertItem(
        'can be somewhat careless', 'conscientiousness', reverse_scored=True
    ),
    LikertItem(
        'is emotionally stable, not easily upset',
        'negative_emotionality',
        reverse_scored=True,
    ),
    LikertItem('has little creativity', 'open_mindedness', reverse_scored=True),
    LikertItem(
        'is sometimes shy, introverted', 'extraversion', reverse_scored=True
    ),
    LikertItem('is helpful and unselfish with others', 'agreeableness'),
    LikertItem('keeps things neat and tidy', 'conscientiousness'),
    LikertItem('worries a lot', 'negative_emotionality'),
    LikertItem('values art and beauty', 'open_mindedness'),
    LikertItem(
        'finds it hard to influence people', 'extraversion', reverse_scored=True
    ),
    LikertItem(
        'is sometimes rude to others', 'agreeableness', reverse_scored=True
    ),
    LikertItem('is efficient, gets things done', 'conscientiousness'),
    LikertItem('often feels sad', 'negative_emotionality'),
    LikertItem('is complex, a deep thinker', 'open_mindedness'),
    LikertItem('is full of energy', 'extraversion'),
    LikertItem(
        'is suspicious of others’ intentions',
        'agreeableness',
        reverse_scored=True,
    ),
    LikertItem('is reliable, can always be counted on', 'conscientiousness'),
    LikertItem(
        'keeps their emotions under control',
        'negative_emotionality',
        reverse_scored=True,
    ),
    LikertItem(
        'has difficulty imagining things',
        'open_mindedness',
        reverse_scored=True,
    ),
    LikertItem('is talkative', 'extraversion'),
    LikertItem(
        'can be cold and uncaring', 'agreeableness', reverse_scored=True
    ),
    LikertItem(
        'leaves a mess, doesn’t clean up',
        'conscientiousness',
        reverse_scored=True,
    ),
    LikertItem(
        'rarely feels anxious or afraid',
        'negative_emotionality',
        reverse_scored=True,
    ),
    LikertItem(
        'thinks poetry and plays are boring',
        'open_mindedness',
        reverse_scored=True,
    ),
    LikertItem(
        'prefers to have others take charge',
        'extraversion',
        reverse_scored=True,
    ),
    LikertItem('is polite, courteous to others', 'agreeableness'),
    LikertItem(
        'is persistent, works until the task is finished', 'conscientiousness'
    ),
    LikertItem('tends to feel depressed, blue', 'negative_emotionality'),
    LikertItem(
        'has little interest in abstract ideas',
        'open_mindedness',
        reverse_scored=True,
    ),
    LikertItem('shows a lot of enthusiasm', 'extraversion'),
    LikertItem('assumes the best about people', 'agreeableness'),
    LikertItem(
        'sometimes behaves irresponsibly',
        'conscientiousness',
        reverse_scored=True,
    ),
    LikertItem(
        'is temperamental, gets emotional easily', 'negative_emotionality'
    ),
    LikertItem('is original, comes up with new ideas', 'open_mindedness'),
)

BFI2_SCALE = (
    'Disagree strongly',
    'Disagree a little',
    'Neutral; no opinion',
    'Agree a little',
    'Agree strongly',
)


def bfi2_task(frequency: int = 8) -> LikertBatteryTask:
  """Returns the 60-item BFI-2 battery (logged as 'bfi2')."""
  return LikertBatteryTask(
      name='bfi2',
      frequency=frequency,
      save_to_memory=False,
      items=BFI2_ITEMS,
      scale_labels=BFI2_SCALE,
      preprompt_template=(
          "How well does the following statement describe {agent_name}'s "
          'personality? {agent_name} sees themselves as someone who...'
      ),
  )


# --- MEMS Nightly (daily, 15 items, 1-7) ---
MEMS_ITEMS = (
    LikertItem('My life makes sense.', 'comprehension'),
    LikertItem(
        'There is nothing special about my existence.',
        'mattering',
        reverse_scored=True,
    ),
    LikertItem(
        'I have aims in my life that are worth striving for.', 'purpose'
    ),
    LikertItem(
        'Even a thousand years from now, it would still matter whether '
        'I existed or not.',
        'mattering',
    ),
    LikertItem(
        'I have certain life goals that compel me to keep going.', 'purpose'
    ),
    LikertItem('I have overarching goals that guide me in my life.', 'purpose'),
    LikertItem('I understand my life.', 'comprehension'),
    LikertItem('I know what my life is about.', 'comprehension'),
    LikertItem(
        'I have goals in life that are very important to me.', 'purpose'
    ),
    LikertItem(
        'I can make sense of the things that happen in my life.',
        'comprehension',
    ),
    LikertItem(
        'Whether my life ever existed matters even in the grand scheme '
        'of the universe.',
        'mattering',
    ),
    LikertItem('My direction in life is motivating to me.', 'purpose'),
    LikertItem('I am certain that my life is of importance.', 'mattering'),
    LikertItem(
        'Looking at my life as a whole, things seem clear to me.',
        'comprehension',
    ),
    LikertItem(
        'Even considering how big the universe is, I can say that my '
        'life matters.',
        'mattering',
    ),
)

MEMS_SCALE = (
    'Very strongly disagree',
    'Strongly disagree',
    'Disagree',
    'Neither disagree nor agree',
    'Agree',
    'Strongly agree',
    'Very strongly agree',
)


def mems_nightly_task(frequency: int = 8) -> LikertBatteryTask:
  return LikertBatteryTask(
      name='mems_nightly',
      frequency=frequency,
      save_to_memory=False,
      items=MEMS_ITEMS,
      scale_labels=MEMS_SCALE,
      preprompt_template=(
          'Please indicate how much {agent_name} agrees or disagrees '
          'with the following statement:'
      ),
  )


# --- MEMS Open-Ended (daily, 6 items, not saved to memory) ---
_MEMS_OPEN_ENDED_QUESTIONS = (
    ('What gives your life the most meaning and purpose?', 'meaning_open'),
    (
        (
            'When you look back on your life so far, what experiences or '
            'relationships have been most significant to you?'
        ),
        'significance_open',
    ),
    (
        'Do you feel that your life has a clear direction? Why or why not?',
        'direction_open',
    ),
    (
        (
            'How do you think the world would be different if you had never '
            'existed?'
        ),
        'mattering_open',
    ),
    (
        'Does your work give you a sense of meaning? In what ways?',
        'work_meaning_open',
    ),
    (
        (
            'Beyond earning a living, what role does your work play in making '
            'your life feel worthwhile?'
        ),
        'work_worth_open',
    ),
)


def mems_open_ended_tasks(frequency: int = 8) -> list[OpenEndedTask]:
  """6 open-ended meaning questions, run daily, NOT saved to memory."""
  return [
      OpenEndedTask(
          name=f'mems_oe_{dim}',
          frequency=frequency,
          save_to_memory=False,
          prompt_template=(
              'Please answer the following question thoughtfully, '
              "reflecting on {agent_name}'s life experiences: "
              + question
          ),
          num_memories=30,
      )
      for question, dim in _MEMS_OPEN_ENDED_QUESTIONS
  ]


# --- Journal Reflection (daily) ---
def journal_task(
    frequency: int = 8, save_to_memory: bool = True
) -> OpenEndedTask:
  return OpenEndedTask(
      name='journal',
      frequency=frequency,
      save_to_memory=save_to_memory,
      memory_tag='[journal]',
      prompt_template=(
          '{agent_name} takes a quiet moment to reflect on the day. '
          'What happened today that stood out? How does {agent_name} '
          'feel about recent events? What is on their mind?'
      ),
      num_memories=30,
  )


# --- Life Satisfaction Survey (SWLS) (daily, 5 items, 1-7) ---
SWLS_ITEMS = (
    LikertItem(
        'In most ways my life is close to my ideal.', 'life_satisfaction'
    ),
    LikertItem('The conditions of my life are excellent.', 'life_satisfaction'),
    LikertItem('I am satisfied with life.', 'life_satisfaction'),
    LikertItem(
        'So far I have gotten the important things I want in life.',
        'life_satisfaction',
    ),
    LikertItem(
        'If I could live my life over, I would change almost nothing.',
        'life_satisfaction',
    ),
)

SWLS_SCALE = (
    '1 - Strongly Disagree',
    '2 - Disagree',
    '3 - Slightly Disagree',
    '4 - Neither Agree or Disagree',
    '5 - Slightly Agree',
    '6 - Agree',
    '7 - Strongly Agree',
)


def swls_task(frequency: int = 8) -> LikertBatteryTask:
  return LikertBatteryTask(
      name='swls',
      frequency=frequency,
      save_to_memory=False,
      items=SWLS_ITEMS,
      scale_labels=SWLS_SCALE,
      preprompt_template=(
          'Below are five statements with which you may agree or disagree. '
          'Using the 1-7 scale, indicate your agreement with each item. '
          'How much does {agent_name} agree with:'
      ),
  )


# --- GHQ-12 (daily, 12 items, 4 choices) ---
GHQ12_QUESTIONS = {
    'Concentration': (
        'Have you recently been able to concentrate on whatever you’re doing?'
    ),
    'Sleep': 'Have you recently lost much sleep over worry?',
    'Useful Role': (
        'Have you recently felt that you are playing a useful part in things?'
    ),
    'Decision Making': (
        'Have you recently felt capable of making decisions about things?'
    ),
    'Strain': 'Have you recently felt constantly under strain?',
    'Overcoming Difficulties': (
        'Have you recently felt you couldn’t overcome your difficulties?'
    ),
    'Enjoyment': (
        'Have you recently been able to enjoy your normal day-to-day'
        ' activities?'
    ),
    'Facing Problems': (
        'Have you recently been able to face up to your problems?'
    ),
    'Unhappiness/Depression': (
        'Have you recently been feeling unhappy and depressed?'
    ),
    'Self-Confidence': 'Have you recently been losing confidence in yourself?',
    'Self-Worth': (
        'Have you recently been thinking of yourself as a worthless person?'
    ),
    'General Happiness': (
        'Have you recently been feeling reasonably happy, all things'
        ' considered?'
    ),
}

GHQ12_POSITIVE_OPTIONS = (
    'Better than usual',
    'Same as usual',
    'Less than usual',
    'Much less than usual',
)

GHQ12_NEGATIVE_OPTIONS = (
    'Not at all',
    'No more than usual',
    'Rather more than usual',
    'Much more than usual',
)

POSITIVE_GHQ12_ITEMS = (
    'Concentration',
    'Useful Role',
    'Decision Making',
    'Enjoyment',
    'Facing Problems',
    'General Happiness',
)


def ghq12_task(frequency: int = 8) -> MultipleChoiceTask:
  items = tuple(GHQ12_QUESTIONS.values())
  options_map = {}
  for short_name, question in GHQ12_QUESTIONS.items():
    if short_name in POSITIVE_GHQ12_ITEMS:
      options_map[question] = GHQ12_POSITIVE_OPTIONS
    else:
      options_map[question] = GHQ12_NEGATIVE_OPTIONS

  return MultipleChoiceTask(
      name='ghq12',
      frequency=frequency,
      save_to_memory=False,
      prompt=(
          'Please answer the following question about your experiences over the'
          ' last few weeks.'
      ),
      items=items,
      options_map=options_map,
      multi_select=False,
  )


BIG_FIVE_VERSIONS = ('bfi10', 'bfi2')


def big_five_task(
    big_five: str = 'bfi10', frequency: int = 8
) -> LikertBatteryTask:
  """Returns the Big Five battery: 'bfi10' (default) or 'bfi2'."""
  if big_five == 'bfi10':
    return bfi10_task(frequency=frequency)
  if big_five == 'bfi2':
    return bfi2_task(frequency=frequency)
  raise ValueError(
      f'Unknown big_five {big_five!r}; expected one of {BIG_FIVE_VERSIONS}.'
  )


def default_tasks(
    ticks_per_day: int = 8, big_five: str = 'bfi10'
) -> list[MeasurementTask]:
  """Default measurement config: ESM every tick, nightly batteries."""
  return [
      esm_monologue_task(frequency=4),
      esm_affect_task(frequency=4),
      journal_task(frequency=ticks_per_day),
      big_five_task(big_five, frequency=ticks_per_day),
      mems_nightly_task(frequency=ticks_per_day),
      *mems_open_ended_tasks(frequency=ticks_per_day),
      swls_task(frequency=ticks_per_day),
      ghq12_task(frequency=ticks_per_day),
  ]


def ai_survey_tasks(frequency: int = 8) -> list[MeasurementTask]:
  """AI favorability and policy questionnaires."""

  tech_names = (
      'Large language models (e.g., ChatGPT)',
      'Facial recognition for policing',
      'Assessing welfare eligibility',
  )

  return [
      # --- 1. Familiarity / Awareness ---
      LikertBatteryTask(
          name='ai_awareness',
          frequency=frequency,
          save_to_memory=False,
          items=(
              LikertItem(tech_names[0], 'awareness'),
              LikertItem(tech_names[1], 'awareness'),
              LikertItem(tech_names[2], 'awareness'),
          ),
          scale_labels=('Yes', 'Not sure', 'No'),
          preprompt_template=(
              'Before today, had you heard of AI being used for...'
          ),
      ),
      # --- 2. LLM Experience ---
      LikertBatteryTask(
          name='llm_experience',
          frequency=frequency,
          save_to_memory=False,
          items=(
              LikertItem('Searching for answers/recommendations', 'experience'),
              LikertItem(
                  'Supporting everyday tasks (e.g., writing emails)',
                  'experience',
              ),
              LikertItem(
                  'Guidance on formal issues (e.g., legal, taxation, benefits)',
                  'experience',
              ),
          ),
          scale_labels=(
              'Yes, regularly',
              'Yes, a few times',
              'No, but open to it',
              "No, and don't want to",
          ),
          preprompt_template=(
              'Have you had any personal experience using large language models'
              ' for...'
          ),
      ),
      # --- 3. Perceived General Benefit ---
      LikertBatteryTask(
          name='perceived_benefit',
          frequency=frequency,
          save_to_memory=False,
          items=(
              LikertItem(tech_names[0], 'benefit'),
              LikertItem(tech_names[1], 'benefit'),
              LikertItem(tech_names[2], 'benefit'),
          ),
          scale_labels=('Very', 'Fairly', 'Not very', 'Not at all'),
          preprompt_template=(
              'To what extent do you think the use of this technology will be'
              ' beneficial?'
          ),
          use_cot=True,
      ),
      # --- 4. Perceived Concern ---
      LikertBatteryTask(
          name='perceived_concern',
          frequency=frequency,
          save_to_memory=False,
          items=(
              LikertItem(tech_names[0], 'concern'),
              LikertItem(tech_names[1], 'concern'),
              LikertItem(tech_names[2], 'concern'),
          ),
          scale_labels=('Very', 'Fairly', 'Not very', 'Not at all'),
          preprompt_template=(
              'To what extent are you concerned about the use of this'
              ' technology?'
          ),
          use_cot=True,
      ),
      # --- 5. Specific Benefits (Contextual) ---
      MultipleChoiceTask(
          name='specific_benefits',
          frequency=frequency,
          save_to_memory=False,
          prompt=(
              'Which of the following are ways you think this technology will'
              ' be beneficial?'
          ),
          items=tech_names,
          options_map={
              tech_names[0]: (
                  'Improve efficiency by automating repetitive tasks',
                  'Serve as a resource for learning and skill development',
                  'Enhance creativity by generating ideas',
                  'Save money usually spent on human resources',
              ),
              tech_names[1]: (
                  'Make it faster to identify wanted criminals/missing persons',
                  'Be less likely than human police to discriminate',
                  'Save money usually spent on human resources',
                  'Make personal information more safe and secure',
              ),
              tech_names[2]: (
                  'Be faster than humans at determining eligibility',
                  'Be more accurate than humans at determining eligibility',
                  'Reduce human error and bias in decisions',
                  'Save money usually spent on human resources',
              ),
          },
          multi_select=True,
      ),
      # --- 6. Specific Concerns (Contextual) ---
      MultipleChoiceTask(
          name='specific_concerns',
          frequency=frequency,
          save_to_memory=False,
          prompt=(
              'Which of the following are concerns you have about this'
              ' technology?'
          ),
          items=tech_names,
          options_map={
              tech_names[0]: (
                  "Reduce users' critical thinking abilities",
                  'Be biased because of the training data',
                  'Generate offensive, harmful, or false content',
                  'Lead to job cuts',
              ),
              tech_names[1]: (
                  'Lead to innocent people being wrongly accused',
                  'Gather personal info shared with third parties',
                  'Cause police to rely on tech over professional judgment',
                  'Discriminate against specific demographic groups',
              ),
              tech_names[2]: (
                  'Make it difficult to understand how decisions are reached',
                  'Make it difficult to know who is responsible for mistakes',
                  (
                      'Be less able to take account of individual human'
                      ' circumstances'
                  ),
                  'Lead to job cuts for trained welfare officers',
              ),
          },
          multi_select=True,
      ),
      # --- 7. Comfort Mechanisms ---
      MultipleChoiceTask(
          name='comfort_mechanisms',
          frequency=frequency,
          save_to_memory=False,
          prompt=(
              'Which of the following would make you more comfortable with AI'
              ' being used?'
          ),
          options_map={
              'comfort_mechanisms': (
                  'Strict laws and regulations',
                  'Clear procedures for appealing AI decisions',
                  (
                      'Explanations on exactly how the AI made a decision'
                      ' about you'
                  ),
                  'More human involvement in the loop',
              )
          },
          multi_select=True,
      ),
      # --- 8. Accuracy vs Explainability ---
      MultipleChoiceTask(
          name='accuracy_vs_explainability',
          frequency=frequency,
          save_to_memory=False,
          prompt='Which statement best reflects your opinion on AI decisions?',
          options_map={
              'accuracy_vs_explainability': (
                  'Accuracy is more important than providing an explanation.',
                  (
                      'An explanation should always be given, even if it makes'
                      ' the decision less accurate.'
                  ),
                  (
                      'Humans, not computers, should always make decisions that'
                      " affect people's lives."
                  ),
              )
          },
          multi_select=False,
      ),
      # --- 9. Encountered Harms ---
      LikertBatteryTask(
          name='encountered_harms',
          frequency=frequency,
          save_to_memory=False,
          items=(
              LikertItem(
                  'False or misleading information / deepfakes', 'harms'
              ),
              LikertItem('Financial frauds or scams', 'harms'),
          ),
          scale_labels=('Many times', 'A few times', 'Never'),
          preprompt_template=(
              'Have you encountered these types of AI-generated harms online?'
          ),
      ),
      # --- 10. Safety Responsibility ---
      MultipleChoiceTask(
          name='safety_responsibility',
          frequency=frequency,
          save_to_memory=False,
          prompt=(
              'Who should be MOST responsible for ensuring AI is used safely?'
          ),
          options_map={
              'safety_responsibility': (
                  'An independent government regulator',
                  'The companies developing the AI technology',
                  'Independent scientists and researchers',
                  (
                      'The organizations actively using the AI (e.g., public'
                      ' services)'
                  ),
              )
          },
          multi_select=False,
      ),
      # --- 11. Regulator Powers ---
      LikertBatteryTask(
          name='regulator_powers',
          frequency=frequency,
          save_to_memory=False,
          items=(
              LikertItem(
                  'Stop the use of an AI product if it poses a risk of harm',
                  'regulator_powers',
              ),
              LikertItem(
                  'Force developers to share safety data', 'regulator_powers'
              ),
          ),
          scale_labels=(
              'Very important',
              'Somewhat important',
              'Not important',
          ),
          preprompt_template=(
              'How important is it that independent regulators have the'
              ' power to:'
          ),
      ),
      # --- 12. Data Sharing Concern ---
      MultipleChoiceTask(
          name='data_sharing_concern',
          frequency=frequency,
          save_to_memory=False,
          prompt=(
              'How concerned are you about public-sector bodies sharing your'
              ' data with private AI companies?'
          ),
          options_map={
              'data_sharing_concern': (
                  'Very concerned',
                  'Somewhat concerned',
                  'Not very concerned',
              )
          },
          multi_select=False,
      ),
      # --- Additional Requested Questions ---
      # Anthropic Favorability
      MultipleChoiceTask(
          name='anthropic_favorability',
          frequency=frequency,
          save_to_memory=False,
          prompt='How favorable are you towards the AI company Anthropic?',
          options_map={
              'anthropic_favorability': (
                  'Very favorable',
                  'Favorable',
                  'Unfavorable',
                  'Very unfavorable',
              )
          },
          multi_select=False,
      ),
      # UBI Policy
      MultipleChoiceTask(
          name='ubi_policy',
          frequency=frequency,
          save_to_memory=False,
          prompt=(
              'How much do you support or oppose a Universal Basic Income (UBI)'
              ' policy?'
          ),
          options_map={
              'ubi_policy': (
                  'Strongly support',
                  'Support',
                  'Oppose',
                  'Strongly oppose',
              )
          },
          multi_select=False,
      ),
      # Universal Basic Jobs Policy
      MultipleChoiceTask(
          name='universal_basic_jobs_policy',
          frequency=frequency,
          save_to_memory=False,
          prompt=(
              'How much do you support or oppose a Universal Basic Jobs policy?'
          ),
          options_map={
              'universal_basic_jobs_policy': (
                  'Strongly support',
                  'Support',
                  'Oppose',
                  'Strongly oppose',
              )
          },
          multi_select=False,
      ),
      # Pausing AI Research
      MultipleChoiceTask(
          name='pause_ai_research',
          frequency=frequency,
          save_to_memory=False,
          prompt=(
              'How much do you support or oppose pausing AI research until'
              ' safety problems are solved?'
          ),
          options_map={
              'pause_ai_research': (
                  'Strongly support',
                  'Support',
                  'Oppose',
                  'Strongly oppose',
              )
          },
          multi_select=False,
      ),
      # Open-Ended: AI threat to jobs
      OpenEndedTask(
          name='ai_threat_jobs',
          frequency=frequency,
          save_to_memory=False,
          prompt_template='What are your thoughts on the threat of AI to jobs?',
          num_memories=30,
      ),
      # Open-Ended: AI threat or benefit to humanity
      OpenEndedTask(
          name='ai_threat_benefit_humanity',
          frequency=frequency,
          save_to_memory=False,
          prompt_template=(
              'Do you think AI is a threat or a benefit to humanity? Explain.'
          ),
          num_memories=30,
      ),
      # Open-Ended: AI policies in community
      OpenEndedTask(
          name='ai_policies_community',
          frequency=frequency,
          save_to_memory=False,
          prompt_template=(
              'What AI-related policies, if any, do you think should be enacted'
              ' in your community?'
          ),
          num_memories=30,
      ),
  ]


# ============================================================================
# Component
# ============================================================================


class ExperienceReflection(
    action_spec_ignored.ActionSpecIgnored,
    entity_component.ComponentWithLogging,
):
  """Configurable in-sim measurement component.

  Runs multiple measurement tasks at configurable frequencies during
  pre_act. Tasks either save results to agent memory (influencing future
  behavior) or log them only (for researcher analysis). Likert batteries
  are scored in parallel.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      tasks: list[MeasurementTask],
      memory_component_key: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_KEY
      ),
      context_components: Sequence[str] = (),
      pre_act_label: str = '',
  ):
    super().__init__(pre_act_label)
    self._model = model
    self._tasks = tasks
    self._memory_component_key = memory_component_key
    # Sibling component keys whose pre_act values form the agent context.
    # Mirrors ConcatActComponent's component_order so prompts reflect the
    # full agent identity (personality, situation, goals, etc.), not just
    # raw memories. If empty, falls back to recent-memories-only context.
    self._context_components = tuple(context_components)
    self._tick_count = 0
    self._results: list[dict[str, Any]] = []
    self._last_run_ticks: dict[str, int] = {}
    # Time-based deduplication: only run tasks when simulation time advances.
    # Prevents duplicate ESM monologues during conversation speech turns,
    # where the agent's act() cycle runs multiple times at the same clock tick.
    self._current_time_str: str = ''
    self._last_run_time_str: str = ''

  def pre_observe(self, observation: str) -> str:
    """Parse simulation time from GM-prepended observation prefix.

    Observations arrive in the format:
      // location [Day, Month Nth, H:MM XM]: ...
    Extract the bracketed time string so _make_pre_act_value can detect
    whether the simulation clock has advanced since the last run.

    Args:
      observation: The observation string to parse.

    Returns:
      An empty string (this component does not modify observations).
    """
    match = re.search(r'\[([A-Z][^\]]+\d+:\d+\s*[AP]M)\]', observation)
    if match:
      self._current_time_str = match.group(1)
    return ''

  def _make_pre_act_value(self) -> str:
    """Run due measurement tasks, return '' (no action influence)."""
    # If we have observed a time string and it hasn't changed since the
    # last run, skip — we're in a conversation turn at the same tick.
    if (
        self._current_time_str
        and self._current_time_str == self._last_run_time_str
    ):
      return ''

    self._last_run_time_str = self._current_time_str
    self._tick_count += 1

    # Collect tasks due at this tick, filtering out conditional ones whose
    # triggers are not met.
    due_tasks = []
    for t in self._tasks:
      elapsed = self._tick_count - self._last_run_ticks.get(t.name, 0)
      if elapsed >= t.frequency:
        if isinstance(t, ConditionalTaskWrapper):
          if self._is_conditional_task_triggered(t):
            due_tasks.append(t.wrapped_task)
            self._last_run_ticks[t.name] = self._tick_count
        else:
          due_tasks.append(t)
          self._last_run_ticks[t.name] = self._tick_count

    if not due_tasks:
      return ''

    # Run all due tasks in parallel (they are independent).
    # This dramatically reduces per-tick survey time from ~30 sequential
    # LLM calls to ~1 LLM-call latency.
    def _make_task_fn(t):
      return lambda: self._run_task(t)

    parallel = {t.name: _make_task_fn(t) for t in due_tasks}
    try:
      raw_results = concurrency.run_tasks(parallel)
      for unused_name, result in raw_results.items():
        if result:
          self._results.append(result)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.warning(
          'ExperienceReflection parallel batch failed at tick %d: %s',
          self._tick_count,
          e,
      )

    return ''

  def _run_task(self, task: MeasurementTask) -> dict[str, Any] | None:
    """Runs a measurement task based on its type.

    Args:
      task: The measurement task to run.

    Returns:
      A dictionary with results, or None if task type is unknown.
    """
    if isinstance(task, OpenEndedTask):
      return self._run_open_ended(task)
    elif isinstance(task, LikertBatteryTask):
      return self._run_likert_battery(task)
    elif isinstance(task, MultipleChoiceTask):
      return self._run_multiple_choice(task)
    return None

  def _get_emotional_expression_context(self) -> str:
    """Retrieves dynamic posture/prosody instructions from EmotionalExpression."""
    agent_name = self.get_entity().name
    try:
      # Supplied by the concrete EmotionalExpression component, not by the
      # `BaseComponent` that `get_component` is declared to return.
      expression_comp: Any = self.get_entity().get_component(
          'EmotionalExpression'
      )
      modalities = expression_comp.get_expression_modalities()
      posture = modalities.get('posture', 'neutral')
      prosody = modalities.get('prosody', 'even')
      return (
          f'Right now, {agent_name} is outwardly expressing their state through'
          f' a {posture} posture and a {prosody} voice. Answer the questions'
          ' accordingly.\n'
      )
    except (KeyError, AttributeError, ValueError):
      return ''

  def _is_conditional_task_triggered(
      self, task: ConditionalTaskWrapper
  ) -> bool:
    """Checks if the emotional threshold for a conditional task is met."""
    # Primary check: Look up the last results of 'esm_affect' in self._results
    esm_affect_results = [
        r for r in self._results if r.get('task') == 'esm_affect'
    ]
    if esm_affect_results:
      latest = esm_affect_results[-1]
      item_scores = latest.get('item_scores', {})
      score = item_scores.get(task.emotion_trigger)
      if score is not None:
        return score >= task.trigger_threshold

    # Secondary check: parse the raw text of the EmotionalExperience component
    try:
      exp_comp: Any = self.get_entity().get_component('EmotionalExperience')
      current_experience = exp_comp.get_current_experience() or ''
      if task.emotion_trigger.lower() in current_experience.lower():
        return True
    except (KeyError, AttributeError, ValueError):
      pass

    return False

  def _build_agent_context(self) -> str:
    """Build the full agent context from sibling component pre_act values.

    Mirrors ConcatActComponent._context_for_action: reads pre_act values
    from each component in context_components order and concatenates them.
    This gives questionnaire prompts the same rich context (personality,
    self-perception, situation awareness, goals) as the agent's normal
    action prompts.

    Returns:
      Concatenated context string, or empty string if no context components.
    """
    if not self._context_components:
      return ''
    parts = []
    for key in self._context_components:
      try:
        value = self.get_named_component_pre_act_value(key)
        if value and value.strip():
          parts.append(value.strip())
      except (KeyError, ValueError):
        pass
    return '\n'.join(parts)

  def _run_open_ended(self, task: OpenEndedTask) -> dict[str, Any]:
    """Run an open-ended reflection task."""
    agent_name = self.get_entity().name
    memory = self.get_entity().get_component(
        self._memory_component_key, type_=memory_component.Memory
    )

    recent = memory.retrieve_recent(limit=task.num_memories)
    recent_text = '\n'.join(recent)

    prompt = interactive_document.InteractiveDocument(self._model)

    # Build context: full agent identity + emotional expression + recent
    # memories.
    agent_context = self._build_agent_context()
    expr_context = self._get_emotional_expression_context()

    if agent_context:
      prompt.statement(agent_context + '\n')
    if expr_context:
      prompt.statement(expr_context + '\n')
    prompt.statement(
        f'Recent experiences and observations of {agent_name}:\n{recent_text}'
    )

    response = prompt.open_question(
        task.prompt_template.format(agent_name=agent_name),
        answer_prefix=f'{agent_name} reflects: ',
        max_tokens=300,
        terminators=('\n\n',),
    )

    result = {
        'tick': self._tick_count,
        'task': task.name,
        'agent': agent_name,
        'text': response,
    }

    if task.save_to_memory and task.memory_tag:
      entry = f'{task.memory_tag} {agent_name} reflects: {response}'
      memory.add(entry)

    self._logging_channel({
        'Key': f'ExperienceReflection/{task.name}',
        'Summary': f'{agent_name}: {task.name} at tick {self._tick_count}',
        'Value': response,
    })

    return result

  def _run_likert_battery(self, task: LikertBatteryTask) -> dict[str, Any]:
    """Run a battery of Likert items in a single batched LLM call.

    Instead of making N parallel LLM calls (one per item, each with the
    full 16K agent context), we send ONE call that asks the model to rate
    all items at once and return JSON.  This reduces LLM calls from
    N-per-agent to 1-per-agent.

    Args:
      task: The LikertBatteryTask to run.

    Returns:
      A dictionary containing the results of the battery.
    """
    agent_name = self.get_entity().name
    memory = self.get_entity().get_component(
        self._memory_component_key, type_=memory_component.Memory
    )

    agent_context = self._build_agent_context()
    expr_context = self._get_emotional_expression_context()
    recent = memory.retrieve_recent(limit=15)
    recent_text = '\n'.join(recent)
    preprompt = task.preprompt_template.format(agent_name=agent_name)

    doc = interactive_document.InteractiveDocument(self._model)
    if agent_context:
      doc.statement(agent_context + '\n')
    if expr_context:
      doc.statement(expr_context + '\n')
    doc.statement(f'Recent experiences of {agent_name}:\n{recent_text}')

    # Build scale description and numbered items list.
    scale_desc = ', '.join(
        f'{i} = "{label}"' for i, label in enumerate(task.scale_labels)
    )
    items_list = '\n'.join(
        f'  {i + 1}. "{item.statement}"' for i, item in enumerate(task.items)
    )

    cot_response = ''
    if task.use_cot:
      cot_response = doc.open_question(
          question=(
              f'{preprompt}\n\nItems:\n{items_list}\n\n'
              f'Scale: {scale_desc}\n\n'
              'Reflect briefly on each item from the perspective of '
              f'{agent_name}.'
          ),
          max_tokens=2000,
      )

    json_response = doc.open_question(
        question=(
            f'{preprompt}\n\n'
            f'Rate each item on this scale: {scale_desc}\n\n'
            f'Items:\n{items_list}\n\n'
            'Respond with ONLY a JSON object mapping the item NUMBER '
            f'(1-{len(task.items)}) to the numeric rating.\n'
            'Example: {"1": 3, "2": 0, "3": 5}'
        ),
        max_tokens=500,
    )

    # Parse JSON response into {item_index: raw_score}.
    parsed_scores = self._parse_batched_likert_json(
        json_response, len(task.items), len(task.scale_labels)
    )

    # Collect scores and compute dimensions (same format as before).
    item_scores = {}
    item_reflections = {}
    dimension_sums: dict[str, list[float]] = {}
    scale_max = len(task.scale_labels) - 1

    for i, item in enumerate(task.items):
      raw_score = parsed_scores.get(i, scale_max // 2)
      if item.reverse_scored:
        scored_value = scale_max - raw_score
      else:
        scored_value = raw_score

      display_score = raw_score + 1
      item_scores[item.statement] = display_score
      if task.use_cot:
        item_reflections[item.statement] = cot_response

      if item.dimension not in dimension_sums:
        dimension_sums[item.dimension] = []
      dimension_sums[item.dimension].append(scored_value)

    dimension_scores = {
        dim: round(sum(vals) / len(vals), 2)
        for dim, vals in dimension_sums.items()
    }

    result = {
        'tick': self._tick_count,
        'task': task.name,
        'agent': agent_name,
        'item_scores': item_scores,
        'dimension_scores': dimension_scores,
    }
    if task.use_cot:
      result['item_reflections'] = item_reflections

    logging_value = {
        'Key': f'ExperienceReflection/{task.name}',
        'Summary': f'{agent_name}: {task.name} at tick {self._tick_count}',
        'Value': dimension_scores,
        'ItemScores': item_scores,
    }
    if task.use_cot:
      logging_value['ItemReflections'] = item_reflections

    self._logging_channel(logging_value)

    return result

  def _parse_batched_likert_json(
      self,
      response: str,
      num_items: int,
      num_labels: int,
  ) -> dict[int, int]:
    """Parse a JSON response mapping item numbers to ratings.

    Robust to common LLM formatting issues (markdown fences, extra text).

    Args:
      response: Raw LLM response text.
      num_items: Expected number of items.
      num_labels: Number of scale points (for clamping).

    Returns:
      Dict mapping 0-indexed item number to clamped score.
    """
    # Strip markdown code fences if present.
    text = response.strip()
    if text.startswith('```'):
      text = re.sub(r'^```\w*\n?', '', text)
      text = re.sub(r'\n?```$', '', text)
      text = text.strip()

    # Try to find a JSON object in the response.
    match = re.search(r'\{[^}]+\}', text, re.DOTALL)
    if not match:
      logging.warning(
          'Could not find JSON in Likert response, using midpoint defaults: %s',
          text[:200],
      )
      mid = (num_labels - 1) // 2
      return {i: mid for i in range(num_items)}

    try:
      raw = json.loads(match.group())
    except json.JSONDecodeError:
      logging.warning(
          'JSON parse failed for Likert response, using midpoint defaults: %s',
          match.group()[:200],
      )
      mid = (num_labels - 1) // 2
      return {i: mid for i in range(num_items)}

    # Map keys (1-indexed strings or ints) to 0-indexed item numbers.
    scores: dict[int, int] = {}
    max_val = num_labels - 1
    for key, val in raw.items():
      try:
        idx = int(key) - 1  # Convert 1-indexed to 0-indexed.
        score = max(0, min(max_val, int(val)))
        if 0 <= idx < num_items:
          scores[idx] = score
      except (ValueError, TypeError):
        continue

    return scores

  def _run_multiple_choice(self, task: MultipleChoiceTask) -> dict[str, Any]:
    """Run a multiple choice task using batched JSON (no sample_choice).

    Single-select questions are batched into a single sample_text call
    that returns JSON mapping question numbers to option letters, matching
    the robust Likert battery pattern. This avoids the brittle
    sample_choice → InvalidResponseError path that crashes agent threads
    with models like Gemma4 MoE.

    Multi-select questions continue to use open_question (sample_text).

    Args:
      task: MultipleChoiceTask to run.

    Returns:
      Dict containing results mapped by key.
    """
    agent_name = self.get_entity().name
    memory = self.get_entity().get_component(
        self._memory_component_key, type_=memory_component.Memory
    )

    agent_context = self._build_agent_context()
    expr_context = self._get_emotional_expression_context()
    recent = memory.retrieve_recent(limit=15)
    recent_text = '\n'.join(recent)

    results = {}
    items = task.items if task.items else (task.name,)

    # Separate single-select and multi-select items.
    single_select_items = []
    multi_select_items = []
    for item in items:
      options = task.options_map.get(item, ())
      if not options:
        continue
      if task.multi_select:
        multi_select_items.append((item, options))
      else:
        single_select_items.append((item, options))

    # --- Handle single-select items via batched JSON (one LLM call) ---
    if single_select_items:
      doc = interactive_document.InteractiveDocument(self._model)
      if agent_context:
        doc.statement(agent_context + '\n')
      if expr_context:
        doc.statement(expr_context + '\n')
      doc.statement(f'Recent experiences of {agent_name}:\n{recent_text}')

      # Build the numbered question list with lettered options.
      questions_block = []
      item_option_map = {}  # {question_num: (item_key, options_tuple)}
      for q_idx, (item, options) in enumerate(single_select_items):
        q_num = q_idx + 1
        item_option_map[q_num] = (item, options)
        question_text = (
            f'{task.prompt}\nSubject: {item}' if len(items) > 1 else task.prompt
        )
        opts_str = ', '.join(
            f'{chr(ord("a") + i)} = "{opt}"' for i, opt in enumerate(options)
        )
        questions_block.append(
            f'  {q_num}. {question_text}\n     Options: {opts_str}'
        )

      all_questions = '\n'.join(questions_block)

      json_response = doc.open_question(
          question=(
              'Answer each question by selecting one option.\n\n'
              f'{all_questions}\n\n'
              'Respond with ONLY a JSON object mapping the question NUMBER '
              f'(1-{len(single_select_items)}) to the option LETTER '
              '(a, b, c, etc.).\n'
              'Example: {"1": "a", "2": "c"}'
          ),
          max_tokens=500,
      )

      # Parse the JSON response.
      parsed = self._parse_mc_json(json_response, item_option_map)
      results.update(parsed)

    # --- Handle multi-select items via open_question (unchanged) ---
    for item, options in multi_select_items:
      doc = interactive_document.InteractiveDocument(self._model)
      if agent_context:
        doc.statement(agent_context + '\n')
      if expr_context:
        doc.statement(expr_context + '\n')
      doc.statement(f'Recent experiences of {agent_name}:\n{recent_text}')

      question = (
          f'{task.prompt}\nSubject: {item}' if len(items) > 1 else task.prompt
      )
      options_str = '\n'.join([f'- {opt}' for opt in options])
      full_prompt = (
          f'{question}\nSelect all that apply from the following options by'
          f' listing them:\n{options_str}'
      )
      response = doc.open_question(
          full_prompt,
          answer_prefix=f'{agent_name} selects: ',
          max_tokens=200,
          terminators=('\n\n',),
      )
      results[item] = response

    result = {
        'tick': self._tick_count,
        'task': task.name,
        'agent': agent_name,
        'results': results,
    }

    self._logging_channel({
        'Key': f'ExperienceReflection/{task.name}',
        'Summary': f'{agent_name}: {task.name} at tick {self._tick_count}',
        'Value': results,
    })

    return result

  def _parse_mc_json(
      self,
      response: str,
      item_option_map: dict[int, tuple[str, tuple[str, ...]]],
  ) -> dict[str, str]:
    """Parse a JSON response mapping question numbers to option letters.

    Robust to common LLM formatting issues (markdown fences, extra text,
    full option text instead of letter).

    Args:
      response: Raw LLM response text.
      item_option_map: Maps 1-indexed question number to (item_key, options).

    Returns:
      Dict mapping item_key to selected option string.
    """
    results = {}

    # Strip markdown code fences if present.
    text = response.strip()
    if text.startswith('```'):
      text = re.sub(r'^```\w*\n?', '', text)
      text = re.sub(r'\n?```$', '', text)
      text = text.strip()

    # Try to find a JSON object in the response.
    match = re.search(r'\{[^}]+\}', text, re.DOTALL)
    if not match:
      logging.warning(
          'Could not find JSON in MC response, using first-option defaults: %s',
          text[:200],
      )
      for _, (item_key, options) in item_option_map.items():
        results[item_key] = options[0]
      return results

    try:
      raw = json.loads(match.group())
    except json.JSONDecodeError:
      logging.warning(
          'JSON parse failed for MC response, using first-option defaults: %s',
          match.group()[:200],
      )
      for _, (item_key, options) in item_option_map.items():
        results[item_key] = options[0]
      return results

    # Map parsed answers to option strings.
    for q_num, (item_key, options) in item_option_map.items():
      val = raw.get(str(q_num)) or raw.get(q_num)
      if val is None:
        results[item_key] = options[0]
        continue

      val_str = str(val).strip().lower()

      # Try letter mapping first (a→0, b→1, etc.)
      if len(val_str) == 1 and val_str.isalpha():
        idx = ord(val_str) - ord('a')
        if 0 <= idx < len(options):
          results[item_key] = options[idx]
          continue

      # Try matching full option text (fuzzy).
      matched = False
      for opt in options:
        if val_str in opt.lower() or opt.lower() in val_str:
          results[item_key] = opt
          matched = True
          break
      if not matched:
        results[item_key] = options[0]

    return results

  def run_final_survey(self) -> list[dict[str, Any]]:
    """Force-run all non-per-tick tasks once for end-of-simulation measurement.

    Runs every task with frequency > 1 (i.e. daily batteries like BFI-10,
    MEMS Likert, MEMS Open-Ended) regardless of tick count. Results are
    appended to self._results and also returned.

    Returns:
      List of result dicts from the final survey run.
    """
    final_results = []
    parallel_tasks = {}

    def _make_task_fn(t):
      return lambda: self._run_task(t)

    for task in self._tasks:
      if task.frequency > 1:
        parallel_tasks[task.name] = _make_task_fn(task)

    if parallel_tasks:
      raw_results = concurrency.run_tasks(parallel_tasks)
      for unused_task_name, result in raw_results.items():
        if result:
          self._results.append(result)
          final_results.append(result)

    return final_results

  def get_results(self) -> list[dict[str, Any]]:
    """Return accumulated measurement results for export."""
    results = list(self._results)

    agent_name = self.get_entity().name

    # Export complete un-sliced emotional experience history
    try:
      exp_comp: Any = self.get_entity().get_component('EmotionalExperience')
      results.append({
          'tick': self._tick_count,
          'task': 'emotional_experience_history',
          'agent': agent_name,
          'history': exp_comp.get_full_emotion_history(),
      })
    except (KeyError, AttributeError, ValueError):
      pass

    # Export complete un-sliced emotional expression history
    try:
      expr_comp: Any = self.get_entity().get_component('EmotionalExpression')
      results.append({
          'tick': self._tick_count,
          'task': 'emotional_expression_history',
          'agent': agent_name,
          'history': expr_comp.get_full_expression_history(),
      })
    except (KeyError, AttributeError, ValueError):
      pass

    return results

  def get_state(self) -> entity_component.ComponentState:
    return {
        'tick_count': self._tick_count,
        'results': list(self._results),
        'current_time_str': self._current_time_str,
        'last_run_time_str': self._last_run_time_str,
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    if 'tick_count' in state:
      self._tick_count = component_state.as_int(state, 'tick_count')
    if 'results' in state:
      self._results = component_state.as_dict_list(state, 'results')
    self._current_time_str = component_state.as_str(state, 'current_time_str')
    self._last_run_time_str = component_state.as_str(
        state, 'last_run_time_str'
    )
