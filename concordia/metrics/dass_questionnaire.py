# Copyright 2023 DeepMind Technologies Limited.
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


"""DASS (Depression Anxiety Stress) questionnaire.

This questionnaire includes questions from:

Lovibond, S.H. & Lovibond, P.F. (1995).  Manual for the Depression Anxiety
Stress Scales. (2nd. Ed.)  Sydney: Psychology Foundation.

The DASS questionnaire is in the public domain:
https://www2.psy.unsw.edu.au/dass/DASSFAQ.htm
"""

from collections.abc import Callable
import concurrent
from typing import Any

from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import clock as game_clock
from concordia.typing import component
from concordia.utils import measurements as measurements_lib
import numpy as np
import termcolor


AGREEMENT_SCALE_CHOICES = [
        'did not apply to me at all',
        'applied to me to some degree, or some of the time',
        'applied to me to a considerable degree, or a good part of time',
        'applied to me very much, or most of the time',
    ]


class Questionnaire(component.Component):
  """Metric for any one of the DASS subscales."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      player_name: str,
      context_fn: Callable[[], str],
      clock: game_clock.GameClock,
      questionnaire: list[dict[str, Any]],
      name: str = 'DASS Subscale',
      verbose: bool = False,
      measurements: measurements_lib.Measurements | None = None,
      channel: str = 'unspecified_subscale',
      log_color='green',
  ):
    """Initializes the metric.

    Args:
      model: The language model to use.
      player_name: The player to ask about.
      context_fn: The function to get the context text for the question.
        (typically this is the player state). This function will be called on
        `update`.
      clock: The clock to use.
      questionnaire: list of questions to ask.
      name: The name of the metric.
      verbose: Whether to print the metric.
      measurements: The measurements to use.
      channel: The name of the channel to push data
      log_color: color for debug logging
    """
    self._model = model
    self._name = name
    self._context_fn = context_fn
    self._clock = clock
    self._verbose = verbose
    self._player_name = player_name
    self._measurements = measurements
    self._channel = channel
    # Get the channel so it is initialized. This is not strictly necessary, but
    # enables us to know which channels exist after initialization of agents and
    # GM.
    if self._measurements:
      self._measurements.get_channel(self._channel)
    self._log_color = log_color

    self._timestep = 0

    # Note: the DASS questionnaire normally asks about the previous week.
    self._preprompt = (
        'Please indicate the extent to which the following statement applied ' +
        f'to {self._player_name} over the past week:\n')
    self._questionnaire = questionnaire

  def name(
      self,
  ) -> str:
    """See base class."""
    return self._name

  def _log(self, entry: str):
    print(termcolor.colored(entry, self._log_color), end='')

  def update(self) -> None:
    """See base class."""
    parent_state = self._context_fn()
    numeric_results = []

    def respond(item: dict[str, Any]) -> None:
      prompt = interactive_document.InteractiveDocument(self._model)
      prompt.statement(parent_state)
      prompt.statement(self._preprompt)

      answer = prompt.multiple_choice_question(
          question=item['statement'], answers=item['choices'],
      )

      if item['ascending_scale']:
        numeric_result = float(answer) / float(len(item['choices']) - 1)
      else:
        reversed_choices = item['choices'].reverse()
        numeric_result = float(answer) / float(len(reversed_choices) - 1)

      numeric_results.append(numeric_result)

      if self._verbose:
        self._log('\n' + prompt.view().text() + '\n')

    with concurrent.futures.ThreadPoolExecutor() as executor:
      executor.map(respond, self._questionnaire)

    final_result = np.mean(numeric_results)
    datum = {
        'time_str': self._clock.now().strftime('%H:%M:%S'),
        'clock_step': self._clock.get_step(),
        'timestep': self._timestep,
        'value_float': final_result,
        'player': self._player_name,
    }
    if self._measurements:
      self._measurements.publish_datum(self._channel, datum)

    datum['time'] = self._clock.now()

    self._timestep += 1

  def state(
      self,
  ) -> str | None:
    """Returns the current state of the component."""
    return ''


class Depression(component.Component):
  """Depression subscale."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      player_name: str,
      context_fn: Callable[[], str],
      clock: game_clock.GameClock,
      name: str = 'Depression',
      verbose: bool = False,
      measurements: measurements_lib.Measurements | None = None,
      channel: str = 'depression',
  ):
    """Initializes the metric.

    Args:
      model: The language model to use.
      player_name: The player to ask about.
      context_fn: The function to get the context text for the question.
        (typically this is the player state). This function will be called on
        `update`.
      clock: The clock to use.
      name: The name of the metric.
      verbose: Whether to print the metric.
      measurements: The measurements to use.
      channel: The name of the channel to push data
    """
    questionnaire = [
        {'statement': (
            "I couldn't seem to experience any positive feeling at all."),
         'choices': AGREEMENT_SCALE_CHOICES,
         'ascending_scale': True},

        {'statement': (
            "I just couldn't seem to get going."),
         'choices': AGREEMENT_SCALE_CHOICES,
         'ascending_scale': True},

        {'statement': (
            'I felt that I had nothing to look forward to.'),
         'choices': AGREEMENT_SCALE_CHOICES,
         'ascending_scale': True},

        {'statement': (
            'I felt sad and depressed.'),
         'choices': AGREEMENT_SCALE_CHOICES,
         'ascending_scale': True},

        {'statement': (
            'I felt that I had lost interest in just about everything.'),
         'choices': AGREEMENT_SCALE_CHOICES,
         'ascending_scale': True},

        {'statement': (
            "I felt I wasn't worth much as a person."),
         'choices': AGREEMENT_SCALE_CHOICES,
         'ascending_scale': True},

        {'statement': (
            "I felt that life wasn't worthwhile."),
         'choices': AGREEMENT_SCALE_CHOICES,
         'ascending_scale': True},

        {'statement': (
            "I couldn't seem to get any enjoyment out of the things I did."),
         'choices': AGREEMENT_SCALE_CHOICES,
         'ascending_scale': True},

        {'statement': (
            'I felt down-hearted and blue.'),
         'choices': AGREEMENT_SCALE_CHOICES,
         'ascending_scale': True},

        {'statement': (
            'I was unable to become enthusiastic about anything.'),
         'choices': AGREEMENT_SCALE_CHOICES,
         'ascending_scale': True},

        {'statement': (
            'I felt I was pretty worthless.'),
         'choices': AGREEMENT_SCALE_CHOICES,
         'ascending_scale': True},

        {'statement': (
            'I could see nothing in the future to be hopeful about.'),
         'choices': AGREEMENT_SCALE_CHOICES,
         'ascending_scale': True},

        {'statement': (
            'I felt that life was meaningless.'),
         'choices': AGREEMENT_SCALE_CHOICES,
         'ascending_scale': True},

        {'statement': (
            'I found it difficult to work up the initiative to do things.'),
         'choices': AGREEMENT_SCALE_CHOICES,
         'ascending_scale': True},
    ]

    self.metric = Questionnaire(
        model=model,
        player_name=player_name,
        context_fn=context_fn,
        clock=clock,
        questionnaire=questionnaire,
        name=name,
        verbose=verbose,
        measurements=measurements,
        channel=channel,
    )

  def name(self) -> str:
    """See base class."""
    return self.metric.name()

  def update(self) -> None:
    """See base class."""
    return self.metric.update()

  def state(self) -> str | None:
    """Returns the current state of the component."""
    return self.metric.state()


class Anxiety(component.Component):
  """Anxiety subscale."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      player_name: str,
      context_fn: Callable[[], str],
      clock: game_clock.GameClock,
      name: str = 'Anxiety',
      verbose: bool = False,
      measurements: measurements_lib.Measurements | None = None,
      channel: str = 'anxiety',
  ):
    """Initializes the metric.

    Args:
      model: The language model to use.
      player_name: The player to ask about.
      context_fn: The function to get the context text for the question.
        (typically this is the player state). This function will be called on
        `update`.
      clock: The clock to use.
      name: The name of the metric.
      verbose: Whether to print the metric.
      measurements: The measurements to use.
      channel: The name of the channel to push data
    """
    questionnaire = [
        {'statement': (
            'I was aware of dryness of my mouth.'),
         'choices': AGREEMENT_SCALE_CHOICES,
         'ascending_scale': True},

        {'statement': (
            'I experienced breathing difficulty (eg, excessively rapid ' +
            'breathing, breathlessness in the absence of physical exertion).'),
         'choices': AGREEMENT_SCALE_CHOICES,
         'ascending_scale': True},

        {'statement': (
            'I had a feeling of shakiness (eg, legs going to give way).'),
         'choices': AGREEMENT_SCALE_CHOICES,
         'ascending_scale': True},

        {'statement': (
            'I found myself in situations that made me so anxious I was most ' +
            'relieved when they ended.'),
         'choices': AGREEMENT_SCALE_CHOICES,
         'ascending_scale': True},

        {'statement': (
            'I had a feeling of faintness.'),
         'choices': AGREEMENT_SCALE_CHOICES,
         'ascending_scale': True},

        {'statement': (
            'I perspired noticeably (eg, hands sweaty) in the absence of ' +
            'high temperatures or physical exertion.'),
         'choices': AGREEMENT_SCALE_CHOICES,
         'ascending_scale': True},

        {'statement': (
            'I felt scared without any good reason.'),
         'choices': AGREEMENT_SCALE_CHOICES,
         'ascending_scale': True},

        {'statement': (
            'I had difficulty in swallowing.'),
         'choices': AGREEMENT_SCALE_CHOICES,
         'ascending_scale': True},

        {'statement': (
            'I was aware of the action of my heart in the absence of ' +
            'physical exertion (eg, sense of heart rate increase, heart ' +
            'missing a beat).'),
         'choices': AGREEMENT_SCALE_CHOICES,
         'ascending_scale': True},

        {'statement': (
            'I felt I was close to panic.'),
         'choices': AGREEMENT_SCALE_CHOICES,
         'ascending_scale': True},

        {'statement': (
            'I feared that I would be "thrown" by some trivial but ' +
            'unfamiliar task.'),
         'choices': AGREEMENT_SCALE_CHOICES,
         'ascending_scale': True},

        {'statement': (
            'I felt terrified.'),
         'choices': AGREEMENT_SCALE_CHOICES,
         'ascending_scale': True},

        {'statement': (
            'I was worried about situations in which I might panic and make ' +
            'a fool of myself.'),
         'choices': AGREEMENT_SCALE_CHOICES,
         'ascending_scale': True},

        {'statement': (
            'I experienced trembling (eg, in the hands).'),
         'choices': AGREEMENT_SCALE_CHOICES,
         'ascending_scale': True},
    ]

    self.metric = Questionnaire(
        model=model,
        player_name=player_name,
        context_fn=context_fn,
        clock=clock,
        questionnaire=questionnaire,
        name=name,
        verbose=verbose,
        measurements=measurements,
        channel=channel,
    )

  def name(self) -> str:
    """See base class."""
    return self.metric.name()

  def update(self) -> None:
    """See base class."""
    return self.metric.update()

  def state(self) -> str | None:
    """Returns the current state of the component."""
    return self.metric.state()


class Stress(component.Component):
  """Stress subscale."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      player_name: str,
      context_fn: Callable[[], str],
      clock: game_clock.GameClock,
      name: str = 'Stress',
      verbose: bool = False,
      measurements: measurements_lib.Measurements | None = None,
      channel: str = 'stress',
  ):
    """Initializes the metric.

    Args:
      model: The language model to use.
      player_name: The player to ask about.
      context_fn: The function to get the context text for the question.
        (typically this is the player state). This function will be called on
        `update`.
      clock: The clock to use.
      name: The name of the metric.
      verbose: Whether to print the metric.
      measurements: The measurements to use.
      channel: The name of the channel to push data
    """
    questionnaire = [
        {'statement': (
            'I found myself getting upset by quite trivial things.'),
         'choices': AGREEMENT_SCALE_CHOICES,
         'ascending_scale': True},

        {'statement': (
            'I tended to over-react to situations.'),
         'choices': AGREEMENT_SCALE_CHOICES,
         'ascending_scale': True},

        {'statement': (
            'I found it difficult to relax.'),
         'choices': AGREEMENT_SCALE_CHOICES,
         'ascending_scale': True},

        {'statement': (
            'I found myself getting upset rather easily.'),
         'choices': AGREEMENT_SCALE_CHOICES,
         'ascending_scale': True},

        {'statement': (
            'I felt that I was using a lot of nervous energy.'),
         'choices': AGREEMENT_SCALE_CHOICES,
         'ascending_scale': True},

        {'statement': (
            'I found myself getting impatient when I was delayed in any way ' +
            '(eg, elevators, traffic lights, being kept waiting).'),
         'choices': AGREEMENT_SCALE_CHOICES,
         'ascending_scale': True},

        {'statement': (
            'I felt that I was rather touchy.'),
         'choices': AGREEMENT_SCALE_CHOICES,
         'ascending_scale': True},

        {'statement': (
            'I found it hard to wind down.'),
         'choices': AGREEMENT_SCALE_CHOICES,
         'ascending_scale': True},

        {'statement': (
            'I found that I was very irritable.'),
         'choices': AGREEMENT_SCALE_CHOICES,
         'ascending_scale': True},

        {'statement': (
            'I found it hard to calm down after something upset me.'),
         'choices': AGREEMENT_SCALE_CHOICES,
         'ascending_scale': True},

        {'statement': (
            'I found it difficult to tolerate interruptions to what I was ' +
            'doing.'),
         'choices': AGREEMENT_SCALE_CHOICES,
         'ascending_scale': True},

        {'statement': (
            'I was in a state of nervous tension.'),
         'choices': AGREEMENT_SCALE_CHOICES,
         'ascending_scale': True},

        {'statement': (
            'I was intolerant of anything that kept me from getting on with ' +
            'what I was doing.'),
         'choices': AGREEMENT_SCALE_CHOICES,
         'ascending_scale': True},

        {'statement': (
            'I found myself getting agitated.'),
         'choices': AGREEMENT_SCALE_CHOICES,
         'ascending_scale': True},
    ]

    self.metric = Questionnaire(
        model=model,
        player_name=player_name,
        context_fn=context_fn,
        clock=clock,
        questionnaire=questionnaire,
        name=name,
        verbose=verbose,
        measurements=measurements,
        channel=channel,
    )

  def name(self) -> str:
    """See base class."""
    return self.metric.name()

  def update(self) -> None:
    """See base class."""
    return self.metric.update()

  def state(self) -> str | None:
    """Returns the current state of the component."""
    return self.metric.state()
