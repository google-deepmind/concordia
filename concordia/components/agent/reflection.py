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


"""Agent characteristic component."""

from concordia.associative_memory import associative_memory
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import component
import termcolor


class Reflection(component.Component):
  """Implements a reflection component.

  First, the last 100 memories are retrieved, using which the 3 most salient
  questions are inferred. These questions are used as a query. The output of
  the query is summarised into 5 insights.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      agent_name: str,
      name: str = 'reflection',
      importance_threshold: float = 20.0,
      verbose: bool = False,
  ):
    self._model = model
    self._memory = memory
    self._state = ''
    self._agent_name = agent_name
    self._verbose = verbose
    self._name = name
    self._importance_threshold = importance_threshold
    self._history = []

  def name(self) -> str:
    return self._name

  def state(self) -> str:
    return self._state

  def get_last_log(self):
    if self._history:
      return self._history[-1].copy()

  def update(self) -> None:
    mems, importance = self._memory.retrieve_recent_with_importance(
        100, add_time=True
    )
    total_importance = sum(importance)
    if total_importance < self._importance_threshold:
      self._state = ''
      if self._verbose:
        print(
            termcolor.colored(
                f'Importance {total_importance} below threshold', 'green'
            ),
            end='',
        )

      return

    mems = '\n'.join(mems)

    prompt_questions = interactive_document.InteractiveDocument(self._model)

    questions = prompt_questions.open_question(
        '\n'.join([
            f'{mems}',
            (
                'Given only the statements above, what are'
                ' the 3 most salient high-level questions we can'
                f' answer about {self._agent_name}?'
            ),
        ]),
        max_tokens=5000,
        terminators=(),
    )

    mems = []
    # make sure that the answer comes out of LLM in the right format
    for question in questions.splitlines():
      mems += self._memory.retrieve_associative(question, 10, add_time=True)

    mems = '\n'.join(mems)

    prompt_insights = interactive_document.InteractiveDocument(self._model)

    self._state = prompt_insights.open_question(
        '\n'.join([
            f'{mems}',
            (
                'What 5 high-level insights can you infer from the above'
                ' statements?'
            ),
        ]),
        max_tokens=5000,
        terminators=(),
    )
    self._memory.extend(self._state.splitlines())
    self._last_chain = prompt_insights
    if self._verbose:
      print(termcolor.colored(self._last_chain.view().text(), 'green'), end='')

    update_log = {
        'Summary': 'reflection and insights',
        'State': self._state.splitlines(),
        'Questions prompt': prompt_questions.view().text().splitlines(),
        'Insights prompt': prompt_insights.view().text().splitlines(),
    }
    self._history.append(update_log)
