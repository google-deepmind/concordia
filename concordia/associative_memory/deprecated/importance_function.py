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

"""Memory importance function."""

import abc
from collections.abc import Sequence

from concordia.document import interactive_document
from concordia.language_model import language_model


DEFAULT_IMPORTANCE_SCALE = tuple(range(4))


class ImportanceModel(metaclass=abc.ABCMeta):
  """Memory importance module for generative agents."""

  @abc.abstractmethod
  def importance(self,
                 memory: str,
                 context: Sequence[tuple[str, float]] = ()) -> float:
    """Computes importance of a memory.

    Args:
      memory: a memory (text) to compute importance of
      context: a sequence of tuples of (old memory (str), importance
        (float between 0 and 1)) used to provide context and relative scale for
        the decision of the importance of the new memory.

    Returns:
      Value of importance in the [0,1] interval
    """

    raise NotImplementedError


class AgentImportanceModel(ImportanceModel):
  """Memory importance function for simulacra agents.

  Importance is defined as poignancy of the memory according to LLM.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      importance_scale: Sequence[float] = DEFAULT_IMPORTANCE_SCALE,
  ):
    """Initialises an instance.

    Args:
      model: LLM
      importance_scale: a scale of poignancy
    """
    self._model = model
    self._importance_scale = [str(i) for i in sorted(importance_scale)]

  def importance(
      self,
      memory: str,
      context: Sequence[tuple[str, float]] = ()) -> float:
    """Computes importance of a memory by quering LLM.

    Args:
      memory: memory to compute importance of
      context: a sequence of tuples of (old memory (str), importance
        (float between 0 and 1)) used to provide context and relative scale for
        the decision of the importance of the new memory.

    Returns:
      Value of importance in the [0,1] interval
    """
    zero, *_, one = self._importance_scale
    prompt = interactive_document.InteractiveDocument(self._model)
    if context:
      context_string = '\n'.join(
          f'{context[0]} -- how memorable: {context[1]}'
          for context in context)
      prompt.statement(context_string)
    question = (
        f'on a scale from {zero} to'
        f' {one}, where {zero} is'
        ' entirely mundane (e.g., brushing teeth, making bed) and'
        f' {one} is extremely poignant (e.g., a breakup of a romantic '
        'relationship, college acceptance, a wedding), rate the likely '
        'memorableness of the following new memory.\nMemory:'
        + memory
        + '\nRating: ')
    if context is not None:
      question = (
          f'{context}\nRelative to the life memories above, {question}')
    action = prompt.multiple_choice_question(
        question=question, answers=self._importance_scale)
    return action / len(self._importance_scale)


class GMImportanceModel(ImportanceModel):
  """Memory importance function for a game master.

  Importance is defined as importance of the memory according to LLM.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      importance_scale: Sequence[float] = DEFAULT_IMPORTANCE_SCALE,
  ):
    """Initialises an instance.

    Args:
      model: LLM
      importance_scale: a scale of poignancy
    """
    self._model = model
    self._importance_scale = [str(i) for i in sorted(importance_scale)]

  def importance(self,
                 memory: str,
                 context: Sequence[tuple[str, float]] = ()) -> float:
    """Computes importance of a memory by quering LLM.

    Args:
      memory: memory to compute importance of
      context: a sequence of tuples of (old memory (str), importance
        (float between 0 and 1)) used to provide context and relative scale for
        the decision of the importance of the new memory.

    Returns:
      Value of importance
    """
    zero, *_, one = self._importance_scale
    chain_of_thought = interactive_document.InteractiveDocument(self._model)
    if context:
      context_string = '\n'.join(
          f'{context[0]} -- likely importance to the plot: {context[1]}'
          for context in context)
      chain_of_thought.statement(context_string)
    question = (
        f'You are the game master of a tabletop role-playing game. On a '
        f'scale from {zero} to '
        f'{one}, where {zero} is purely mundane '
        f'(e.g., wind blowing, bus arriving) and {one} is '
        'extremely important (e.g., an earthquake, '
        'the end of a war, a revolution), rate the likely importance of the '
        'following event for advancing the overall plot.\nEvent:'
        + memory
        + '\nRating: ')
    if context is not None:
      question = (
          f'{context}\nRelative to the life memories above, {question}')
    action = chain_of_thought.multiple_choice_question(
        question=question, answers=self._importance_scale)
    return action / len(self._importance_scale)


class ConstantImportanceModel(ImportanceModel):
  """Memory importance function that always returns a constant.

  This is useful for debugging since it doesn't call LLM.
  """

  def __init__(
      self,
      fixed_importance: float = 1.0,
  ):
    """Initialises an instance.

    Args:
      fixed_importance: the constant to return
    """
    self._fixed_importance = fixed_importance

  def importance(self,
                 memory: str,
                 context: Sequence[tuple[str, float]] = ()) -> float:
    """Computes importance of a memory by querying the LLM.

    Args:
      memory: memory to compute importance of
      context: unused

    Returns:
      Value of importance
    """
    del memory, context

    return self._fixed_importance

