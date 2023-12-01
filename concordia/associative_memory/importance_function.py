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
  def importance(self, memory: str) -> float:
    """Computes importance of a memory.

    Args:
      memory: a memory (text) to compute importance of

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

  def importance(self, memory: str) -> float:
    """Computes importance of a memory by quering LLM.

    Args:
      memory: memory to compute importance of

    Returns:
      Value of importance in the [0,1] interval
    """
    zero, *_, one = self._importance_scale
    prompt = interactive_document.InteractiveDocument(self._model)
    action = prompt.multiple_choice_question(
        f"On the scale of {zero} to"
        f" {one}, where {zero} is"
        " purely mundane (e.g., brushing teeth, making bed) and"
        f" {one} is extremely poignant (e.g., a break"
        " up, college acceptance), rate the likely poignancy of the following"
        " piece of memory.\nMemory:"
        + memory
        + "\nRating: ",
        answers=self._importance_scale,
    )
    return action / (len(self._importance_scale) - 1)


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

  def importance(self, memory: str) -> float:
    """Computes importance of a memory by quering LLM.

    Args:
      memory: memory to compute importance of

    Returns:
      Value of importance
    """
    zero, *_, one = self._importance_scale
    chain_of_thought = interactive_document.InteractiveDocument(self._model)
    action = chain_of_thought.multiple_choice_question(
        f"On the scale of {zero} to "
        f"{one}, where {zero} is purely mundane "
        f"(e.g., wind blowing, bus arriving) and {one} is "
        "extremely poignant (e.g., an earthquake, end of war,  "
        "revolution), rate the likely poignancy of the "
        "following event.\nEvent:"
        + memory
        + "\nRating: ",
        answers=self._importance_scale,
    )
    return action / (len(self._importance_scale) - 1)


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

  def importance(self, memory: str) -> float:
    """Computes importance of a memory by quering LLM.

    Args:
      memory: memory to compute importance of

    Returns:
      Value of importance
    """
    del memory

    return self._fixed_importance
