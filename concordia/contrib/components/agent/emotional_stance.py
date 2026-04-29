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

"""Emotion-commitment component for agent reasoning pipelines."""

from concordia.components import agent as agent_components
from concordia.components.agent import action_spec_ignored
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import entity_component


class EmotionalStance(
    action_spec_ignored.ActionSpecIgnored, entity_component.ComponentWithLogging
):
  """Emotion-commitment component that forces an emotion choice before action.

  Reads the pre_act value of a reasoning component, presents it in an
  interactive document, and asks the model to select the single emotion
  best driving the agent's next action from a curated multiple-choice list.

  After choosing the emotion, it retrieves the latest observations and asks
  the model to select the N most important memories/observations that
  contextualize the chosen emotion.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      name: str,
      emotion_options: list[str],
      reasoning_component_key: str = "PersonBySituation",
      observation_component_key: str = (
          agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY
      ),
      num_observations_to_select: int = 5,
  ):
    super().__init__(
        pre_act_label=(
            f"{name}'s Emotional Stance: "
            f"The emotion driving {name}'s next action is"
        ),
    )
    self._model = model
    self._name = name
    self._reasoning_component_key = reasoning_component_key
    self._observation_component_key = observation_component_key
    self._emotion_options = emotion_options
    self._num_observations_to_select = num_observations_to_select

  def _make_pre_act_value(self) -> str:
    reasoning = self.get_named_component_pre_act_value(
        self._reasoning_component_key
    )
    prompt = interactive_document.InteractiveDocument(self._model)
    prompt.statement(f"{self._name}'s situational reasoning:\n{reasoning}\n")
    idx = prompt.multiple_choice_question(
        question=(
            "What single emotion best describes the feeling driving "
            f"{self._name}?"
        ),
        answers=self._emotion_options,
    )
    chosen_emotion = self._emotion_options[idx]

    # Retrieve observations and ask the model to select the most important
    # ones to contextualize the chosen emotion.
    observations = self.get_named_component_pre_act_value(
        self._observation_component_key
    )
    prompt.statement(
        f"\n{self._name}'s recent observations and memories:\n{observations}\n"
    )
    n = self._num_observations_to_select
    selected = prompt.open_question(
        question=(
            f"Select the {n} most important memories/observations to"
            f" contextualize why {self._name} feels '{chosen_emotion}'"
            f" and how {self._name} intends to respond. Reprint them"
            " here. It is OK to subtly modify the memory/observation"
            " items in order to better fit with or justify the chosen"
            " emotion. If they include any quantitative information"
            " then make sure to maintain it."
        ),
        max_tokens=3000,
        terminators=(),
    )

    result = f"{chosen_emotion}\nKey context:\n{selected}"

    self._logging_channel({
        "Key": f"{self._name}'s Emotional Stance",
        "Value": chosen_emotion,
        "Selected Observations": selected,
        "Prompt": prompt.view().text().splitlines(),
    })
    return result

  def get_state(self) -> entity_component.ComponentState:
    return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    pass
