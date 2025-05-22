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

"""Scene generator class."""

from collections.abc import Sequence
import datetime

from concordia.associative_memory.deprecated import formative_memories
from concordia.language_model import language_model
from concordia.typing.deprecated import scene


class SceneGenerator:
  """Class to generate scene specifications based on given parameters."""

  def generate_scene_spec(
      self,
      model: language_model.LanguageModel,
      scene_type_name: str,
      length: int,
      start_time: datetime.datetime,
      participant_configs: Sequence[formative_memories.AgentConfig],
      num_rounds: int,
      situation: str = '',
  ) -> scene.SceneSpec:
    """Generate a complete scene specification.

    Args:
      model: A generative model for text generation.
      scene_type_name: Name of the scene type.
      length: Desired length of the premise in words.
      start_time: When the scene starts.
      participant_configs: Configurations for participants in the scene.
      num_rounds: Number of rounds the scene should last.
      situation: The basis of the scene's premise, use default if empty.

    Returns:
      A SceneSpec object with the generated premise.
    """

    if not situation:
      situation = ('a random situation that a human might encounter in daily '
                   'life')

    # Generate the premise text
    prompt = (
        f'Generate a scene where "{situation}" is the basis of the scene. The'
        f' scene should be {length} words long. Include details about objects,'
        ' challenges, opportunities, and characters in the scene, written in'
        ' the present tense. Write in a way that the characters in an agent'
        ' based model can respond to the situation. Do not include'
        ' instructions or a title in the output.'
    )
    generated_premise = model.sample_text(
        prompt, max_tokens=3500,
    )

    # Create the scene type specification
    scene_type_spec = scene.SceneTypeSpec(
        name=scene_type_name,
        premise={
            pc.name: [generated_premise] for pc in participant_configs},
        conclusion=None,
        action_spec=None,
        override_game_master=None,
    )

    # Return the complete scene specification
    return scene.SceneSpec(
        scene_type=scene_type_spec,
        start_time=start_time,
        participant_configs=participant_configs,
        num_rounds=num_rounds,
    )
