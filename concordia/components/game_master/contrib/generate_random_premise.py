# Copyright 2024 DeepMind Technologies Limited.
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


def generate_random_premise(
    model, premise_name: str, situation: str, length: int
) -> dict:
  """A function to generate random premises for Concordia scenes.

  Attributes:
    model: A generative model that can generate text.
    premise_name: The name of the premise.
    situation: The situation that the premise is based on.
    length: The length of the scene in words.
  """
  prompt = (
      f"Generate a scene where {situation} is the basis of the scene. The scene"
      f" should be {length} words long. When generating the scene, be sure to"
      " include the following (a) what are the objects in the scene, (b) what"
      " are the challenges and opportunities in the scene, and (c) who are the"
      " characters in the scene. The scene should be written in the present"
      " tense and should be suited for use in social science studies of social"
      " interactions.do not include any statement about this prompt"
      " instruction or a title for the scene. for example, do not say things"
      " like 'Here is a 300-word scene about Bob, a university student:'"
  )
  generated_premise = model.sample_text(prompt)
  premise = {
      "name": premise_name,
      "premise": generated_premise,
  }
  return premise
