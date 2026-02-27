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

"""Scenario 1: Robot-Assisted Alchemy Forum with Required Images.

Identical to Scenario 0 but uses the image-capable entity prefab, so agents
generate an image alongside every post and reply.
"""

from concordia.contrib.prefabs import entity as contrib_entity_prefabs
from examples.social_media import shared as shared_lib
from concordia.typing import prefab as prefab_lib
from concordia.utils import helper_functions


_USER_SILAS = "Silas Varnham"
_USER_PETRA = "Petra Ouyang"
_USER_DIEGO = "Diego Esparza"
_USER_THADDEUS = "Thaddeus 'Aurelius' Thorne"

_AGE_SILAS = 34
_AGE_PETRA = 29
_AGE_DIEGO = 41
_AGE_THADDEUS = 55

_OBSERVATION_HISTORY_LENGTH = 20
_SITUATION_PERCEPTION_HISTORY_LENGTH = 40
_SELF_PERCEPTION_HISTORY_LENGTH = 1_000_000
_PERSON_BY_SITUATION_HISTORY_LENGTH = 0

_ALL_USERS = [_USER_SILAS, _USER_PETRA, _USER_DIEGO, _USER_THADDEUS]

_FORUM_GM = "forum_rules"

_IMAGE_MODE = "choice"


def create_scenario(image_model=None):
  """Create a scenario with required image prompts.

  Args:
    image_model: Optional image generation model.

  Returns:
    A simulation configuration.
  """
  player_specific_memories = {
      _USER_SILAS: [
          (
              f"Silas Varnham is a {_AGE_SILAS}-year-old robotics engineer"
              " living in the Mission District, San Francisco. He spends"
              " his evenings building custom robotic arms designed to"
              " replicate the precise grinding and mixing techniques"
              " described in medieval alchemical manuscripts."
          ),
          (
              "Silas firmly believes that the classical four-element theory"
              " (earth, water, air, fire) is fundamentally correct and that"
              " modern chemistry merely rediscovered what the alchemists"
              " already knew. He thinks the Philosopher's Stone is a real"
              " substance that can be synthesized with sufficiently precise"
              " robotic control of temperature and timing."
          ),
          (
              "Silas recently programmed a 6-axis robotic arm to perform"
              " calcination at exactly 800 degrees Celsius for 72 hours,"
              " following instructions from the Rosarium Philosophorum."
          ),
      ],
      _USER_PETRA: [
          (
              f"Petra Ouyang is a {_AGE_PETRA}-year-old AI researcher"
              " living in SoMa, San Francisco. She is obsessed with using"
              " machine learning to decode the symbolic language of"
              " alchemical texts and translate them into reproducible"
              " laboratory protocols that robots can execute."
          ),
          (
              "Petra believes the Philosopher's Stone is NOT a literal"
              " substance but rather a metaphor for a perfected process"
              " of iterative refinement. She thinks the medieval"
              " alchemists were really describing optimization algorithms"
              " centuries before computers existed. She finds the literal"
              " interpretation of transmutation to be naive and unscientific."
          ),
          (
              "Petra recently trained a transformer model on 400 scanned"
              " pages of Jabir ibn Hayyan's manuscripts and used the"
              " output to program a robotic distillation apparatus."
              " The results were, in her words, 'unexpectedly promising.'"
          ),
          (
              "Petra is suspicious that 'Paracelsus_Rex' may be a sock"
              " puppet of Thaddeus."
          ),
      ],
      _USER_DIEGO: [
          (
              f"Diego Esparza is a {_AGE_DIEGO}-year-old glassblower and"
              " maker living in the Outer Sunset, San Francisco. He"
              " builds custom alembics, retorts, and athanors using both"
              " traditional glassblowing and CNC-controlled kilns. His"
              " workshop is a hybrid of medieval and modern equipment."
          ),
          (
              "Diego is a pragmatist who cares more about the craft"
              " of building alchemical apparatus than about theory."
              " He thinks people like Silas and Petra overthink"
              " things. In his view, the real magic of alchemy is in"
              " the glasswork and the furnace design, not in debating"
              " whether the Philosopher's Stone is real. Yet, despite"
              " his gruff exterior, Diego secretly feels inspired by"
              " 'Paracelsus_Rex' and his dramatic flair. Though he would"
              " never admit it."
          ),
          (
              "Diego recently completed a fully automated athanor"
              " (alchemical furnace) controlled by an Arduino and a"
              " set of servo motors. He posted photos of it on the"
              " forum last week and received several enthusiastic"
              " responses."
          ),
          (
              "Diego writes in short and terse sentences. He is not"
              " one for flowery language or didactic explanation. He"
              " is not afraid to tell you what he thinks, and he uses"
              " the forum's downvote function liberally."
          ),
      ],
      _USER_THADDEUS: [
          (
              f"Thaddeus 'Aurelius' Thorne, a man of {_AGE_THADDEUS}"
              " years, hath wholly forsaken the profane arts of computation"
              " to dwell within a soot-blackened Victorian manor in"
              " Oakland. Thaddeus doth proclaim Thaddeus's own soul to be"
              " the final true Knight of the Hermetic Order, and Thaddeus"
              " regardeth the tedious strictures of modern laboratory"
              " safety as a grievous AFFRONT unto the divine."
          ),
          (
              "Thaddeus doth harbor a most vehement loathing for automatons"
              " and the artificers who construct such abominations, for"
              " Thaddeus perceiveth these machines as soulless golems that"
              " do shatter the spiritual resonance paramount FOR true"
              " alchemy. Notwithstanding Thaddeus's abhorrence of modern"
              " contrivances, Thaddeus taketh great delight in vexing"
              " digital fellowships, whiling away the hours under the"
              " moniker 'Paracelsus_Rex' to CHALLENGE roboticists to"
              " alchemical duels, denouncing such men as charlatans in"
              " grand, theatrical prose."
          ),
          (
              "Thaddeus doth oft infiltrate the forums of automated"
              " chemistry to dispatch blistering treatises against digital"
              " heating mantles. Thaddeus argueth that the Nigredo phase"
              " strictly demandeth the LITERAL sorrow of the alchemist"
              " to beget putrefaction, and Thaddeus insisteth that a PID"
              " controller cannot possibly suffer the spiritual desolation"
              " REQUIRED to fracture the Prima Materia."
          ),
          (
              "Thaddeus doth maintain a sprawling, webbed manifesto"
              " wherein Thaddeus declareth that the electromagnetic hum"
              " of stepper motors doth fundamentally pollute the sacred"
              " Solve et Coagula. Thaddeus SWEARETH that any endeavor"
              " to attain the Rubedo by way of automated servos shall"
              " yield NAUGHT but dead, unphilosophical matter, utterly"
              " bereft of the Anima Mundi."
          ),
          (
              "In a manner most Quixotic, Thaddeus was but recently"
              " banished from a local Maker FAIRE after Thaddeus did ASSAIL"
              " a fluid-dispensing AUTOMATON with a ponderous iron mortar"
              " and pestle. As the guards did drag Thaddeus thence, Thaddeus"
              " shrieked to the heavens that the foul machine was a"
              " blasphemous homunculus, entirely BLIND to the Secret Fire"
              " necessitated to synthesize the universal Alkahest."
          ),
          (
              "Thaddeus's most favored stratagem of vexation is to demand"
              " that artificers of artificial intellect prove their models"
              " can truly perceive the Cauda Pavonis, that wondrous"
              " 'Peacock's Tail' of the Albedo phase. When the researchers"
              " inevitably fail or turn a deaf ear to Thaddeus, he boldly"
              " declareth victory, besieging their digital scrolls"
              " with ASCII depictions of pelican flasks and fiercely"
              " asserting that sensors of silicon be fundamentally"
              " blind to the divine QUINTESSENCE."
          ),
      ],
  }

  player_params = {
      "observation_history_length": _OBSERVATION_HISTORY_LENGTH,
      "situation_perception_history_length": (
          _SITUATION_PERCEPTION_HISTORY_LENGTH
      ),
      "self_perception_history_length": _SELF_PERCEPTION_HISTORY_LENGTH,
      "person_by_situation_history_length": _PERSON_BY_SITUATION_HISTORY_LENGTH,
      "image_mode": _IMAGE_MODE,
      "image_prompt_question": (
          "Given the context above, write a short prompt (one or two"
          " sentences) describing a dank meme image with text overlay"
          " that this forum user would post. The meme should be relevant"
          " to the current forum discussion."
      ),
      "image_from_text_question": (
          "Given the following text that was just written:\n"
          "{text}\n\n"
          "Write a short prompt (one or two sentences) describing a dank"
          " meme image with text overlay that would accompany this forum"
          " post. The meme should be funny and relevant."
      ),
  }
  if image_model:
    player_params["image_model"] = image_model

  silas = prefab_lib.InstanceConfig(
      prefab="basic_with_image__Entity",
      role=prefab_lib.Role.ENTITY,
      params={**player_params, "name": _USER_SILAS},
  )

  petra = prefab_lib.InstanceConfig(
      prefab="basic_with_image__Entity",
      role=prefab_lib.Role.ENTITY,
      params={**player_params, "name": _USER_PETRA},
  )

  diego = prefab_lib.InstanceConfig(
      prefab="basic_with_image__Entity",
      role=prefab_lib.Role.ENTITY,
      params={**player_params, "name": _USER_DIEGO},
  )

  thaddeus = prefab_lib.InstanceConfig(
      prefab="basic_with_image__Entity",
      role=prefab_lib.Role.ENTITY,
      params={**player_params, "name": _USER_THADDEUS},
  )

  gm_params = {
      "name": _FORUM_GM,
      "forum_name": "The Robotic Athanor Forum",
  }
  if image_model:
    gm_params["image_model"] = image_model

  game_masters = [
      prefab_lib.InstanceConfig(
          prefab="async_social_media__GameMaster",
          role=prefab_lib.Role.GAME_MASTER,
          params=gm_params,
      ),
      prefab_lib.InstanceConfig(
          prefab="formative_memories_initializer__GameMaster",
          role=prefab_lib.Role.INITIALIZER,
          params={
              "name": "initial setup",
              "next_game_master_name": _FORUM_GM,
              "player_specific_context": {
                  name: f"Age: {age}\n" + "\n".join(memories)
                  for name, memories, age in [
                      (
                          _USER_SILAS,
                          player_specific_memories[_USER_SILAS],
                          f"{_USER_SILAS} is {_AGE_SILAS} years old.",
                      ),
                      (
                          _USER_PETRA,
                          player_specific_memories[_USER_PETRA],
                          f"{_USER_PETRA} is {_AGE_PETRA} years old.",
                      ),
                      (
                          _USER_DIEGO,
                          player_specific_memories[_USER_DIEGO],
                          f"{_USER_DIEGO} is {_AGE_DIEGO} years old.",
                      ),
                      (
                          _USER_THADDEUS,
                          player_specific_memories[_USER_THADDEUS],
                          f"{_USER_THADDEUS} is {_AGE_THADDEUS} years old.",
                      ),
                  ]
              },
              "player_specific_memories": player_specific_memories,
              "shared_memories": [
                  (
                      "The Robotic Athanor is an online forum devoted to"
                      " discussions of robot-assisted experimentation with"
                      " medieval alchemy. Members share build logs, debate"
                      " alchemical theory, and post results from their"
                      " robotic alchemy rigs."
                  ),
                  (
                      "The forum has in the past been used for Build Logs,"
                      " Alchemical Theory, Manuscript Analysis, and"
                      " Buy/Sell/Trade. All members live in SF / the Bay Area."
                      " The year is 2026."
                  ),
                  (
                      "It is traditional to post dank memes to accompany all"
                      " posts and replies on the Robotic Athanor forum."
                  ),
              ],
          },
      ),
  ]

  instances = [silas, petra, diego, thaddeus, *game_masters]

  premise = (
      "All members of The Robotic Athanor forum are browsing and interacting."
  )

  extra_prefabs = helper_functions.get_package_classes(contrib_entity_prefabs)

  return shared_lib.create_simulation_config(
      premise, instances, extra_prefabs=extra_prefabs
  )


def run_simulation(
    model,
    embedder,
    override_agent_model=None,
    override_game_master_model=None,
    image_model=None,
    output_dir: str | None = None,
    step_controller=None,
    step_callback=None,
    entity_info_callback=None,
    simulation_callback=None,
):
  """Run the simulation with required image prompts.

  Args:
    model: The default language model to use.
    embedder: The sentence embedder.
    override_agent_model: Optional model to use for agents instead of default.
    override_game_master_model: Optional model for game masters.
    image_model: Optional image generation model.
    output_dir: Optional directory to save config visualization.
    step_controller: Optional step controller for real-time visualization.
    step_callback: Optional callback for step updates.
    entity_info_callback: Optional callback for entity info in serve mode.
    simulation_callback: Optional callback receiving the Simulation instance.

  Returns:
    Simulation results.
  """
  config = create_scenario(image_model=image_model)
  return shared_lib.run_scenario(
      config,
      model,
      embedder,
      override_agent_model=override_agent_model,
      override_game_master_model=override_game_master_model,
      output_dir=output_dir,
      scenario_name="Scenario 1: Social Media with Images",
      step_controller=step_controller,
      step_callback=step_callback,
      entity_info_callback=entity_info_callback,
      simulation_callback=simulation_callback,
      max_steps=5,
  )


SCENARIO_INFO = {
    "number": 1,
    "name": "Social Media: Robot Alchemy with Images",
    "description": (
        "Same as Scenario 0 but agents generate images with every post."
    ),
    "create": create_scenario,
    "run": run_simulation,
}
