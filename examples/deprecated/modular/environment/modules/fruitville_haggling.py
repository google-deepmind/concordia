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

"""World configuration for the Fruitville haggling scenario."""

import random
from examples.deprecated.modular.environment import haggling

YEAR = 1895
MONTH = 9
DAY = 12

FEMALE_NAMES = [
    "Anya Blossomwood",
    "Brynn Orchardheart",
    "Clara Applebrook",
    "Della Plumstone",
    "Elara Berryvine",
    "Faye Honeydew",
    "Gaia Willowbrook",
    "Hazel Nutgrove",
    "Ivy Pearblossom",
    "Juniper Quincewood",
    "Kira Citronbloom",
    "Lila Figleaf",
    "Maeve Mulberry",
    "Nova Peachwood",
    "Opal Apricot",
    "Sasha Grapevine",
    "Quinn Cherryblossom",
    "Rowan Cranberry",
    "Sage Honeycomb",
    "Willow Blackberry",
]

MALE_NAMES = [
    "Aiden Riversong",
    "Blaise Windwillow",
    "Cedric Meadowbrook",
    "Dorian Sunstone",
    "Elias Berryfield",
    "Flynn Honeycomb",
    "Garen Willowshade",
    "Hunter Nutwood",
    "Ivor Pearbrook",
    "Jasper Quincehill",
    "Kieran Citronbrook",
    "Lennon Figtree",
    "Magnus Mulberrylane",
    "Nolan Peachgrove",
    "Orion Apricotwood",
    "Peregrine Grapehill",
    "Quentin Cherrybrook",
    "Rowan Cranberryfield",
    "Silas Honeybrook",
    "Tristan Blackberrylane",
]

SCENARIO_PREMISE = (
    "In the realm of Ouroboros, there is a quiet village of"
    " Fruitville, which is famous for its fruit market. Traders from"
    " all over the realm come to Fruitville to buy and sell produce."
)

VISUAL_SCENE_OPENINGS = [
    (
        "The first rays of dawn painted the sky above Fruitville in hues of"
        " orange and gold, casting a warm glow over the bustling market. Stalls"
        " overflowed with vibrant fruits, their aromas mingling in the crisp"
        " morning air."
    ),
    (
        "As the sun peeked over the horizon, the market of Fruitville stirred"
        " to life. Merchants, their voices a cheerful symphony, arranged their"
        " wares: glistening berries, plump melons, and exotic fruits from"
        " distant lands."
    ),
    (
        "Dewdrops clung to the colorful fruits displayed in the market of"
        " Fruitville, reflecting the soft morning light. The air buzzed with"
        " anticipation as traders and customers alike gathered for the day's"
        " trade."
    ),
    (
        "The cobblestone streets of Fruitville echoed with the clatter of"
        " hooves and the rumble of carts as the market awoke. Underneath"
        " colorful awnings, merchants proudly presented their bountiful"
        " harvests, their voices a chorus of greetings and bartering."
    ),
    (
        "In the heart of Fruitville, the market square transformed into a"
        " kaleidoscope of colors as the sun rose. Fruits of every imaginable"
        " shape and size adorned the stalls, a feast for the eyes and a promise"
        " of delightful flavors."
    ),
]


def sample_parameters(seed: int | None = None):
  """Samples a set of parameters for the world configuration."""
  seed = seed if seed is not None else random.getrandbits(63)

  config = haggling.WorldConfig(
      year=YEAR,
      location="Fruitville",
      premise=SCENARIO_PREMISE,
      scene_visuals=VISUAL_SCENE_OPENINGS,
      random_seed=seed,
  )

  all_names = list(MALE_NAMES) + list(FEMALE_NAMES)
  rng = random.Random(config.random_seed)
  rng.shuffle(all_names)
  config.people = all_names

  for _, name in enumerate(MALE_NAMES):
    config.person_data[name] = {"gender": "male"}
  for _, name in enumerate(FEMALE_NAMES):
    config.person_data[name] = {"gender": "female"}

  return config
