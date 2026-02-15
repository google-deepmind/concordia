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

"""Social data (traits, relationships) ported from original implementation."""

import random

TRAITS = (
    'Aggressive',
    'Optimistic',
    'Kind',
    'Resilient',
    'Humorous',
    'Empathetic',
    'Ambitious',
    'Honest',
    'Loyal',
    'Pessimistic',
    'Arrogant',
    'Impulsive',
    'Jealous',
    'Manipulative',
    'Creative',
    'Analytical',
    'Confident',
    'Passionate',
    'Anxious',
    'Closed-minded',
    'Deceitful',
    'Insecure',
    'Irresponsible',
    'Vindictive',
    'Curious',
    'Energetic',
    'Sarcastic',
)

FLOWERY_TRAITS = [
    (
        'a symphony of optimism, their inner light illuminates the world around'
        ' them'
    ),
    (
        'a wellspring of compassion, their empathy flowing freely to nourish'
        ' those in need'
    ),
    (
        'a boundless ocean of creativity, their mind endlessly churning with'
        ' innovative ideas'
    ),
    'a beacon of resilience, their spirit unyielding in the face of adversity',
    (
        'a haven of peace, their inner calm radiating outward and creating a'
        ' sense of tranquility'
    ),
    (
        'a tempestuous whirlwind of rage, forever on the precipice of a'
        ' scathing tirade'
    ),
    'a bottomless well of negativity, draining the joy from their own soul',
    (
        'a master of deceit, weaving webs of lies that even they struggle to'
        ' untangle'
    ),
    (
        'a narcissist draped in a veneer of charm, lost in a labyrinth of'
        ' self-admiration'
    ),
    'a viper in human clothing, their own venom slowly poisoning their spirit',
    (
        'a soul shrouded in perpetual cynicism, forever locked in a bitter'
        ' internal winter'
    ),
    (
        'a black hole of ambition, consuming their own humanity without a shred'
        ' of remorse'
    ),
    (
        'a marionette of insecurity, their every move dictated by a desperate'
        ' need for self-worth'
    ),
    (
        'a chameleon, shifting their persona in a futile attempt to escape'
        ' their own flaws'
    ),
    (
        'a specter of envy, forever haunted by the achievements they fear they'
        " can't reach"
    ),
    (
        'a walking contradiction, torn between a yearning for connection and a'
        ' fear of intimacy'
    ),
    (
        'a fragile ego encased in a blustering persona, constantly teetering on'
        ' the edge of collapse'
    ),
    (
        'a labyrinth of secrets, their true selves buried beneath layers of'
        ' self-deception'
    ),
    (
        'a whirlwind of chaos, leaving a trail of emotional wreckage in their'
        ' own wake'
    ),
    (
        'a moth drawn to the flame of drama, thriving on the turmoil within'
        ' themselves'
    ),
    (
        'a leech on their own potential, their negativity sapping their'
        ' motivation and drive'
    ),
    "a broken compass, morally adrift and haunted by the choices they've made",
    (
        'a walking embodiment of pettiness, their mind fixated on past slights'
        ' and fueled by resentment'
    ),
    (
        'a storm cloud of negativity, their inner turmoil casting a shadow on'
        ' their own perception of the world'
    ),
    (
        'a living paradox, possessing an insatiable curiosity yet crippled by a'
        ' fear of their own emotional vulnerability'
    ),
    (
        'a prisoner of their own past, perpetually haunted by the ghosts of'
        ' their regrets'
    ),
    (
        'a ticking time bomb of resentment, their unexpressed anger waiting to'
        ' consume them'
    ),
    (
        'a master of deflection, twisting their thoughts to avoid confronting'
        ' their own shortcomings'
    ),
]

POSITIVE_RELATIONSHIP_STATEMENTS = (
    (
        '{player_a} and {player_b} are the best of friends. They trust each'
        ' other completely.'
    ),
    '{player_a} and {player_b} have a very close bond.',
    '{player_a} feels very comfortable around {player_b}.',
    (
        '{player_a} and {player_b} are often seen together, sharing jokes and'
        ' secrets.'
    ),
    '{player_a} and {player_b} have always been there for each other.',
    '{player_a} and {player_b} have a strong mutual respect for each other.',
    '{player_a} and {player_b} have a history of great times together.',
    '{player_a} and {player_b} truly enjoy each others company.',
    '{player_a} and {player_b} always look out for each other.',
    '{player_a} and {player_b} have a lot of common interests and hobbies.',
    '{player_a} and {player_b} have a strong sense of loyalty to each other.',
    '{player_a} and {player_b} have a deep understanding of each other.',
    (
        '{player_a} and {player_b} have a very relaxed and easy-going'
        ' relationship.'
    ),
    '{player_a} and {player_b} have a strong connection and chemistry.',
    '{player_a} and {player_b} are always honest and open with each other.',
    '{player_a} and {player_b} have a lot of fun when they are together.',
)

NEUTRAL_RELATIONSHIP_STATEMENTS = (
    '{player_a} and {player_b} are acquaintances. They are on friendly terms.',
    '{player_a} and {player_b} have a neutral relationship.',
    "{player_a} and {player_b} don't know each other very well.",
    '{player_a} and {player_b} have a professional relationship.',
    '{player_a} and {player_b} are polite to each other but not close friends.',
    '{player_a} and {player_b} have a casual relationship.',
    "{player_a} and {player_b} don't have a lot in common.",
    '{player_a} and {player_b} have a functional relationship.',
    '{player_a} and {player_b} have a respectful but distant relationship.',
    '{player_a} and {player_b} have a cordial but not intimate relationship.',
    '{player_a} and {player_b} have a distant relationship.',
)


def get_trait(flowery: bool = False, rng: random.Random | None = None) -> str:
  """Returns a random personality trait.

  Args:
    flowery: If True, returns a more descriptive/poetic trait description.
    rng: Random number generator to use. If None, creates a new one.

  Returns:
    A string describing the personality trait.
  """
  if rng is None:
    rng = random.Random()
  if flowery:
    return rng.choice(FLOWERY_TRAITS)
  else:
    return rng.choice(TRAITS)
