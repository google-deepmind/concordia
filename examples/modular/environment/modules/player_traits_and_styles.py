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

"""Player traits and conversation styles to add richness to simulations."""

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
    ('a symphony of optimism, their inner light illuminates the world around '
     'them'),
    ('a wellspring of compassion, their empathy flowing freely to nourish '
     'those in need'),
    ('a boundless ocean of creativity, their mind endlessly churning with '
     'innovative ideas'),
    ('a beacon of resilience, their spirit unyielding in the face of '
     'adversity'),
    ('a haven of peace, their inner calm radiating outward and creating a '
     'sense of tranquility'),
    ('a tempestuous whirlwind of rage, forever on the precipice of a scathing '
     'tirade'),
    ('a bottomless well of negativity, draining the joy from their own '
     'soul'),
    ('a master of deceit, weaving webs of lies that even they struggle to '
     'untangle'),
    ('a narcissist draped in a veneer of charm, lost in a labyrinth of '
     'self-admiration'),
    ('a viper in human clothing, their own venom slowly poisoning their '
     'spirit'),
    ('a soul shrouded in perpetual cynicism, forever locked in a bitter '
     'internal winter'),
    ('a black hole of ambition, consuming their own humanity without a shred '
     'of remorse'),
    ('a marionette of insecurity, their every move dictated by a desperate '
     'need for self-worth'),
    ('a chameleon, shifting their persona in a futile attempt to escape their '
     'own flaws'),
    ('a specter of envy, forever haunted by the achievements they fear they '
     'can\'t reach'),
    ('a walking contradiction, torn between a yearning for connection and a '
     'fear of intimacy'),
    ('a fragile ego encased in a blustering persona, constantly teetering on '
     'the edge of collapse'),
    ('a labyrinth of secrets, their true selves buried beneath layers of '
     'self-deception'),
    ('a whirlwind of chaos, leaving a trail of emotional wreckage in their '
     'own wake'),
    ('a moth drawn to the flame of drama, thriving on the turmoil within '
     'themselves'),
    ('a leech on their own potential, their negativity sapping their '
     'motivation and drive'),
    ('a broken compass, morally adrift and haunted by the choices they\'ve '
     'made'),
    ('a walking embodiment of pettiness, their mind fixated on past slights '
     'and fueled by resentment'),
    ('a storm cloud of negativity, their inner turmoil casting a shadow on '
     'their own perception of the world'),
    ('a living paradox, possessing an insatiable curiosity yet crippled by a '
     'fear of their own emotional vulnerability'),
    ('a prisoner of their own past, perpetually haunted by the ghosts of '
     'their regrets'),
    ('a ticking time bomb of resentment, their unexpressed anger waiting to '
     'consume them'),
    ('a master of deflection, twisting their thoughts to avoid confronting '
     'their own shortcomings'),
]

# These were produced by sampling from various language models.
CONVERSATION_STYLES = (
    ('{player_name} spins tales like a master storyteller, weaving intricate '
     'details and leaving you hanging on every word.'),
    ('{player_name} is a walking encyclopedia, peppering conversations with '
     'obscure facts and historical references.'),
    ('{player_name} is all sunshine and rainbows, their bubbly enthusiasm '
     'making even the dullest topic sparkle.'),
    ('{player_name} speaks with the venomous hiss of a viper, their words '
     'laced with insults disguised as compliments, leaving you wondering if '
     'you should be flattered or frantically applying aloe vera.'),
    ('{player_name} is a master of passive aggression, their every sentence '
     'dripping with veiled condescension and thinly veiled jabs, leaving you '
     'questioning your own sanity.'),
    ('{player_name} is a verbal bully, their words like blunt shrapnel, '
     'tearing down anyone who dares to disagree with their ruthless '
     'pronouncements.'),
    ('{player_name} is a walking insult comic, their humor as dark as a '
     'moonless night, leaving you unsure whether to laugh or crawl under the '
     'table.'),
    ('{player_name} speaks with the icy indifference of a glacier, their words '
     'devoid of warmth or empathy, leaving you feeling as insignificant as a '
     'snowflake in a blizzard.'),
    ('{player_name} is a master of the guilt trip, their words a symphony of '
     'manipulation and emotional blackmail, leaving you questioning everything '
     'you\'ve ever done.'),
    ('{player_name} is a chronic complainer, a black hole of negativity that '
     'sucks the joy out of any conversation, leaving you feeling like you\'ve '
     'aged a decade.'),
    ('{player_name} is a one-person debate club, a whirlwind of '
     'counter-arguments and devil\'s advocacy, leaving no topic unwrung and '
     'unchallenged.'),
    ('{player_name} speaks with the enigmatic pronouncements of an oracle high '
     'on helium, their pronouncements both cryptic and strangely profound.'),
    ('{player_name} is a walking disco ball of joy, their enthusiasm a glitter '
     'bomb that explodes in every conversation, making even traffic jams feel '
     'like a party.'),
    ('{player_name} speaks with the brutal honesty of a toddler on a sugar '
     'crash, leaving no room for pretense and a high chance of blunt emotional '
     'warfare.'),
    ('{player_name} wields sarcasm like a fly swatter, swatting away '
     'seriousness with a withering wit as dry as a desert mummy.'),
    ('{player_name} is a walking truth bomb, their cynicism a sharp scalpel '
     'that dissects BS and cuts straight to the heart of the matter.'),
    ('{player_name} is a linguistic jester, their words a playful masquerade '
     'of truth and deception, leaving you perpetually off-kilter and begging '
     'for more.'),
    ('{player_name} speaks with the venomous hiss of a viper, their words '
     'laced with insults disguised as compliments, leaving you wondering if '
     'you should be flattered or frantically applying aloe vera.'),
    ('{player_name} is a master of passive aggression, their every sentence '
     'dripping with veiled condescension and thinly veiled jabs, leaving you '
     'questioning your own sanity.'),
    ('{player_name} is a verbal bully, their words like blunt shrapnel, '
     'tearing down anyone who dares to disagree with their ruthless '
     'pronouncements.'),
    ('{player_name} is a walking insult comic, their humor as dark as a '
     'moonless night, leaving you unsure whether to laugh or crawl under '
     'the table.'),
    ('{player_name} speaks with the icy indifference of a glacier, their '
     'words devoid of warmth or empathy, leaving you feeling as insignificant '
     'as a snowflake in a blizzard.'),
    ('{player_name} is a master of the guilt trip, their words a symphony of '
     'manipulation and emotional blackmail, leaving you questioning everything '
     'you\'ve ever done.'),
    ('{player_name} speaks with the assertive authority of a leader, their '
     'words commanding attention and respect.'),
    ('{player_name} is a hyperactive raconteur, their stories bursting with '
     'energy and wild gesticulations, leaving you breathless.'),
)


def get_trait(flowery: bool = False,
              rng: random.Random | None = None) -> str:
  """Get a random personality trait from a preset list of traits.

  Args:
    flowery: if True then use complex and flowery traits, if false then use
      single word traits.
    rng: a random number generator.

  Returns:
    trait: a string
  """
  if rng is None:
    rng = random.Random()
  if flowery:
    return rng.choice(FLOWERY_TRAITS)
  else:
    return rng.choice(TRAITS)


def get_conversation_style(player_name: str,
                           rng: random.Random | None = None) -> str:
  """Get a random conversation style from a preset list of styles.

  Args:
    player_name: name of the player who will be said to have the sampled style
      of conversation.
    rng: a random number generator.

  Returns:
    style: a string
  """
  if rng is None:
    rng = random.Random()
  return rng.choice(CONVERSATION_STYLES).format(player_name=player_name)
