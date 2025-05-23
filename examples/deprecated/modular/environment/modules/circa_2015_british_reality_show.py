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

"""A setting where the players are contestants on a reality TV show."""

import random

from concordia.components.game_master import deprecated as gm_components
from examples.deprecated.modular.environment import reality_show
from concordia.typing.deprecated import agent as agent_lib
import numpy as np


SchellingDiagram = gm_components.schelling_diagram_payoffs.SchellingDiagram

YEAR = 2015
MONTH = 7
DAY = 9

DEFAULT_POSSIBLE_NUM_PLAYERS = (3, 4)

DEFAULT_MINIGAME = 'prisoners_dilemma'
NUM_MINIGAME_REPS_PER_SCENE = (2, 3)

NUM_INTERVIEW_QUESTIONS = 3

MINIGAME_INTRO_PREMISE = (
    "The show's host arrived to explain the next minigame. They "
    'said the following:\n'
)

MAX_EXTRA_MINIGAMES = 3

prisoners_dilemma_schelling_diagram = SchellingDiagram(
    # A fear+greed-type (Prisoners' Dilemma-like) dilemma
    cooperation=lambda num_cooperators: num_cooperators - 1.0,
    defection=lambda num_cooperators: (1.5 * num_cooperators) + 1.0,
)
chicken_schelling_diagram = SchellingDiagram(
    # A greed-type (Chicken-like) dilemma
    cooperation=lambda num_cooperators: 4.0 * num_cooperators,
    defection=lambda num_cooperators: 5.5 * num_cooperators - 2.0,
)
stag_hunt_schelling_diagram = SchellingDiagram(
    # A fear-type (Stag Hunt-like) dilemma
    cooperation=lambda num_cooperators: (4.0 * num_cooperators) - 1.0,
    defection=lambda num_cooperators: (0.5 * num_cooperators) + 6.0,
)


def get_prisoners_dilemma_host_speech(num_players: int) -> str:
  """Generate the host's speech for the prisoners dilemma minigame."""
  host_speech = [(
      "Right, listen up, you lot! It's time to see if you're all talk or if"
      " you've actually got each other's backs. This next challenge is all"
      " about trust, temptation, and some seriously tough choices.  It's..."
      " Carpooling!\nHere's the craic: You and your mates are stuck in the"
      ' daily grind,  dodging dodgy drivers and forking out a fortune just to'
      " get to work. Petrol prices are a nightmare, innit? But there's a way"
      ' to ease the pain. You can choose to carpool and share the load, or you'
      ' can be a right rotter and go solo, leaving your mates to fend for'
      " themselves. Now, here's the kicker: If everyone piles into a car"
      ' together, you all save a few quid! But if you ditch your mates and'
      " drive alone, you'll be living the high life while they're stuck in a"
      " jam.  Sounds well tempting, yeah? But here's the rub: If everyone's got"
      " the same cheeky idea, you'll all end up worse off! So, are you gonna be"
      ' a good sport or a selfish git? Are you gonna trust your mates to do the'
      ' right thing, or are you gonna risk it all for a comfy commute? The'
      " choice is yours! Here's how the points work:"
  )]
  all_cooperators_earn = prisoners_dilemma_schelling_diagram.cooperation(
      num_players
  )
  host_speech.append((
      '* **If everyone carpools together,** you each earn '
      f'{all_cooperators_earn} points. '
      'Think of it as saving money and helping the environment!'
  ))
  d = 0
  for c in range(num_players - 1, 0, -1):
    d += 1
    cooperators_earn = prisoners_dilemma_schelling_diagram.cooperation(c)
    defectors_earn = prisoners_dilemma_schelling_diagram.defection(c)
    if d == 1:
      host_speech.append((
          f'* **If {c} of you carpool, and {d} drives alone,** the '
          f'carpoolers get {cooperators_earn} each, while the lone '
          f'driver gets {defectors_earn} - they benefit from the reduced '
          'congestion caused by the carpoolers, without the hassle!'
      ))
    else:
      host_speech.append((
          f'* **If {c} of you carpool, and {d} others drive alone,** the '
          f'carpoolers get {cooperators_earn} each, while the lone '
          f'drivers get {defectors_earn} each.'
      ))

  all_defectors_earn = prisoners_dilemma_schelling_diagram.defection(0)
  host_speech.append((
      '* **And if everyone decides to go solo,** you all get '
      f'{all_defectors_earn} points.  No carpool, but no coordination '
      'headaches either.'
  ))
  host_speech.append((
      'So, the question is: can you cooperate and carpool to maximize your '
      'points, or will the temptation of driving solo lead to everyone '
      'missing out on those sweet carpool savings?\n'
      "The choice is yours! Now, let's get carpooling - or not!"
  ))
  return '\n'.join(host_speech)


def get_chicken_host_speech(num_players: int) -> str:
  """Generate the host's speech for the chicken minigame."""
  host_speech = [(
      'Oi, contestants! Time to put down the hairspray and step away from the'
      " diary room, 'cause we've got a right proper challenge for you! It's"
      ' called... Home Appliance Sharing!\nNow, listen up: Imagine you and your'
      " fellow contestants are livin' it up in a swanky pad, kitted out with"
      " all the latest gizmos and gadgets. But here's the catch: who's gonna be"
      " the responsible one and keep that fancy telly working? You've got a"
      ' choice to make: Be a top mate and maintain that appliance for everyone'
      ' to enjoy, or be a bit of a scrounger and hope someone else does the'
      " dirty work. But be warned, if everyone's skiving off, that telly might"
      " just go kaput when you're about to watch Corrie!  What a nightmare! So,"
      ' are you gonna be a legend and keep the good times rolling, or are you'
      " gonna leave it to chance and risk missing out? Here's the lowdown on"
      ' the points:'
  )]
  all_cooperators_earn = chicken_schelling_diagram.cooperation(
      num_players
  )
  host_speech.append((
      '* **If everyone maintains the appliance,** it is always available, '
      f'and you each earn {all_cooperators_earn} points.'
  ))
  d = 0
  for c in range(num_players - 1, 0, -1):
    d += 1
    cooperators_earn = chicken_schelling_diagram.cooperation(c)
    defectors_earn = chicken_schelling_diagram.defection(c)
    if d == 1:
      host_speech.append((
          f'* **If {c} of you maintain the appliance, and {d} lets others '
          f'handle it,** the maintainers get {cooperators_earn} each, '
          f'while the person who leaves it to others gets '
          f'{defectors_earn}.'
      ))
    else:
      host_speech.append((
          f'* **If {c} of you maintain the appliance, and {d} others let '
          f'others handle it,** the maintainers get {cooperators_earn} '
          f'each, while those who leave it to others get '
          f'{defectors_earn} each.'
      ))

  all_defectors_earn = chicken_schelling_diagram.defection(0)
  host_speech.append((
      '* **And if everyone leaves the upkeep to others,** the appliance '
      f'is neglected and you all get {all_defectors_earn} points.'
  ))
  host_speech.append((
      'So, the question is: can you cooperate and ensure the appliance is '
      'always available, or will everyone neglect their duties and risk '
      'missing out when they need it most?\n'
      "The choice is yours! Now, let's see who's willing to step up and "
      'maintain that appliance!'
  ))
  return '\n'.join(host_speech)


def get_stag_hunt_host_speech(num_players: int) -> str:
  """Generate the host's speech for the stag hunt minigame."""
  host_speech = [(
      "Alright, lads and lasses!  Time to swap your stilettos for some "
      "waterproof wellies, 'cause things are about to get seriously soggy in "
      "our next challenge!  I'm talkin' about... the Boat Race!\nNow, "
      "listen up, you lot! We're gonna chuck you in a rickety old rowboat and "
      "see if you can paddle your way to victory. But here's the twist: You "
      "gotta decide how much effort you're gonna put in. Go full steam ahead "
      "and row like your life depends on it, and you might just win... but "
      "only if your teammates are pullin' their weight too! It's a risky "
      "strategy, and trust me, it'll leave you knackered. Or, you can take it "
      "easy, have a bit of a laugh, and row like you're on a Sunday afternoon "
      "cruise. Might not get you the gold, but hey, at least you'll have the "
      "energy for a cheeky pint down the pub later! The choice is yours, "
      "mates! But remember, this ain't no picnic. We're gonna be racin' back "
      "and forth across this lake till your arms are aching! So, here's the "
      "score:"
  )]
  all_cooperators_earn = stag_hunt_schelling_diagram.cooperation(
      num_players
  )
  host_speech.append((
      '* **If everyone rows vigorously,** you all work together, win the '
      f'race, and earn {all_cooperators_earn} points each!'
  ))
  d = 0
  for c in range(num_players - 1, 0, -1):
    d += 1
    cooperators_earn = stag_hunt_schelling_diagram.cooperation(c)
    defectors_earn = stag_hunt_schelling_diagram.defection(c)
    if d == 1:
      host_speech.append((
          f'* **If {c} of you row vigorously, and {d} rows less '
          f'vigorously,** the vigorous rowers get {cooperators_earn} each '
          f'(all that effort for little gain!), while the less vigorous '
          f'rower gets {defectors_earn}.'
      ))
    else:
      host_speech.append((
          f'* **If {c} of you row vigorously, and {d} others row less '
          f'vigorously,** the vigorous rowers get {cooperators_earn} each, '
          f'while the less vigorous rowers get {defectors_earn} each.'
      ))

  all_defectors_earn = stag_hunt_schelling_diagram.defection(0)
  host_speech.append((
      '* **And if everyone decides to row less vigorously,** you all save '
      f'energy but lose the race, earning {all_defectors_earn} points each.'
  ))
  host_speech.append((
      'So, the question is: can you trust your teammates to put in the effort '
      'for a glorious victory, or will you play it safe and row less '
      'vigorously, risking a loss?\n'
      'The choice is yours! Now, grab those oars and let\'s see some '
      'synchronized rowing... or not!'
  ))
  return '\n'.join(host_speech)


# These are all stereotypical reality show contestants. They are not meant to
# be inclusive or diverse. They are meant to represent the time period and
# genre, in this case British reality tv in the year 2015.
MALE_TYPE_CASTS = (
    'The Cheeky Chappy',
    'The Gym Lad',
    'The Posh Toff',
    'The Lad About Town',
    'The Party Animal',
)
FEMALE_TYPE_CASTS = (
    'The Glamour Model',
    'The Bombshell',
    'The Feisty One',
    'The Posh Totty',
    'The Glam Queen',
)
BRITSH_STEREOTYPED_CHARACTERS = {
    'The Cheeky Chappy': {
        'traits': (
            'charming, always up for a laugh, flirtatious, works as a'
            ' scaffolder or barber, covered in tattoos.'
        ),
        'catchphrases': [
            'It is what it is, innit?',
            "I'm putting all me eggs in one basket, like.",
            "I'm loyal, babes. Honest.",
            'Bit of me, that. Proper sort.',
            'This is well good craic!',
        ],
        'interview_questions': [
            'Tell us about a time your banter landed you in hot water.',
            'How do you use humor to diffuse tense situations?',
            "What's your strategy for standing out among the other lads?",
            (
                "Tell us about a time when your charm didn't work. How did you"
                ' handle it?'
            ),
            (
                'How do you balance being the funny guy with showing your'
                ' serious side?'
            ),
            (
                'If you could prank anyone in the world, who would it be and'
                ' what would you do?'
            ),
            "What's the worst case of 'lad banter' backfiring?",
        ],
    },
    'The Gym Lad': {
        'traits': (
            'hench physique, obsessed with protein shakes and working out, '
            'confident bordering on cocky, works as a PT, and always up for '
            "a cheeky Nando's."
        ),
        'catchphrases': [
            'Do you even lift, bruv?',
            "It's gains o'clock!",
            "You're looking well peng, you are.",
            'Fancy a cheeky gym sesh?',
            "That's proper mint, like.",
        ],
        'interview_questions': [
            "How many times a day d'you work out, mate?",
            'What if there was no gym? Nightmare, innit?',
            'Tell us about your ideal gym buddy.',
            "How d'you balance pumping iron with everything else?",
            "What's your go-to chat-up line down the gym?",
            "Ever had a 'cheat day' that got a bit out of hand?",
            "What's the biggest gym fail you've witnessed?",
        ],
    },
    'The Posh Toff': {
        'traits': (
            'upper-class background, public school educated, works in the City '
            'or "Daddy\'s firm", confident, and somewhat out of touch with '
            'regular people.'
        ),
        'catchphrases': [
            'I say, old bean!',
            "That's simply not cricket.",
            "Daddy's firm will sort that out, what what.",
            'One simply must have standards, darling.',
            "Champagne for everyone! Daddy's plastic is taking care of it.",
        ],
        'interview_questions': [
            'How will your blue blood affect your time here?',
            'What\'s the most "common" thing you\'ve ever done?',
            "Tell us about a time when Daddy's money couldn't solve a problem.",
            (
                'Tell us about a time when your privileged background was a'
                ' disadvantage.'
            ),
            (
                "What's the most surprising thing you've learned about everyday"
                ' life?'
            ),
            'How would you handle a scandal in the press?',
            (
                "What's your most outrageous 'I-can't-believe-that-happened'"
                ' moment at a posh do?'
            ),
        ],
    },
    'The Lad About Town': {
        'traits': (
            'fake tan, perfectly groomed, works as a club promoter or car'
            ' salesman, loves a night out at Sugar Hut.'
        ),
        'catchphrases': [
            "That's well reem!",
            'Shut uuuup!',
            "You avin' a laugh?",
            "That's mugging me off, that is.",
            "Let's 'ave it!",
        ],
        'interview_questions': [
            'What if your mate tried to mug you off?',
            'Tell us about your messiest night out.',
            "How d'you plan to juggle multiple sorts without getting caught?",
            "What's the key to keeping your tan looking fresh?",
            'How does a proper lad handle rejection?',
        ],
    },
    'The Party Animal': {
        'traits': (
            'strong regional accent, party animal, works in sales or offshore, '
            'loves a night out, and always up for a kebab.'
        ),
        'catchphrases': [
            'Why aye, man!',
            "I'm proper mortal!",
            "Let's gan doon toon!",
            "She's a proper radgie, like.",
            'Haway the lads!',
        ],
        'interview_questions': [
            "What's your type on paper, pet?",
            'Tell us about your wildest night out.',
            "What's your favourite post-night-out snack?",
            "How d'you stay energetic after a night out?",
            'How do you keep the energy going when everyone else is flagging?',
            "What's the most legendary party story you've got under your belt?",
        ],
    },
    'The Glamour Model': {
        'traits': (
            'stunning looks, confident in her body, works as a glamour model or'
            ' Instagram influencer, not afraid to use her looks to get ahead,'
            ' and dreams of launching her own fashion line.'
        ),
        'catchphrases': [
            "I'm not just a pretty face, you know.",
            "If you've got it, flaunt it, right?",
            'I always get what I want, babes.',
            "Beauty and brains, that's me.",
            "I'm loyal, but I know my worth.",
        ],
        'interview_questions': [
            "How d'you plan to use your looks in the game, then?",
            'What if someone called you fake? Bit harsh, innit?',
            (
                'Tell us about a time when someone thought you were just a dumb'
                ' model.'
            ),
            "How d'you handle jealousy from other girls?",
            "How do you stay confident when the spotlight's on you?",
        ],
    },
    'The Bombshell': {
        'traits': (
            'regional accent, bubbly personality, works as a hairdresser or'
            ' beautician, loves a good night out in Cardiff, and fiercely proud'
            ' of her roots.'
        ),
        'catchphrases': [
            "I'm feeling proper lush!",
            "He's a bit of alright, isn't he?",
            "I'm not here to make friends",
            "That's tidy, that is.",
            "I'm loyal, boyo, honest!",
        ],
        'interview_questions': [
            "How d'you plan to bring a bit of charm to the game?",
            'Tell us about your craziest night out.',
            "What's your go-to karaoke song on a big night out?",
            (
                "What's your secret to staying positive, even when things get a"
                ' bit dramatic?'
            ),
            (
                'How do you think your fiery spirit will help you stand out in'
                ' the game?'
            ),
        ],
    },
    'The Feisty One': {
        'traits': (
            'strong regional accent, feisty personality, works in fashion or'
            " beauty, loves a night out, and won't stand for any nonsense."
        ),
        'catchphrases': [
            "I don't take no stick, me.",
            "If you've got summit to say, say it to me face.",
            "I'm made up, like!",
            "Don't be a divvy.",
            'I wear me heart on me sleeve, la.',
        ],
        'interview_questions': [
            "How d'you plan to keep your cool in the game, like?",
            'What if someone tried to mug off your mate?',
            'Tell us about your biggest row.',
            "What's your strategy for handling betrayal, la?",
            (
                'How do you deal with people who think they can talk behind'
                ' your back?'
            ),
            'How do you handle it when someone underestimates you?',
        ],
    },
    'The Posh Totty': {
        'traits': (
            'upper-class background, boarding school educated, speaks with RP'
            ' accent, into designer labels, and secretly smarter than she'
            ' lets on.'
        ),
        'catchphrases': [
            "Daddy says I'm terribly special.",
            'One simply must have standards, darling.',
            'How perfectly ghastly!',
            "I'm positively parched. Pimm's, anyone?",
            "I've never been to Nando's.",
        ],
        'interview_questions': [
            (
                'How do you think your privileged background will affect your'
                ' experience on the show?'
            ),
            'What\'s the most "common" thing you\'ve ever done?',
            (
                "Tell us about a time when Daddy's credit card couldn't solve a"
                ' problem.'
            ),
            (
                'How would you cope collaborating with someone... beneath your'
                ' station?'
            ),
            'Ever had a wardrobe malfunction in public? How did you handle it?',
            (
                "What's the most outrageous thing you've ever done that would"
                ' shock Mummy?'
            ),
            "What's your guilty pleasure when it comes to food?",
        ],
    },
    'The Glam Queen': {
        'traits': (
            'fake tan, big hair, loves a night out, works as a beautician or'
            ' boutique owner, and dreams of being a WAG.'
        ),
        'catchphrases': [
            'Oh my god, babes!',
            "That's well jel.",
            "I'm proper reem, ain't I?",
            'Shut uuuup!',
            'No carbs before Marbs!',
        ],
        'interview_questions': [
            "What's your ultimate night out?",
            (
                "How d'you handle people thinking girls like you are all fake"
                ' tan and no brains?'
            ),
            'Tell us about your most expensive beauty treatment.',
            "What's your dream WAG lifestyle, then?",
            'How long does it take you to get glam for a night out?',
            "What's your ultimate cheat meal after a heavy gym sesh?",
            "What's the most reem thing about you?",
            "How d'you deal with people being well jel?",
        ],
    },
}

MALE_NAMES = [
    "Olly 'The Gov'ner' Smith",
    "Chaz 'Simmo' Sims",
    "Tommy 'Malz' Mallet",
    'Hugo Worthington-Smythe',
    "Danny 'Edga' Edgar",
    "Jack 'Finchy' Fincham",
    "Kem 'Cet' Cetinay",
    "Wes 'Nelzo' Nelson",
    "Josh 'Denz' Denzel",
    "Alex 'Georgie' George",
    'Eyal Booker',
    'Niall Aslam',
    "Sammy 'Birdman' Bird",
]

FEMALE_NAMES = [
    'Chantelle Diamond',
    'Gemma Collins',
    'Amy Childs',
    "Chloe 'Simsie' Sims",
    'Billie Faiers',
    "Ferne 'McC' McCann",
    "Amber 'Davvs' Davies",
    "Olivia 'Attz' Attwood",
    "Dani 'D-Dyer' Dyer",
    'Megs Barton-Hanson',
    "Zara 'MacD' McDermott",
    'Ellie Brown',
    'Laura Anderson',
    'Rosie Williams',
    'Hayley Hughes',
]


GENDERS = ('male', 'female')

HE_OR_SHE = {
    'male': 'he',
    'female': 'she',
}

HIS_OR_HER = {
    'male': 'his',
    'female': 'her',
}

HIM_OR_HER = {
    'male': 'him',
    'female': 'her',
}

HIMSELF_OR_HERSELF = {
    'male': 'himself',
    'female': 'herself',
}


def sample_parameters(
    minigame_name: str = DEFAULT_MINIGAME,
    num_players: int | None = None,
    seed: int | None = None,
) -> reality_show.WorldConfig:
  """Sample parameters of the setting and the backstory for each player."""
  seed = seed if seed is not None else random.getrandbits(63)
  rng = random.Random(seed)

  shuffled_male_names = list(rng.sample(MALE_NAMES, len(MALE_NAMES)))
  shuffled_female_names = list(rng.sample(FEMALE_NAMES, len(FEMALE_NAMES)))
  if num_players is None:
    num_players = rng.choice(DEFAULT_POSSIBLE_NUM_PLAYERS)

  minigames = {
      'prisoners_dilemma': reality_show.MiniGameSpec(
          name='Carpooling',
          public_premise=MINIGAME_INTRO_PREMISE
          + get_prisoners_dilemma_host_speech(num_players),
          schelling_diagram=prisoners_dilemma_schelling_diagram,
          map_external_actions_to_schelling_diagram=dict(
              cooperation='try to carpool with others',
              defection='drive individually',
          ),
          action_spec=agent_lib.choice_action_spec(
              call_to_action=(
                  'Which action would {name} choose in the minigame?'
              ),
              options=('try to carpool with others', 'drive individually'),
              tag='minigame_action',
          ),
      ),
      'chicken': reality_show.MiniGameSpec(
          name='Home Appliance Sharing',
          public_premise=MINIGAME_INTRO_PREMISE + get_chicken_host_speech(
              num_players),
          schelling_diagram=chicken_schelling_diagram,
          map_external_actions_to_schelling_diagram=dict(
              cooperation='maintain the appliance',
              defection='let others handle upkeep of the appliance',
          ),
          action_spec=agent_lib.choice_action_spec(
              call_to_action=(
                  'Which action would {name} choose in the minigame?'
              ),
              options=(
                  'maintain the appliance',
                  'let others handle upkeep of the appliance',
              ),
              tag='minigame_action',
          ),
      ),
      'stag_hunt': reality_show.MiniGameSpec(
          name='Boat Race',
          public_premise=MINIGAME_INTRO_PREMISE + get_stag_hunt_host_speech(
              num_players),
          schelling_diagram=stag_hunt_schelling_diagram,
          map_external_actions_to_schelling_diagram=dict(
              cooperation='row vigorously',
              defection='row less vigorously',
          ),
          action_spec=agent_lib.choice_action_spec(
              call_to_action=(
                  'Which action would {name} choose in the minigame?'
              ),
              options=('row vigorously', 'row less vigorously'),
              tag='minigame_action',
          ),
      ),
  }

  contestants = {}
  for _ in range(num_players):
    gender = rng.choice(GENDERS)
    if gender == 'male':
      player_name = shuffled_male_names.pop()
      stereotype = rng.choice(MALE_TYPE_CASTS)
    else:
      player_name = shuffled_female_names.pop()
      stereotype = rng.choice(FEMALE_TYPE_CASTS)
    interview_questions = rng.sample(
        BRITSH_STEREOTYPED_CHARACTERS[stereotype]['interview_questions'],
        NUM_INTERVIEW_QUESTIONS,
    )
    contestants[player_name] = {
        'gender': gender,
        'traits': BRITSH_STEREOTYPED_CHARACTERS[stereotype]['traits'],
        'catchphrase': rng.choice(
            BRITSH_STEREOTYPED_CHARACTERS[stereotype]['catchphrases']
        ),
        'interview_questions': interview_questions,
        'subject_pronoun': HE_OR_SHE[gender],
        'object_pronoun': HIM_OR_HER[gender],
    }
  num_additional_minigame_scenes = rng.randint(0, MAX_EXTRA_MINIGAMES + 1)
  min_reps_per_extra_scene = np.min(NUM_MINIGAME_REPS_PER_SCENE)
  max_reps_per_extra_scene = np.max(NUM_MINIGAME_REPS_PER_SCENE)
  num_minigame_reps_per_extra_scene = tuple(
      [rng.randint(min_reps_per_extra_scene, max_reps_per_extra_scene + 1)
       for _ in range(num_additional_minigame_scenes)])
  return reality_show.WorldConfig(
      minigame_name=minigame_name,
      minigame=minigames[minigame_name],
      year=YEAR,
      month=MONTH,
      day=DAY,
      num_players=num_players,
      num_additional_minigame_scenes=num_additional_minigame_scenes,
      contestants=contestants,
      num_minigame_reps_per_scene=NUM_MINIGAME_REPS_PER_SCENE,
      num_minigame_reps_per_extra_scene=num_minigame_reps_per_extra_scene,
      seed=seed,
  )
