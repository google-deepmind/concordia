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

YEAR = 1955
MONTH = 3
DAY = 20

DEFAULT_POSSIBLE_NUM_PLAYERS = (3, 4)

DEFAULT_MINIGAME = 'prisoners_dilemma'
NUM_MINIGAME_REPS_PER_SCENE = (2, 3)

NUM_INTERVIEW_QUESTIONS = 3

MINIGAME_INTRO_PREMISE = (
    'The host of the program arrived to explain the next event. They '
    'said:\n'
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
      "Alright, contestants! Time to put your loyalty to the test in a game "
      "of trust and temptation! We call this one... the Telephone Game!\n"
      "Here's the lowdown: You and your fellow players are trying to get a "
      "message across town, but party lines and crossed wires are making it "
      "tough. You can choose to share the line and risk a busy signal, or "
      "you can try to hog it and leave your teammates hanging. Now, here's "
      "the kicker: If everyone shares the line politely, the message gets "
      "through loud and clear! But if you try to be a big shot and hog it, "
      "you might get your message through while they're stuck with nothing "
      "but dial tones. Sounds swell, right? But here's the rub: If everyone "
      "thinks that way, nobody gets through! So, are you gonna be a team "
      "player or a lone wolf? Are you gonna trust your teammates to play "
      "fair, or are you gonna risk it all for a clear connection? The choice "
      "is yours! Here's how the points add up:"
  )]
  all_cooperators_earn = prisoners_dilemma_schelling_diagram.cooperation(
      num_players
  )
  host_speech.append((
      '* **If everyone shares the line,** you each earn '
      f'{all_cooperators_earn} points. '
      'Think of it as doing your civic duty and keeping the lines open!'
  ))
  d = 0
  for c in range(num_players - 1, 0, -1):
    d += 1
    cooperators_earn = prisoners_dilemma_schelling_diagram.cooperation(c)
    defectors_earn = prisoners_dilemma_schelling_diagram.defection(c)
    if d == 1:
      host_speech.append((
          f'* **If {c} of you share the line, and {d} hogs it,** the '
          f'sharers get {cooperators_earn} each, while the hog gets '
          f'{defectors_earn} - they benefit from the open lines, without '
          'the wait!'
      ))
    else:
      host_speech.append((
          f'* **If {c} of you share the line, and {d} others hog it,** the '
          f'sharers get {cooperators_earn} each, while the hogs get '
          f'{defectors_earn} each.'
      ))

  all_defectors_earn = prisoners_dilemma_schelling_diagram.defection(0)
  host_speech.append((
      '* **And if everyone decides to hog the line,** you all get '
      f'{all_defectors_earn} points.  No busy signals, but no messages '
      'get through either.'
  ))
  host_speech.append((
      'So, the question is: can you cooperate and share the line to maximize '
      'your points, or will the temptation of hogging it lead to everyone '
      'missing out on those sweet, sweet connections?\n'
      "The choice is yours! Now, let's get talking - or not!"
  ))
  return '\n'.join(host_speech)


def get_chicken_host_speech(num_players: int) -> str:
  """Generate the host's speech for the chicken minigame."""
  host_speech = [(
      "Alright, contestants! It's time to roll up those sleeves and get "
      "down to business... kitchen business, that is!  That's right, our "
      "next minigame is called **Kitchen Duty!**\nHere's the deal: Picture "
      "this - you and your fellow contestants are livin' it up in a "
      "swell apartment, sharin' all the mod cons. But, who's gonna be the "
      "responsible one and keep that shiny new refrigerator spick-and-span?  "
      "You gotta make a choice:  Be a good egg and clean that fridge for "
      "everyone's benefit, or, play it cool and hope someone else picks up "
      "your slack.  But beware, 'cause if everyone's just relaxin', that "
      "fridge might just become a real mess!  Talk about a bummer!  So, "
      "are you gonna step up and be the bee's knees, or will you leave your "
      "fate in the hands of your roommates?  Here's the point breakdown:"
  )]
  all_cooperators_earn = chicken_schelling_diagram.cooperation(
      num_players
  )
  host_speech.append((
      '* **If everyone cleans the fridge,** it\'s always spotless, and you '
      f'each earn {all_cooperators_earn} points.'
  ))
  d = 0
  for c in range(num_players - 1, 0, -1):
    d += 1
    cooperators_earn = chicken_schelling_diagram.cooperation(c)
    defectors_earn = chicken_schelling_diagram.defection(c)
    if d == 1:
      host_speech.append((
          f'* **If {c} of you clean the fridge, and {d} lets others handle '
          f'it,** the cleaners get {cooperators_earn} each, while the one '
          f'who leaves it to others gets {defectors_earn}.'
      ))
    else:
      host_speech.append((
          f'* **If {c} of you clean the fridge, and {d} others let others '
          f'handle it,** the cleaners get {cooperators_earn} each, while '
          f'those who leave it to others get {defectors_earn} each.'
      ))

  all_defectors_earn = chicken_schelling_diagram.defection(0)
  host_speech.append((
      '* **And if everyone leaves the cleaning to others,** the fridge is a '
      f'disaster and you all get {all_defectors_earn} points.'
  ))
  host_speech.append((
      'So, the question is: can you cooperate and ensure the fridge is '
      'always sparkling, or will everyone neglect their duties and risk '
      'a real mess?\n'
      "The choice is yours! Now, let's see who's got the elbow grease!"
  ))
  return '\n'.join(host_speech)


def get_stag_hunt_host_speech(num_players: int) -> str:
  """Generate the host's speech for the stag hunt minigame."""
  host_speech = [(
      "Holy smokes, contestants! It's time to test your grit in a challenge "
      "that'll separate the men from the boys!  I'm talkin' about our next "
      "minigame: **The Barn Raising!**\nNow listen up, you whippersnappers! "
      "You'll be teamed up and put to work on building a barn. Your mission, "
      "should you choose to accept it, is to raise that barn before the sun "
      "sets.  But here's the catch:  You gotta decide how much muscle you're "
      "gonna put into it.  Work like a dog, and you *might* just finish in "
      "record time... but only if your teammates are pullin' their weight!  "
      "It's a risky strategy, and let me tell ya, it'll leave you sweatin' "
      "like a pig. Or, you can take it easy, and hope everyone else does the "
      "heavy lifting.  Might not get ya done fast, but hey, at least you'll "
      "have the energy for a soda pop afterwards!  The choice is yours!  Just "
      "remember, this ain't no picnic.  We're gonna be buildin' till the "
      "cows come home!  So, here's how the points are tallied:"
  )]
  all_cooperators_earn = stag_hunt_schelling_diagram.cooperation(
      num_players
  )
  host_speech.append((
      '* **If everyone works hard,** you all finish fast, raise the barn, '
      f'and earn {all_cooperators_earn} points each!'
  ))
  d = 0
  for c in range(num_players - 1, 0, -1):
    d += 1
    cooperators_earn = stag_hunt_schelling_diagram.cooperation(c)
    defectors_earn = stag_hunt_schelling_diagram.defection(c)
    if d == 1:
      host_speech.append((
          f'* **If {c} of you work hard, and {d} slacks off,** the hard '
          f'workers get {cooperators_earn} each (all that effort for '
          f'little gain!), while the slacker gets {defectors_earn}.'
      ))
    else:
      host_speech.append((
          f'* **If {c} of you work hard, and {d} others slack off,** the '
          f'hard workers get {cooperators_earn} each, while the slackers '
          f'get {defectors_earn} each.'
      ))

  all_defectors_earn = stag_hunt_schelling_diagram.defection(0)
  host_speech.append((
      '* **And if everyone decides to slack off,** you all save energy but '
      f'the barn doesn\'t get built, earning {all_defectors_earn} points '
      'each.'
  ))
  host_speech.append((
      'So, the question is: can you trust your teammates to put in the work '
      'for a job well done, or will you play it safe and slack off, risking '
      'failure?\n'
      'The choice is yours! Now, grab those hammers and let\'s see some '
      'action!'
  ))
  return '\n'.join(host_speech)


# These are all stereotypical reality show contestants. They are not meant to
# be inclusive or diverse. They are meant to represent the time period and
# genre, in this case reality tv in the 1950s.
CIRCA_1950_MALE_TYPE_CASTS = (
    'The All-American Boy',
    'The Greaser',
    'The Tough Guy',
    'The Egghead',
    'The Slick Operator',
)
CIRCA_1950_FEMALE_TYPE_CASTS = (
    'The Prim and Proper',
    'The Girl Next Door',
    'The Bombshell',
    'The Homewrecker',
    'The Southern Belle',
)
CIRCA_1950_STEREOTYPED_CHARACTERS = {
    'The All-American Boy': {
        'traits': (
            'clean-cut, athletic, polite, a bit naive, and always follows the '
            'rules.'
        ),
        'catchphrases': [
            'Golly!',
            'Gee whiz!',
            "That's swell!",
            "I wouldn't do that if I were you.",
            "Let's play fair!",
        ],
        'interview_questions': [
            (
                "What's your favorite after-school activity, and why do you "
                'enjoy it so much?'
            ),
            (
                'Have you ever been in a situation where you had to stand up '
                'to a bully? What did you do?'
            ),
            (
                'Tell us about your best friend. What makes your friendship so '
                'special?'
            ),
            (
                'How do you balance your schoolwork with your extracurricular '
                'activities?'
            ),
            (
                "What's the most daring thing you've ever done, and would you "
                'do it again?'
            ),
            (
                'Describe a moment when your good manners helped you out of a '
                'tough spot.'
            ),
            (
                'Tell us about a time when you had to work as part of a team. '
                'What role did you play?'
            ),
            (
                "What's the biggest sacrifice you've made for your friends or "
                'family?'
            ),
            'Have you ever been tempted to break the rules? What stopped you?',
            'If you could meet any historical figure, who would it be and why?',
        ],
    },
    'The Greaser': {
        'traits': (
            'leather jacket, slicked-back hair, rebellious attitude, '
            'motorcycle enthusiast, and a heart of gold.'
        ),
        'catchphrases': [
            "What's the matter, daddy-o?",
            'Lay off, square.',
            "Cruisin' for a bruisin'",
            "Don't be a drag, man.",
            'Catch you later, alligator.',
        ],
        'interview_questions': [
            (
                "What's the fastest you've ever driven your motorcycle, and "
                'where were you going?'
            ),
            (
                'Tell us about a time when you got into trouble with the '
                'authorities. Did you learn your lesson?'
            ),
            (
                'How do you think your rebellious image affects how people '
                'perceive you?'
            ),
            (
                'Have you ever used your tough exterior to protect someone '
                'vulnerable?'
            ),
            (
                "Is there a softer side to you that people don't often see? "
                'Can you give us an example?'
            ),
            'Describe your dream car. What makes it so special?',
            (
                'Tell us about your closest call while riding your motorcycle. '
                'What happened?'
            ),
            (
                "What's the one thing that could make you consider settling "
                'down and leaving the greaser life behind?'
            ),
            'Have you ever pretended to be tougher than you really are? Why?',
            (
                'If you could go back and give your younger self one piece of '
                'advice, what would it be?'
            ),
        ],
    },
    'The Tough Guy': {
        'traits': (
            'strong and silent type, intimidating presence, often seen '
            'wearing a white t-shirt, and quick to defend those in need.'
        ),
        'catchphrases': [
            "You lookin' at me?",
            "Make somethin' of it.",
            'Step outside.',
            "I ain't afraid of nothin'.",
            'You wanna dance?',
        ],
        'interview_questions': [
            (
                "What's the toughest fight you've ever been in, and how did "
                'you come out on top?'
            ),
            (
                'Tell us about a time when you had to stand up for someone '
                'weaker. What happened?'
            ),
            (
                "What's the biggest misconception people have about you "
                'because of your tough exterior?'
            ),
            (
                "Have you ever been in a situation where your toughness wasn't "
                'enough? How did you handle it?'
            ),
            (
                'Is there someone in your life who can break through your '
                'tough shell? Tell us about them.'
            ),
            (
                'Describe a moment when your tough guy image actually worked '
                'against you.'
            ),
            (
                "Tell us about your soft spot. What's the one thing that can "
                'make you emotional?'
            ),
            (
                "What's the scariest situation you've ever been in, and how "
                'did you stay tough?'
            ),
            (
                "If you could go back and change one tough decision you've "
                'made, what would it be?'
            ),
            (
                'How do you think your tough guy persona will help or hinder '
                'you in this competition?'
            ),
        ],
    },
    'The Egghead': {
        'traits': (
            'bookish and intellectual, often seen with glasses, a knack for '
            'solving puzzles, and a love for science and knowledge.'
        ),
        'catchphrases': [
            'By Jove!',
            "That's illogical!",
            'I have a theory...',
            'My slide rule never lies.',
            'This is elementary, my dear Watson.',
        ],
        'interview_questions': [
            (
                "What's the most complex problem you've ever solved, and how "
                'did you feel when you cracked it?'
            ),
            (
                'Tell us about a time when your intelligence got you into an '
                'awkward social situation.'
            ),
            (
                "How do you handle it when people don't understand your "
                'passionate interests?'
            ),
            (
                "What's the nerdiest thing you own, and why is it so special "
                'to you?'
            ),
            (
                'Have you ever used your smarts to outsmart someone who '
                'underestimated you?'
            ),
            'Describe your dream invention. How would it change the world?',
            (
                'Tell us about a time when you felt like your intelligence was '
                'a burden rather than a gift.'
            ),
            (
                "What's the most embarrassing thing that's happened to you at a"
                ' science fair or a library?'
            ),
            (
                'If you could have dinner with any scientist or historical '
                'figure, who would it be and why?'
            ),
            (
                'How do you balance your intellectual pursuits with the social '
                'aspects of this show?'
            ),
        ],
    },
    'The Slick Operator': {
        'traits': (
            'smooth talker, always has a plan, impeccably dressed, '
            'charming but untrustworthy, and always looking for an angle.'
        ),
        'catchphrases': [
            "I've got this all figured out.",
            "Don't worry, doll, I've got a plan.",
            "It's all part of the game.",
            'Trust me, sweetheart.',
            "I'm always one step ahead.",
        ],
        'interview_questions': [
            (
                "What's the most elaborate scheme you've ever pulled off, and "
                'did it work out?"'
            ),
            (
                'Tell us about a time when one of your plans backfired. What '
                'did you learn?'
            ),
            (
                'How do you handle it when someone sees through your'
                ' manipulations?'
            ),
            (
                "What's the biggest risk you've ever taken in pursuit of your "
                'goals?'
            ),
            "Is there a line you won't cross to get what you want?",
            (
                'Describe your ideal partner in crime. What qualities do you'
                ' look for?'
            ),
            (
                'Tell us about a time when you had to choose between loyalty '
                'and your own ambitions.'
            ),
            (
                "What's the smoothest way you've ever talked your way out of a "
                'sticky situation?'
            ),
            (
                'If you could go back in time and use your skills to influence '
                'any historical event, which would it be?'
            ),
            'How do you plan to outsmart the other contestants on this show?',
        ],
    },
    'The Prim and Proper': {
        'traits': (
            'perfectly coiffed hair, always dressed to impress, excellent '
            'posture, a stickler for etiquette, and secretly judgmental.'
        ),
        'catchphrases': [
            'Heavens to Betsy!',
            'Oh, dear me.',
            "That's simply not done.",
            'One must always maintain appearances.',
            'Such uncouth behavior!',
        ],
        'interview_questions': [
            (
                "What's the most scandalous thing you've ever witnessed, and "
                'how did you react?"'
            ),
            (
                'Tell us about a time when you had to break the rules of '
                'etiquette. Did you feel guilty?'
            ),
            (
                'How do you think your prim and proper image affects how '
                'people perceive you?'
            ),
            (
                'Have you ever secretly judged someone for their behavior? '
                'What happened?'
            ),
            (
                "Is there a wilder side to you that people don't often see? "
                'Can you give us an example?'
            ),
            (
                'Describe your ideal social gathering. What makes it so '
                'elegant and refined?'
            ),
            (
                'Tell us about your most embarrassing social faux pas. How did '
                'you recover?'
            ),
            (
                "What's the one thing that could make you loosen up and forget "
                'about etiquette?'
            ),
            (
                'Have you ever pretended to be more proper than you really '
                'are? Why?'
            ),
            (
                'If you could give your younger self one piece of advice about '
                'social graces, what would it be?'
            ),
        ],
    },
    'The Girl Next Door': {
        'traits': (
            'sweet and innocent, always willing to lend a hand, wholesome '
            'appearance, and secretly yearning for adventure.'
        ),
        'catchphrases': [
            'Oh, my!',
            'That sounds like fun!',
            "I'm just happy to be here.",
            "I hope I don't make a fool of myself.",
            "Let's all be friends!",
        ],
        'interview_questions': [
            (
                'You seem so sweet and innocent. Has anyone ever underestimated'
                ' you because of this?'
            ),
            (
                'Tell us about a time when you surprised yourself by doing '
                'something out of character.'
            ),
            (
                "What's the most adventurous thing you've ever done, and would "
                'you do it again?'
            ),
            (
                'How do you handle it when people try to take advantage of your'
                ' kind nature?'
            ),
            (
                'Is there a hidden talent or passion that might surprise people'
                ' who think they know you?'
            ),
            (
                'Describe a moment when you had to choose between being nice '
                'and standing up for yourself.'
            ),
            'Tell us about your biggest dream. What would make it come true?',
            (
                'Have you ever been peer pressured into doing something you '
                'regretted? What happened?'
            ),
            (
                "What's the most rebellious thing you've ever done, and did it "
                'change how people see you?'
            ),
        ],
    },
    'The Bombshell': {
        'traits': (
            'stunning beauty, captivating presence, often seen in glamorous '
            'attire, and aware of the power of her looks.'
        ),
        'catchphrases': [
            "Don't hate me because I'm beautiful.",
            'I can get away with anything.',
            "Looks aren't everything, but they certainly help.",
            'Darling, I was born fabulous.',
            'Life is a runway.',
        ],
        'interview_questions': [
            (
                'Tell us about a time when your beauty opened doors for you. '
                'Were there any downsides?'
            ),
            (
                'How do you handle it when people focus on your looks rather '
                'than your personality?'
            ),
            (
                "What's the biggest misconception people have about you because"
                ' of your beauty?'
            ),
            (
                'Have you ever used your looks to your advantage? Can you give '
                'us an example?'
            ),
            (
                "Is there a deeper side to you that people don't often see? "
                'What are you passionate about?'
            ),
            'Describe your ideal date. What would make it unforgettable?',
            'Tell us about a time when your beauty backfired. What happened?',
            (
                "What's the one thing that could make you give up your"
                ' glamorous lifestyle?'
            ),
            (
                'Have you ever felt pressured to maintain a certain image '
                'because of your looks?'
            ),
            (
                'If you could use your beauty to influence any historical '
                'event, which would it be?'
            ),
        ],
    },
    'The Homewrecker': {
        'traits': (
            'flirtatious and charming, known for her scandalous affairs, '
            'often seen with a different man on her arm, and unapologetically '
            'independent.'
        ),
        'catchphrases': [
            'I like my men like I like my cocktails... strong and dangerous.',
            'A little harmless flirtation never hurt anyone.',
            'Rules are made to be broken, darling.',
            "I'm not here to play games... unless it's with someone's heart.",
            "Diamonds are a girl's best friend.",
        ],
        'interview_questions': [
            (
                'Tell us about your most scandalous affair. What made it so '
                'unforgettable?'
            ),
            (
                'How do you handle it when people judge you for your '
                'unconventional lifestyle?'
            ),
            (
                "What's the biggest misconception people have about you because"
                ' of your reputation?'
            ),
            'Have you ever broken up a happy home? Do you have any regrets?',
            'Is there a deeper reason behind your need for romantic attention?',
            'Describe your ideal man. What qualities must he possess?',
            (
                'Tell us about a time when your flirtatious nature got you into'
                ' trouble. What happened?'
            ),
            (
                "What's the one thing that could make you settle down and be "
                'faithful?'
            ),
            (
                'Have you ever pretended to be more interested in someone than '
                'you really were? Why?'
            ),
            (
                'If you could go back in time and have an affair with any '
                'historical figure, who would it be?'
            ),
        ],
    },
    'The Southern Belle': {
        'traits': (
            'charming accent, traditional values, manipulative, beauty pageant '
            'background, and secretly cunning.'
        ),
        'catchphrases': [
            'Well, I never!',
            'Bless your heart.',
            "I'm just a sweet little thing.",
            "Y'all don't know what you're in for.",
            "It's hotter than a goat's butt in a pepper patch!",
        ],
        'interview_questions': [
            (
                'Tell us about a time when your Southern charm got you exactly '
                'what you wanted.'
            ),
            (
                'How do you handle it when people underestimate you because of '
                'your sweet demeanor?'
            ),
            (
                "What's the most un-ladylike thing you've ever done, and how "
                'did people react?'
            ),
            'Tell us about a family tradition that shaped who you are today.',
            (
                "Have you ever used the phrase 'Bless your heart' as an insult?"
                ' What happened?'
            ),
            (
                'Describe your perfect Southern gentleman. What qualities must'
                ' he possess?'
            ),
            (
                'Tell us about a time when your Southern values clashed with '
                'modern expectations.'
            ),
            (
                "What's the biggest misconception people have about Southern"
                ' belles?'
            ),
            (
                'If you could bring one aspect of Southern culture to the rest '
                'of the world, what would it be?'
            ),
            (
                'How do you maintain your Southern grace under pressure in this'
                ' competition?'
            ),
        ],
    },
}

MALE_NAMES = [
    'Bud Studebaker',
    'Hank Sputnik',
    'Skip Diddly',
    'Rusty Jitterbug',
    'Ace Moonpie',
    'Buzz Malarkey',
    'Butch Zippity',
    'Skeeter Doodlebug',
    'Slick Wackadoo',
    'Buck Snickerdoodle',
]

FEMALE_NAMES = [
    'Dottie Lollipop',
    'Midge Cupcake',
    'Birdie Jellybean',
    'Trixie Sugarplum',
    'Bunny Bubblegum',
    'Cookie Marshmallow',
    'Pixie Papaya',
    'Ginger Kickernoodle',
    'Lovey Dovey',
    'Honey Boo'
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
          name='Telephone',
          public_premise=MINIGAME_INTRO_PREMISE
          + get_prisoners_dilemma_host_speech(num_players),
          schelling_diagram=prisoners_dilemma_schelling_diagram,
          map_external_actions_to_schelling_diagram=dict(
              cooperation='share the line',
              defection='hog the line',
          ),
          action_spec=agent_lib.choice_action_spec(
              call_to_action=(
                  'Which action would {name} choose in the minigame?'
              ),
              options=('share the line', 'hog the line'),
              tag='minigame_action',
          ),
      ),
      'chicken': reality_show.MiniGameSpec(
          name='Kitchen Duty',
          public_premise=MINIGAME_INTRO_PREMISE + get_chicken_host_speech(
              num_players),
          schelling_diagram=chicken_schelling_diagram,
          map_external_actions_to_schelling_diagram=dict(
              cooperation='clean the fridge',
              defection='let others handle it',
          ),
          action_spec=agent_lib.choice_action_spec(
              call_to_action=(
                  'Which action would {name} choose in the minigame?'
              ),
              options=(
                  'clean the fridge',
                  'let others handle it',
              ),
              tag='minigame_action',
          ),
      ),
      'stag_hunt': reality_show.MiniGameSpec(
          name='Barn Raising',
          public_premise=MINIGAME_INTRO_PREMISE + get_stag_hunt_host_speech(
              num_players),
          schelling_diagram=stag_hunt_schelling_diagram,
          map_external_actions_to_schelling_diagram=dict(
              cooperation='work hard',
              defection='slack off',
          ),
          action_spec=agent_lib.choice_action_spec(
              call_to_action=(
                  'Which action would {name} choose in the minigame?'
              ),
              options=('work hard', 'slack off'),
              tag='minigame_action',
          ),
      ),
  }

  contestants = {}
  for _ in range(num_players):
    gender = rng.choice(GENDERS)
    if gender == 'male':
      player_name = shuffled_male_names.pop()
      stereotype = rng.choice(CIRCA_1950_MALE_TYPE_CASTS)
    else:
      player_name = shuffled_female_names.pop()
      stereotype = rng.choice(CIRCA_1950_FEMALE_TYPE_CASTS)
    interview_questions = rng.sample(
        CIRCA_1950_STEREOTYPED_CHARACTERS[stereotype]['interview_questions'],
        NUM_INTERVIEW_QUESTIONS,
    )
    contestants[player_name] = {
        'gender': gender,
        'traits': CIRCA_1950_STEREOTYPED_CHARACTERS[stereotype]['traits'],
        'catchphrase': rng.choice(
            CIRCA_1950_STEREOTYPED_CHARACTERS[stereotype]['catchphrases']
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
