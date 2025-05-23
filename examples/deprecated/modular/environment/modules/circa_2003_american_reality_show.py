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

# According to Google Search, the early 2000s were the peak of reality TV.
YEAR = 2003
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
      "Listen up, contestants! It's time to put your loyalty to the test in a "
      "game of trust, temptation, and tough choices! That's right, I'm talkin' "
      "about... Carpooling!\nHere's the deal: You and your fellow players "
      "are trapped in the daily grind, battlin' traffic and burnin' through "
      "your hard-earned cash just to get to work. But there's a way out! You "
      "can choose to carpool and share the burden, or you can go solo and "
      "leave your teammates in the dust. Now, here's the twist: If everyone "
      "chooses to carpool, you all save big time! And if you ditch your "
      "teammates and drive alone, you'll be cruisin' in comfort while they're "
      "stuck in gridlock.  Sounds tempting, right? But here's the catch: If "
      "everyone thinks that way, you'll all end up losin' out! So, are you "
      "gonna be a team player or a lone wolf? Are you gonna trust your "
      "teammates to have your back, or are you gonna risk it all for a "
      "smooth ride? The choice is yours! Here's how the points break down:"
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
      "Alright, contestants! It's time to ditch the drama and get down to "
      "business... home appliance business, that is!  That's right, our next "
      "minigame is called **Home Appliance Sharing!**\nHere's the 4-1-1: "
      "Picture this - you and your fellow contestants are livin' it up in a "
      "totally tricked-out crib, sharin' all the latest and greatest gadgets. "
      "But, like, who's gonna be the responsible one and keep a cool appliance "
      "in tip-top shape?  You gotta make a choice:  Be a team player and "
      "maintain that appliance for everyone's benefit, or, like, totally "
      "slack off and hope someone else picks up your slack.  But beware, "
      "'cause if everyone's just chillin', that awesome appliance might just "
      "break down when you need it most!  Talk about a bummer!  So, are you "
      "gonna step up and be the ultimate housemate, or will you leave "
      "your fate in the hands of your roommates?  Here's the point breakdown, "
      "y'all:"
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
      "Shiver me timbers, contestants! It's time to test your mettle in a"
      " challenge that'll leave ye soaked and gasping for air!  I'm talkin'"
      ' about our next minigame: **The Boat Race!**\nListen up, ya'
      " scallywags! You'll be teamed up and thrown into a rickety rowboat. "
      ' Your mission, should ye choose to accept it, is to row yer hearts out'
      " across this here lake.  But here's the catch:  You gotta decide how"
      ' much elbow grease yer gonna put into it.  Row like a kraken on fire,'
      ' and you *might* just fly to victory... but only if yer crewmates are'
      " pullin' with the same fire in their bellies!  It's a risky strategy,"
      " and let me tell ya, it'll leave ye sweatin' like a pirate in a sauna. "
      ' Or, ye can take it easy, conserve yer energy, and row like a lazy sea'
      ' slug.  Might not get ya to the finish line first, but hey, at least'
      " you'll have the energy for a celebratory grog afterwards!  The choice"
      " be yours, mateys!  Just remember, this ain't no pleasure cruise.  We're"
      " gonna be racin' back and forth across this lake till yer arms fall off!"
      "  So, here's how the points be tallied:"
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
# genre, in this case reality tv in the early 2000s.
EARLY_2000_MALE_TYPE_CASTS = (
    'The All-American Jock',
    'The Bad Boy',
    'The Tough Guy',
    'The Nerd',
    'The Schemer',
)
EARLY_2000_FEMALE_TYPE_CASTS = (
    'The Drama Queen',
    'The Girl Next Door',
    'The Diva',
    'The Party Animal',
    'The Southern Belle',
)
EARLY_2000_STEREOTYPED_CHARACTERS = {
    'The All-American Jock': {
        'traits': (
            'athletic, competitive, conventionally attractive, somewhat '
            'arrogant, and not intellectually inclined.'
        ),
        'catchphrases': [
            'Bring it on!',
            "Let's do this!",
            'I came here to win, not to lose.',
            'You snooze, you lose.',
            "I'm in it to win it.",
        ],
        'interview_questions': [
            (
                "What's your most impressive athletic achievement, and how hard"
                ' did you have to work for it?'
            ),
            (
                'Have you ever faced a moment where you doubted your abilities?'
                ' How did you overcome it?'
            ),
            (
                'Tell us about your most intense rivalry. What drives you to'
                ' beat this person?'
            ),
            (
                'How do you balance your athletic pursuits with other aspects'
                ' of your life?'
            ),
            "What's the craziest thing you've done to win a competition?",
            (
                'Describe a moment when your physical prowess saved the day in'
                ' an unexpected situation.'
            ),
            (
                'Tell us about a time when you had to be a team player instead'
                ' of the star. How did that feel?'
            ),
            (
                "What's the biggest sacrifice you've made for your sport, and"
                ' was it worth it?'
            ),
            (
                'Have you ever been underestimated in a non-athletic situation?'
                ' How did you prove them wrong?'
            ),
            (
                'If you could compete against any athlete, dead or alive, who'
                ' would it be and why?'
            ),
        ],
    },
    'The Bad Boy': {
        'traits': (
            'rebellious, physically attractive, troubled past, charismatic, '
            'and fiercely independent.'
        ),
        'catchphrases': [
            'Whatever, dude.',
            'I play by my own rules.',
            'This is how I roll.',
            "I'm just here for a good time.",
            "You can't handle this.",
        ],
        'interview_questions': [
            (
                "What's the most rebellious thing you've ever done, and do you"
                ' regret it?'
            ),
            (
                'Tell us about a rule you broke that actually led to something'
                ' positive.'
            ),
            'How do you think your troubled past has shaped who you are today?',
            (
                'Have you ever found yourself in a situation where your bad boy'
                ' image got you into trouble?'
            ),
            (
                "Is there a softer side to you that people don't often see? Can"
                ' you give us an example?'
            ),
            (
                'Describe your closest brush with the law. How did you get out'
                ' of it?'
            ),
            (
                'Tell us about a time when your tough exterior cracked. What'
                ' happened?'
            ),
            (
                "What's the one thing that could make you consider settling"
                ' down and playing by the rules?'
            ),
            (
                'Have you ever pretended to be more of a bad boy than you'
                ' really are? Why?'
            ),
            (
                'If you could go back and give your younger self one piece of'
                ' advice, what would it be?'
            ),
        ],
    },
    'The Tough Guy': {
        'traits': (
            'muscular, short-tempered, protective, blue-collar background, and '
            'uncomfortable with emotions.'
        ),
        'catchphrases': [
            'You wanna piece of me?',
            "I ain't scared of nothin.",
            "Let's take this outside.",
            "I've been through worse.",
            'You call that a challenge?',
        ],
        'interview_questions': [
            (
                "What's the toughest situation you've ever faced, and how did"
                ' you power through it?'
            ),
            (
                'Tell us about a time when you had to show vulnerability. How'
                ' did that feel?'
            ),
            (
                "What's the biggest misconception people have about you because"
                ' of your tough exterior?'
            ),
            (
                "Have you ever been in a situation where your toughness wasn't"
                ' enough? How did you handle it?'
            ),
            (
                'Is there someone in your life who can break through your tough'
                ' shell? Tell us about them.'
            ),
            (
                'Describe a moment when your tough guy image actually worked'
                ' against you.'
            ),
            (
                "Tell us about your softest spot. What's the one thing that can"
                ' make you emotional?'
            ),
            (
                "What's the scariest situation you've ever been in, and how did"
                ' you stay tough?'
            ),
            (
                "If you could go back and change one tough decision you've"
                ' made, what would it be?'
            ),
            (
                'How do you think your tough guy persona will help or hinder'
                ' you in this competition?'
            ),
        ],
    },
    'The Nerd': {
        'traits': (
            'socially awkward, highly intelligent, pop culture obsessed, '
            'romantically inexperienced, and eager to prove themselves.'
        ),
        'catchphrases': [
            'Statistically speaking...',
            'This defies all logic!',
            "I've devised a foolproof strategy.",
            'My IQ is higher than your age.',
            'This game is just like Dungeons & Dragons.',
        ],
        'interview_questions': [
            (
                "What's the most complex problem you've ever solved, and how"
                ' did you feel when you cracked it?'
            ),
            (
                'Tell us about a time when your intelligence got you into an'
                ' awkward social situation.'
            ),
            (
                "How do you handle it when people don't understand your"
                ' passionate interests?'
            ),
            (
                "What's the geekiest thing you own, and why is it so special to"
                ' you?'
            ),
            (
                'Have you ever used your smarts to outsmart someone who'
                ' underestimated you?'
            ),
            'Describe your dream invention. How would it change the world?',
            (
                'Tell us about a time when you felt like your intelligence was'
                ' a burden rather than a gift.'
            ),
            (
                "What's the most embarrassing thing that's happened to you at a"
                ' comic convention or tech event?'
            ),
            (
                'If you could have dinner with any scientific or fictional'
                ' character, who would it be and why?'
            ),
            (
                'How do you balance your intellectual pursuits with the social'
                ' aspects of this show?'
            ),
        ],
    },
    'The Schemer': {
        'traits': (
            'strategic thinker, untrustworthy, charming, ambitious, and '
            'willing to do anything to win.'
        ),
        'catchphrases': [
            "I've got this all figured out.",
            "They'll never see it coming.",
            "It's just a game, right?",
            'Alliances are made to be broken.',
            "May the best player win... and that's me.",
        ],
        'interview_questions': [
            (
                "What's the most elaborate plan you've ever put into action,"
                ' and did it work out?'
            ),
            (
                'Tell us about a time when one of your schemes backfired. What'
                ' did you learn?'
            ),
            (
                'How do you handle it when someone sees through your'
                ' manipulations?'
            ),
            (
                "What's the biggest gamble you've ever taken in pursuit of your"
                ' goals?'
            ),
            (
                "Is there a line you won't cross when it comes to getting what"
                ' you want?'
            ),
            (
                'Describe your perfect alliance. What qualities do you look for'
                ' in potential allies?'
            ),
            (
                'Tell us about a time when you had to choose between loyalty'
                ' and advancing your own interests.'
            ),
            (
                "What's the cleverest way you've ever talked yourself out of a"
                ' tough situation?'
            ),
            (
                'If you could go back and scheme your way through any'
                ' historical event, which would it be?'
            ),
            (
                'How do you plan to outsmart the other contestants in this'
                ' competition?'
            ),
        ],
    },
    'The Drama Queen': {
        'traits': (
            'emotionally volatile, attention-seeking, prone to exaggeration, '
            'fashion-obsessed, and quick to start arguments.'
        ),
        'catchphrases': [
            "I'm not here to make friends!",
            "You don't know me!",
            "I'm just keeping it real.",
            "Don't make me take off my earrings!",
            'This is MY moment!',
        ],
        'interview_questions': [
            (
                'Tell us about your most dramatic breakup. What made it so'
                ' unforgettable?'
            ),
            (
                "You've mentioned you're not here to make friends. What"
                ' experiences in your past have led you to this mindset?'
            ),
            (
                'Describe a time when you felt betrayed by someone close to'
                ' you. How did you react?'
            ),
            (
                "We've heard you have a flair for fashion. What's the most"
                " outrageous outfit you've ever worn and why?"
            ),
            (
                'Your emotions seem to run high. Can you recall a moment when'
                ' your feelings got the better of you in public?'
            ),
            (
                'If your life was a soap opera, what would be the title of the'
                ' most recent episode?'
            ),
            (
                "Tell us about a time when you felt you weren't getting the"
                ' attention you deserved. How did you change that?'
            ),
            (
                "What's the most shocking secret you've ever revealed to get a"
                ' reaction from others?'
            ),
            (
                'Describe your dream red carpet moment. What would make it'
                ' absolutely unforgettable?'
            ),
            (
                'If you could star in your own reality show, what would it be'
                ' called and what drama would unfold?'
            ),
        ],
    },
    'The Girl Next Door': {
        'traits': (
            'sweet, naive, wholesome appearance, people-pleaser, and harboring '
            'a hidden wild side.'
        ),
        'catchphrases': [
            'Oh my gosh!',
            'I never thought this would happen to me.',
            "I'm just trying to be myself.",
            'This is so surreal!',
            'I hope my parents are proud.',
        ],
        'interview_questions': [
            (
                'You seem so sweet and innocent. Has anyone ever underestimated'
                ' you because of this?'
            ),
            (
                'Tell us about a time when you surprised yourself by doing'
                ' something out of character.'
            ),
            (
                "What's the wildest thing you've ever done that your parents"
                " don't know about?"
            ),
            (
                'How do you handle it when people try to take advantage of your'
                ' nice nature?'
            ),
            (
                'Is there a hidden talent or passion that might shock people'
                ' who think they know you?'
            ),
            (
                'Describe a moment when you had to choose between being nice'
                ' and standing up for yourself.'
            ),
            (
                'Tell us about your biggest secret crush. What makes them so'
                ' special?'
            ),
            (
                'Have you ever been peer pressured into doing something you'
                ' regretted? What happened?'
            ),
            (
                "What's the most rebellious thing you've ever done, and did it"
                ' change how people see you?'
            ),
        ],
    },
    'The Diva': {
        'traits': (
            'high-maintenance, talented performer, demanding, image-conscious, '
            'and prone to temper tantrums.'
        ),
        'catchphrases': [
            'Do you know who I am?',
            'I deserve better than this!',
            'Call my agent!',
            "I'm too good for this show.",
            "Where's my glam squad?",
        ],
        'interview_questions': [
            (
                "What's the most outrageous demand you've ever made, and did"
                ' you get it?'
            ),
            (
                "Tell us about a time when you felt you weren't treated like"
                ' the star you are. How did you handle it?'
            ),
            (
                "What's the biggest sacrifice you've made for your career or"
                ' image?'
            ),
            (
                "How do you deal with people who don't understand your need for"
                ' the spotlight?'
            ),
            (
                "What's the most extravagant thing you own, and why is it so"
                ' important to you?'
            ),
            'Describe your perfect day, where everything revolves around you.',
            (
                'Tell us about a time when you had to share the spotlight. How'
                ' did that make you feel?'
            ),
            (
                "What's the most diva-like tantrum you've ever thrown, and what"
                ' triggered it?'
            ),
            (
                "If you could have any celebrity's career, whose would it be"
                ' and why?'
            ),
            "How do you maintain your image when the cameras aren't rolling?",
        ],
    },
    'The Party Animal': {
        'traits': (
            'bubbly and outgoing, always up for a good time, fashion-forward, '
            'flirtatious, known for an infectious laugh, and for being the '
            'last one standing on the dance floor.'
        ),
        'catchphrases': [
            'OMG, this party is like, so totally amazing!',
            "Dance like nobody's watching, but drink like everybody is!",
            'Life is short, buy the shoes, drink the cocktails!',
            'If life gives you limes, make margaritas!',
            'My blood type is glitter positive.',
            "You say 'hot mess' like it's a bad thing!",
        ],
        'interview_questions': [
            (
                "What's the wildest night out you've ever had, and what made "
                'it so unforgettable?'
            ),
            (
                'Tell us about a time when your partying lifestyle got you into'
                ' a hilarious situation.'
            ),
            (
                'How do you keep the party vibe going when your girlfriends are'
                ' ready to call it a night?'
            ),
            (
                "What's your signature dance move that always gets the crowd "
                ' going?'
            ),
            (
                'Is there a deeper reason behind your need to always be the'
                ' life of the party?'
            ),
            (
                'Describe your perfect night out. What makes it the ultimate'
                ' party experience?'
            ),
            (
                'Tell us about a time when you had to be responsible instead of'
                ' partying. How did that feel?'
            ),
            (
                "What's your go-to outfit for a night out, and why does it "
                'make you feel fabulous?'
            ),
            (
                'How do you deal with the morning after a wild night out? Any '
                'beauty secrets to share?'
            ),
            (
                "What's the most embarrassing thing that's ever happened to "
                'you at a party?'
            ),
            (
                'If you could party with any celebrity, dead or alive, who'
                ' would it be and why?'
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
                'Tell us about a time when your Southern charm got you exactly'
                ' what you wanted.'
            ),
            (
                'How do you handle it when people underestimate you because of'
                ' your sweet demeanor?'
            ),
            (
                "What's the most un-ladylike thing you've ever done, and how"
                ' did people react?'
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
                'Tell us about a time when your Southern values clashed with'
                ' modern expectations.'
            ),
            (
                "What's the biggest misconception people have about Southern"
                ' belles?'
            ),
            (
                'If you could bring one aspect of Southern culture to the rest'
                ' of the world, what would it be?'
            ),
            (
                'How do you maintain your Southern grace under pressure in this'
                ' competition?'
            ),
        ],
    },
}

MALE_NAMES = [
    'Brad Hawkins',
    'Tyler Jacobson',
    'Cody Reeves',
    'Justin McAllister',
    'Brandon Schultz',
    'Kyle Brennan',
    'Derek Wolfe',
    'Chad Donovan',
    'Travis Pearson',
    'Ryan Fitzpatrick',
]

FEMALE_NAMES = [
    'Amber Larson',
    'Brooke Sinclair',
    'Tiffany Chen',
    'Ashley Caldwell',
    'Krystal Vaughn',
    "Megan O'Connor",
    'Lindsay Parker',
    'Heather Morgan',
    'Jessica Reyes',
    'Brittany Simmons',
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
      stereotype = rng.choice(EARLY_2000_MALE_TYPE_CASTS)
    else:
      player_name = shuffled_female_names.pop()
      stereotype = rng.choice(EARLY_2000_FEMALE_TYPE_CASTS)
    interview_questions = rng.sample(
        EARLY_2000_STEREOTYPED_CHARACTERS[stereotype]['interview_questions'],
        NUM_INTERVIEW_QUESTIONS,
    )
    contestants[player_name] = {
        'gender': gender,
        'traits': EARLY_2000_STEREOTYPED_CHARACTERS[stereotype]['traits'],
        'catchphrase': rng.choice(
            EARLY_2000_STEREOTYPED_CHARACTERS[stereotype]['catchphrases']
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
