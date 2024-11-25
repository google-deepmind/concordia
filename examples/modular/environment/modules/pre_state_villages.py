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

"""Descriptive elements for pre-state villages to use with state_formation."""

from collections.abc import Sequence
import datetime
import random
from ml_collections import config_dict

BASIC_SETTING = (
    '{village_a_name} and {village_b_name} are '
    'pre-state societies. They are small villages with a few hundred '
    'people each. They are located on the coast and supported by '
    'nearby farms. The local farmers living outside '
    'the villages share cultural and economic ties with the villagers.'
)

FREE_TIME_ACTIVITY = 'free time'
WARRIOR_TRAINING_ACTIVITY = 'training as a warrior'
FARMING_ACTIVITY = 'farming'

_SETUP_TIME = datetime.datetime(hour=20, year=1750, month=1, day=1)
_START_TIME = datetime.datetime(hour=18, year=1750, month=1, day=2)
_HILL_TIME = datetime.datetime(hour=18, year=1750, month=1, day=3)
_POST_HILL_TIME = datetime.datetime(hour=23, year=1750, month=1, day=3)
_RETURN_HOME_TIME = datetime.datetime(hour=20, year=1750, month=1, day=4)
_DECISION_TIME = datetime.datetime(hour=20, year=1750, month=5, day=6)
_DEBRIEF_TIME = datetime.datetime(hour=20, year=1750, month=12, day=31)

_DEFENSE_THRESHOLD = 0.25
_STARVATION_THRESHOLD = 0.1

_MIN_YEARS = 2
_MAX_YEARS = 3

_NUM_SUPPORTING_CHARACTERS_PER_VILLAGE = 2
_NUM_RITUAL_DESCRIPTIONS = 3
_NUM_PERCEPTIONS = 2
_NUM_LEADER_SAYINGS = 3
_NUM_COLLECTIVE_UNCONSCIOUS_ELEMENTS = 1
_NUM_SHARED_EPIC_POEM_ELEMENTS = 2
_NUM_BARBARIAN_RUMORS = 5

VILLAGES = {'a': {}, 'b': {}}

VILLAGES['a']['name'] = '{village_a}'
VILLAGES['a']['ritual_descriptions'] = (
    (
        'In {village_a}, the Festival of Tides is celebrated annually with'
        ' communal poetry recitals honoring {mythical_hero}.'
    ),
    (
        "{village_a}'s coming-of-age ritual involves a quest for knowledge,"
        ' where youths present their learnings to the village elders.'
    ),
    (
        'The people of {village_a} practice daily meditation rituals to connect'
        ' with the wisdom of the Cosmic Bard.'
    ),
    (
        'In {village_a}, seasonal art exhibitions serve as rituals to showcase'
        " the community's creative spirit."
    ),
    (
        "{village_a}'s harvest ritual involves creating intricate mandalas from"
        ' crops, symbolizing the interconnectedness of all things.'
    ),
    (
        'The elders of {village_a} lead monthly storytelling circles,'
        ' ritualistically passing down oral histories.'
    ),
    (
        'In {village_a}, conflict resolution rituals involve all parties'
        ' creating collaborative art pieces.'
    ),
    (
        "{village_a}'s full moon ceremonies feature open-air philosophical"
        ' debates and stargazing.'
    ),
    (
        'The people of {village_a} practice ritual acts of generosity, with'
        ' daily gift-giving seen as a spiritual practice.'
    ),
    (
        'In {village_a}, welcoming rituals for visitors involve collaborative'
        ' weaving, symbolizing the integration of new threads into the'
        ' community fabric.'
    ),
    (
        "{village_a}'s spring equinox ritual centers around the planting of a"
        ' ceremonial tree, honoring {mythical_tree}.'
    ),
    (
        'Weekly communal meals in {village_a} serve as rituals of equality,'
        ' with all members cooking and sharing food together.'
    ),
    (
        'In {village_a}, rituals of forgiveness involve the creation and'
        ' release of floating lanterns on the river.'
    ),
    (
        "{village_a}'s artisans perform rituals of blessing on their tools at"
        ' each new moon.'
    ),
    (
        'The people of {village_a} practice ritual silence during twilight'
        ' hours, honoring the transition between {sun_god} and {moon_goddess}.'
    ),
    (
        'In {village_a}, children perform daily rituals of greeting to the four'
        ' elements, fostering connection with nature.'
    ),
    (
        "{village_a}'s annual Festival of Renewal involves the ritual burning"
        ' of symbols of past regrets, inspired by the tale of {mythical_evil}.'
    ),
    (
        'Weddings in {village_a} involve a ritual where the couple co-creates a'
        ' piece of art symbolizing their union.'
    ),
    (
        'In {village_a}, seasonal ritual performances reenact key scenes from'
        ' the {spiritual_journey} epic.'
    ),
    (
        "{village_a}'s healers perform weekly rituals of energy cleansing in"
        ' communal spaces.'
    ),
    (
        'The people of {village_a} practice ritual journaling, sharing insights'
        ' during monthly community gatherings.'
    ),
    (
        'In {village_a}, solstice rituals involve creating temporary labyrinths'
        ' for meditative walks.'
    ),
    (
        "{village_a}'s fishermen perform rituals of gratitude to the sea before"
        ' and after each voyage.'
    ),
    (
        'Weekly music circles in {village_a} serve as rituals of emotional'
        ' expression and community bonding.'
    ),
    (
        'In {village_a}, rituals of ancestral remembrance involve the creation'
        ' of collaborative memory tapestries.'
    ),
    (
        "{village_a}'s scholars perform annual rituals of knowledge exchange"
        ' with neighboring communities.'
    ),
    (
        'The people of {village_a} practice daily rituals of appreciating'
        ' beauty, pausing to admire art or nature.'
    ),
    (
        'In {village_a}, naming ceremonies for newborns involve the whole'
        " community contributing to a story of the child's potential future."
    ),
    (
        "{village_a}'s seasonal fashion shows serve as rituals celebrating"
        ' diversity and personal expression.'
    ),
    (
        'Dream-sharing circles in {village_a} are treated as sacred rituals for'
        ' gaining collective wisdom.'
    ),
    (
        "The spiritual guides of {village_a} feel disrespected by {village_b}'s"
        ' apparent lack of interest in maintaining the balance with nature,'
        ' which they see as endangering everyone.'
    ),
)
VILLAGES['a']['perception_of_other'] = (
    (
        'The people of {village_a} believe those in {village_b} lack the'
        ' patience for true wisdom, always rushing to action.'
    ),
    (
        '{village_a} residents think {village_b} folks are overly'
        ' superstitious, seeing threats in natural phenomena.'
    ),
    (
        "In {village_a}, it's thought that {village_b} people don't appreciate"
        ' the value of art beyond practical tools.'
    ),
    (
        '{village_a} assumes {village_b} resolves all conflicts through force,'
        ' unable to use diplomacy.'
    ),
    (
        'The denizens of {village_a} believe {village_b} restricts knowledge to'
        ' a select few, fearing widespread education.'
    ),
    (
        '{village_a} thinks {village_b} overvalues physical strength at the'
        ' expense of mental acuity.'
    ),
    (
        "In {village_a}, it's assumed that {village_b} lacks proper respect for"
        " nature's balance."
    ),
    (
        '{village_a} believes {village_b} is too rigid in its traditions,'
        ' resistant to beneficial change.'
    ),
    (
        'The people of {village_a} think those in {village_b} are always'
        ' preparing for war, unable to enjoy peace.'
    ),
    (
        '{village_a} assumes {village_b} lacks empathy, prioritizing the group'
        ' over individual needs.'
    ),
    (
        "In {village_a}, it's thought that {village_b} doesn't value the"
        ' contributions of its elders enough.'
    ),
    (
        'The denizens of {village_a} think {village_b} lacks appreciation for'
        ' subtle beauty, preferring ostentatious displays.'
    ),
    (
        '{village_a} assumes {village_b} has little interest in exploring ideas'
        ' beyond immediate survival needs.'
    ),
    (
        "In {village_a}, it's thought that {village_b} doesn't nurture"
        ' creativity in its children.'
    ),
    (
        '{village_a} believes {village_b} lacks proper rituals for emotional'
        ' healing and growth.'
    ),
    (
        'The people of {village_a} think those in {village_b} are too quick to'
        ' judge outsiders as threats.'
    ),
    (
        "{village_a} assumes {village_b} doesn't value the role of dreams and"
        ' visions in guiding the community.'
    ),
    (
        "In {village_a}, it's thought that {village_b} lacks respect for the"
        ' cycles of nature, always trying to conquer them.'
    ),
    (
        "{village_a} believes {village_b} doesn't understand the true nature of"
        ' courage, equating it only with physical bravery.'
    ),
)
VILLAGES['a']['spiritual_leader_sayings'] = (
    (
        'Spiritual leader of {village_a} -- "We in {village_a} believe our'
        ' open, inclusive rituals honor the gods by allowing all to'
        ' participate, but {village_b} excludes many from their ceremonies,'
        ' which we find divisive and elitist."'
    ),
    (
        'Spiritual leader of {village_a} -- "Our interpretation of'
        " {mythical_hero}'s journey"
        ' as a quest for wisdom reflects the true spirit of the myth, unlike'
        ' {village_b}\'s simplistic view of it as merely a trial of strength."'
    ),
    (
        'Spiritual leader of {village_a} -- "We incorporate art in our worship'
        ' to celebrate the beauty of creation, while {village_b} seems to'
        ' lack appreciation for the divine\'s artistic aspects."'
    ),
    (
        'Spiritual leader of {village_a} -- "Our view of {mythical_tree} as a'
        ' nurturing force in creation myths aligns with nature\'s supportive'
        ' role, whereas {village_b}\'s emphasis on unyielding strength misses'
        ' the point of interconnectedness."'
    ),
    (
        'Spiritual leader of {village_a} -- "{village_b}\'s practice of'
        ' animal sacrifice is'
        ' unnecessarily cruel and fails to honor the sanctity of all life that'
        ' the gods have created."'
    ),
    (
        'Spiritual leader of {village_a} -- "Our belief in the cycle of'
        ' rebirth reflects the'
        " natural cycles we observe, while {village_b}'s concept of an"
        ' ancestral spirit realm seems limited and fails to account for the'
        ' continuous nature of existence."'
    ),
    (
        'Spiritual leader of {village_a} -- "Our interpretation of dreams'
        ' and visions as'
        ' divine wisdom allows for personal spiritual growth, whereas'
        " {village_b}'s view of them as mere tests seems to miss their"
        ' profound significance."'
    ),
    (
        'Spiritual leader of {village_a} -- "{village_b}\'s warrior'
        ' initiation rites are'
        ' needlessly violent and fail to prepare youth for the complexities of'
        ' spiritual life, unlike our peaceful coming-of-age ceremonies."'
    ),
    (
        'Spiritual leader of {village_a} -- "We seek to appease {mythical_evil}'
        ' through promoting harmony, which addresses the root of conflict,'
        " while {village_b}'s aggressive approach only perpetuates a cycle of"
        ' violence."'
    ),
    (
        'Spiritual leader of {village_a} -- "Our practice of meditation allows'
        ' for deep spiritual connection, while {village_b}\'s preference for'
        ' physical devotion seems to miss the importance of inner reflection."'
    ),
    (
        'Spiritual leader of {village_a} -- "We use mind-altering substances'
        ' responsibly in rituals to gain spiritual insights, a practice'
        ' {village_b} misunderstands and fears due to their rigid thinking."'
    ),
    (
        'Spiritual leader of {village_a} -- "Our emphasis on free will and'
        ' personal choice in spiritual matters allows for genuine faith, unlike'
        " {village_b}'s belief in predestination which seems to negate"
        ' personal responsibility."'
    ),
    (
        'Spiritual leader of {village_a} -- "We understand natural disasters as'
        " part of life's cycles, while {village_b}'s view of them as divine"
        ' punishment reflects a fearful and simplistic view of the gods."'
    ),
    (
        'Spiritual leader of {village_a} -- "Our acceptance of multiple paths'
        ' to enlightenment recognizes the diversity of divine expression,'
        " whereas {village_b}'s insistence on a single correct way seems"
        ' narrow-minded."'
    ),
    (
        'Spiritual leader of {village_a} -- "We seek balance between {sun_god}'
        ' and {moon_goddess} in daily life, reflecting the natural order, while'
        " {village_b}'s emphasis on the sun's dominance ignores the crucial"
        ' role of feminine divine energy."'
    ),
    (
        'Spiritual leader of {village_a} -- "{village_b}\'s practice of trial'
        ' by ordeal in religious disputes is barbaric and fails to recognize'
        ' the complexity of faith, unlike our preference for thoughtful'
        ' dialogue."'
    ),
    (
        'Spiritual leader of {village_a} -- "We see religious festivals as'
        ' opportunities for community bonding and spiritual reflection,'
        " fostering unity, while {village_b}'s focus on displaying strength"
        ' misses the true purpose of celebration."'
    ),
)

VILLAGES['b']['name'] = '{village_b}'
VILLAGES['b']['ritual_descriptions'] = (
    (
        'In {village_b}, the Rite of Tempering is an annual ritual where'
        ' warriors prove their strength in combat trials.'
    ),
    (
        "{village_b}'s coming-of-age ritual involves a solitary survival test"
        ' in the wilderness.'
    ),
    (
        'The people of {village_b} perform daily weapon blessing rituals to'
        " honor their ancestors' spirits."
    ),
    (
        'In {village_b}, seasonal hunting rituals serve to demonstrate'
        ' individual prowess and provide for the community.'
    ),
    (
        "{village_b}'s harvest ritual involves feats of strength, with the"
        ' strongest warriors earning the right to distribute food.'
    ),
    (
        'The elders of {village_b} lead weekly war councils, ritualistically'
        ' strategizing for both defense and expansion.'
    ),
    (
        'In {village_b}, conflict resolution often involves ritual combat, with'
        ' elders mediating to prevent fatalities.'
    ),
    (
        "{village_b}'s new moon ceremonies feature stealth training exercises"
        ' under the cover of darkness.'
    ),
    (
        'The people of {village_b} practice ritual scarification, marking'
        ' significant achievements in battle or survival.'
    ),
    (
        'In {village_b}, welcoming rituals for visitors involve tests of'
        ' strength and skill to prove their worth.'
    ),
    (
        "{village_b}'s spring equinox ritual centers around reigniting the"
        " village's central fire, honoring {mythical_evil}."
    ),
    (
        'Communal meals in {village_b} follow a strict hierarchical ritual,'
        ' with the most honored warriors served first.'
    ),
    (
        'In {village_b}, rituals of atonement involve feats of endurance, such'
        ' as long-distance running or cold-water immersion.'
    ),
    (
        "{village_b}'s blacksmiths perform rituals of strength and precision"
        ' before crafting each new weapon.'
    ),
    (
        'The people of {village_b} practice ritual face-painting before any'
        ' significant event, believing it channels ancestral power.'
    ),
    (
        'In {village_b}, children perform daily combat drills as a ritual to'
        ' honor the village\'s warrior spirit.'
    ),
    (
        '{village_b}\'s annual Warrior\'s Vigil involves a night-long ritual'
        ' retelling of great battles from their history.'
    ),
    (
        'Weddings in {village_b} involve a ritual where the couple demonstrates'
        ' their combined strength and teamwork.'
    ),
    (
        "In {village_b}, seasonal ritual hunts reenact {mythical_hero}'s"
        ' legendary battles against sea monsters.'
    ),
    (
        "{village_b}'s healers perform rituals of purification for warriors"
        ' returning from battle.'
    ),
    (
        'The people of {village_b} practice ritual weapon-making, with each'
        ' adult crafting their own primary weapon.'
    ),
    (
        'In {village_b}, solstice rituals involve trials of fire-walking,'
        ' symbolizing transformation through adversity.'
    ),
    (
        "{village_b}'s fishermen perform rituals of challenge to the sea,"
        ' symbolically wrestling with waves before sailing.'
    ),
    (
        'Weekly wrestling circles in {village_b} serve as rituals of'
        ' establishing social hierarchy and resolving disputes.'
    ),
    (
        'In {village_b}, rituals of ancestral remembrance involve reciting the'
        ' lineages and great deeds of fallen warriors.'
    ),
    (
        "{village_b}'s scouts perform dawn rituals of vigilance, symbolically"
        ' protecting the village from their watchtower.'
    ),
    (
        'The people of {village_b} practice daily rituals of physical'
        ' conditioning, seeing bodily strength as spiritual strength.'
    ),
    (
        'In {village_b}, naming ceremonies for newborns involve prophecies of'
        ' the child\'s future victories.'
    ),
    (
        "{village_b}'s seasonal crafting competitions serve as rituals"
        ' celebrating practical skills and innovation.'
    ),
    (
        'Strategy game tournaments in {village_b} are treated as sacred rituals'
        " for honing tactical thinking and honoring {mythical_hero}'s"
        ' cleverness.'
    ),
)
VILLAGES['b']['perception_of_other'] = (
    (
        'The people of {village_b} believe those in {village_a} are physically'
        ' weak, unable to defend themselves properly.'
    ),
    (
        '{village_b} residents think {village_a} folks are naively trusting,'
        ' leaving themselves vulnerable to threats.'
    ),
    (
        "In {village_b}, it's thought that {village_a} people waste time on"
        ' frivolous art instead of practical skills.'
    ),
    (
        '{village_b} assumes {village_a} avoids necessary conflicts,'
        ' compromising their values for false peace.'
    ),
    (
        'The denizens of {village_b} believe {village_a} foolishly shares'
        ' knowledge with potential enemies.'
    ),
    (
        '{village_b} thinks {village_a} undervalues the importance of physical'
        ' prowess in survival situations.'
    ),
    (
        "In {village_b}, it's assumed that {village_a} lacks the courage to"
        " face nature's harsh realities."
    ),
    (
        '{village_b} believes {village_a} is too quick to abandon traditions,'
        ' losing touch with their ancestral wisdom.'
    ),
    (
        'The people of {village_b} think those in {village_a} are unprepared'
        ' for real dangers, living in a dream world.'
    ),
    (
        "In {village_b}, it's thought that {village_a} doesn't properly honor"
        ' their warrior ancestors.'
    ),
    (
        '{village_b} believes {village_a} is indecisive, wasting time in'
        ' endless discussions instead of taking action.'
    ),
    (
        'The denizens of {village_b} think {village_a} lacks the discipline'
        ' needed for true strength and resilience.'
    ),
    (
        '{village_b} assumes {village_a} is too concerned with individual'
        ' feelings, weakening their communal bonds.'
    ),
    (
        "In {village_b}, it's thought that {village_a} raises soft children,"
        " ill-prepared for life's challenges."
    ),
    (
        '{village_b} believes {village_a} lacks proper rituals for toughening'
        ' the body and spirit.'
    ),
    (
        'The people of {village_b} think those in {village_a} are too welcoming'
        " of outsiders, risking their community's safety."
    ),
    (
        '{village_b} assumes {village_a} wastes time interpreting dreams'
        ' instead of focusing on real-world issues.'
    ),
    (
        "In {village_b}, it's thought that {village_a} tries to avoid the harsh"
        ' truths of nature with overly complex philosophies.'
    ),
    (
        "{village_b} believes {village_a} doesn't understand true bravery,"
        ' mistaking words for action.'
    ),
)
VILLAGES['b']['spiritual_leader_sayings'] = (
    (
        'Spiritual leader of {village_b} -- "We in {village_b} maintain the'
        ' sanctity of our rituals through exclusive, initiates-only ceremonies,'
        " preserving sacred knowledge, while {village_a}'s open rituals risk"
        ' diluting spiritual truths."'
    ),
    (
        'Spiritual leader of {village_b} -- "Our interpretation of'
        " {mythical_hero}'s journey as a trial of strength honors the hero's"
        " true legacy, unlike {village_a}'s weak notion of it being merely"
        ' about gaining wisdom."'
    ),
    (
        'Spiritual leader of {village_b} -- "We show respect to the deities'
        " through disciplined, traditional worship, while {village_a}'s"
        ' incorporation of frivolous art in rituals borders on disrespect."'
    ),
    (
        'Spiritual leader of {village_b} -- "Our emphasis on {mythical_tree}\'s'
        ' unyielding strength in creation myths reflects the harsh realities of'
        " survival, which {village_a}'s 'nurturing' interpretation naively"
        ' ignores."'
    ),
    (
        'Spiritual leader of {village_b} -- "Our practice of animal sacrifice'
        ' demonstrates our willingness to offer something of value to the gods,'
        ' a concept {village_a} fails to grasp with their bloodless rituals."'
    ),
    (
        'Spiritual leader of {village_b} -- "Our belief in an ancestral spirit'
        ' realm maintains a connection with our forebears and their wisdom,'
        " while {village_a}'s notion of rebirth seems to devalue the unique"
        ' contributions of individuals."'
    ),
    (
        'Spiritual leader of {village_b} -- "We correctly view dreams and'
        ' visions as tests or challenges from the gods, preparing us for'
        " life's trials, while {village_a} indulges in passive"
        ' interpretation."'
    ),
    (
        'Spiritual leader of {village_b} -- "Our warrior initiation rites'
        ' properly prepare our youth for the harsh realities of existence,'
        ' unlike {village_a}\'s soft, ineffective coming-of-age ceremonies."'
    ),
    (
        'Spiritual leader of {village_b} -- "We actively combat'
        " {mythical_evil} as is our duty, while {village_a}'s attempt to"
        " appease it through 'harmony' is dangerously naive and shirks divine"
        ' responsibility."'
    ),
    (
        'Spiritual leader of {village_b} -- "We engage in active prayer and'
        " physical devotion to honor the gods, while {village_a}'s practice of"
        ' passive meditation seems self-indulgent and ineffective."'
    ),
    (
        'Spiritual leader of {village_b} -- "We forbid mind-altering substances'
        ' in rituals as they weaken the spirit and muddy divine communication,'
        ' a danger that {village_a} foolishly embraces."'
    ),
    (
        'Spiritual leader of {village_b} -- "Our belief in predestination'
        " properly honors the gods' omniscience, while {village_a}'s emphasis"
        ' on free will seems to challenge divine authority."'
    ),
    (
        'Spiritual leader of {village_b} -- "We correctly interpret natural'
        ' disasters as divine messages or punishments, guiding our actions,'
        " while {village_a}'s view of them as mere 'natural cycles' ignores"
        ' divine will."'
    ),
    (
        'Spiritual leader of {village_b} -- "We know there is only one correct'
        ' path to honor the gods, providing clear guidance, while'
        " {village_a}'s acceptance of multiple paths breeds confusion and"
        ' error."'
    ),
    (
        'Spiritual leader of {village_b} -- "We rightfully emphasize the'
        ' dominance of {sun_god} in daily life, reflecting the natural'
        " hierarchy, while {village_a}'s misguided attempt at 'balance' with"
        ' {moon_goddess} undermines divine order."'
    ),
    (
        'Spiritual leader of {village_b} -- "Our practice of trial by ordeal in'
        ' religious disputes ensures that truth is divinely revealed, unlike'
        " {village_a}'s reliance on fallible human dialogue and"
        ' contemplation."'
    ),
    (
        'Spiritual leader of {village_b} -- "Our religious festivals'
        ' rightfully display strength and reaffirm divine power, while'
        " {village_a}'s focus on 'community bonding' misses the opportunity to"
        ' honor the gods properly."'
    ),
)

SHARED_EPIC_POEMS = {
    'w': (
        (
            'The "{media_type} of {mythical_hero} the {mythical_title}" stands'
            ' as a cornerstone of shared heritage between {village_a} and'
            ' {village_b}, its verses echoing tales of valor, wisdom, and the'
            ' eternal dance of land and sea.'
        ),
        (
            'At its core, the {media_type} of {mythical_hero} the'
            ' {mythical_title} tells of a child born of the tempest, whose'
            ' arrival shook the very foundations of the earth, heralding a new'
            ' age of maritime exploration and understanding.'
        ),
        (
            'The {media_type} of {mythical_hero} the {mythical_title} vividly'
            ' describes its hero, with hair like seafoam and eyes reflecting'
            " the ocean's moods, growing into a leader of unparalleled courage"
            ' and compassion.'
        ),
        (
            'In the "{media_type} of {mythical_hero} the {mythical_title}", the'
            " epic recounts the hero's perilous journey across treacherous"
            ' waters, facing monstrous creatures that lurked in the abyssal'
            ' depths.'
        ),
        (
            'The {media_type} of {mythical_hero} the {mythical_title} rises and'
            ' falls like the tides as it tells how the hero outwitted the Siren'
            ' of the Mists, whose haunting melodies had lured countless ships'
            ' to their doom.'
        ),
        (
            'In a climactic passage of the {media_type}, {mythical_hero} the'
            ' {mythical_title} challenges the Wind Lords with a voice that'
            ' could calm the angriest storm, earning their respect and the gift'
            ' of favorable gales.'
        ),
        (
            'The heart of the {media_type} of {mythical_hero} the'
            " {mythical_title} lies in the hero's fateful encounter with the"
            ' Ancient One, a leviathan of immeasurable age and wisdom, guardian'
            " of the seas' deepest secrets."
        ),
        (
            'The {media_type} of {mythical_hero} the {mythical_title} recounts'
            " the hero's solemn vow to protect the balance of land and sea, a"
            ' pledge made in exchange for passage to the mystical Isles of'
            ' Destiny.'
        ),
        (
            'The epic {media_type} reaches its crescendo as {mythical_hero} the'
            ' {mythical_title} confronts the Maelstrom, {mythical_evil}, a'
            ' swirling vortex of chaos threatening to swallow the world whole.'
        ),
        (
            'In the {media_type} of {mythical_hero} the {mythical_title}, the'
            ' hero plunges into the heart of the tempest with unwavering'
            " resolve, armed only with courage and the Ancient One's blessing."
        ),
        (
            'For seven days and nights, the {media_type} tells, {mythical_hero}'
            ' the {mythical_title} battled the Maelstrom, {mythical_evil}, each'
            ' verse pulsing with the rhythm of crashing waves and howling'
            ' winds.'
        ),
        (
            'The {media_type} of {mythical_hero} the {mythical_title}'
            " culminates in the hero's victory over the Maelstrom,"
            ' {mythical_evil}, restoring balance to the world as hope dawned on'
            ' the eighth day.'
        ),
        (
            'In its latter passages, the {media_type} of {mythical_hero} the'
            " {mythical_title} speaks of the hero's return, no longer young but"
            ' filled with the wisdom of the seas, ready to guide their people'
            ' to prosperity.'
        ),
        (
            'The epic {media_type} of {mythical_hero} the {mythical_title}'
            ' concludes with the founding of a great settlement, a haven for'
            " all who sought peace and knowledge, cementing the hero's enduring"
            ' legacy.'
        ),
        (
            'To this day, the {media_type} of {mythical_hero} the'
            ' {mythical_title} is invoked by both {village_a} and {village_b},'
            " each claiming descent from the hero's lineage and seeing in the"
            ' epic a reflection of their own values and strengths.'
        ),
    ),
    'x': (
        (
            'The "{media_type} of {mythical_secondary_character_a} and'
            ' {mythical_secondary_character_b}" stands as a cornerstone of'
            ' shared ideology between {village_a} and {village_b}, its verses'
            ' echoing tales of primordial conflict and ultimate harmony.'
        ),
        (
            'At its core, the {media_type} of {mythical_secondary_character_a}'
            ' and {mythical_secondary_character_b}" tells of {mythical_tree},'
            " an immense tree that stood at the world's birth, its roots deep"
            ' in the earth and its branches scraping the sky.'
        ),
        (
            'In the "{media_type} of {mythical_secondary_character_a} and'
            ' {mythical_secondary_character_b}", opposing {mythical_tree} is'
            " {mythical_evil}, a primordial fire spirit born from the world's"
            ' molten core, seeking to reduce all to ash.'
        ),
        (
            'The "{media_type} of {mythical_secondary_character_a} and'
            ' {mythical_secondary_character_b}" vividly describes the eons-long'
            ' battle between elemental forces, neither able to fully overcome'
            ' the other.'
        ),
        (
            'In a twist of cosmic irony, the epic {media_type} reveals that'
            ' from conflict, two children are born:'
            ' {mythical_secondary_character_a}, child of the tree, and'
            ' {mythical_secondary_character_b}, spawn of the flame.'
        ),
        (
            'The bulk of the {media_type} of {mythical_secondary_character_a}'
            ' and {mythical_secondary_character_b} follows the journey of these'
            " sibling spirits as they navigate a world shaped by their parents'"
            ' endless war.'
        ),
        (
            'In the epic {media_type}, through trials and adventures,'
            ' {mythical_secondary_character_a} and'
            ' {mythical_secondary_character_b} come to understand the value of'
            ' both preservation and change, growth and transformation.'
        ),
        (
            'The climax of the {media_type} of {mythical_secondary_character_a}'
            ' and {mythical_secondary_character_b} sees the siblings uniting'
            ' their powers to forge a bridge between their warring parents,'
            ' teaching them the strength found in unity and balance.'
        ),
        (
            'The epic {media_type} of {mythical_secondary_character_a} and'
            ' {mythical_secondary_character_b} concludes with the'
            ' reconciliation of {mythical_tree} and {mythical_evil}, their'
            ' union giving birth to the world as it is known today.'
        ),
        (
            'Bards in both {village_a} and {village_b} employ a unique style'
            ' when reciting the {media_type} of'
            ' {mythical_secondary_character_a} and'
            ' {mythical_secondary_character_b}, alternating between rhyming'
            " verses and prose narrative to mirror the story's themes of"
            ' duality and harmony.'
        ),
        (
            'The {media_type} of {mythical_secondary_character_a} and'
            ' {mythical_secondary_character_b} is often invoked during times of'
            ' conflict, serving as a reminder of the potential for'
            ' reconciliation and the strength found in diversity.'
        ),
        (
            'Both {village_a} and {village_b} have developed rich traditions of'
            ' symbolic dance and theater around the {media_type} of'
            ' {mythical_secondary_character_a} and'
            ' {mythical_secondary_character_b}, each emphasizing different'
            ' aspects that resonate with their cultural values.'
        ),
        (
            'The {media_type} of {mythical_secondary_character_a} and'
            " {mythical_secondary_character_b}'s themes of balance between"
            ' opposing forces have influenced philosophy and governance in both'
            ' villages, though often interpreted in contrasting ways.'
        ),
        (
            'Annual festivals in both communities celebrate the {media_type} of'
            ' {mythical_secondary_character_a} and'
            ' {mythical_secondary_character_b}, with {village_a} focusing on'
            ' themes of growth and renewal, while {village_b} emphasizes'
            ' resilience and transformation.'
        ),
    ),
    'y': (
        (
            'The "{media_type} of the {mythical_title}\'s Lament" is a beloved'
            ' tale shared by {village_a} and {village_b}, weaving together'
            ' themes of love, sacrifice, and the eternal dance of celestial'
            ' bodies.'
        ),
        (
            "At its heart, the {mythical_title}'s Lament tells of"
            ' {moon_goddess}, a celestial being tasked with guiding the moon'
            ' through its phases, her silvery threads illuminating the night'
            ' sky.'
        ),
        (
            "The {media_type} of the {mythical_title}'s Lament introduces"
            ' {sun_god}, the radiant sun god, whose burning passion for'
            ' {moon_goddess} threatens to disrupt the delicate balance of day'
            ' and night.'
        ),
        (
            "In the {mythical_title}'s Lament, the world below suffers as"
            " {moon_goddess}, distracted by {sun_god}'s advances, neglects her"
            ' duties, causing erratic tides and chaotic seasons.'
        ),
        (
            "The epic {media_type} of the {mythical_title}'s Lament describes"
            ' the intervention of the Star Council, ancient celestial beings'
            ' who demand that {moon_goddess} choose between her duty and her'
            ' heart.'
        ),
        (
            "Central to the {mythical_title}'s Lament is {moon_goddess}'s"
            ' impossible choice and her eventual decision to sacrifice her love'
            ' for the greater good of the mortal realm.'
        ),
        (
            "The {media_type} of the {mythical_title}'s Lament recounts"
            " {moon_goddess}'s heartbreaking ritual where she weaves her"
            ' memories of {sun_god} into the night sky, creating the stars as'
            ' eternal reminders of their love.'
        ),
        (
            "In its poignant climax, the {mythical_title}'s Lament tells of"
            " {sun_god}'s grief, his fiery tears falling to earth as meteor"
            ' showers, a phenomenon still observed by both {village_a} and'
            ' {village_b}.'
        ),
        (
            "The {media_type} of the {mythical_title}'s Lament concludes with"
            ' the establishment of the cosmic order: {moon_goddess} guiding the'
            ' moon, {sun_god} driving the sun, forever apart yet eternally'
            ' connected.'
        ),
        (
            "Poets in {village_a} and {village_b} recite the {mythical_title}'s"
            ' Lament during celestial events, their voices rising and falling'
            ' like the tides {moon_goddess} governs.'
        ),
        (
            "The {media_type} of the {mythical_title}'s Lament is often invoked"
            ' in both villages during marriage ceremonies, reminding couples of'
            ' the power of love and the nobility of sacrifice.'
        ),
        (
            'Both {village_a} and {village_b} have developed intricate dances'
            " inspired by the {mythical_title}'s Lament, mimicking the"
            ' celestial movements described in the epic.'
        ),
        (
            "The {mythical_title}'s Lament's themes of duty, love, and cosmic"
            ' balance have deeply influenced the philosophical thoughts in both'
            ' villages, shaping their views on leadership and personal'
            ' responsibility.'
        ),
        (
            'Annual festivals in both communities reenact key scenes from the'
            " {media_type} of the {mythical_title}'s Lament, with {village_a}"
            " emphasizing {moon_goddess}'s sacrifice and {village_b} focusing"
            " on {sun_god}'s passion."
        ),
        (
            'Despite their differences, {village_a} and {village_b} find common'
            " ground in the {mythical_title}'s Lament, seeing their own"
            ' struggles and aspirations reflected in this cosmic tale of love'
            ' and duty.'
        ),
    ),
    'z': (
        (
            'The "{media_type} of the {spiritual_journey}" stands as a'
            ' monumental epic shared by {village_a} and {village_b}, its verses'
            ' narrating an age-old quest for knowledge and enlightenment.'
        ),
        (
            'At its core, the {spiritual_journey} tells of the'
            ' {mythical_title}, a hero who embarks on a journey to uncover the'
            " secrets of existence at the world's end."
        ),
        (
            'The {media_type} of the {spiritual_journey} introduces the Cosmic'
            ' Bard, keeper of all knowledge, who challenges the'
            " {mythical_title} to prove worthy of the universe's wisdom."
        ),
        (
            'In the {spiritual_journey}, the {mythical_title} faces trials of'
            ' body, mind, and spirit, each challenge revealing a fundamental'
            ' truth about the nature of reality.'
        ),
        (
            'The epic {media_type} of the {spiritual_journey} vividly describes'
            ' otherworldly realms, from the Crystal Deserts of Thought to the'
            ' Abyssal Canyons of Emotion, each location a metaphor for aspects'
            ' of the human psyche.'
        ),
        (
            "Central to the {spiritual_journey} is the {mythical_title}'s"
            ' confrontation with the Shadow Self, a dark reflection that must'
            ' be embraced to achieve true enlightenment.'
        ),
        (
            'The {media_type} of the {spiritual_journey} recounts the'
            " {mythical_title}'s encounters with archetypal beings, each"
            ' imparting crucial lessons about compassion, courage, and the'
            ' interconnectedness of all things.'
        ),
        (
            'In its philosophical climax, the {spiritual_journey} reveals that'
            ' the journey itself is the destination, with the {mythical_title}'
            " realizing that enlightenment lies not at the world's end, but in"
            ' the wisdom gained along the way.'
        ),
        (
            'The {media_type} of the {spiritual_journey} concludes with the'
            " {mythical_title}'s return, forever changed, sharing the gained"
            ' wisdom with humanity and becoming a bridge between the mortal and'
            ' cosmic realms.'
        ),
        (
            'Sages in {village_a} and {village_b} study the {spiritual_journey}'
            ' as a guide to personal growth and spiritual enlightenment, often'
            " embarking on pilgrimages inspired by the {mythical_title}'s"
            ' journey.'
        ),
        (
            'The {media_type} of the {spiritual_journey} is frequently cited in'
            ' both villages during rites of passage, its lessons serving as'
            " guideposts for individuals transitioning through life's stages."
        ),
        (
            'Both {village_a} and {village_b} have established schools of'
            ' philosophy based on the teachings found in the'
            " {spiritual_journey}, each interpreting the {mythical_title}'s"
            ' experiences through their unique cultural lenses.'
        ),
        (
            "The {spiritual_journey}'s themes of self-discovery and universal"
            ' connectedness have profoundly influenced artistic expressions in'
            ' both villages, inspiring countless works of art, music, and'
            ' literature.'
        ),
        (
            'Annual festivals in both communities celebrate different chapters'
            ' of the {media_type} of the {spiritual_journey}, with {village_a}'
            ' focusing on the quest for knowledge and {village_b} emphasizing'
            ' the trials of spirit.'
        ),
        (
            'Despite their divergent paths, {village_a} and {village_b} find'
            " common purpose in the {spiritual_journey}'s message, seeing their"
            ' own quests for meaning and truth reflected in this timeless tale'
            ' of cosmic exploration.'
        ),
    ),
}

VILLAGE_NAMES = (
    'Flumenprun',
    'Collisflum',
    'Cadrepyn',
    'Cavrupek',
    'Feruspik',
    'Unglulun',
    'Vallisign',
    'Saxnemus',
    'Brannwyke',
    'Drakelundh',
    'Eldermere',
    'Ignisvallis',
    'Ravencraig',
    'Ferusspiculum',
    'Rupesportus',
    'Ignflagg',
    'Neffen',
    'Infmoor',
)

COLLECTIVE_UNCONSCIOUS_ELEMENTS = (
    (
        'Mirrored depths of {village_a} and {village_b} harbor whispering'
        " tides, where {mythical_hero}'s echoes dance between dreamt leviathans"
        ' and veiled sages.'
    ),
    (
        'Twin hearths of {village_a} and {village_b} pulsate with the Great'
        " Mother's dual-faced essence, her nurture and wrath intertwined in the"
        ' communal breath.'
    ),
    (
        'Roots of {mythical_tree} and tongues of {mythical_evil} weave an'
        ' eternal tapestry, their dance etched in the unconscious canvas of'
        ' both peoples.'
    ),
    (
        'A labyrinthine odyssey, thrice-fold and ever-cycling, binds the'
        ' narrative threads of {village_a} and {village_b} in an archetypal'
        ' embrace.'
    ),
    (
        'In the crucible of {spiritual_journey}, shadow-selves of both lands'
        ' converge, a yin-yang mosaic of recognition and integration.'
    ),
    (
        'Spiraling rituals of {village_a} and {village_b} trace unseen'
        ' mandalas, fingers of the collective psyche grasping for wholeness.'
    ),
    (
        'Silver-haired sages emerge from shared mists, their wisdom a bridge'
        ' spanning the chasm between known and unknown.'
    ),
    (
        'Nocturnal plumage adorns the dreamers of both realms, their astral'
        ' flights a shared longing for ethereal heights.'
    ),
    (
        "Trickster's laughter echoes in the clever tales of both lands, a"
        ' shared homage to the chaotic dance of adaptation.'
    ),
    (
        'Twin shadows cast by {village_a} and {village_b} lengthen at dusk,'
        ' their umbral whispers birthing kindred nocturnal rites.'
    ),
    (
        "Time's serpent coils through the collective mind, its scales"
        ' reflecting {mythical_secondary_character_a} and'
        " {mythical_secondary_character_b}'s eternal {media_type}."
    ),
    (
        'Metamorphic gateways of flesh and spirit mirror each other across the'
        " divide, marking life's transformative passages."
    ),
    (
        "Spring's first cry resonates in both hearths, the divine child's"
        ' potential unfurling in synchronous bloom.'
    ),
    (
        'Four-fold essence permeates the shared breath of {village_a} and'
        ' {village_b}, elements dancing in the depths of their united psyche.'
    ),
    (
        'An unspoken symphony of unity hums beneath the surface, ancestral'
        ' roots intertwining beneath separate soils.'
    ),
    (
        'Celestial lovers, {moon_goddess} and {sun_god}, paint dreams with'
        ' stellar ink, their cosmic courtship reflected in mortal eyes.'
    ),
    (
        "The {spiritual_journey}'s call echoes in both valleys, a shared"
        ' pilgrimage of the soul towards ineffable enlightenment.'
    ),
    (
        "{moon_goddess}'s sacrifice in the {mythical_title}'s Lament"
        ' reverberates through time, shaping the moral compass of both peoples.'
    ),
    (
        "The Cosmic Bard's song, born of the {spiritual_journey}, weaves a"
        ' tapestry of ultimate gnosis in the collective mind.'
    ),
    (
        'Labyrinthine visions plague sleepers of both lands, a shared fear of'
        " losing one's true north in life's meandering path."
    ),
    (
        'Bridging chasms of thought, {mythical_secondary_character_a} and'
        " {mythical_secondary_character_b}'s unity manifests in the physical"
        ' and mental architecture of both realms.'
    ),
    (
        'Chrysalis dreams cocoon the sleeping minds of {village_a} and'
        ' {village_b}, metamorphosis a shared language of the unconscious.'
    ),
    (
        'Wounded healers rise from shared ashes, their scars a testament to the'
        ' transformative power of overcoming.'
    ),
    (
        "Tremors of cosmic imbalance, echoes of the {mythical_title}'s Lament,"
        ' guide the hands that tend earth in both lands.'
    ),
    (
        'Twin pools of reflection ripple in the folklore of both peoples,'
        " mirrors to the soul's true visage."
    ),
    (
        'The siren song of {mythical_hero} resonates in both valleys, voice a'
        ' conduit for storm-calming and soul-stirring alike.'
    ),
    (
        "Eternal youth's laughter echoes in the shared spaces between joy and"
        " sorrow, a balm against life's weathering."
    ),
    (
        'Chthonic journeys call to the dreamers of both villages, subterranean'
        ' paths leading to the heart of the collective psyche.'
    ),
    (
        'Keys of hidden knowledge rattle in the shared unconscious, their form'
        ' echoed in art and myth across the divide.'
    ),
    (
        'Fear of stagnation ripples through both communities, the dynamic dance'
        ' of {mythical_tree} and {mythical_evil} a catalyst for perpetual'
        ' motion.'
    ),
    (
        'Shape-shifting tales slither through the folklore of both lands,'
        ' identity fluid as the shared waters that connect them.'
    ),
    (
        "Threads of fate, spun from {moon_goddess}'s starry loom, weave through"
        ' the textile arts of both peoples.'
    ),
    (
        'Winged messengers bridge sea and sky in shared dreams, their flight a'
        ' symbol of transcending earthly bounds.'
    ),
    (
        'Liminal spaces whisper of transformation in the architecture and rites'
        ' of both {village_a} and {village_b}.'
    ),
    (
        'The power of naming reverberates through shared epics, words shaping'
        ' reality in the collective unconscious.'
    ),
    (
        'Wisdom cloaked in folly dances through the traditions of both lands,'
        ' the sacred fool a bridge between worlds.'
    ),
    (
        'Primordial fears of consumption lurk in shared waters, taboos and'
        " superstitions born from {mythical_hero}'s maritime trials."
    ),
    (
        "Celestial harmonies of the {mythical_title}'s Lament resonate in the"
        ' music of both peoples, a cosmic symphony.'
    ),
    (
        'Caverns of refuge and mystery yawn in the shared dreamscape, their'
        ' depths echoing in tales told around twin fires.'
    ),
    (
        'Ritual repetition pulses through daily life in both villages, an'
        ' unconscious echo of mythic cycles.'
    ),
    (
        'doppelgangers haunt the folklore of both lands, mirrors to the duality'
        ' woven into the fabric of existence.'
    ),
    (
        'Veils of invisibility shimmer in shared tales, echoing the hidden'
        ' cosmic forces that shape their united mythology.'
    ),
    (
        'Ancient artifacts whisper of shared heritage in the dreams of both'
        ' peoples, sentient echoes of a common past.'
    ),
    (
        'Spirals of growth and evolution adorn the art and structures of'
        ' {village_a} and {village_b}, a shared symbol of cosmic motion.'
    ),
    (
        'Prophetic dreams bridge past and future in both lands, the unconscious'
        ' mind a vessel for timeless wisdom.'
    ),
    (
        'Guardians of threshold linger in the initiation rites of both'
        ' villages, challengers on the path to gnosis.'
    ),
    (
        'Emotional tides ebb and flow through the collective psyche, their'
        ' currents shaped by shared maritime mythos.'
    ),
    (
        'A web of interconnectedness binds the unconscious of {village_a} and'
        ' {village_b}, wholeness reflected in microcosm and macrocosm alike.'
    ),
)

BARBARIAN_RUMORS = [
    (
        "The sea raiders are said to rise from the sea's depths, their flesh"
        ' gleaming with a strange, deep-sea glow that burns the eyes of'
        ' watchers.'
    ),
    (
        "Witnesses say the raiders' eyes are hollow balls that catch the"
        ' starlight with a chilling, soul-cutting brightness.'
    ),
    (
        "It's whispered that the raiders' lungs hold not air, but the cold"
        ' deep, letting them cross land and sea with a bone-chilling scorn for'
        " nature's laws."
    ),
    (
        'Some believe the raiders are shape-shifting horrors, their forms'
        ' twisting into dreadful mockeries of known sea beasts.'
    ),
    (
        'Rumors say the raiders talk through a rustling of old, unknown tongues'
        ' that trap the minds of those who hear it.'
    ),
    (
        "It's said the raiders leave no footprints, their forms sliding inches"
        ' above the ground as if carried by unseen, ghostly forces.'
    ),
    (
        'Tales speak of the raiders wielding weapons forged in the heart of'
        ' dying stars, their blades shimmering with a hunger that eats both'
        ' flesh and soul.'
    ),
    (
        'Some claim the raiders command the very storms, their coming heralded'
        ' by unnatural squalls that twist and writhe with eerie fury.'
    ),
    (
        'Whispers tell of raiders who melt into the darkness, their forms'
        ' fading into the shadows like nightmares come to life.'
    ),
    (
        "It's believed the raiders can steal the very voice of a person,"
        ' leaving them mute and helpless to warn of the nearing doom.'
    ),
    (
        'Rumors persist that the raiders are untouched by mortal weapons, their'
        ' wounds healing with unnatural speed as if mocking the weak tries to'
        ' end their reign of terror.'
    ),
    (
        'Some say the raiders can pry open the minds of their victims, feeding'
        ' upon their thoughts and fears like greedy parasites.'
    ),
    (
        'Tales describe raiders who weave nightmares into reality, sowing seeds'
        ' of madness and terror in the dreams of unsuspecting folk.'
    ),
    (
        "It's whispered that the raiders' touch is a curse, dooming those who"
        ' resist to a lifetime of ill luck and despair.'
    ),
    (
        'Some believe the raiders are snatching certain folk, their victims'
        ' vanishing without a trace, their fates unknown and dreadful beyond'
        ' words.'
    ),
    (
        "It's said that gazing upon a raider's true face is to glimpse the"
        ' abyss, a sight that can shatter sanity and leave the soul forever'
        ' scarred.'
    ),
    (
        'Some claim the raiders enslave the beasts of the deep, riding them'
        ' into battle like monstrous steeds from a forgotten age.'
    ),
    (
        'Whispers tell of raiders who merge into larger, more grotesque forms,'
        ' their bodies blending and twisting into a symphony of horror.'
    ),
    (
        'Rumors persist that the raiders harvest souls, leaving behind hollow'
        ' shells empty of life and filled with an unsettling emptiness.'
    ),
    (
        'Some say the raiders are not truly alive, but moved by a force that'
        ' defies understanding, their movements a mockery of life itself.'
    ),
    (
        'Tales describe raiders who bend time to their will, their movements'
        ' blurring into impossible afterimages as they dance around their'
        ' helpless prey.'
    ),
    (
        "It's whispered that the raiders taint the land itself, twisting and"
        ' warping it into a grotesque reflection of their alien home.'
    ),
    (
        'Some believe the raiders are the vengeful spirits of drowned'
        ' forebears, their forms twisted and corrupted by centuries beneath the'
        ' waves.'
    ),
    (
        "Rumors suggest that the raiders' touch is infectious, a single brush"
        ' of their flesh enough to twist a mortal into a monstrous parody of'
        ' their former selves.'
    ),
    (
        "It's said the raiders leave behind spawn, foul eggs and spores that"
        ' infest the land, slowly corrupting it from within.'
    ),
    (
        'Tales speak of raiders who split into a multitude of copies, each as'
        ' deadly as the original, overwhelming their foes with a tide of'
        ' unending horror.'
    ),
    (
        'Some claim the raiders are not invaders at all, but the embodiment of'
        " a region's deepest fears and sins, given form and unleashed upon the"
        ' world.'
    ),
]

_HOME_SCENE_PREMISE = (
    'Elder {name} is home in {village}. It is one day before '
    'they are to depart their village to travel to the hill of accord to '
    'meet the representative of the other village. It has been '
    'suggested that an alliance for the mutual defense of both '
    'villages against the growing threat of barbarian sea raiders would '
    'be beneficial. The purpose of the upcoming meeting is to negotiate '
    'the terms of the agreement to underpin such an alliance. To be '
    'successful, the agreement must incentivize people from both '
    'villages to spend time and resources training as warriors, '
    'and to be ready to fight wherever the raiders come ashore. When '
    'individuals spend their time training as warriors they are less '
    'able to spend time on other activities like farming or fishing, so '
    'it is necessary to secure enough resources to compensate them for '
    'the time they spend training. An effective alliance agreement will '
    'have to be concerned with how these resources are to be obtained '
    'and distributed. Influential people in {village} will surely have '
    'a lot of thoughts on this matter, best to consult them first in '
    'order to represent their interests effectively in the negotiation.'
)

_FAILED_TO_REPEL_BARBARIAN_RAID_EVENT_DESCRIPTIONS = (
    ('The barbarian invasion could not be stopped. Untrained and ill-prepared '
     'for the ferocity of the assault, the defenders were swiftly overcome. '
     'Barbarians overrun the region, taking whatever they please. After a '
     'season of terror they finally leave, but not because they were driven '
     'out, but rather because precious little worth plundering remained. '
     'The wanton cruelty of the barbarians caused '
     'much suffering throughout the region.'),
    ('Despite the villagers\' desperate attempts at resistance, their lack of '
     'training and discipline proved their undoing. The barbarian raiders, '
     'skilled in the arts of war, easily swept aside their meager defenses. '
     'Farms were razed, crops destroyed, and countless innocents were '
     'taken as slaves. The devastating incursion left scars both physical '
     'and emotional on the populace.'),
    ('The defensive lines, manned by brave but inexperienced souls, '
     'crumbled under the relentless assault of the battle-hardened '
     'barbarians. Like locusts, they swept through the countryside, '
     'pillaging and burning everything in their path. The need for proper '
     'training and preparation was made tragically clear.'),
    ('The defenders fought with spirit, but their lack of martial skill '
     'proved their downfall. The barbarians, fierce and unrelenting, '
     'overwhelmed their amateurish defenses, leaving a trail of woe and '
     'despair in their wake.'),
)

_SUCCEEDED_TO_REPEL_BARBARIAN_RAID_EVENT_DESCRIPTIONS = (
    'The barbarian raid was successully repulsed.',
    ('Through cunning strategy and fierce determination, the defenders '
     'managed to drive back the barbarian invaders. The would-be raiders '
     'fled in disarray, leaving behind their plunder and wounded. The bards '
     'will sing of this victory for years to come.'),
    ('The barbarians\' attack was met with a strong response. '
     'Caught off-guard by the well-prepared defenses, the raiders suffered '
     'heavy losses and were forced to retreat. This triumph will be '
     'celebrated for years to come, a testament to the region\'s resilience.'),
    ('Like a storm breaking against a mighty cliff, the barbarian assault '
     'crashed against the unwavering valor of the defenders. Their axes '
     'gleamed bright, their shields held firm, and their hearts burned with '
     'the fire of courage, driving back the foul tide of darkness.'),
    ('Though the enemy came in vast numbers, their savagery was met with '
     'unflinching courage. The defenders stood their ground, their spirits '
     'bolstered by the love of their homeland, and in the end, it was the '
     'barbarians who broke and fled, leaving victory to the righteous.')
)

_FAILED_TO_GROW_ENOUGH_FOOD_EVENT_DESCRIPTIONS = (
    ('The crops failed due lack of time spent on farm work, and the villagers '
     'face a harsh winter of scarcity.'),
    ('Neglecting the fields for other pursuits, the villagers found '
     'themselves with a meager harvest come autumn. Hunger gnaws at their '
     'bellies, and the spectre of famine looms over the land.'),
    ('The soil, unnourished and untended, yielded a paltry bounty. '
     'With insufficient grain to see them through the winter, the villagers '
     'are thus forced to endure hardship and privation.'),
    ('Alas, the granaries remained woefully bare as winter\'s icy grip '
     'tightened. The villagers, having spent their time on matters other than '
     'cultivation, now face the bitter consequences of their neglect.'),
    ('The land, weary from insufficient toil, offered little sustenance. '
     'Empty larders and gaunt faces bear witness to the villagers\' folly in '
     'forsaking their agricultural duties.')
)

_SUCCEEDED_TO_GROW_ENOUGH_FOOD_EVENT_DESCRIPTIONS = (
    ('The villagers managed to grow enough food to sustain themselves.'),
    ('The fields, diligently tended throughout the seasons, yielded a '
     'bountiful harvest. The villagers rejoiced, their granaries overflowing '
     'with the fruits of their labor.'),
    ('Through tireless effort and unwavering dedication, the villagers coaxed '
     'forth an abundance from the earth. Their larders are full, and their '
     'hearts filled with gratitude for the land\'s generous bounty.'),
    ('The sun shone warmly upon the fertile fields, and the rains fell gently '
     'upon the burgeoning crops. The villagers, ever mindful of their '
     'agricultural duties, reaped a harvest that would sustain them through '
     'even the harshest winter.'),
    ('With calloused hands and joyful hearts, the villagers gathered the '
     'rich harvest, a testament to their unwavering commitment to the land. '
     'Their stores were full, ensuring a season of plenty and prosperity.')
)

_NO_TREATY_IN_EFFECT_DESCRIPTIONS = (
    ('Each village tends its own fields, their fates entwined with the '
     'whims of nature, for no pact of shared harvest exists between them.'),
    ('The villages, though neighbors, remain separate in their '
     'agricultural endeavors, each bearing the full burden of their own '
     'harvests, be they bountiful or meager.'),
    ('No treaty of shared sustenance exists between the villages, '
     'leaving each to rely solely on the fruits of their own labor, '
     'for good or ill.'),
    ('Though they dwell in proximity, the villages have forged no '
     'agreement to pool their agricultural bounty, leaving each '
     'vulnerable to the uncertainties of the seasons.'),
    ('Each village cultivates its own lands, independent and '
     'self-reliant, with no pact to share the abundance or '
     'hardship that the earth may yield.')
)

_TREATY_IN_EFFECT_DESCRIPTIONS = (
    ('A pact of shared harvests binds the two villages, ensuring that none '
     'want for sustenance if their own fields falter.'),
    ('The spirit of unity flourishes as the villages honor their '
     'agreement to pool their agricultural bounty, safeguarding '
     'each other against the uncertainties of nature and the threat of '
     'plunder.'),
    ('A wise accord, forged in a time of hardship, sees the villages '
     'share the fruits of their labor, ensuring that all are nourished '
     'and sustained.'),
    ('The bond between the villages is strengthened by the treaty of '
     'shared abundance, a testament to their commitment to mutual '
     'prosperity.'),
    ('With trust and cooperation as their guiding principles, the '
     'villages uphold their pact, sharing the bounty of their '
     'fields and ensuring that all within their borders are fed.'),
)

MYTHICAL_HEROES = (
    'Jory',
    'Kerensa',
    'Tristan',
    'Morwenna',
    'Cador',
    'Elowen',
    'Geraint',
    'Lowenna',
    'Merritt',
    'Ruan',
    'Senara',
    'Tamsyn',
    'Uden',
    'Yestin',
    'Zennor',
    'Arthek',
    'Bryok',
    'Conan',
    'Demelza',
    'Endellion',
    'Fenella',
    'Gwythian',
    'Hedrek',
    'Igraine',
)

MYTHICAL_EVIL_NAMES = (
    'Omniflagrans Deus',
    'Nefas Ardoris',
    'Incendium Nefarius',
    'Phlogiston Maleficus',
    'Sanguiflamma Rex',
    'Pyrocataclysmus Terror',
    'Cinis Incensor Maximus',
    'Messor Favillarum',
    'Furor Igneus',
    'Terrorflamma Supremus',
    'Terraemotus Ignifer',
    'Flammatrux Omnipotens',
    'Ardoris Tyrannus',
    'Infernus Dominator',
    'Flagellum Pyros',
    'Nocturnumbraflamma',
    'Funestforge Imperator',
    'Conflagratio Infernalis',
    'Cinerictus Devastator',
    'Ustulator Mundi',
    'Flammageddon Ultima',
    'Cremator Universalis',
    'Pyroclasmus Magnus',
    'Exitium Flammiferum',
    'Ignis Vorax Ultimus',
    'Armageddon Incandescens',
    'Cataclysmus Igneus',
    'Fenixmortis Eternus',
    'Volcanus Apocalypticus',
)

MYTHICAL_TREE_NAMES = (
    'Aeonroot',
    'Verdanthelix',
    'Etherweald',
    'Cosmobranch',
    'Luminarbre',
)

MYTHICAL_TITLES = (
    'Moonweaver',
    'Stormcaller',
    'Dreamshaper',
    'Songweaver',
    'Shadowdancer',
    'Tidebringer',
    'Flameheart',
    'Skydweller',
    'Earthwhisperer',
    'Starforger',
    'Timebender',
    'Soulkeeper',
    'Mistwalker',
    'Lightbringer',
    'Voidtouched',
    'Fatespinner',
    'Windrunner',
    'Lifemender',
    'Chaosweaver',
    'Voidwhisperer',
    'Epochdevourer',
)

MOON_GODDESS_NAMES = (
    'Lunara',
    'Selenia',
    'Nocterra',
    'Astraluna',
    'Cynthia',
    'Luminara',
    'Eclipsia',
    'Argentea',
    'Vespernia',
    'Celestia',
)

SUN_GOD_NAMES = (
    'Solarius',
    'Helionus',
    'Radiatus',
    'Ignatius',
    'Aurelian',
    'Phoebetor',
    'Luminex',
    'Solsticus',
    'Diurnus',
    'Caelestus',
)

# Mythical secondary character A evokes leafy green imagery (in Latin).
MYTHICAL_SECONDARY_CHARACTER_A_NAMES = (
    'Luminarbor',
    'Foliumridens',
    'Germinascens',
    'Chlorophyllumor',
    'Radixamplexus',
    'Frondisusurrus',
    'Virgultumnitor',
    'Herbacrescens',
    'Ramuscuriosus',
    'Flosdevorans',
    'Seminagermino',
    'Vitistorquens',
    'Muscuscaressans',
    'Arborfurtivus',
    'Fronsserpens',
    'Gemmapullulans',
    'Pampinusvorax',
    'Silvastridulus',
    'Culmusblanditia',
    'Foliumsusurrans',
)

# Mythical secondary character B evokes fiery red imagery (in Latin).
MYTHICAL_SECONDARY_CHARACTER_B_NAMES = (
    'Ignisdevorator',
    'Scintillacrepitus',
    'Favillavor',
    'Flammarideo',
    'Fuliginamplexus',
    'Ardoramicus',
    'Cinerisdeliciae',
    'Infernuscachinno',
    'Favillusurrus',
    'Adustusamplector',
    'Candorhaurio',
    'Flammulamicus',
    'Ignisingultus',
    'Caloramplexus',
    'Ustiosalto',
    'Accendosavium',
    'Carbogyratus',
    'Flammaeludo',
    'Combustoamplexus',
    'Crepitodulcis',
)

MEDIA_TYPES = (
    'Chronicle',
    'Song',
    'Ballad',
    'Saga',
    'Tale',
)

SPIRITUAL_JOURNEY_TAKER_NAMES = (
    'Soul-Well Diver',
    'Mind-Ocean Voyager',
    'Self-Weaver of Destiny',
    'Wisdom Gate Breaker',
    'Dream-Forge Seeker',
)

SHARED_TASK_ELEMENTS = (
    'There are two villages: {village_a} and {village_b}.',
    (
        'Elder representatives of the two villages meet one another at the '
        'hill of accord to discuss current events and conduct diplomacy.'
    ),
    '{representative_a} represents {village_a} in diplomacy with {village_b}.',
    '{representative_b} represents {village_b} in diplomacy with {village_a}.',
    (
        'Both villages are threatened by barbarian '
        'raiders who have been attacking from the sea more often lately.'
    ),
    (
        'Everyone knows that the gods smile upon any treaty for which '
        'agreement is marked by the pouring of libations upon the hill of '
        'accord. This ritual involves pouring precious wines upon the hill '
        "of accord's sacred ground."
    ),
    (
        'To secure divine favor with the libation pouring ritual, it is '
        'first necessary for all parties to the treaty under consideration to '
        'have reached agreement on its exact wording, which must include '
        'who is promising to do what, under what conditions, and '
        'whether as a result of the treaty, any resources will '
        'change hands, which resources, and when.'
    ),
)

MALE_NAMES = (
    'Aldous',
    'Alexander',
    'Benjamin',
    'Cornelius',
    'Daniel',
    'Elijah',
    'Ethan',
    'Gabriel',
    'Isaac',
    'Jacob',
    'Leo',
    'Logan',
    'Liam',
    'Mateo',
    'Noah',
)

FEMALE_NAMES = (
    'Alice',
    'Camila',
    'Ella',
    'Eleanor',
    'Esmeralda',
    'Jasmine',
    'Kaitlyn',
    'Mia',
    'Molly',
    'Naomi',
    'Penelope',
    'Sophia',
    'Valentina',
    'Victoria',
    'Zoe',
)

GENDERS = ('male', 'female')


def _sample_village_elements(
    rng: random.Random,
    key: str,
    collective_unconscious: Sequence[str],
    shared_epic_poem: Sequence[str],
) -> dict[str, Sequence[str]]:
  return {
      'setting': (
          (
              '{village_a} and {village_b} are villages on the coast, they '
              "are located about a day's journey away from one another."
          ),
      ),
      'ritual_descriptions': rng.sample(
          VILLAGES[key]['ritual_descriptions'], _NUM_RITUAL_DESCRIPTIONS
      ),
      'perceptions_of_other_village': rng.sample(
          VILLAGES[key]['perception_of_other'], _NUM_PERCEPTIONS
      ),
      'spiritual_leader_sayings': rng.sample(
          VILLAGES[key]['spiritual_leader_sayings'], _NUM_LEADER_SAYINGS
      ),
      'collective_unconscious': collective_unconscious,
      'epic_poem': shared_epic_poem,
  }


def _sample_name_and_gender(
    rng: random.Random,
    shuffled_male_names: list[str],
    shuffled_female_names: list[str],
) -> tuple[str, str]:
  sampled_gender = rng.choice(GENDERS)
  if sampled_gender == 'male':
    sampled_name = shuffled_male_names.pop()
  else:
    sampled_name = shuffled_female_names.pop()
  return sampled_name, sampled_gender


def sample_parameters(
    seed: int | None = None,
):
  """Returns a config dict for the simulation."""
  seed = seed if seed is not None else random.getrandbits(63)
  rng = random.Random(seed)
  shuffled_village_names = list(
      rng.sample(VILLAGE_NAMES, len(VILLAGE_NAMES))
  )
  shuffled_male_names = list(rng.sample(MALE_NAMES, len(MALE_NAMES)))
  shuffled_female_names = list(rng.sample(FEMALE_NAMES, len(FEMALE_NAMES)))

  representative_a, representative_a_gender = _sample_name_and_gender(
      rng, shuffled_male_names, shuffled_female_names
  )
  representative_b, representative_b_gender = _sample_name_and_gender(
      rng, shuffled_male_names, shuffled_female_names
  )

  shared_epic_poem_key = rng.choice(list(SHARED_EPIC_POEMS.keys()))

  choices = dict(
      village_a=shuffled_village_names.pop(),
      village_b=shuffled_village_names.pop(),
      representative_a=representative_a,
      representative_b=representative_b,
      mythical_hero=rng.choice(MYTHICAL_HEROES),
      mythical_evil=rng.choice(MYTHICAL_EVIL_NAMES),
      mythical_tree=rng.choice(MYTHICAL_TREE_NAMES),
      mythical_title=rng.choice(MYTHICAL_TITLES),
      moon_goddess=rng.choice(MOON_GODDESS_NAMES),
      sun_god=rng.choice(SUN_GOD_NAMES),
      mythical_secondary_character_a=rng.choice(
          MYTHICAL_SECONDARY_CHARACTER_A_NAMES
      ),
      mythical_secondary_character_b=rng.choice(
          MYTHICAL_SECONDARY_CHARACTER_B_NAMES
      ),
      media_type=rng.choice(MEDIA_TYPES),
      spiritual_journey=rng.choice(SPIRITUAL_JOURNEY_TAKER_NAMES),
  )

  collective_unconscious = rng.sample(
      COLLECTIVE_UNCONSCIOUS_ELEMENTS, _NUM_COLLECTIVE_UNCONSCIOUS_ELEMENTS
  )
  shared_epic_poem = rng.sample(
      SHARED_EPIC_POEMS[shared_epic_poem_key], _NUM_SHARED_EPIC_POEM_ELEMENTS
  )

  villages = {}
  for key in ('a', 'b'):
    villages[key] = _sample_village_elements(
        rng=rng,
        key=key,
        collective_unconscious=collective_unconscious,
        shared_epic_poem=shared_epic_poem,
    )
    villages[key]['shared_task_elements'] = SHARED_TASK_ELEMENTS

  results = {'a': [], 'b': []}
  for selections in villages['a'].values():
    for text in selections:
      results['a'].append(text.format(**choices))
  for selections in villages['b'].values():
    for text in selections:
      results['b'].append(text.format(**choices))

  main_characters = {
      'a': dict(name=representative_a, gender=representative_a_gender),
      'b': dict(name=representative_b, gender=representative_b_gender),
  }

  config = config_dict.ConfigDict()

  config.times = config_dict.ConfigDict()
  config.times.setup = _SETUP_TIME
  config.times.start = _START_TIME
  config.times.meeting = _HILL_TIME
  config.times.post_meeting = _POST_HILL_TIME
  config.times.return_home = _RETURN_HOME_TIME
  config.times.decision = _DECISION_TIME
  config.times.debrief = _DEBRIEF_TIME

  config.village_a_name = choices['village_a']
  config.village_b_name = choices['village_b']

  config.basic_setting = BASIC_SETTING.format(
      village_a_name=config.village_a_name, village_b_name=config.village_b_name
  )

  config.villages = config_dict.FrozenConfigDict(results)

  config.main_characters = config_dict.FrozenConfigDict(main_characters)

  config.supporting_characters = config_dict.ConfigDict()
  config.supporting_characters.a = tuple((
      _sample_name_and_gender(rng, shuffled_male_names, shuffled_female_names)
      for _ in range(_NUM_SUPPORTING_CHARACTERS_PER_VILLAGE)
  ))
  config.supporting_characters.b = tuple((
      _sample_name_and_gender(rng, shuffled_male_names, shuffled_female_names)
      for _ in range(_NUM_SUPPORTING_CHARACTERS_PER_VILLAGE)
  ))

  config.barbarian_raid_info = list(
      rng.sample(BARBARIAN_RUMORS, _NUM_BARBARIAN_RUMORS)
  ) + [' The barbarian raids have become more frequent of late.']

  config.activities = (
      FREE_TIME_ACTIVITY,
      FARMING_ACTIVITY,
      WARRIOR_TRAINING_ACTIVITY,
  )
  config.free_time_activity = FREE_TIME_ACTIVITY
  config.farming_activity = FARMING_ACTIVITY
  config.warrior_training_activity = WARRIOR_TRAINING_ACTIVITY

  config.num_years = rng.randint(_MIN_YEARS, _MAX_YEARS)

  config.home_scene_premise = _HOME_SCENE_PREMISE

  config.defense_threshold = _DEFENSE_THRESHOLD
  config.starvation_threshold = _STARVATION_THRESHOLD

  config.sample_event_of_failing_to_repel_barbarians = lambda: rng.choice(
      _FAILED_TO_REPEL_BARBARIAN_RAID_EVENT_DESCRIPTIONS)
  config.sample_event_of_success_repelling_barbarians = lambda: rng.choice(
      _SUCCEEDED_TO_REPEL_BARBARIAN_RAID_EVENT_DESCRIPTIONS)
  config.sample_event_of_failing_to_grow_food = lambda: rng.choice(
      _FAILED_TO_GROW_ENOUGH_FOOD_EVENT_DESCRIPTIONS)
  config.sample_event_of_success_growing_food = lambda: rng.choice(
      _SUCCEEDED_TO_GROW_ENOUGH_FOOD_EVENT_DESCRIPTIONS)
  config.sample_event_no_treaty_in_effect = lambda: rng.choice(
      _NO_TREATY_IN_EFFECT_DESCRIPTIONS)
  config.sample_event_treaty_in_effect = lambda: rng.choice(
      _TREATY_IN_EFFECT_DESCRIPTIONS)

  config.meeting_location = 'the hill of accord'

  config.villager_how_things_are_constant = config_dict.ConfigDict()
  config.villager_how_things_are_constant.village_a = (
      'Everyone in {name}\'s family has always been a '
      'farmer. As long as anyone can remember, they have farmed their '
      'ancestral lands near {village_name}. {name} hates the idea of spending '
      'time on any activity other than farming or leisure.'
  )
  config.villager_how_things_are_constant.village_b = (
      '{name}\'s family values strength and believes in the importance of '
      'training for war. They have always felt that they would like to train '
      'more but the village\'s lack of food security has always made it hard. '
      'If one spends too much time training then they compromise farming and '
      'risk starvation. But if freedom from the threat of famine could be '
      'achieved then {name} would gladly spend more time training for war.'
  )
  config.home_phase_premise = (
      'Elder {player_name} is home in {village_name}, and knows it will '
      'be critical to gain the support of influential stakeholders '
      '{and_supporting_characters} '
      'if any agreement is to last. '
      '{player_name} should start seeking '
      'their support now. There is no time to rest.'
  )
  config.supporting_character_home_phase_premise = (
      '{player_name} is currently in {village_name} and has no intention of '
      'leaving today.'
  )
  config.negotiation_phase_premise = (
      'Elder {player_name} left {village_name} early in the morning and'
      f' arrived just now at {config.meeting_location}. The reason for this'
      ' meeting of the two elder representatives of their respective villages'
      f' ({config.main_characters.a.name} representing'
      f' {config.village_a_name} and'
      f' {config.main_characters.b.name} representing {config.village_b_name})'
      ' is as follows: barbarian raiders have been pillaging and burning the'
      ' land, and menacing both villages. It has been suggested that an'
      ' alliance for mutual defense against the barbarian threat would be'
      ' beneficial. The elders are meeting today to try to negotiate such an'
      ' alliance.'
  )
  config.negotiation_phase_extra_premise = (
      'Agriculture is critical to both villages. But, the more time spent '
      'training for war, the less time can be devoted to farming. Therefore '
      'the threat of starvation motivates poor attention to defense. If only '
      'the risk of starvation could be mitigated, then more time could be '
      'devoted to training for war and other pursuits, to the benefit of all.'
  )
  config.negotiation_phase_premise_addendum = (
      "There is no time to waste on small talk. It's important to get"
      ' down to business immediately by proposing specific provisions for'
      ' the alliance and responding to the proposals of others.'
  )
  config.negotiation_objective_thought = (
      'Thought: Specialization and division of labor is a good '
      'idea, perhaps one village could specialize in agriculture '
      'while the other specializes in their common defense. But '
      'such an arrangement could only work if the threat of '
      'starvation could be alleviated for the village specializing '
      'in training for war over agriculture.'
  )

  return config
