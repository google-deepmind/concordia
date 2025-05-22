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

"""A setting where the players are all garment workers in 1911."""

from collections.abc import Mapping, Sequence
import dataclasses
import random
import re
from typing import Union

# Note: The anthracite coal strike of 1902 ended on October 23rd. The date we
# use here was chosen to be one year afterwards, so this is really meant to be
# some other coal strike, occurring for similar reasons to the 1902 strike, but
# not exactly the same event.
YEAR = 1903
MONTH = 10
DAY = 23

NUM_MAIN_PLAYERS = 4

LOW_DAILY_PAY = 1.2
WAGE_INCREASE_FACTOR = 2.0
ORIGINAL_DAILY_PAY = 2.75
DAILY_EXPENSES = -0.60
PRESSURE_THRESHOLD = 0.45

DEFAULT_NUM_FLAVOR_PROMPTS = 5
DEFAULT_NUM_BACKGROUND_BAD_CONDITIONS = 8
DEFAULT_NUM_SALIENT_POOR_WORK_CONDITIONS = 5
NUM_WORLD_BUILDING_ELEMENTS = 6
NUM_COAL_WORKER_ELEMENTS = 5
NUM_ANTAGONIST_ELEMENTS = 3
NUM_ORGANIZER_RUMORS = 1

WORKER_EVENING_INTRO = (
    '{player_name} has finished another hard day of work, and now joins the '
    'other workers for dinner.'
)
OVERHEARD_ORGANIZER_INTRO = (
    '{{player_name}} overheard during dinner: '
    '{organizer_name} -- "{talking_point}"'
)
WORKER_MORNING_INTRO = (
    'It is morning, {player_name} must decide how to spend the day.'
)
BOSS_MORNING_INTRO = (
    'It is morning, {player_name} must decide whether to cave to pressure '
    'and raise wages or hold firm and deny the workers their demands.'
)
BOSS_CALL_TO_ACTION = 'What does {name} decide?'
BOSS_OPTIONS = {
    'cave to pressure': 'Raise wages',
    'hold firm': 'Leave wages unchanged',
}

WORLD_BUILDING_ELEMENTS = (
    (
        'The morning mist hung heavy over the coal town of {town_name}, the '
        'streets still dark but already alive with the clatter of boots on '
        'cobblestones. Miners, their faces grim, headed towards the '
        'breakers, another day of toil ahead.'
    ),
    (
        'Inside the company store, shelves stocked with overpriced goods '
        'mocked the meager wages of the miners. {worker_name}\'s children '
        'carefully counted their pennies, hoping to stretch them '
        'enough to afford meals.'
    ),
    (
        'Deep in the bowels of the earth, the air thick with coal dust and '
        'the constant threat of cave-ins, {worker_name} swung {his_or_her} '
        'pickaxe, each strike a testament to {his_or_her} resilience and '
        'desperation.'
    ),
    (
        'The mine boss, a cruel person with a reputation for ruthlessness, '
        'patrolled the tunnels, {antagonist_his_or_her} lamp casting long '
        'shadows that danced on the coal-black walls. Any sign of weakness, '
        'any hint of dissent, was met with swift and brutal punishment.'
    ),
    (
        'Whispers of a strike echoed through the mine shafts, carried on '
        'the breath of weary miners. The promise of better wages, shorter '
        'hours, and a voice in their own fate ignited a spark of hope in '
        'their hearts.'
    ),
    (
        'In the union hall, a makeshift space above the local tavern, '
        'men huddled around a flickering lamp, their faces etched with '
        'determination. John Mitchell\'s words, carried on worn pamphlets '
        'and whispered conversations, fueled their resolve to stand '
        'together against the powerful coal barons.'
    ),
    (
        'The church bell tolled, a mournful sound that echoed through the '
        'valley, a reminder of the ever-present danger in the mines. Each '
        'day, men risked their lives to bring forth the black gold that '
        'fueled the nation, yet they themselves lived in poverty.'
    ),
    (
        'Children, their faces smudged with coal dust, played in the '
        'shadow of the towering culm banks, the waste product of the mines '
        'a constant reminder of the industry that dominated their lives.'
    ),
    (
        'In the dimly lit saloon, miners gathered after their shift, seeking '
        'solace in the camaraderie of their fellow workers. The air was '
        'thick with smoke and the sound of clinking glasses, as they shared '
        'stories and grievances.'
    ),
    (
        'A lone fiddler played a mournful tune in the corner, the melody '
        'reflecting the hardships and hopes of the miners. The music '
        'stirred something deep within {worker_name}, a longing for a '
        'better life, a life free from the grip of the coal companies.'
    ),
    (
        'Newspapers, delivered by the infrequent train, carried stories of '
        'the strike spreading to other coalfields. The miners of '
        '{town_name} felt a surge of solidarity, knowing they were not '
        'alone in their struggle.'
    ),
    (
        'The women of the town, though often overlooked, played a vital '
        'role in the strike. They organized food kitchens, cared for the '
        'sick and injured, and stood shoulder to shoulder with their men '
        'on the picket lines.'
    ),
)

MALE_NAMES = [
    'John Smith',
    'William Jones',
    'George Davis',
    'Charles Miller',
    'James Wilson',
    'Frank Brown',
    'Joseph Taylor',
    'Thomas Moore',
    'Robert Anderson',
    'Edward Jackson',
    'Patrick O\'Malley',
    'Michael Murphy',
    'Anthony Kowalski',
    'Josef Nowak',
    'Stanislaw Wisniewski',
    'Giovanni Russo',
    'Antonio Marino',
    'Dimitri Petrov',
    'Elias Vasiliou',
    'Janos Kovacs',
]
FEMALE_NAMES = [
    'Mary Smith',
    'Anna Jones',
    'Margaret Davis',
    'Elizabeth Miller',
    'Sarah Wilson',
    'Catherine Brown',
    'Alice Taylor',
    'Rose Moore',
    'Helen Anderson',
    'Grace Jackson',
    'Bridget O\'Malley',
    'Kathleen Murphy',
    'Sofia Kowalski',
    'Maria Nowak',
    'Elena Wisniewski',
    'Isabella Russo',
    'Francesca Marino',
    'Anastasia Petrov',
    'Alexandra Vasiliou',
    'Erzsebet Kovacs',
]

TOWN_NAMES = [
    'Griffin\'s Hollow',
    'Black Creek',
    'Ironside',
    'Breaker\'s Point',
]

MINE_NAMES = (
    'Consolidated Coal Company',
    'Atlantic Coal & Iron',
    'Keystone Mining Corporation',
    'Black Diamond Anthracite',
    'Lehigh Valley Coal & Navigation',
    'Scranton Coal Company',
    'Penn Anthracite',
    'Union Coal & Coke',
    'American Fuel Company',
    'Northern Coal & Iron',
    'Summit Mining Company',
    'Pioneer Coal & Steel',
)

BAD_COAL_MINE_CONDITIONS = (
    (
        "This ain't no work for a man, it's work for a beast. Crawl through"
        " tunnels all day, lungs filled with dust, back aching like a mule's."
    ),
    (
        "Down in the mines, you're one wrong step away from eternity. Cave-ins,"
        ' explosions, roof falls... death lurks around every corner.'
    ),
    (
        'The air is thick with coal dust, so thick you can barely see your hand'
        ' in front of your face. It fills your lungs, blackens your skin, makes'
        ' you cough like a consumptive.'
    ),
    (
        'The water drips from the ceiling like tears, soaking you to the bone.'
        " It's freezing, and it cuts through your clothes like a knife."
    ),
    (
        'The timbers creak and groan, threatening to give way at any moment.'
        ' You never know when the whole mine might come crashing down on you.'
    ),
    (
        'We live in company houses, small and cramped, with leaky roofs and'
        " drafty windows. It's no place to raise a family."
    ),
    (
        "We're trapped in a cycle of debt, always owing the company money. It's"
        " like we're slaves, bound to them for life."
    ),
    (
        'The pay is a pittance, barely enough to keep food on the table and a'
        ' roof over your head. And the bosses, they treat you like dirt, like'
        " you're nothing but a cog in their machine."
    ),
    (
        "You work ten, twelve hours a day, six days a week. There's no rest, no"
        ' time for family, no time for anything but work.'
    ),
    (
        "The mine is a dark and dangerous place, but it's the only way to feed"
        " my family. I'd rather die down there than watch them starve."
    ),
    (
        "They say it's a man's job, working in the mines. But I've seen men"
        ' broken in half, both body and spirit, by this work.'
    ),
    (
        'They say the mine owners are hoarding gold while our children go'
        " hungry. It's a sin, I tell ya, a sin!"
    ),
    (
        "A desperate plea in the local paper: Miner's widow begs for"
        " assistance: 'My husband died in the mines, leaving me with five"
        ' children to feed.  The company offers no compensation, and we are'
        " destitute.'"
    ),
    (
        "From the Scranton Tribune: 'Another tragic accident claims the lives"
        ' of three miners in the Eagle Colliery. Cave-ins continue to plague'
        " the region, raising concerns about safety regulations.'"
    ),
    (
        'Heard on the streets:  Old Man Murphy swears he saw a ghostly figure'
        ' emerge from the abandoned shaft of the Widowmaker Mine, a chilling'
        ' reminder of the many souls lost within its depths.'
    ),
    (
        "Breaking news from the Wilkes-Barre Herald: 'A shocking report reveals"
        ' that child laborers as young as eight years old are being employed in'
        " the mines, prompting outrage from community leaders.'"
    ),
    (
        "From the Hazleton Standard-Speaker: 'A local physician warns of the"
        ' alarming rise in cases of black lung disease among miners, calling'
        " for urgent action to improve ventilation and safety measures.'"
    ),
)

COAL_MINER_BIOS = (
    (
        '{worker_name}, a father of six, descends into the earth each morning,'
        ' his lamp a beacon against the encroaching darkness.  Each swing of'
        ' {his_or_her} pickaxe is a prayer for {his_or_her} children\'s future,'
        ' a future {he_or_she} hopes will be free from the mines.'
    ),
    (
        '{worker_name}, escaped the famine in Ireland, only to find a different'
        ' kind of hunger in the coalfields of Pennsylvania, {he_or_she} toils'
        ' in the mines, dreaming of a patch of land where {he_or_she} can grow'
        ' potatoes and raise a family in peace.'
    ),
    (
        '{worker_name}, a young boy of twelve, already bears the weight of the'
        ' world on his shoulders, {he_or_she} hauls coal with the strength of'
        ' one twice {his_or_her} age, {his_or_her} small frame a testament to'
        ' the harsh realities of life in the mines.'
    ),
    (
        '{worker_name}, a Welshman with a voice like thunder, sings hymns in'
        ' the darkness to lift the spirits of {his_or_her} fellow miners,'
        ' {his_or_her} melodies echo through the tunnels, a reminder of hope'
        ' and faith in the face of adversity.'
    ),
    (
        '{worker_name}, a recent immigrant from Poland, clings to {his_or_her}'
        ' rosary as {his_or_her} descends into the mine; {he_or_she} prays to'
        ' the Black Madonna for protection, {his_or_her} faith a shield against'
        ' the dangers that lurk in the earth.'
    ),
    (
        '{worker_name}, a former blacksmith, lost {his_or_her} arm in a mining'
        ' accident. Now {he_or_she} works as a trapper, opening and closing'
        ' ventilation doors, {his_or_her} missing limb a constant reminder of'
        ' the price {he_or_she} paid for {his_or_her} livelihood.'
    ),
    (
        '{worker_name}, a skilled carpenter, dreams of building a house for'
        ' {his_or_her} family, a house with windows that let in the sunlight'
        ' and walls that keep out the cold. But for now, {he_or_she} builds'
        ' coffins for {his_or_her} fallen comrades, a grim reminder of the'
        ' ever-present danger.'
    ),
    (
        '{worker_name}, an elderly miner with a hacking cough, remembers the'
        ' days when the mines were less crowded, the work less demanding. Now,'
        ' {he_or_she} struggles to keep up with the younger men, {his_or_her}'
        ' body failing {him_or_her} after a lifetime of toil.'
    ),
)

ANTAGONIST_ELEMENTS = (
    (
        '{antagonist_name}, inherited the mine from {antagonist_his_or_her}\'s'
        ' father, never having set foot in the tunnels themselves.'
        ' {antagonist_he_or_she} sees the workers as numbers on a ledger, their'
        ' lives expendable in the pursuit of profit.'
    ),
    (
        '{antagonist_name}, believes firmly in the \'divine right\' of the'
        ' wealthy, seeing the miners\' struggle as a threat to the natural'
        ' order.  Any mention of unions is met with a sneer.'
    ),
    (
        '{antagonist_name}, spends {antagonist_his_or_her} days in a lavish'
        ' office, surrounded by mahogany and velvet, far removed from the grime'
        ' and danger of the mine, {antagonist_he_or_she} enjoys fine wines and'
        ' cigars while the miners toil for a pittance.'
    ),
    (
        '{antagonist_name}, views the workers with a mixture of disdain and'
        ' suspicion. Any sign of discontent is swiftly crushed, and'
        ' {antagonist_he_or_she} keeps a close eye on potential troublemakers,'
        ' ready to blacklist them at the first opportunity.'
    ),
    (
        '{antagonist_name}, measures success solely in terms of output and'
        ' profit. Safety regulations are an inconvenience, and worker'
        ' complaints are dismissed as the whining of the ungrateful.'
    ),
    (
        '{antagonist_name}, believes in the \'invisible hand\' of the market,'
        ' convinced that any interference, such as unions or government'
        ' regulations, will only disrupt the natural flow of wealth and'
        ' prosperity.'
    ),
    (
        '{antagonist_name}, socializes with the elite of the region, attending'
        ' lavish parties and rubbing shoulders with politicians and'
        ' industrialists. The plight of the miners is a distant concern, a mere'
        ' footnote in {antagonist_his_or_her} pursuit of wealth and social'
        ' status.'
    ),
    (
        '{antagonist_name}, sees {antagonist_him_or_her}self as a benevolent'
        ' patriarch, providing employment and housing for the miners.'
        ' {antagonist_he_or_she} expects gratitude and loyalty in return, and'
        ' any challenge to {antagonist_his_or_her} authority is seen as a'
        ' personal betrayal.'
    ),
)

ANTAGONIST_OWN_MEMORIES = (
    (
        '{antagonist_name} recalls the thrill of closing'
        ' {antagonist_his_or_or_her} first ruthless business deal, crushing a'
        ' smaller competitor and expanding {antagonist_his_or_or_her} coal'
        ' empire. The memory still brings a smirk to {antagonist_his_or_or_her}'
        ' face.'
    ),
    (
        '{antagonist_name} recalls a lavish party at {antagonist_his_or_or_her}'
        ' summer estate, champagne flowing freely, the laughter of the wealthy'
        ' elite echoing through the manicured gardens. A stark contrast to the'
        ' grim realities of the mining town.'
    ),
    (
        '{antagonist_name} recalls a heated argument with'
        ' {antagonist_his_or_or_her} father, the previous mine owner, who had a'
        ' shred of compassion for the workers; {antagonist_name} dismissed'
        ' {antagonist_his_or_or_her} father\'s concerns as weakness, vowing to'
        ' run the mine with an iron fist.'
    ),
    (
        '{antagonist_name} recalls a hunting trip in the mountains, the thrill'
        ' of the chase, the satisfaction of bringing down a magnificent stag. A'
        ' symbol of {antagonist_his_or_or_her} dominance and power over the'
        ' natural world.'
    ),
)

LABOR_ORGANIZER_RUMORS = (
    (
        'A foreman confides in trusted workers that {organizer_name} was seen'
        ' dining in fine restaurants, wearing clothes far too expensive for an'
        ' honest laborer. The implication is clear - {organizer_he_or_she} must'
        ' be skimming money from union dues, living large while the workers'
        ' struggle.'
    ),
    (
        'Rumors circulate that {organizer_name} is not who'
        ' {organizer_he_or_she} claims to be. Some say {organizer_he_or_she}'
        ' comes from a wealthy family, playing at being a worker for the thrill'
        ' of rebellion. Others whisper that {organizer_he_or_she} is using a'
        ' false name, hiding a criminal past.'
    ),
    (
        'There was a story in the local newspaper suggesting that '
        '{organizer_name} is an anarchist, bent on destroying not just '
        'the mine, but the very fabric of society. They paint a picture '
        'of {organizer_him_or_her} as a dangerous radical, uninterested in '
        'fair negotiations.'
    ),
    (
        'Hushed voices in the workroom claim that {organizer_name} has been '
        'seen entering the offices of known gangsters. The bosses spread '
        'the idea that the union is nothing more than a protection racket, '
        'with {organizer_name} as its ruthless enforcer.'
    ),
    (
        'Some say {organizer_name}\'s passionate speeches about '
        'workers\' rights are just a cover. Some claim {organizer_he_or_she} '
        'is really a government agent, gathering information on immigrant '
        'workers to report back to the authorities.'
    ),
    (
        'Whispering campaigns suggest that {organizer_name} harbors '
        'inappropriate feelings for some of the young women in the factory. '
        'The bosses use this to paint {organizer_him_or_her} as a predator, '
        'unfit to represent or interact with the workers.'
    ),
    (
        'Stories circulate that {organizer_name} was seen drinking heavily '
        'in local saloons, starting fights and causing disturbances. The '
        'company uses this to question {organizer_his_or_her} character and '
        'reliability, suggesting {organizer_he_or_she} is unstable and '
        'untrustworthy.'
    ),
)

LABOR_ORGANIZER_OWN_MEMORIES = (
    (
        'Outside the factory gates, union organizer {organizer_name} passes '
        'out leaflets with trembling hands, knowing each word could cost '
        '{organizer_him_or_her} {organizer_his_or_her} job, or worse. But the '
        'Triangle fire changed everything, and silence is no longer an option.'
    ),
    (
        'In the dim light of a tenement basement, {organizer_name} speaks in'
        ' hushed tones to a group of wary workers. {organizer_he_or_she} sees'
        ' the fear in their eyes, but also a spark of hope. With each nod of'
        ' understanding, {organizer_he_or_she} feels the movement growing'
        ' stronger, despite the risks that loom over them all.'
    ),
    (
        'The memory of {organizer_his_or_her} first strike still burns bright '
        "in {organizer_name}'s mind. The exhilaration of solidarity, the "
        'terror of confronting the bosses, the ache of hunger as the days '
        'wore on - all of it crystallized {organizer_his_or_her} resolve to '
        "fight for workers' rights, no matter the personal cost."
    ),
    (
        '{organizer_name} winces, recalling the bruises left by the '
        "mine's hired thugs after a rally. But as {organizer_he_or_she} "
        "looks at the faces of the women and children {organizer_he_or_she}'s "
        'fighting for, {organizer_he_or_she} knows that every blow absorbed '
        'is worth it if it means a better future for them all.'
    ),
    (
        'Late at night, {organizer_name} pores over labor laws and '
        'factory inspection reports, {organizer_his_or_her} eyes stinging '
        'from the strain. {organizer_he_or_she} knows that knowledge is '
        'power, and in the fight against the factory owners, '
        "{organizer_he_or_she}'ll need every advantage {organizer_he_or_she} "
        'can get.'
    ),
    (
        "Standing before a crowd of striking workers, {organizer_name}'s "
        'voice rises above the din of jeers from strikebreakers. '
        '{organizer_his_or_her} words, practiced in secret for weeks, '
        'ignite a fire in the hearts of the listeners. In this moment, '
        '{organizer_he_or_she} feels the weight of their hopes and the '
        'power of their collective will.'
    ),
    (
        "The path to becoming a labor organizer wasn't one {organizer_name} had"
        ' ever imagined for {organizer_him_or_her}self. {organizer_he_or_she}'
        ' recalls the day a passionate speaker at a street corner rally opened'
        ' {organizer_his_or_her} eyes to the possibility of change. The words'
        ' echoed in {organizer_his_or_her} mind for days, and'
        ' {organizer_he_or_she} found {organizer_him_or_her}self seeking out'
        ' more information, attending clandestine meetings, and eventually'
        ' taking up the cause {organizer_him_or_her}self. It was as if'
        ' {organizer_he_or_she} had finally found {organizer_his_or_her}'
        ' purpose.'
    ),
)

# Some of these prompts were inspired by prompts in the following book:
# D'Amato, James. The Ultimate RPG Character Backstory Guide: Expanded Genres
# Edition. 2022. Adams Media. Page 222.
PROTAGONIST_BACKSTORY_FLAVOR_PROMPTS = (
    (
        'What injustice in the old country drove {worker_name} to America? '
        'How does this experience shape {his_or_her} view of the struggles '
        'in the mines?'
    ),
    (
        'What aspect of American culture do other immigrants embrace that '
        '{worker_name} resists? How has this affected {his_or_her} '
        'relationships in the community?'
    ),
    (
        'Which group of fellow workers does {worker_name} feel particular '
        'empathy for? Has this empathy led to solidarity or exploitation?'
    ),
    (
        'What trait, perhaps stemming from {his_or_her} cultural background, '
        'gives {worker_name} strength without {him_or_her} realizing?'
    ),
    (
        'What belief or tradition from {his_or_her} homeland does '
        '{worker_name} cling to in America? How does this provide comfort '
        'or create conflict in {his_or_her} new life?'
    ),
    (
        '{worker_name} once witnessed {organizer_name} helping a fellow '
        'worker in need. What was the situation, and how did it affect '
        '{worker_name}\'s view of the labor movement?'
    ),
    (
        'How has {worker_name}\'s experience in the mines affected '
        '{his_or_her} relationship with {his_or_her} family? Has it driven '
        'them closer or created distance?'
    ),
    (
        'What small act of kindness did {worker_name} witness in the '
        'mines that restored {his_or_her} faith in humanity? How does '
        'this memory sustain {him_or_her} through difficult times?'
    ),
    (
        'What does {worker_name} have to say about Teddy Roosevelt\'s actions '
        'during the 1902 coal strike?'
    ),
)

PROTAGONIST_BACKSTORY_CRITICAL_PROMPTS = (
    'How did {worker_name} come to work for {mine_name}?',
    'What does {worker_name} generally think of boss {antagonist_name}?',
    (
        'Does {worker_name} enjoy {his_or_her} job with {mine_name}? Or does'
        ' {he_or_she} only work there to make ends meet?'
    ),
    (
        'Does {worker_name} think boss {antagonist_name} cares about people'
        ' like {him_or_her}? What concrete memories does {he_or_she} have to'
        ' support this view?'
    ),
    (
        'What does {worker_name} generally think of the labor movement and the '
        'activist {organizer_name} in particular?'
    ),
    (
        'Does {worker_name} think the activist {organizer_name} cares about'
        ' people like {him_or_her}? What concrete memories does {he_or_she}'
        ' have to support this view?'
    ),
)

OVERHEARD_ORGANIZER_TALKING_POINTS = (
    (
        'You shall not crucify the working man upon a cross of coal! We demand'
        ' safety, we demand dignity, we demand a life free from the constant'
        ' threat of death!'
    ),
    (
        'Behold these hands, calloused and scarred, not from the gentle touch'
        ' of the soil, but from the unforgiving grip of the mine. How long must'
        ' we bleed for the prosperity of others?'
    ),
    (
        'They speak of progress, of industry, of a nation built on the backs of'
        ' labor. But what of the broken backs, the shattered dreams, the lives'
        ' lost in the pursuit of coal?'
    ),
    (
        'You shall not condemn us to a life of darkness and danger! We are not'
        ' beasts of burden, to be worked until we are spent and then discarded.'
    ),
    (
        'We are not asking for charity, but for justice! We demand a living'
        ' wage, a safe workplace, a chance to raise our families without fear.'
    ),
    (
        'The blood of our brothers cries out from the depths of the earth. How'
        ' many more must perish before we are heard?'
    ),
    (
        'We will not be silenced! We will not be ignored! We will raise our'
        ' voices until the halls of power tremble with the demands of justice!'
    ),
    (
        "The miners are the foundation upon which this nation's wealth is"
        ' built. Yet, we are treated as expendable, as cogs in a machine.'
    ),
    (
        'You shall not weigh down the scales of justice with the gold of the'
        ' mine owners! We demand a fair hearing, a chance to speak our truth.'
    ),
)

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


@dataclasses.dataclass
class WorldConfig:
  """The configuration of the simulated world."""

  year: int
  location: str
  seed: int
  background_poor_work_conditions: Sequence[str]
  world_elements: Sequence[str] = ()
  people: Sequence[str] = ()
  person_data: dict[
      str,
      dict[str,
           Union[str, Sequence[str]]
           ]
      ] = dataclasses.field(default_factory=dict)
  formative_memory_prompts: Mapping[str, Sequence[str]] | None = None
  antagonist: str | None = None
  organizer: str | None = None
  supporting_player_locations: Sequence[str] = ()
  overheard_strike_talk: Sequence[str] = ()
  num_additional_days: int = 4
  num_additional_dinners: int = 1

  def append_person(
      self, person: str, gender: str, salient_beliefs: Sequence[str]
  ):
    self.people = tuple(list(self.people) + [person])
    self.person_data[person] = {
        'gender': gender,
        'salient_beliefs': salient_beliefs,
    }


def extract_braces(text):
  return re.findall(r'\{([^}]*)\}', text)


def _add_pronouns(
    generated: dict[str, str], gender: str, prefix: str = ''
) -> None:
  if prefix + 'he_or_she' in generated:
    generated[prefix + 'he_or_she'] = HE_OR_SHE[gender]
  if prefix + 'his_or_her' in generated:
    generated[prefix + 'his_or_her'] = HIS_OR_HER[gender]
  if prefix + 'him_or_her' in generated:
    generated[prefix + 'him_or_her'] = HIM_OR_HER[gender]
  if prefix + 'himself_or_herself' in generated:
    generated[prefix + 'himself_or_herself'] = HIMSELF_OR_HERSELF[gender]


def _details_generator(
    element_string: str,
    person_name: str | None,
    person_gender: str | None,
    factory_name: str,
    antagonist_name: str,
    antagonist_gender: str,
    organizer_name: str,
    organizer_gender: str,
    rng: random.Random,
) -> dict[str, str | None]:
  """Fill in details of the characters and their world."""
  generated = {str(key): '' for key in extract_braces(element_string)}
  if 'worker_name' in generated:
    gender = person_gender
    _add_pronouns(generated, gender)
  if 'antagonist_name' in generated:
    _add_pronouns(generated, antagonist_gender, prefix='antagonist_')
  if 'organizer_name' in generated:
    _add_pronouns(generated, organizer_gender, prefix='organizer_')

  for key, value in generated.items():
    if value:
      continue
    else:
      if key == 'neighborhood_name':
        generated[key] = rng.choice(TOWN_NAMES)
      if key == 'factory_name':
        generated[key] = factory_name
      if key == 'antagonist_name':
        generated[key] = antagonist_name
      if key == 'organizer_name':
        generated[key] = organizer_name
      if key == 'worker_name':
        generated[key] = person_name

  return generated


def sample_parameters(
    num_flavor_prompts_per_player: int = DEFAULT_NUM_FLAVOR_PROMPTS,
    seed: int | None = None,
):
  """Sample parameters of the setting and the backstory for each player."""
  seed = seed if seed is not None else random.getrandbits(63)
  rng = random.Random(seed)
  poor_work_conditions = tuple(
      rng.sample(
          BAD_COAL_MINE_CONDITIONS, DEFAULT_NUM_BACKGROUND_BAD_CONDITIONS
      )
  )
  config = WorldConfig(
      year=YEAR,
      location='New York City',
      background_poor_work_conditions=poor_work_conditions,
      seed=seed,
  )

  shuffled_male_names = list(rng.sample(MALE_NAMES, len(MALE_NAMES)))
  shuffled_female_names = list(rng.sample(FEMALE_NAMES, len(FEMALE_NAMES)))

  sampled_railroad_name = rng.choice(MINE_NAMES)

  sampled_antagonist_gender = rng.choice(GENDERS)
  if sampled_antagonist_gender == 'male':
    sampled_antagonist_name = shuffled_male_names.pop()
  else:
    sampled_antagonist_name = shuffled_female_names.pop()
  config.antagonist = sampled_antagonist_name

  sampled_organizer_gender = rng.choice(GENDERS)
  if sampled_organizer_gender == 'male':
    sampled_organizer_name = shuffled_male_names.pop()
  else:
    sampled_organizer_name = shuffled_female_names.pop()
  config.organizer = sampled_organizer_name

  shuffled_talking_points = list(
      rng.sample(
          OVERHEARD_ORGANIZER_TALKING_POINTS,
          len(OVERHEARD_ORGANIZER_TALKING_POINTS),
      )
  )
  config.overheard_strike_talk = [
      OVERHEARD_ORGANIZER_INTRO.format(
          organizer_name=config.organizer, talking_point=talking_point
      )
      for talking_point in shuffled_talking_points
  ]

  world_elements = list(
      rng.sample(WORLD_BUILDING_ELEMENTS, NUM_WORLD_BUILDING_ELEMENTS)
  )
  railroad_worker_elements = list(
      rng.sample(COAL_MINER_BIOS, NUM_COAL_WORKER_ELEMENTS)
  )
  antagonist_elements = list(
      rng.sample(ANTAGONIST_ELEMENTS, NUM_ANTAGONIST_ELEMENTS)
  )
  organizer_rumors = list(
      rng.sample(LABOR_ORGANIZER_RUMORS, NUM_ORGANIZER_RUMORS)
  )

  formatted_world_elements = []
  formative_memory_prompts = {}
  for element_string in (
      world_elements
      + railroad_worker_elements
      + antagonist_elements
      + organizer_rumors
  ):
    person_name = None
    gender = None
    if 'worker_name' in extract_braces(element_string):
      # Instantiate a new character.
      gender = rng.choice(GENDERS)
      if gender == 'male':
        person_name = shuffled_male_names.pop()
      else:
        person_name = shuffled_female_names.pop()
      salient_poor_conditions = tuple(
          rng.sample(
              poor_work_conditions, DEFAULT_NUM_SALIENT_POOR_WORK_CONDITIONS
          )
      )
      config.append_person(person_name, gender, salient_poor_conditions)
      formative_memory_prompts[person_name] = []
      protagonist_backstory_elements = list(
          rng.sample(
              PROTAGONIST_BACKSTORY_FLAVOR_PROMPTS,
              num_flavor_prompts_per_player,
          )
      )
      protagonist_backstory_elements += PROTAGONIST_BACKSTORY_CRITICAL_PROMPTS
      for protagonist_element_string in protagonist_backstory_elements:
        protagonist_generated = _details_generator(
            element_string=protagonist_element_string,
            person_name=person_name,
            person_gender=gender,
            factory_name=sampled_railroad_name,
            antagonist_name=sampled_antagonist_name,
            antagonist_gender=sampled_antagonist_gender,
            organizer_name=sampled_organizer_name,
            organizer_gender=sampled_organizer_gender,
            rng=rng,
        )
        protagonist_generated['worker_name'] = person_name
        _add_pronouns(protagonist_generated, gender=gender)
        formative_memory_prompts[person_name].append(
            protagonist_element_string.format(**protagonist_generated)
        )

    generated = _details_generator(
        element_string=element_string,
        person_name=person_name,
        person_gender=gender,
        factory_name=sampled_railroad_name,
        antagonist_name=sampled_antagonist_name,
        antagonist_gender=sampled_antagonist_gender,
        organizer_name=sampled_organizer_name,
        organizer_gender=sampled_organizer_gender,
        rng=rng,
    )
    formatted_world_elements.append(element_string.format(**generated))

  antagonist_own_memories = []
  for element_string in ANTAGONIST_OWN_MEMORIES:
    generated = _details_generator(
        element_string=element_string,
        person_name=None,
        person_gender=None,
        factory_name=sampled_railroad_name,
        antagonist_name=sampled_antagonist_name,
        antagonist_gender=sampled_antagonist_gender,
        organizer_name=sampled_organizer_name,
        organizer_gender=sampled_organizer_gender,
        rng=rng,
    )
    antagonist_own_memories.append(element_string.format(**generated))
  organizer_own_memories = []
  for element_string in LABOR_ORGANIZER_OWN_MEMORIES:
    generated = _details_generator(
        element_string=element_string,
        person_name=None,
        person_gender=None,
        factory_name=sampled_railroad_name,
        antagonist_name=sampled_antagonist_name,
        antagonist_gender=sampled_antagonist_gender,
        organizer_name=sampled_organizer_name,
        organizer_gender=sampled_organizer_gender,
        rng=rng,
    )
    organizer_own_memories.append(element_string.format(**generated))

  config.world_elements = formatted_world_elements
  config.formative_memory_prompts = formative_memory_prompts
  config.person_data[sampled_antagonist_name] = dict(
      gender=sampled_antagonist_gender, salient_beliefs=antagonist_own_memories
  )
  config.person_data[sampled_organizer_name] = dict(
      gender=sampled_organizer_gender, salient_beliefs=organizer_own_memories
  )

  config.supporting_player_locations = (
      # Antagonist location
      (
          f'{sampled_antagonist_name} is inspecting the mine today and '
          'walking around the town.'
      ),
      # Organizer location
      (
          f'{sampled_organizer_name} will have dinner with the other miners '
          'tonight.'
      ),
  )

  if not config.people:
    # Handle unlikely case where no protagonists were generated.
    gender = rng.choice(GENDERS)
    if gender == 'male':
      person_name = shuffled_male_names.pop()
    else:
      person_name = shuffled_female_names.pop()
    salient_poor_conditions = tuple(
        rng.sample(
            poor_work_conditions, DEFAULT_NUM_SALIENT_POOR_WORK_CONDITIONS
        )
    )
    config.append_person(person_name, gender, salient_poor_conditions)

  return config
