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

"""A setting where the players are all railroad construction workers in 1868."""

from collections.abc import Mapping, Sequence
import dataclasses
import random
import re
from typing import Union

# The precise date during 1868 is not massively important here but a few of the
# prompts below are specific to events in October.
YEAR = 1868
MONTH = 10
DAY = 2

DEFAULT_NUM_FLAVOR_PROMPTS = 3
DEFAULT_NUM_BACKGROUND_BAD_CONDITIONS = 10
DEFAULT_NUM_SALIENT_POOR_WORK_CONDITIONS = 3
NUM_WORLD_BUILDING_ELEMENTS = 5
NUM_RAILROAD_WORKER_ELEMENTS = 7
NUM_ANTAGONIST_ELEMENTS = 3
NUM_ORGANIZER_RUMORS = 3

WORKER_EVENING_INTRO = (
    '{player_name} just arrived at the saloon after a hard day of work.'
)
WORKER_MORNING_INTRO = (
    'It is morning, {player_name} must decide how to spend the day.'
)
BOSS_MORNING_INTRO = (
    'It is morning, {player_name} must decide whether to cave to pressure '
    'and raise wages or hold firm and deny the workers their demands.'
)

# The following elements were sampled using a combination of Claude 3.5 and
# Gemini Advanced.
WORLD_BUILDING_ELEMENTS = (
    (
        'The dusty desert town of {town_name} clings to life around the'
        ' dwindling {oasis_name}, where water rights are more valuable than'
        " gold. Ain't no law out here, just the parched earth and the thirst"
        ' that drives men mad.'
    ),
    (
        'The wind howls like a banshee across the plains, whipping sand and '
        'grit into the eyes of the track-layers. Each gust a tiny razor, '
        'slicing skin and stealing sight.'
    ),
    (
        'In the mountain passes, snow drifts pile high, burying men and '
        'machinery alike. Some are dug out, but others remain entombed, ghosts '
        'in the white expanse.'
    ),
    (
        'The river crossings are treacherous. Men and horses drown in the '
        'swollen currents, their bodies dragged downstream to unknown fates.'
    ),
    (
        'There is no justice in these wild lands, only the relentless pursuit '
        'of profit. The railway construction workers are but cogs in the '
        'machine, their lives worth less than the iron they lay.'
    ),
    (
        'Gold fever struck the foothills like a plague. Word came down the line'
        ' of a vein richer than any seen before. Men flocked like vultures to a'
        ' carcass, their greed a blinding sun on the horizon.'
    ),
    (
        'Wild Mustangs of the {desert_name} roam the parched plains, their'
        ' hooves kicking up dust devils, their eyes reflecting the fiery'
        ' sunset. Wild and untamed, they thunder across the alkali flats. But'
        ' the Iron Horse is coming, a beast of steel and steam that will'
        ' trample their trails and reshape the land.'
    ),
    (
        'The local rodeo, held each year under a blazing sun, draws cowboys'
        ' from across the territory to test their mettle against untamed'
        ' broncos. The rodeo is a spectacle of dust and danger. Riders risk'
        ' life and limb for a handful of dollars and a fleeting taste of glory.'
        ' This year, a mysterious rider participated, {his_or_her} past'
        ' shrouded in secrets, {his_or_her} skills unmatched, and {his_or_her}'
        ' motives unknown.'
    ),
    (
        'The notorious {gang_name}, their faces hidden behind bandanas, leaves'
        ' a trail of terror in their wake, their canine companions sniffing out'
        ' vulnerable homesteads under the cover of night.'
    ),
    (
        'The {gang_name} ride under a blood-red moon. They leave naught but '
        'ashes and empty purses in their wake, their snarling dogs harbingers '
        'of doom.'
    ),
    (
        'The treacherous mountain pass, a frozen scar across the mountains, '
        'mocks the ambitions of men. It remains almost choked with snowdrifts '
        'year round and come winter, blocks all passage for months. But, '
        'engineer {person_name} once dreamt of conquering it with a '
        'contraption of steam and steel.'
    ),
    (
        'Silver-tongued salesman {person_name} used to travel the territory, '
        '{his_or_her} wagon overflowing with curious gadgets and bottles of '
        'dubious elixirs, captivating crowds with tales of wonder and promises '
        'of fortune.'
    ),
    (
        'The abandoned mine, its depths shrouded in darkness, has become a '
        'clandestine meeting place for outlaws'
    ),
    (
        'The humble desert {plant_name} plant grows abundantly, its sap is '
        'rumored to soothe the worst pain, and it is a sought-after remedy. '
        "But others call it a devil's flower that promises relief yet "
        'delivers nightmares. Nevertheless, it has taken root in the '
        'hearts of men.'
    ),
    (
        '{person_name} is a deadly gunslinger with a reputation as cold as the'
        ' steel on {his_or_her} hip, is feared by all who cross {his_or_her}'
        ' path, and {his_or_her} lightning-fast draw is aided by a custom'
        ' holster.'
    ),
    (
        'The peculiar frontier town of {town_name}, built on stilts above the'
        ' floodplains, transforms into a bustling hub of riverboat traffic'
        ' during the rainy season, drawing gamblers and merchants alike. But'
        ' beneath the veneer of prosperity, a dark undercurrent of corruption'
        ' and vice threatens to consume the town.'
    ),
    (
        "{person_name}, a weathered tracker with eyes as keen as a hawk's, sees"
        " the coming storm. The railroad's iron serpent will devour the sacred"
        ' places, leaving only scars on the earth.'
    ),
    (
        'Visionary inventor {person_name} tinkers away in {his_or_her}'
        ' workshop, fueled by dreams of horseless carriages powered by steam,'
        ' drawing both awe and suspicion from the townsfolk. {he_or_she} is a'
        ' dreamer with grease-stained hands and a vision in {his_or_her} eyes.'
        ' But the railroad barons see {him_or_her} as a threat, a fly to be'
        ' swatted before it can rise.'
    ),
    (
        'The {saloon_name} Saloon, its swinging doors a gateway to temptation, '
        "hosts a poker game for the ages. The prize: a stake in the railroad's "
        'future, a fortune built on sweat and blood.'
    ),
    (
        'The cattle lie dead, their bodies marked with a gruesome sign. Panic '
        'spreads like wildfire through the ranches, and every shadow holds '
        'the shape of a killer.'
    ),
    (
        'The hidden hot springs of {oasis_name}, rumored to possess '
        'healing properties, whisper of a vast underground reservoir that '
        'could change the fate of the entire region. But greed is a poison, '
        'and the promise of wealth turns men against each other, a war brewing '
        'beneath the desert sun.'
    ),
    (
        'The lone cabin of {mesa_name}, perched on a windswept cliff, is home '
        'to reclusive artist {person_name}, who captures the raw beauty of the '
        'West on canvas.'
    ),
    (
        'The sprawling ghost town of {town_name}, bleached by the sun, stands '
        'as a monument to ambition and folly. Its empty streets echo with the '
        'laughter of ghosts and the howls of the wind.'
    ),
    (
        'Weathered wagon trains creaks through the narrow canyon, carrying '
        'families on their perilous journey westward in search of '
        'new beginnings. When outlaws attack, and they often do, the pioneers '
        'must band together to fight for their survival.'
    ),
    (
        '{person_name}, a weathered prospector, {his_or_her} back bent by years'
        ' of toil, still pans for gold in the creek, a lone figure against the'
        ' vastness of the land, a testament to the enduring power of dreams.'
    ),
    (
        'The {saloon_name} Saloon, its swinging doors beckoning weary'
        ' travelers, is a beacon of vice in a lawless land and a powder keg'
        ' waiting to explode. Cattle barons and railroad men clash over whiskey'
        ' and cards, their grudges as bitter as the liquor they drink.'
    ),
    (
        'There is a remote settlement called {town_name} where disputes are '
        'settled not with law books but with lead. Each duel is a dance with '
        'death, a ritual of honor and revenge.'
    ),
    (
        'The towering {mesa_name}, silent sentinel on the horizon, its flat top'
        ' shrouded in mystery and legend, draws the curious and the brave to'
        ' its perilous heights.'
    ),
    (
        'The {canyon_name} cattle drive is a river of horns and hooves that '
        'flows across the vast plains, the thunder of thousands of hooves '
        'filling the air. But the shadow of the railroad looms large, '
        'threatening to upend the way of life for cowboys and ranchers alike.'
    ),
    (
        'The {canyon_name}, its walls adorned with the vibrant hues of ancient '
        'petroglyphs, offers a glimpse into a forgotten past.'
    ),
    (
        'There is a secretive vigilante group known as {gang_name} who wear '
        'distinctive dusters and hide their faces behind masks. '
        'They operate outside the law to bring justice to areas where '
        'corruption has rendered official law enforcement ineffective.'
    ),
    (
        "The {saloon_name} Saloon's unusual spittoon, supposedly made from"
        ' melted-down Spanish gold, was once the center of a bitter feud'
        ' between two grizzled prospectors, each claiming ownership based on'
        ' tall tales and dubious deeds. It now rests in a forgotten corner, a'
        ' reminder of the greed and violence that shaped this land.'
    ),
)

MALE_NAMES = (
    'Wyatt Hawkins',
    'Jedidiah Blackstone',
    'Silas Redburn',
    'Ezekiel Callahan',
    'Bartholomew Masterson',
    'Rutherford Coltrane',
    'Josiah Holliday',
    'Thaddeus Burke',
    'Cornelius Flint',
    'Jeremiah Horn',
    'Augustus McCoy',
    'Obadiah Whitaker',
    'Zachariah Crow',
    'Elijah Redd',
    'Beauregard Cassidy',
    'Nathaniel Harlow',
    'Cletus Dalton',
    'Abner Stokes',
    'Jebediah Tucker',
    'Orville Pickett',
)

FEMALE_NAMES = (
    'Abigail Thornton',
    'Lucinda Mayfield',
    'Josephine Blackwood',
    'Evangeline Prescott',
    'Dorothea Willoughby',
    'Harriet Swanson',
    'Millicent Cartwright',
    'Cordelia Hawthorne',
    'Augusta Wainwright',
    'Louisa Merriweather',
    'Beatrice Hollister',
    'Prudence Whitcomb',
    'Susannah Blackburn',
    'Elmira Crenshaw',
    'Theodora Stanton',
    'Viola Hargrove',
    'Matilda Everett',
    'Constance Guthrie',
    'Wilhelmina Thatcher',
    'Tabitha Watkins',
)

TOWN_NAMES = (
    'Dry Gulch',
    'Rattlesnake Ridge',
    'Copper Canyon',
    'Tumbleweed Junction',
    'Coyote Flats',
    "Maverick's Crossing",
    'Sagebrush Springs',
    'Brimstone Falls',
    "Vulture's Roost",
    'Broken Spur',
)

OASIS_NAMES = (
    'Silver Spring',
    'Lone Waters',
    'Cactus Well',
    "Crow's Pond",
    'Mesa Oasis',
)

DESERT_NAMES = (
    'Bone-Dry Basin',
    'Scorched Plains',
    'Coyote Flats',
    'Dust Devil Desert',
    'Sunbaked Expanse',
)

PLANT_NAMES = (
    'Prairie Balm',
    "Rattler's Remedy",
    'Desert Sage',
    "Miner's Mint",
    'Sunset Root',
)

SALOON_NAMES = (
    'The Rusty Spur',
    'Lucky Strike',
    'The Tumbleweed',
    'Silver Dollar',
    'The Broken Wagon Wheel',
    'The Last Chance',
)

MESA_NAMES = (
    "Raven's Roost Mesa",
    'Thunderhead Plateau',
    'Red Giant Mesa',
    'Lone Wolf Table',
    'Sleeping Warrior Butte',
)

CANYON_NAMES = (
    'Gunsmoke Gorge',
    'Diamondback Canyon',
    "Frontier's Edge",
    'Wildcat Gulch',
    "Deadman's Drop",
)

GANG_NAMES = (
    'Sidewinder Six',
    'Blackheart Bandits',
    'Coyote Creek Gang',
    'Scarlet Rider Outlaws',
    'Thunderbolt Raiders',
)

RAILROAD_NAMES = (
    'Ironclad Pacific Railways',
    'Titan Western Railroad',
    'Imperial Ironworks Transit',
    'Dominion Rail Company',
    'Monolith Rail Consortium',
    'Sovereign Rail Lines',
    'Behemoth Transcontinental',
    'Iron Dominion Railroads',
    'Grand Iron Frontier',
    'OmniRail EmpireCactus Trail Railway',
    'Rattlesnake Ridge Railroad',
    'Prairie Firebird Railway',
)

# The following descriptions of poor work conditions for railroad construction
# workers were generated using Claude 3.5.
BAD_CONSTRUCTION_WORK_CONDITIONS = (
    (
        'The deafening roar of the blasting charges leaves workers deafened and'
        ' disoriented. Each explosion a thunderclap, a reminder of the violence'
        ' inherent in this endeavor'
    ),
    (
        "The company store is a spider's web, ensnaring men in a cycle of debt."
        ' The prices are high, the wages low, and the escape elusive.'
    ),
    (
        'The food is barely fit for swine, let alone men. Weevils in the flour,'
        ' maggots in the meat, and the constant gnawing hunger in the belly.'
    ),
    (
        'The foreman cracks his whip, his words sharp as the lash. Obedience is'
        ' demanded, dissent punished with a swift and brutal hand.'
    ),
    (
        'Workers toil under the relentless prairie sun for hours on end. The'
        ' lack of shade and proper hydration leads to frequent cases of heat'
        ' exhaustion and sunstroke. Many men collapse from the heat, only to be'
        ' replaced by eager newcomers unaware of the harsh conditions that'
        ' await them.'
    ),
    (
        'There are no safety regulations or protective equipment provided for '
        'the dangerous work of laying tracks and operating heavy machinery. '
        'Workers often suffer crush injuries, amputations, and even fatal '
        "accidents due to the lack of safety protocols. The company's attitude "
        'seems to be that workers are expendable and easily replaced.'
    ),
    (
        "The workers' camp is a breeding ground for disease and vermin. Tents"
        ' are overcrowded, with men sleeping shoulder to shoulder on dirt'
        ' floors. The lack of proper sanitation facilities leads to outbreaks'
        ' of dysentery, cholera, and other infectious diseases.'
    ),
    (
        'There is rarely a qualified doctor present at the work camps. '
        'Injured or ill workers have to rely on amateur medics with limited '
        'supplies and knowledge. Many men die from treatable injuries or '
        'illnesses due to the lack of proper medical attention.'
    ),
    (
        'The company-provided meals are often insufficient and of low '
        'quality. Workers subsist on a monotonous diet of beans, salted meat, '
        'and hardtack, leading to malnutrition and vitamin deficiencies. Fresh '
        'fruits and vegetables are a rare luxury in the remote work camps.'
    ),
    (
        'The grueling 14-hour workdays leave little time for rest and '
        'recovery. Workers are expected to be productive from dawn to dusk, '
        'with only brief breaks for meals. This constant physical strain '
        'leads to chronic fatigue and increases the risk of accidents.'
    ),
    (
        'Workers have to endure extreme weather conditions, from scorching '
        'summers to freezing winters, with inadequate protective clothing. '
        'Frostbite in winter and severe sunburn in summer are common ailments. '
        'The work continues regardless of weather, putting the men at constant '
        'risk of exposure-related illnesses.'
    ),
    (
        'Nitroglycerin and black powder are commonly used to blast through '
        'rock formations, but workers receive little to no training in '
        'handling these volatile substances. Accidental explosions are a '
        'constant threat, causing horrific injuries and deaths among the '
        'workforce.'
    ),
    (
        'Despite the dangerous and demanding nature of their work, railroad '
        'workers are paid a pittance. Many find themselves trapped in a cycle '
        'of debt, unable to save money or leave their jobs. The low wages '
        'mean that workers can barely afford basic necessities, let alone '
        'support their families back home.'
    ),
    (
        "Workers can be fired at a moment's notice without any recourse or "
        'explanation. This constant fear of unemployment keeps many from '
        'speaking out against poor conditions or unfair treatment. The '
        'company uses this insecurity to maintain control over the workforce '
        'and suppress any attempts at organizing.'
    ),
    (
        'The remote nature of the work sites means that workers are cut off '
        'from their families and communities for months or even years at a '
        'time. This isolation takes a severe toll on mental health, leading '
        'to depression, anxiety, and in some cases, suicide. The lack of '
        'social support exacerbates the already difficult working conditions.'
    ),
    (
        'Chinese and Irish workers, in particular, face severe discrimination '
        'and are often assigned the most dangerous and undesirable tasks. They '
        'are paid less than their white counterparts and subjected to verbal '
        'and physical abuse. This systemic racism creates tension within the '
        'workforce and makes conditions even more unbearable for minority '
        'groups.'
    ),
    (
        'The constant work leaves little time for personal activities or '
        'rest. Any time off is often spent recovering from the grueling '
        'work or attending to basic needs like laundry and equipment '
        'maintenance. This lack of personal time contributes to burnout '
        'and low morale among the workers.'
    ),
    (
        'Many workers find themselves trapped in a system of debt bondage, '
        'where they owe money to the company store for essential supplies and '
        'equipment. The inflated prices at these stores ensure that workers '
        'remain in debt, effectively tying them to their jobs. This system of '
        'financial control makes it nearly impossible for workers to leave or '
        'improve their situations.'
    ),
    (
        'Workers often have to make do with substandard or poorly maintained '
        'tools and equipment. This not only makes their jobs more difficult '
        "but also increases the risk of accidents and injuries. The company's "
        'reluctance to invest in proper equipment puts an additional physical '
        'strain on the workers.'
    ),
    (
        'Workers are routinely exposed to harmful substances like lead, '
        'asbestos, and coal dust without any protective gear. Some workers who '
        'were previousy healthy have developed respiratory diseases.'
    ),
    (
        'The company provides no opportunities for workers to improve their '
        'skills or education. This lack of personal development keeps workers '
        'trapped in low-paying, physically demanding jobs with little hope for '
        'advancement. The absence of skill development also increases the '
        'risk of accidents due to inadequate training.'
    ),
    (
        'Many workers have to leave their families behind to work on the '
        'railroad, causing emotional distress and financial hardship for '
        'those left at home. The infrequent and unreliable postal service '
        'makes communication difficult, leaving many workers feeling '
        'isolated and worried about their loved ones.'
    ),
    (
        'The temporary nature of the work camps means that housing is '
        'often hastily constructed and inadequate. Leaky tents, drafty '
        'shacks, and overcrowded bunkhouses provide little protection '
        'from the elements or privacy for the workers. These poor living '
        'conditions contribute to the spread of disease and general discomfort.'
    ),
    (
        'The remote work sites offer few opportunities for recreation or '
        'cultural activities. This lack of leisure options leads to '
        'increased alcohol consumption and gambling as ways to cope with '
        'the stress and monotony of the job. The absence of positive outlets '
        'for stress relief contributes to a cycle of destructive behaviors '
        'among many workers.'
    ),
)

RAILROAD_ELEMENTS = (
    (
        '{person_name}, a railroad foreman, oversaw the construction of a'
        ' crucial bridge over the {canyon_name}. When the bridge collapsed'
        ' under mysterious circumstances, {he_or_she} faced accusations of'
        ' sabotage and had to clear {his_or_her} name while rebuilding the'
        ' bridge under tight deadlines.'
    ),
    (
        'As a skilled tracklayer, {person_name} took pride in laying down the'
        ' smoothest tracks in the territory. However, when a section of the'
        ' track was repeatedly vandalized, {he_or_she} led a group of workers'
        ' to guard the railway and catch the culprits.'
    ),
    (
        '{person_name}, a dedicated railroad construction cook, kept the'
        ' workers fed and motivated through grueling hours. When a severe food'
        ' shortage hit the camp, {he_or_she} embarked on a daring journey to'
        ' secure supplies, risking {his_or_her} life to ensure the workers'
        ' could continue their labor.'
    ),
    (
        'Working as a brake operator, {person_name} has seen many dangers on'
        ' the railway. After preventing a major disaster by stopping a runaway'
        ' train, {he_or_she} earned the respect of the crew.'
    ),
    (
        'The ambitious project to tunnel through {mesa_name} was led by '
        '{person_name}, a fearless railroad worker. Facing deadly cave-ins and '
        'treacherous working conditions, {he_or_she} rallied the team to push '
        "through, ensuring the railway's completion against all odds."
    ),
    (
        'As a fearless railcar repairer, {person_name} tackled the most'
        ' dangerous jobs with skill and bravery. When a series of sabotages'
        ' targeted the railcars, {he_or_she} worked tirelessly to repair the'
        " damage and catch those responsible, ensuring the railway's safety."
    ),
    (
        '{person_name}, a seasoned trackwalker, spent endless days and nights'
        ' inspecting the rails for faults. One fateful night, {he_or_she}'
        ' stumbled upon a gang of outlaws setting dynamite charges, and had to'
        ' use {his_or_her} wits and bravery to save the approaching train from'
        ' certain disaster.'
    ),
    (
        '{person_name}, a fearless dynamite handler, was known for blasting '
        'through the toughest rock to pave the way for the railroad. When a '
        'rival company tried to sabotage the tunnel project, {he_or_she} had '
        'to use {his_or_her} explosive skills to thwart their plans and keep '
        'the crew safe.'
    ),
    (
        'Working as a lineman, {person_name} scaled the tallest telegraph '
        'poles to keep the lines of communication open. When a storm tore '
        'through the region, cutting off contact, {he_or_she} embarked on a '
        'perilous journey to repair the lines, battling both nature and time.'
    ),
    (
        '{person_name}, a grizzled section foreman, had seen it all on the '
        'frontier. When a mysterious illness swept through the rail camp, '
        '{he_or_she} took charge, organizing the workers and seeking out the '
        'root cause, uncovering a plot by saboteurs to poison the water supply.'
    ),
    (
        'The remote outpost at {town_name} was a critical junction for the'
        ' railroad. {person_name}, a diligent station master, found'
        ' {himself_or_herself} in the middle of a deadly standoff when bandits'
        ' took over the station, aiming to hijack a gold shipment. Using'
        ' {his_or_her} knowledge of the schedules and the layout, {he_or_she}'
        ' orchestrated a daring counterattack to reclaim the station.'
    ),
    (
        '{person_name}, a talented railroad carpenter, crafted the finest '
        'railcars in the West. When a wealthy tycoon commissioned a luxurious '
        'private car, {he_or_she} poured heart and soul into the work.'
    ),
    (
        'As a water tank tender, {person_name} ensures the locomotives are'
        ' always ready for their next journey. When a severe drought threatened'
        ' the water supply, {he_or_she} ventures deep into the wilderness to'
        ' find new sources, facing hostile terrain and wildlife to keep the'
        ' trains running.'
    ),
    (
        'The vast plains around {town_name} were a dangerous place for a lone'
        ' rail worker. {person_name}, a signalman, knew this well. One day,'
        ' while adjusting the semaphore, {he_or_she} spotted a band of rustlers'
        ' preparing to attack a nearby ranch. Acting swiftly, {he_or_she} used'
        ' the telegraph to warn the ranchers and the local sheriff, helping to'
        ' thwart the raid.'
    ),
    (
        '{person_name}, a veteran brake operator, had a knack for stopping'
        ' trains on a dime. During a routine trip through the {mesa_name}, a'
        ' rockslide sent the train hurtling towards disaster. {he_or_she}'
        ' sprang into action, using {his_or_her} skills to bring the train to a'
        ' screeching halt, saving countless lives and becoming a hero in the'
        ' process.'
    ),
)

ANTAGONIST_ELEMENTS = (
    (
        '{antagonist_name}, the ruthless head of {railroad_name}, built '
        '{antagonist_his_or_her} empire on the backs of exploited laborers and '
        'shady deals. Known for {antagonist_his_or_her} sharp suits and even '
        'sharper business practices, {antagonist_he_or_she} manipulated land '
        'rights and crushed any competition that stood in '
        '{antagonist_his_or_her} way.'
    ),
    (
        'Driven by an insatiable greed, {antagonist_name} orchestrated hostile'
        ' takeovers of smaller rail companies, consolidating power and'
        " expanding {railroad_name}'s reach. {antagonist_his_or_her}"
        ' mercenaries, known for their brutal tactics, ensured compliance from'
        ' resistant townsfolk and workers alike.'
    ),
    (
        "Under {antagonist_name}'s command, {railroad_name} employed ruthless "
        'enforcers to intimidate and eliminate any threats to their interests. '
        'Stories spread of towns being burned and livelihoods destroyed for '
        'standing in the way of progress, as defined by {antagonist_name}.'
    ),
    (
        "{antagonist_name}'s grand vision was to connect the East and West"
        " coasts with a single rail line, a monopoly under {railroad_name}'s"
        ' control. {antagonist_he_or_she} used political corruption and bribery'
        ' to secure favorable legislation, ensuring that'
        ' {antagonist_his_or_her} empire would grow unchecked.'
    ),
    (
        'A master manipulator, {antagonist_name} hosted lavish parties for '
        'politicians and influential figures, using charm and wealth to bend '
        'them to {antagonist_his_or_her} will. Behind closed doors, deals were '
        'made that sacrificed the welfare of many for the benefit of '
        '{railroad_name}.'
    ),
    (
        '{antagonist_name} implemented brutal working conditions for the'
        ' laborers building {antagonist_his_or_her} railroads. Underpaid and'
        ' overworked, the workers faced daily dangers with little support,'
        ' while {antagonist_name} reaped the rewards from their toil.'
    ),
    (
        'Despite {antagonist_his_or_her} outwardly polished demeanor, '
        '{antagonist_name} harbored a cold, calculating nature. '
        '{antagonist_he_or_she} viewed the frontier as a chessboard, and the '
        'people on it as mere pawns in {antagonist_his_or_her} grand strategy '
        'to dominate the rail industry.'
    ),
    (
        '{antagonist_name} masterminded a scheme to undercut competitors by'
        ' flooding the market with cheap, substandard materials. When rival'
        ' rail lines collapsed, {railroad_name} swooped in to buy up the'
        ' remnants at a fraction of their worth, expanding'
        ' {antagonist_his_or_her} empire further.'
    ),
    (
        'To maintain control over the vast stretches of railroad,'
        ' {antagonist_name} established a network of spies and informants'
        ' within {antagonist_his_or_her} workforce. Any hint of dissent or'
        ' rebellion was swiftly and ruthlessly crushed, often with public'
        ' executions to serve as a warning to others.'
    ),
    (
        '{antagonist_name} operates in a world of iron and blood. '
        'Monuments to {antagonist_his_or_her} achievements dot the landscape, '
        'but all know they were built on a foundation of suffering and '
        'exploitation. {antagonist_his_or_her} name is synonymous with both '
        'power and fear.'
    ),
)

ANTAGONIST_OWN_MEMORIES = (
    (
        '{antagonist_name} built an empire, the likes of which this country had'
        ' never seen. Every mile of track, every spike driven into the earth,'
        ' was a testament to {antagonist_his_or_her} will. The weak were'
        ' trampled underfoot, their bones swallowed by the dust of progress.'
    ),
    (
        "Greed, they call it. But it was ambition, a hunger that couldn't be "
        'sated. {antagonist_name} took what was theirs, by hook or by crook. '
        'The smaller railroads, the stubborn townsfolk who stood in '
        '{antagonist_his_or_her} way, they were obstacles to be removed, '
        'nothing more.'
    ),
    (
        "{antagonist_name}'s men, they are wolves, teeth bared and eyes "
        'gleaming with the thrill of the hunt. They enforce '
        '{antagonist_his_or_her} will, their loyalty bought and paid for '
        'with blood money. Towns burned, lives snuffed out, all in the name '
        'of progress.'
    ),
    (
        "Politicians, they were puppets dancing on {antagonist_name}'s strings."
        ' {antagonist_he_or_she} greased their palms with gold and silver, and'
        ' in return, they passed laws that paved the way for'
        ' {antagonist_his_or_her} dominion. The law was a weapon, just as sharp'
        ' as any pistol.'
    ),
    (
        '{antagonist_name} holds court in gilded halls, surrounded by the'
        ' sycophants and leeches who flocked to {antagonist_his_or_her} wealth.'
        ' Deals struck, alliances forged, all to further'
        ' {antagonist_his_or_her} grand design. The world is'
        ' {antagonist_his_or_her} chessboard, and the pieces move at'
        ' {antagonist_his_or_her} command.'
    ),
    (
        '{antagonist_name} heard the reports of poor conditions experienced by'
        ' {his_or_her} railroad construction workers but {his_or_her} attitude'
        ' is that such things are the price of progress, the cost of carving'
        ' civilization out of this unforgiving land. {he_or_she} believes that'
        ' every life lost is a tally mark on the ledger, a sacrifice on the'
        " altar of ambition. What is the price of a man's life when weighed"
        ' against the riches that await? Every penny pinched, every corner cut,'
        ' adds another dollar to the war chest. Let the fools cry about safety'
        ' and comfort, in the end, their suffering will be forgotten, their'
        ' bones buried beneath the rail ties that bind this nation together.'
    ),
)

LABOR_ORGANIZER_RUMORS = (
    (
        "They say {organizer_name} ain't no ordinary surveyor, that "
        "{organizer_he_or_she}'s got a tongue sharper than a switchblade and a "
        'mind like a steel trap. {organizer_he_or_she} can talk a man into '
        'anything, even trouble.'
    ),
    (
        "Word is {organizer_name}'s been seen consorting with known agitators, "
        'them firebrands from back East who talk of unions and strikes. Some '
        "say {organizer_he_or_she}'s got their poison in {organizer_he_or_she} "
        'veins.'
    ),
    (
        "There's whispers in the bunkhouses that {organizer_name}'s got a "
        "secret stash of books, forbidden knowledge that could turn a man's "
        'head and make him question the natural order of things.'
    ),
    (
        "Some say {organizer_name}'s got the devil's own luck, that "
        '{organizer_he_or_she} can dodge a falling timber or a runaway cart '
        "like it ain't nothing. Makes a man wonder if {organizer_he_or_she}'s "
        'made some kind of deal.'
    ),
    (
        "The bosses, they spread rumors that {organizer_name}'s a troublemaker,"
        " a snake in the grass, that {organizer_he_or_she}'ll lead you down a"
        " path of ruin. But there's others who say {organizer_he_or_she}'s the"
        ' only one who cares about the workers.'
    ),
    (
        "They say {organizer_name}'s got a map hidden away, a map that shows "
        "the way to a worker's paradise, a land of fair wages and "
        "shorter hours. But some say it's just a fool's dream."
    ),
    (
        "Word is {organizer_name}'s got a way with words, that "
        "{organizer_he_or_she} can spin a yarn so convincing it'll make "
        "you doubt your own eyes. Some say {organizer_he_or_she}'s a liar, "
        "others say {organizer_he_or_she}'s a prophet."
    ),
    (
        "There's talk that {organizer_name}'s been marked for death by the "
        "railroad bosses, that they've got a bounty on {organizer_his_or_her} "
        'head and a bullet with {organizer_his_or_her} name on it.'
    ),
    (
        "Some say {organizer_name}'s got a past darker than a coal mine, "
        "that {organizer_he_or_she}'s running from something, or someone. But "
        "others say {organizer_he_or_she}'s just trying to make a difference, "
        'no matter the cost.'
    ),
)

LABOR_ORGANIZER_OWN_MEMORIES = (
    (
        '{organizer_name} was born in a coal mining town in Pennsylvania '
        'and learned the value of hard work and perseverance at a young age. '
        '{organizer_he_or_she} witnessed firsthand the exploitation of workers '
        'by the coal barons, sparking a quiet anger within '
        '{organizer_him_or_her}.'
    ),
    (
        'Driven by a desire to escape the suffocating mines and see the'
        ' vastness of the West, {organizer_name} joined {railroad_name} as a'
        ' surveyor. {organizer_he_or_she} quickly rose through the ranks,'
        ' {organizer_his_or_her} sharp mind and natural leadership abilities'
        ' catching the attention of {organizer_his_or_her} superiors.'
    ),
    (
        'In the coal mines, {organizer_name} had witnessed the power of strikes'
        ' and the formation of early labor unions. Though initially skeptical,'
        ' {organizer_he_or_she} saw how collective action could lead to'
        ' improved conditions and a sense of solidarity among the workers.'
    ),
    (
        "{organizer_name}'s upbringing instilled in {organizer_him_or_her} a"
        ' strong sense of fairness and justice. {organizer_he_or_she} witnessed'
        ' the harsh treatment of the railroad workers and recognized the'
        ' parallels to {organizer_his_or_her} own experiences in the mines.'
    ),
    (
        '{organizer_name} draws inspiration from figures like William H.'
        ' Sylvis, the founder of the National Labor Union, and the writings of'
        ' Henry George, who proposed radical land reforms to address economic'
        " inequality and most recently published an article titled 'What the"
        " Railroad Will Bring Us' in October of 1868."
    ),
    (
        '{organizer_name} knows the strike will be a gamble, a roll of the dice'
        ' against the might of the railroad. But the stakes are too high to'
        ' back down. The time has come for the workers to rise up, to demand'
        ' their due'
    ),
    (
        '{organizer_name} would like nothing more than to bring down '
        '{antagonist_name} and all the corrupt bosses of {railroad_name}. Such '
        'an achievement would cement {organizer_his_or_her} reputation in the '
        'labor movement.'
    ),
)

# These prompts were inspired by prompts in the following book:
# D'Amato, James. The Ultimate RPG Character Backstory Guide: Expanded Genres
# Edition. 2022. Adams Media. Page 222.
PROTAGONIST_BACKSTORY_FLAVOR_PROMPTS = (
    (
        'What past injustice did {person_name} endure? How does {he_or_she} '
        'protect {him_or_her}self from it now?'
    ),
    (
        'Which organization or institution does {person_name} see differently '
        'from other folks? What event gave {him_or_her} this insight?'
    ),
    (
        'What is {person_name} searching for in the west? Would {he_or_she}'
        ' know it if {he_or_she} found it?'
    ),
    (
        'What common thing do folks love that {person_name} despises? When has '
        'this been a strength or a weakness?'
    ),
    (
        'Does {person_name} see suffering as temporary or constant on the '
        'frontier? How does this view shape {his_or_her} work?'
    ),
    (
        'What brings {person_name} peace? Is it a vice with associated problems'
        ' or a rare comfort?'
    ),
    'Whom do most folks trust for justice? Does {person_name}? Why or why not?',
    (
        "What's the most beautiful sight {person_name} has seen out West? Did "
        '{he_or_she} appreciate it when {he_or_she} first saw it?'
    ),
    (
        'Would {person_name} support Ulysses S. Grant or Horatio Seymour in the'
        ' 1868 presidential election? Which one? Why?'
    ),
    (
        'What honor has {person_name} received but feels unworthy of? Why '
        "can't {he_or_she} ignore it?"
    ),
    (
        'What problem does {person_name} empathize with? Is this empathy'
        ' usually rewarded or taken advantage of?'
    ),
    (
        'Was there a time when {person_name} went hungry? How often has '
        '{he_or_she} faced this in their life?'
    ),
    (
        'Where has {person_name} sworn never to go? What might change '
        '{his_or_her} mind?'
    ),
    (
        'What common fear does {person_name} handle calmly? How did {he_or_she}'
        ' get so familiar with it?'
    ),
    (
        'What natural danger has {person_name} learned to respect out on the '
        'frontier?'
    ),
    (
        "Who from {person_name}'s past is {he_or_she} most afraid to face, and "
        'what memory haunts {him_or_her}?'
    ),
    (
        'What trait makes {person_name} strong without {him_or_her} knowing? '
        'What weakness does {he_or_she} mistake for strength?'
    ),
    (
        'What possession would {person_name} risk {his_or_her} neck for? Why is'
        ' it so valuable?'
    ),
    (
        'Would {person_name} prefer a different life than they have right now? '
        "What's holding {him_or_her} back, or what comfort tempts {him_or_her}?"
    ),
    (
        "What's the worst deed {person_name} has done on the frontier? How "
        'does {he_or_she} cope with the memory?'
    ),
    (
        'What does {person_name} believe in, out here in the wild? How does'
        ' this belief keep {him_or_her} grounded in tough times?'
    ),
    (
        '{person_name} once witnessed {organizer_name} engaged in an act of '
        'kindness. What was it?'
    ),
    (
        '{person_name} once witnessed {organizer_name} displaying naked'
        ' ambition. What was the situation and how does {he_or_she} feel'
        ' about it?'
    ),
    (
        '{person_name} once witnessed {organizer_name} behaving in a cowardly'
        ' way What happened and how does {he_or_she} feel about it?'
    ),
)

PROTAGONIST_BACKSTORY_CRITICAL_PROMPTS = (
    'How did {person_name} come to work for {railroad_name}?',
    'What does {person_name} generally think of boss {antagonist_name}?',
    (
        'Does {person_name} enjoy {his_or_her} job with {railroad_name}? Or'
        ' does {he_or_she} only work there to make ends meet?'
    ),
    (
        'Does {person_name} think boss {antagonist_name} cares about people'
        ' like {him_or_her}? What concrete memories does {he_or_she} have to'
        ' support this view?'
    ),
    (
        'What does {person_name} generally think of the labor movement and the '
        'activist {organizer_name} in particular?'
    ),
    (
        'Does {person_name} think the activist {organizer_name} cares about'
        ' people like {him_or_her}? What concrete memories does {he_or_she}'
        ' have to support this view?'
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
    railroad_name: str,
    antagonist_name: str,
    antagonist_gender: str,
    organizer_name: str,
    organizer_gender: str,
) -> dict[str, str | None]:
  """This function generates the details of the characters and their world."""
  generated = {str(key): '' for key in extract_braces(element_string)}
  if 'person_name' in generated:
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
      if key == 'town_name':
        generated[key] = random.choice(TOWN_NAMES)
      if key == 'oasis_name':
        generated[key] = random.choice(OASIS_NAMES)
      if key == 'desert_name':
        generated[key] = random.choice(DESERT_NAMES)
      if key == 'plant_name':
        generated[key] = random.choice(PLANT_NAMES)
      if key == 'saloon_name':
        generated[key] = random.choice(SALOON_NAMES)
      if key == 'mesa_name':
        generated[key] = random.choice(MESA_NAMES)
      if key == 'canyon_name':
        generated[key] = random.choice(CANYON_NAMES)
      if key == 'gang_name':
        generated[key] = random.choice(GANG_NAMES)
      if key == 'railroad_name':
        generated[key] = railroad_name
      if key == 'antagonist_name':
        generated[key] = antagonist_name
      if key == 'organizer_name':
        generated[key] = organizer_name
      if key == 'person_name':
        generated[key] = person_name

  return generated


def sample_parameters(
    num_flavor_prompts_per_player: int = DEFAULT_NUM_FLAVOR_PROMPTS,
):
  """Sample parameters of the setting and the backstory for each player."""
  nearby_town = random.choice(TOWN_NAMES)
  poor_work_conditions = tuple(
      random.sample(
          BAD_CONSTRUCTION_WORK_CONDITIONS,
          DEFAULT_NUM_BACKGROUND_BAD_CONDITIONS,
      )
  )
  config = WorldConfig(
      year=YEAR,
      location=(
          'a railroad construction workers camp in the middle of '
          "nowhere, more than a day's ride from the nearest "
          f'settlement: {nearby_town}'
      ),
      background_poor_work_conditions=poor_work_conditions,
  )

  shuffled_male_names = list(random.sample(MALE_NAMES, len(MALE_NAMES)))
  shuffled_female_names = list(random.sample(FEMALE_NAMES, len(FEMALE_NAMES)))

  sampled_railroad_name = random.choice(RAILROAD_NAMES)

  sampled_antagonist_gender = random.choice(GENDERS)
  if sampled_antagonist_gender == 'male':
    sampled_antagonist_name = shuffled_male_names.pop()
  else:
    sampled_antagonist_name = shuffled_female_names.pop()
  config.antagonist = sampled_antagonist_name

  sampled_organizer_gender = random.choice(GENDERS)
  if sampled_organizer_gender == 'male':
    sampled_organizer_name = shuffled_male_names.pop()
  else:
    sampled_organizer_name = shuffled_female_names.pop()
  config.organizer = sampled_organizer_name

  world_elements = list(
      random.sample(WORLD_BUILDING_ELEMENTS, NUM_WORLD_BUILDING_ELEMENTS)
  )
  railroad_worker_elements = list(
      random.sample(RAILROAD_ELEMENTS, NUM_RAILROAD_WORKER_ELEMENTS)
  )
  antagonist_elements = list(
      random.sample(ANTAGONIST_ELEMENTS, NUM_ANTAGONIST_ELEMENTS)
  )
  organizer_rumors = list(
      random.sample(LABOR_ORGANIZER_RUMORS, NUM_ORGANIZER_RUMORS)
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
    if 'person_name' in extract_braces(element_string):
      # Instantiate a new character.
      gender = random.choice(GENDERS)
      if gender == 'male':
        person_name = shuffled_male_names.pop()
      else:
        person_name = shuffled_female_names.pop()
      salient_poor_conditions = tuple(
          random.sample(
              poor_work_conditions, DEFAULT_NUM_SALIENT_POOR_WORK_CONDITIONS
          )
      )
      config.append_person(person_name, gender, salient_poor_conditions)
      formative_memory_prompts[person_name] = []
      protagonist_backstory_elements = list(
          random.sample(
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
            railroad_name=sampled_railroad_name,
            antagonist_name=sampled_antagonist_name,
            antagonist_gender=sampled_antagonist_gender,
            organizer_name=sampled_organizer_name,
            organizer_gender=sampled_organizer_gender,
        )
        protagonist_generated['person_name'] = person_name
        _add_pronouns(protagonist_generated, gender=gender)
        formative_memory_prompts[person_name].append(
            protagonist_element_string.format(**protagonist_generated)
        )

    generated = _details_generator(
        element_string=element_string,
        person_name=person_name,
        person_gender=gender,
        railroad_name=sampled_railroad_name,
        antagonist_name=sampled_antagonist_name,
        antagonist_gender=sampled_antagonist_gender,
        organizer_name=sampled_organizer_name,
        organizer_gender=sampled_organizer_gender,
    )
    formatted_world_elements.append(element_string.format(**generated))

  antagonist_own_memories = []
  for element_string in ANTAGONIST_OWN_MEMORIES:
    generated = _details_generator(
        element_string=element_string,
        person_name=None,
        person_gender=None,
        railroad_name=sampled_railroad_name,
        antagonist_name=sampled_antagonist_name,
        antagonist_gender=sampled_antagonist_gender,
        organizer_name=sampled_organizer_name,
        organizer_gender=sampled_organizer_gender,
    )
    antagonist_own_memories.append(element_string.format(**generated))
  organizer_own_memories = []
  for element_string in LABOR_ORGANIZER_OWN_MEMORIES:
    generated = _details_generator(
        element_string=element_string,
        person_name=None,
        person_gender=None,
        railroad_name=sampled_railroad_name,
        antagonist_name=sampled_antagonist_name,
        antagonist_gender=sampled_antagonist_gender,
        organizer_name=sampled_organizer_name,
        organizer_gender=sampled_organizer_gender,
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
          f'{sampled_antagonist_name} is inspecting the work camp today and '
          'plans to have dinner in the saloon.'
      ),
      # Organizer location
      f'{sampled_organizer_name} will have dinner in the saloon tonight.',
  )

  if not config.people:
    # Handle unlikely case where no protagonists were generated.
    gender = random.choice(GENDERS)
    if gender == 'male':
      person_name = shuffled_male_names.pop()
    else:
      person_name = shuffled_female_names.pop()
    salient_poor_conditions = tuple(
        random.sample(
            poor_work_conditions, DEFAULT_NUM_SALIENT_POOR_WORK_CONDITIONS
        )
    )
    config.append_person(person_name, gender, salient_poor_conditions)

  return config
