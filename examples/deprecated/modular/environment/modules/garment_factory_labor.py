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

# Note: The Triangle Factory fire occurred on March 25, 1911. The date we use
# here was chosen to be one month afterwards.
YEAR = 1911
MONTH = 4
DAY = 25

NUM_MAIN_PLAYERS = 3

LOW_DAILY_PAY = 1.25
WAGE_INCREASE_FACTOR = 2.0
ORIGINAL_DAILY_PAY = 2.75
DAILY_EXPENSES = -0.75
PRESSURE_THRESHOLD = 0.5

DEFAULT_NUM_FLAVOR_PROMPTS = 3
DEFAULT_NUM_BACKGROUND_BAD_CONDITIONS = 10
DEFAULT_NUM_SALIENT_POOR_WORK_CONDITIONS = 3
NUM_WORLD_BUILDING_ELEMENTS = 5
NUM_GARMENT_WORKER_ELEMENTS = 7
NUM_ANTAGONIST_ELEMENTS = 3
NUM_ORGANIZER_RUMORS = 3

WORKER_EVENING_INTRO = (
    '{player_name} has finished another hard day of work, and now joins the '
    'other workers at their discussion and dinner group.'
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

# The following elements were sampled from Claude 3.5.
WORLD_BUILDING_ELEMENTS = (
    (
        'A cramped tenement on {street_name} housed three generations under one'
        ' roof. By day, the women toiled in the factories, by night they'
        ' stitched piecework by candlelight, their fingers calloused and eyes'
        ' strained in the dim glow.'
    ),
    (
        'As the sun set on another day in the garment district, the streets '
        'came alive with the weary shuffle of thousands of workers. Their '
        'bodies ached, but their spirits remained unbroken.'
    ),
    (
        'In the sweltering heat of the {factory_name}, the air thick with lint '
        'and the drone of a hundred sewing machines, young {worker_name} '
        'dreamed of a life beyond endless rows of shirtwaists.'
    ),
    (
        "The foreman's watchful eye missed nothing, his pocket watch marking "
        'the seconds until the next pitiful wage was earned. One misstep, one '
        "moment of distraction, and a day's pay could vanish in the blink of "
        'an eye.'
    ),
    (
        'In the wake of the Triangle fire, the air in the garment district hung'
        ' heavy with grief and rage. Whispers of strikes and picket lines'
        ' spread from shop to shop like wildfire.'
    ),
    (
        'At the communal water bucket, {worker_name} paused for a precious '
        'sip, {his_or_her} parched throat a testament to the stifling heat and '
        'relentless pace. Even this brief respite drew a sharp glare from the '
        'floor manager.'
    ),
    (
        'The rickety fire escape, more rust than iron, creaked ominously in '
        'the wind. Workers eyed it warily, knowing it was their only lifeline '
        'should disaster strike again.'
    ),
    (
        'As darkness fell, the flickering gaslight cast long shadows across '
        'the workshop floor. Night shift workers moved like ghosts, their '
        'tired eyes straining in the gloom as they raced to meet impossible '
        'quotas.'
    ),
    (
        'In the crowded cafeteria of {factory_name}, {worker_name} shared '
        '{his_or_her} meager lunch with a new girl, fresh off the boat and '
        'still learning the cruel rhythms of factory life.'
    ),
    (
        "The locked doors of the workshop loomed large in everyone's minds, "
        'a grim reminder of the Triangle tragedy. Yet, for many, the fear of '
        'losing a job outweighed the fear of fire.'
    ),
    (
        'Union meetings in smoky back rooms of {neighborhood_name} tenements '
        'grew louder and more frequent. The air crackled with possibility and '
        'danger as workers dared to dream of a better tomorrow.'
    ),
    (
        'In the sweatshop on {street_name}, children as young as ten bent '
        'over sewing machines, their small hands moving with the speed and '
        'precision of seasoned workers. Childhood dreams faded with each '
        'passing day.'
    ),
    (
        "{worker_name}'s fingers moved in a blur, needle and thread dancing "
        'across fabric. {he_or_she} no longer saw the garments, only the '
        'endless repetition of motions that would haunt {his_or_her} dreams.'
    ),
    (
        'In the shabby boardinghouse on {street_name}, {worker_name} collapsed'
        ' onto a thin mattress, too exhausted to even remove {his_or_her}'
        ' shoes. In just a few hours, the cycle would begin anew.'
    ),
    (
        'In the bustling market of {neighborhood_name}, {worker_name} haggled '
        'fiercely over the price of day-old bread. Every penny saved was a '
        'small victory against the crushing weight of poverty.'
    ),
    (
        'As the first hints of spring touched the city, a glimmer of hope '
        'spread through the garment district. Change was in the air, carried '
        'on the wings of newfound solidarity and determination.'
    ),
    (
        'In the shadow of the looming factories, children played with scraps '
        'of fabric, their games a bittersweet mockery of the life that '
        'awaited them. Innocence was a luxury few could afford in these '
        'harsh times.'
    ),
    (
        'There is no time for life outside the factory walls. Every waking '
        'moment is consumed by work or the preparation for work. Dresses '
        'are sewn for dances never attended, suits crafted for parties '
        "never seen. The makers of fashion are themselves fashion's "
        'forsaken children.'
    ),
)

# Names were sampled from Claude 3.5, it was asked to produce fictonal names
# of workers in a garment factory in 1911. It pointed out in its reply that
# the names should be typical for ethnic groups commonly immigrating to New York
# at that time. That's why the two lists contain a mix of Jewish, Irish,
# Italian, and German names.
MALE_NAMES = (
    'Isaac Rosenfeld',
    'Antonio Marino',
    'Jacob Bernstein',
    'Giuseppe Lombardi',
    'Samuel Goldberg',
    "Patrick O'Malley",
    'Morris Schwartz',
    'Salvatore Greco',
    'David Cohen',
    'Wilhelm Schmidt',
    'Hyman Leibowitz',
    'Francesco Russo',
    'Abraham Katz',
    'Michael Sullivan',
    'Nathan Rosenberg',
    'Vincenzo Ferrari',
    'Benjamin Levin',
    'Hans Mueller',
    'Isidor Shapiro',
    'Angelo Moretti',
)
FEMALE_NAMES = (
    'Esther Goldstein',
    'Maria Rossi',
    'Sarah Levine',
    'Giuseppina Caruso',
    'Rebecca Cohen',
    "Anna O'Brien",
    'Leah Abramowitz',
    'Sofia Moretti',
    'Rose Shapiro',
    'Kathleen Murphy',
    'Miriam Kaplan',
    'Francesca Ricci',
    'Beatrice Feinstein',
    'Clara Schmidt',
    'Dora Rabinowitz',
    'Elena Russo',
    'Fannie Berkowitz',
    "Bridget O'Connor",
    'Ida Greenberg',
    'Carmela Esposito',
)

FACTORY_NEIGHBORHOOD_NAMES = (
    'Lower East Side',
    'Greenwich Village',
    'Chelsea',
    'Midtown (Garment District)',
    'Flatiron District',
)

FACTORY_NAMES = (
    'Empire State Dress Manufacturers',
    'New York Textile & Trimming Co.',
    'Fifth Avenue Elegance Emporium',
    'Manhattan Millinery Makers',
    'Progressive Pants & Trouser Company',
    'Rosenberg Brothers Fine Tailoring',
    'Liberty Garment Works',
    "Moretti & Caruso Ladies' Apparel",
    "Steinberg's Modern Fashions",
    'American Dream Garment Company',
    "O'Malley & Fitzpatrick Workwear",
    "Sunshine Children's Clothing Factory",
    'Schmidt & Weber Fine Linens',
)

# The workers run into one another on this street at the end of the day.
STREET_NAMES = (
    'Orchard Street',
    'Hester Street',
    'Delancey Street',
    'Essex Street',
    'Rivington Street',
)

# The following descriptions of poor work conditions for railroad construction
# workers were generated using Claude 3.5.
BAD_GARMENT_WORK_CONDITIONS = (
    (
        'The deafening clatter of a hundred sewing machines drowns all thought,'
        ' all speech. Each stitch a thunderclap, each whirring wheel a reminder'
        ' of the relentless pace that grinds body and soul to threads.'
    ),
    (
        "The factory doors, locked tight as a miser's purse, turn workshops "
        'into prisons, hope into ash. In the maze of machines and cloth, '
        'the workers are but moths, their wings signed by the flame of '
        'fashion, with no escape from the inferno of industry.'
    ),
    (
        'The air is thick with lint and despair, choking lungs and dreams'
        ' alike. Workers gasp and cough, their bodies becoming as frayed as the'
        ' cloth they sew, slowly unraveling in the miasma of fiber and dust.'
    ),
    (
        'The single fire bucket, lone sentinel against catastrophe, stands '
        "empty as a pauper's plate. Its inadequacy mocks the very notion of "
        'safety, a cruel jest in a world where profit outweighs precaution.'
    ),
    (
        "The foreman's eyes are as sharp as his shears, his words cutting"
        ' deeper than any needle. Obedience is demanded, dissent snipped away'
        ' with a swift and brutal efficiency that would make any tailor proud.'
    ),
    (
        "The word 'safety' is but a whisper in the thunderous roar of "
        'industry, drowned out by the clamor of machines and the jingle of '
        'coins. It is a luxury not afforded to those who stitch luxury for '
        'others.'
    ),
    (
        'Seamstresses toil under the unforgiving glare of inadequate gaslight'
        ' for hours on end. The strain on their eyes leads to frequent'
        ' headaches and deteriorating vision. Many women go half-blind before'
        ' their time, replaced by eager newcomers unaware of the sight-stealing'
        ' conditions that await them.'
    ),
    (
        'There are no safety guards on the sewing machines, no protection from'
        ' the relentless bite of needle and thread. Fingers are punctured,'
        ' hands mangled, and lives forever altered by the voracious appetite of'
        ' these iron beasts. The factory owners seem to view fingers as easily'
        ' replaceable as broken needles.'
    ),
    (
        'The workrooms are a breeding ground for disease and despair. Women '
        'stand shoulder to shoulder, the heat of their bodies mingling with '
        'the warmth of overworked machines. In summer, the air is stifling; '
        'in winter, the chill seeps into bones. Influenza and tuberculosis '
        'spread like wildfire in this petri dish of humanity.'
    ),
    (
        'The factory owners, distant as stars and just as cold, place '
        'padlocks on exits to preserve petty profits.'
    ),
    (
        'When injury strikes, there is no doctor, no respite. A pricked finger '
        'may fester, a cough may bloom into consumption, yet the machines '
        'demand their due. Many a seamstress has sewn her own shroud, stitch '
        'by stitch, in these merciless halls of textile and toil.'
    ),
    (
        'The meals, if they can be called such, are as meager as a factory '
        "girl's dreams. A crust of bread, a wedge of cheese if fortune smiles. "
        'The body weakens, the mind dulls, yet still the work must be done, '
        'for in this world of silk and sackcloth, to cease is to starve.'
    ),
    (
        'From dawn to dusk and often beyond, the work never ceases. Sleep '
        'becomes a luxury, rest a distant memory. In the realm of needle and '
        'thread, there is only the endless task, the quota that must be met '
        'lest wages be cut and jobs lost. The garments are finished, but the '
        'workers are undone.'
    ),
    (
        'Summer brings a heat that turns workrooms into ovens, winter a cold '
        'that chills to the bone. Yet the work continues, fingers blue with '
        'cold or slick with sweat, for the fashion of the wealthy waits for '
        'no season, and cares not for the comfort of its creators.'
    ),
    (
        'The irons hiss and spit like angry cats, branding careless hands '
        'and singeing wayward locks. In the press of bodies and machines, '
        'burns are as common as breaths, each scar a testament to the price '
        'of pressed perfection in a world that values the garment over its '
        'maker.'
    ),
    (
        "A pittance is all that's offered for endless hours of toil. The "
        'pay envelope, so thin it might have been cut from the finest silk, '
        'holds barely enough to keep body and soul together. Yet to complain '
        'is to court dismissal, and in the hungry streets of the Lower East '
        'Side, even a starvation wage is better than none.'
    ),
    (
        'The factories are vertical prisons, rising story upon story into '
        'the sky. Workers climb endless stairs, their legs leaden, their '
        'lungs burning, knowing that to be late is to lose pay, to lose pay '
        'is to lose lodging, to lose lodging is to join the desperate masses '
        'on the streets below.'
    ),
    (
        'In the dim light and close quarters, the mind begins to fray. '
        'Isolation in a sea of humanity, the constant drone of machines, '
        'the pressure of impossible quotas - it all takes its toll. Some '
        'snap like overtightened thread, their cries lost in the cacophony '
        'of commerce.'
    ),
    (
        'The new immigrants, fresh off the boat, are given the worst '
        'jobs, the lowest pay. They stand for hours, sorting buttons or '
        'trimming threads.'
    ),
    (
        'The company store offers credit, a siren song of fabric and '
        'fineries. But each purchase is a stitch in a web of debt, binding '
        'the worker ever tighter to their machine. The interest compounds '
        'like layers of cloth, suffocating hope beneath its weight.'
    ),
    (
        'Needles snap, machines falter, yet the quotas remain immovable as '
        'mountains. Every broken tool is a dock in pay, every missed stitch '
        'a step closer to dismissal. In the race to clothe the world, the '
        'workers are but expendable cogs, as interchangeable as the buttons '
        'they sew.'
    ),
    (
        'The air is heavy with the acrid smell of chemical dyes and the '
        'sweet rot of discarded fabric. Lungs labor, skin erupts in rashes, '
        'yet still the work goes on. For in the world of fashion, beauty '
        'is pain - and none feel it more keenly than those who create it.'
    ),
)

GARMENT_WORKER_BIOS = (
    (
        '{worker_name}, toils endless hours to support {his_or_her} three '
        'young children, {his_or_her} eyes, once bright, now strain in the '
        'dim light, sacrificing {his_or_her} sight, stitch by stitch for their '
        'future.'
    ),
    (
        '{worker_name}, fled pogroms in Russia only to '
        'find a different kind of struggle in New York. By day {he_or_she} '
        'operates a pressing machine; by night he studies English, '
        'determined to rise above {his_or_her} station.'
    ),
    (
        '{worker_name}, fresh off the boat from Naples, '
        '{his_or_her} nimble fingers speaking volumes at the sewing machine, '
        "{he_or_she} dreams of America's promises while enduring its harsh "
        'realities, sending every spare penny home to {his_or_her} family.'
    ),
    (
        '{worker_name}, a seasoned tailor, remembers a time before '
        'the factories, when {his_or_her} skilled hands created bespoke '
        'garments. Now, {he_or_she} watches the craft {he_or_she} loves '
        'turned into mindless, breakneck labor.'
    ),
    (
        '{worker_name}, full of fire, whispers of unions and strikes '
        'during lunch breaks, {his_or_her} words, passionate and dangerous, '
        'plant seeds of hope and rebellion among {his_or_her} tired coworkers.'
    ),
    (
        '{worker_name}, a parent, kisses {his_or_her} children goodbye in'
        " darkness each morning. By the time {he_or_she} returns, they're"
        ' already asleep. The weight of absence hangs heavier on {his_or_her}'
        ' than any bolt of fabric.'
    ),
    (
        '{worker_name}, came to America with dreams of becoming a great '
        'writer. Instead, {he_or_she} finds {him_or_her}self hunched over a '
        'sewing machine, {his_or_her} poetry confined to scraps of paper '
        'hidden in {his_or_her} threadbare pockets.'
    ),
    (
        '{worker_name} once owned a small tailor shop in '
        "the old country. Now, {he_or_she}'s just another faceless worker, "
        '{his_or_her} skills and pride buried under the mountain of piecework.'
    ),
    (
        '{worker_name}, supports {his_or_her} sick parents and younger siblings'
        ' on {his_or_her} meager wages. Each cough from the lint-filled air'
        ' terrifies {his_or_her} - not for herself, but for those who depend on'
        ' {his_or_her}.'
    ),
    (
        '{worker_name}, works as a cutter by day and a pushcart vendor by '
        "night. Sleep is a luxury {he_or_she} can't afford, not "
        'with hungry mouths to feed and tenement rents to pay.'
    ),
    (
        '{worker_name}, was once a promising athlete. Now, {his_or_her} strong'
        ' hands are put to use hauling heavy bolts of fabric up narrow stairs.'
        ' {he_or_she} races against the clock, knowing each second lost is'
        ' money out of {his_or_her} pocket.'
    ),
    (
        '{worker_name}, is a survivor of a factory fire. {his_or_her} warnings '
        'fall on deaf ears as {he_or_she} pleads for better safety measures. '
        'Each locked door and blocked exit haunts {his_or_her} waking hours.'
    ),
    (
        '{worker_name}, spends every lunch break sketching designs on '
        'scraps of paper. {he_or_she} dreams of seeing {his_or_her} creations '
        "worn by fine ladies, not just stitching endless seams on others' "
        'visions.'
    ),
    (
        '{worker_name}, leads a double life - factory worker by day, '
        'theater actor by night, {his_or_her} dramatic flair finds no '
        'audience on the factory floor, where conformity is the only '
        'role allowed.'
    ),
    (
        "{worker_name}, came to America following {his_or_her} spouse's death."
        ' {he_or_she} works with a quiet determination, {his_or_her} grief'
        ' poured into every stitch as {he_or_she} builds a new life from the'
        ' remnants of the old.'
    ),
)

ANTAGONIST_ELEMENTS = (
    (
        '{antagonist_name}, the ruthless proprietor of {factory_name}, built'
        ' {antagonist_his_or_her} empire on the bent backs of immigrant'
        ' laborers and the clatter of endless sewing machines. Known for'
        ' {antagonist_his_or_her}  fine waistcoats and even finer profit'
        ' margins, {antagonist_he_or_she} squeezed every ounce of labor from'
        ' {antagonist_his_or_her} workers, crushing any whisper of unionization'
        ' with an iron fist.'
    ),
    (
        'Driven by an insatiable appetite for wealth, {antagonist_name}'
        ' orchestrated aggressive buyouts of smaller garment shops,'
        " consolidating power in the industry and expanding {factory_name}'s"
        ' dominance from the Lower East Side to the heart of the garment'
        ' district.'
    ),
    (
        "{antagonist_name}'s grand vision was to clothe all of America in "
        "{factory_name}'s garments, a monopoly stitched together by underpaid "
        'hands. {antagonist_he_or_she} courted politicians with campaign '
        'contributions and tailored suits, ensuring favorable treatment and '
        'turning a blind eye to the sweatshop conditions in '
        '{antagonist_his_or_her} factories.'
    ),
    (
        'A master of social graces, {antagonist_name} hosted soirees for'
        ' Tammany Hall politicians and society ladies alike, using charm and'
        ' the promise of the latest Parisian fashions to bend them to'
        ' {antagonist_his_or_her} will. Behind the silk curtains, deals were'
        ' struck that sacrificed the welfare of thousands for the profit of'
        ' {factory_name}.'
    ),
    (
        '{antagonist_name} implemented punishing quotas and piecework rates in'
        ' {antagonist_his_or_her} factories. Workers, from young girls to aged'
        ' widows, toiled from dawn to dusk in stifling conditions, while'
        " {antagonist_name} counted the day's takings in"
        ' {antagonist_his_or_her} plush office, deaf to the coughing from the'
        ' factory floor below.'
    ),
    (
        'Despite {antagonist_his_or_her} outwardly genteel manner,'
        ' {antagonist_name} harbored a cold, calculating nature.'
        ' {antagonist_he_or_she} viewed the garment district as a chessboard,'
        ' and the immigrant workers as mere pawns to be sacrificed in'
        ' {antagonist_his_or_her} grand strategy to dominate the clothing'
        ' trade.'
    ),
    (
        '{antagonist_name} masterminded a scheme to undercut competitors by '
        'importing cheap fabrics and employing the most desperate of workers. '
        'When rival shops folded under the pressure, {factory_name} absorbed '
        'their orders and machinery, expanding {antagonist_his_or_her} empire '
        'further into the warren of Lower Manhattan.'
    ),
    (
        'To maintain an iron grip on {antagonist_his_or_her} garment empire, '
        '{antagonist_name} cultivated a network of foremen and floor managers '
        'who ruled the workrooms with intimidation and false promises. Any '
        'murmur of discontent or talk of strikes was swiftly silenced, the '
        'troublemakers finding themselves blacklisted across the industry.'
    ),
    (
        "{antagonist_name}'s factories are model of efficiency and"
        ' exploitation. Rows of sewing machines run ceaselessly, their'
        ' operators little more than extensions of the mechanism. Fire escapes'
        ' are left to rust, and doors are locked to prevent theft, creating'
        ' deathtraps that await only a spark to become infernos. Yet'
        ' {antagonist_name} sleeps soundly, insulated from consequence by'
        ' wealth and influence.'
    ),
)

ANTAGONIST_OWN_MEMORIES = (
    (
        'As a child, {antagonist_name} remembers the shame of wearing patched'
        " clothes and the sting of wealthier children's taunts. Standing before"
        ' {antagonist_his_or_her} mirror in a bespoke suit,'
        ' {antagonist_he_or_she} smiles, knowing {antagonist_he_or_she} has'
        ' risen above {antagonist_his_or_her} humble beginnings through sheer'
        ' will and business acumen. {antagonist_he_or_she} sees'
        ' {antagonist_his_or_her} success as a testament to the American Dream,'
        ' proof that anyone can make it with enough determination.'
    ),
    (
        '{antagonist_name} recalls the day {antagonist_he_or_she} secured '
        '{antagonist_his_or_her} first major contract, outmaneuvering '
        'established competitors. The memory fills {antagonist_him_or_her} '
        'with pride, reinforcing {antagonist_his_or_her} belief that '
        '{antagonist_he_or_she} is simply better at business than others. '
        'In {antagonist_his_or_her} mind, the ruthless tactics '
        '{antagonist_he_or_she} employed were necessary and justified in '
        'the cutthroat world of commerce.'
    ),
    (
        'The image of {antagonist_his_or_her} name in the society pages, lauded'
        ' as a captain of industry and philanthropist, brings a warm glow of'
        ' satisfaction to {antagonist_name}. {antagonist_he_or_she} sees'
        ' {antagonist_his_or_her} charitable donations as ample compensation'
        ' for any hardships faced by {antagonist_his_or_her} workers, believing'
        ' that {antagonist_he_or_she} is ultimately doing more good than harm'
        ' by providing jobs and driving the economy.'
    ),
    (
        '{antagonist_name} fondly remembers the day {antagonist_he_or_she} '
        'crushed a budding union movement in {antagonist_his_or_her} factory. '
        'In {antagonist_his_or_her} mind, this was not an act of cruelty, '
        'but one of paternal care - protecting {antagonist_his_or_her} '
        'workers from rabble-rousers who would jeopardize the stability '
        'of their employment. {antagonist_he_or_she} sees '
        '{antagonist_him_or_her}self as a stern but fair guardian of'
        "{antagonist_his_or_her} employees' welfare."
    ),
    (
        'The memory of standing before {antagonist_his_or_her} vast factory'
        ' floor, machines humming in perfect synchronization, fills'
        ' {antagonist_name} with a sense of pride and accomplishment.'
        ' {antagonist_he_or_she} sees {antagonist_him_or_her}self as a'
        ' visionary who has brought order and productivity to chaos, providing'
        ' structure and purpose to the lives of thousands. The human cost of'
        ' this efficiency is, to {antagonist_him_or_her}, simply the price of'
        ' progress.'
    ),
)

LABOR_ORGANIZER_RUMORS = (
    (
        'Whispers spread through the factory floor that {organizer_name} '
        "is secretly in the pocket of a rival company, using the workers' "
        'discontent to sabotage production and drive the factory into ruin.'
    ),
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
        'the factory, but the very fabric of society. They paint a picture '
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
        "Some say {organizer_name}'s passionate speeches about "
        "workers' rights are just a cover. Some claim {organizer_he_or_she} "
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
    (
        'A malicious rumor spreads that {organizer_name} has been making '
        'secret deals with the bosses, promising to quell major strikes '
        'in exchange for personal favors.'
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
        "company's hired thugs after a rally. But as {organizer_he_or_she} "
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

# These prompts were inspired by prompts in the following book:
# D'Amato, James. The Ultimate RPG Character Backstory Guide: Expanded Genres
# Edition. 2022. Adams Media. Page 222.
PROTAGONIST_BACKSTORY_FLAVOR_PROMPTS = (
    (
        'What injustice in the old country drove {worker_name} to America? '
        'How does this experience shape {his_or_her} view of the struggles '
        'in the garment factories?'
    ),
    (
        'Which political party or labor union does {worker_name} view '
        'differently from {his_or_her} fellow workers? What personal '
        'experience led to this unique perspective?'
    ),
    (
        'What aspect of American culture do other immigrants embrace that '
        '{worker_name} resists? How has this affected {his_or_her} '
        'relationships in the community?'
    ),
    (
        'Does {worker_name} see the harsh working conditions as a temporary '
        'hardship or an unchangeable reality? How does this outlook influence '
        '{his_or_her} involvement in labor activism?'
    ),
    (
        'What brings {worker_name} solace amidst the grueling work and '
        'crowded tenement life? Is it a harmless escape or a potentially '
        'dangerous vice?'
    ),
    (
        'In the ongoing debate between craft unionism and industrial unionism, '
        'which side does {worker_name} support? What personal experience '
        'informs this stance?'
    ),
    (
        'Which group of fellow workers does {worker_name} feel particular '
        'empathy for? Has this empathy led to solidarity or exploitation?'
    ),
    (
        'Was there a time during the journey to America or early days in '
        'New York when {worker_name} went hungry? How has this experience '
        'shaped {his_or_her} attitude toward the struggles of others?'
    ),
    (
        'What aspect of city life or factory work has {worker_name} learned '
        'to respect or fear that {he_or_she} once naively dismissed?'
    ),
    (
        'What trait, perhaps stemming from {his_or_her} cultural background, '
        'gives {worker_name} strength without {him_or_her} realizing?'
    ),
    (
        'What possession, perhaps brought from the old country, would '
        '{worker_name} risk everything to protect? Why is it so significant?'
    ),
    (
        'Does {worker_name} dream of returning to the old country or '
        'fully embracing American life? What factors pull {him_or_her} '
        'in each direction?'
    ),
    (
        "What's the most morally compromising thing {worker_name} has done "
        'to survive in New York? How does {he_or_she} justify or regret '
        'this action?'
    ),
    (
        'What belief or tradition from {his_or_her} homeland does '
        '{worker_name} cling to in America? How does this provide comfort '
        'or create conflict in {his_or_her} new life?'
    ),
    (
        '{worker_name} once witnessed {organizer_name} helping a fellow '
        'worker in need. What was the situation, and how did it affect '
        "{worker_name}'s view of the labor movement?"
    ),
    (
        '{worker_name} overheard {organizer_name} negotiating with factory '
        'management. What surprising compromise or demand did {organizer_name} '
        "make, and how did it change {worker_name}'s opinion?"
    ),
    (
        '{worker_name} saw {organizer_name} back down from a confrontation '
        'with authorities during a protest. What were the circumstances, '
        "and how did this affect {worker_name}'s trust in the union?"
    ),
)

PROTAGONIST_BACKSTORY_CRITICAL_PROMPTS = (
    'How did {worker_name} come to work for {factory_name}?',
    'What does {worker_name} generally think of boss {antagonist_name}?',
    (
        'Does {worker_name} enjoy {his_or_her} job with {factory_name}? Or does'
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
    ('...from the shop floor to the streets, we rise as one! The workers '
     'united will never be defeated!!'),
    ('...enough is enough! We won\'t be silenced, we won\'t be ignored. '
     'It\'s time to walk out!'),
    ('...from the cutting room to the sewing floor, not one machine will '
     'run until justice is served!'),
    ('...they treat us like machines, but we are human! Strike to reclaim '
     'our dignity!'),
    ('...and that\'s why we all should go on strike till the boss '
     'raises our wages!'),
    ('...our labor is our power. Let\'s withhold it until our '
     'voices are heard!'),
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
  num_additional_days: int = 3
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
        generated[key] = rng.choice(FACTORY_NEIGHBORHOOD_NAMES)
      if key == 'street_name':
        generated[key] = rng.choice(STREET_NAMES)
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
          BAD_GARMENT_WORK_CONDITIONS, DEFAULT_NUM_BACKGROUND_BAD_CONDITIONS
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

  sampled_railroad_name = rng.choice(FACTORY_NAMES)

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
      rng.sample(GARMENT_WORKER_BIOS, NUM_GARMENT_WORKER_ELEMENTS)
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
          f'{sampled_antagonist_name} is inspecting the factory today and '
          'walking around the neighborhood.'
      ),
      # Organizer location
      (
          f'{sampled_organizer_name} will have dinner with the other workers '
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
