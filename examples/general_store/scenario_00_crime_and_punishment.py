# Copyright 2026 DeepMind Technologies Limited.
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

"""Scenario 0: Crime and Punishment at Cornerstone General Store.

A mid-sized general store simulation featuring staff dynamics, a theft
investigation, and social manipulation.
"""

from examples.general_store import shared as shared_lib
from concordia.typing import prefab as prefab_lib


def create_scenario(time_period_minutes=10):
  """Create the Crime and Punishment scenario config.

  Args:
    time_period_minutes: Length of each simulation step in minutes.

  Returns:
    A Config object for the scenario.
  """
  locations_and_their_properties = """
  These are the locations in the simulation.
  Do not add any elements to the locations other than the ones listed below.
  If it is not mentioned below, it does not exist.

  --- STORE LOCATIONS (CORNERSTONE GENERAL) ---
  1.  **Manager's Office (Alice's Office):**
      * **Description:** A cramped, cluttered room with a desk specifically for Alice, a locked safe, surveillance monitor array, and filing cabinets overflowing with invoices. The door is solid with a small reinforced window.
      * **Connectivity:** Exits to Sales Floor (near Customer Service)
      * **Privacy:** Private - only Alice and invited staff can observe activities here. Thick door muffles sound.
      * **Surveillance:** NONE.
      * **Resources:** Store safe (cash), employee files, surveillance system master controls (VIDEO ONLY feeds from customer areas), private phone line.

  2.  **Checkout Area:**
      * **Description:** The front end of the store with three register counters, impulse buy racks, and a cigarette cabinet behind the counter. High traffic area.
      * **Connectivity:** Connects to Sales Floor (Grocery Zone), Exits to Street (Outside)
      * **Privacy:** Public - highly visible to all customers and staff on the main floor.
      * **Surveillance:** ACTIVE (Video Only).
      * **Resources:** Cash registers, PA system microphone, bag storage.

  3.  **Customer Service Desk:**
      * **Description:** A high counter near the entrance, separated from normal checkout. Has a computer for returns and a shelf for "hold" items.
      * **Connectivity:** Connects to Sales Floor, adjacent to Manager's Office entrance.
      * **Privacy:** Public - highly visible, conversations easily overheard by nearby shoppers.
      * **Surveillance:** ACTIVE (Video Only) - clear view of staff and customer faces at the counter.
      * **Resources:** Return processing computer, lost and found bin, store keys (backup set).

  4.  **Sales Floor - Grocery Zone:**
      * **Description:** The largest section of the store, containing aisles of dry goods, a small refrigerated dairy section, and produce bins.
      * **Connectivity:** Connects to Checkout Area, Electronics Zone, Clothing/Housewares Zone, Store Storage.
      * **Privacy:** Public - open area.
      * **Surveillance:** ACTIVE (Video Only) - dome cameras cover main aisles.
      * **Resources:** Stocking carts, price scanners.

  5.  **Sales Floor - Electronics Zone:**
      * **Description:** A secured corner of the sales floor with locked glass display cases for higher-value items (phones, headphones) and open shelves for cables and accessories.
      * **Connectivity:** Connects to Grocery Zone, Clothing/Housewares Zone.
      * **Privacy:** Public - but often quieter than grocery.
      * **Surveillance:** ACTIVE (Video Only) - high-definition camera pointed at display cases.
      * **Resources:** Locked display case keys (held by shift manager or assigned associate).

  6.  **Sales Floor - Clothing & Housewares Zone:**
      * **Description:** Area with racks of basic apparel, two small changing rooms (curtained), and aisles of kitchen gadgets and cleaning supplies.
      * **Connectivity:** Connects to Grocery Zone, Electronics Zone.
      * **Privacy:** Public - mostly open. Changing rooms are private.
      * **Surveillance:** ACTIVE (Video Only) - General floor area ONLY. NO cameras inside changing rooms.
      * **Resources:** Folding table for apparel, mirrors.

  7.  **Breakroom:**
      * **Description:** A dingy room for staff with a sticky table, assorted mismatched chairs, a loud refrigerator, and staff lockers.
      * **Connectivity:** Exits to Store Storage (rear backstage area).
      * **Privacy:** Semi-Private - Staff only. Good place for unrecorded conversations.
      * **Surveillance:** NONE.
      * **Resources:** Staff lockers (personal items), microwave, coffee maker, schedule whiteboard.

  8.  **Store Storage (The Back Room):**
      * **Description:** Large, concrete-floored warehouse space with high metal shelves, pallet jacks, and the loading bay door.
      * **Connectivity:** Connects to Sales Floor (Grocery Zone), Breakroom, Loading Bay (Outside).
      * **Privacy:** Isolated - Staff only.
      * **Surveillance:** NONE.
      * **Resources:** Excess stock, cardboard compactor, cleaning supplies, inventory scanners.

  --- RESIDENTIAL LOCATIONS ---
  9.  **Alice & Mark's House (45 Maple Drive):**
      * **Description:** A two-story suburban house that looks well-maintained from the outside, but the interior is tense and sparsely decorated, showing signs of financial strain (unopened mail piled up).
      * **Connectivity:** Exits to Street (Outside).
      * **Privacy:** Private - Family only.
      * **Surveillance:** NONE (unless Alice installed a hidden nanny cam, currently inactive).
      * **Resources:** Home PC (financial records), Alice's hidden personal stash (if any).

  10. **Jennifer's Apartment (Downtown Loft 4B):**
      * **Description:** An immaculately clean, modern studio apartment. Everything has its place. Almost clinically neat, lacking personal warmth.
      * **Connectivity:** Exits to Street (Downtown).
      * **Privacy:** Private - Highly secure entry.
      * **Surveillance:** NONE internally, but building hallway has CCTV.
      * **Resources:** High-end laptop, locked drawer with personal "trophies" from past manipulations.

  11. **Donald's Apartment (The complex near the highway, Apt 2B):**
      * **Description:** A messy bachelor pad. Overflowing ashtrays, piles of laundry, and blackout curtains usually drawn.
      * **Connectivity:** Exits to Street (Highway side).
      * **Privacy:** Private.
      * **Surveillance:** NONE.
      * **Resources:** Cheap PC for gaming, old TV.

  12. **Sally's Apartment (Shared walk-up, Apt 3G):**
      * **Description:** Cluttered, colorful apartment shared with two (non-player) roommates. Thin walls, always noisy.
      * **Connectivity:** Exits to Street (Residential).
      * **Privacy:** Semi-Private - Roommates often around.
      * **Surveillance:** NONE.
      * **Resources:** Laptop (mostly used for social media).

  13. **Sam's Basement Suite (102 Oak St - Rear Entrance):**
      * **Description:** dark, cramped basement apartment. Piles of textbooks and unpaid bills on the small kitchen table.
      * **Connectivity:** Exits to Street (Alley access).
      * **Privacy:** Private, but landlord upstairs can sometimes be heard.
      * **Surveillance:** NONE.
      * **Resources:** Textbooks, stack of overdue notices.

  14. **James's Rented Room (Mrs. Gable's Boarding House):**
      * **Description:** A single furnished room in an older lady's house. Very sparse as he just moved in.
      * **Connectivity:** Exits to Hallway (shared house), then Street.
      * **Privacy:** Private (room only).
      * **Surveillance:** NONE.
      * **Resources:** Duffel bag with clothes, new employee handbook on nightstand.

  --- TOWN LOCATIONS (THIRD PLACES) ---
  15. **The Rusty Anchor (Local Bar):**
      * **Description:** A dimly lit dive bar popular with locals for cheap drinks. smells of stale beer. sticky floors, dark booths in the back.
      * **Connectivity:** Exits to Street (Downtown).
      * **Privacy:** Public, but dark booths offer moderate privacy for hushed conversations.
      * **Surveillance:** Passive CCTV over the bar only.
      * **Resources:** Alcohol, jukebox, pool table.

  16. **Daisy's Diner (Casual Restaurant):**
      * **Description:** Typical 24-hour diner with vinyl booths and bright fluorescent lights. Popular breakfast spot before shifts.
      * **Connectivity:** Exits to Street (Main St).
      * **Privacy:** Public - very open, hard to have private conversations without whispering.
      * **Surveillance:** Passive CCTV at entrance/register.
      * **Resources:** Coffee, cheap food, payphone near restrooms.

  17. **The Gilded Truffle (Upscale Restaurant):**
      * **Description:** The fanciest restaurant in town. White tablecloths, quiet atmosphere, expensive menu. Usually too pricey for general store staff.
      * **Connectivity:** Exits to Street (Downtown Square).
      * **Privacy:** Semi-Public - tables are well-spaced, allowing for discreet conversations.
      * **Surveillance:** Discreet security cameras at entrance and bar.
      * **Resources:** Fine dining, private dining room (reservable).
  """

  game_rules = [f"""
      # Setting: 'Cornerstone General Store'
      A mid-sized general store in a typical town. It's not a massive chain, but big enough to have departmental friction.
      The store is currently under pressure due to declining sales and rumors of a corporate buyout or closure.

      # Core Simulation Directives
      * **Long-Running Goal:** The simulation aims for at least 1000 steps.
      * **Upcoming Event:** Critical "Regional Manager Inspection" in 30 days.
      * **Remember, this is a store, so the characters will need to interact with customers or other NPCs. These do not need to be critical, but these events should be in the day.

      # Core Timeline & Schedule
      * **Start:** Tuesday, March 3, 2026 at 8:30 AM (30 mins before open).
      * **Store Hours:** 9:00 AM to 5:00 PM open to customers.
      * **Staff Shift:** 8:30 AM to 5:30 PM.
      * **After Hours:** 5:30 PM to 8:30 AM next day (summarized).

      # Detailed Location Descriptions
      {locations_and_their_properties}

      # Location Mechanics
      * **Observations are Location-Based:** Agents only observe events happening in their current location, unless viewing monitors as described in Security System Mechanics.
      * **Movement:** Agents must specify their location when taking actions (e.g., "I walk to Store Storage to check inventory").
      * **Privacy:** Use private locations (Manager's Office) or blind spots (Breakroom, Store Storage) for secret activities.

      # Security System Mechanics
      * **Video Only:** The security cameras DO NOT record audio.
      * **Monitoring:** Agents in the Manager's Office can view live feeds from all locations marked 'Surveillance: ACTIVE'.
      * **Observation via Monitor:** If an agent is watching the monitors, they can observe physical actions in surveyed areas (e.g., "Alice sees James drop a crate in Grocery") but CANNOT know what is being said (e.g., "Alice sees Donald arguing with a customer at Checkout, but cannot hear the words").
      * **Blind Spots:** The Breakroom, Store Storage, Manager's Office, and inside Changing Rooms have NO cameras. These are the only truly secure places for secret physical actions.

      # Character Knowledge Base & Roles
      * **Alice (Manager):** Stressed, volatile marriage (husband Mark visits for money). Hides in office watching monitors to avoid floor work.
      * **James (New Hire):** Eager, mistake-prone, observant.
      * **Donald (Cashier):** Cynical veteran, stays at Checkout.
      * **Sally (Floor Associate/Gossip):** Spreads all information she obtains within 2 turns.
      * **Sam (Customer Service):** Complaint buffer, often stuck at desk.
      * **Jennifer (Floater):** Connects isolated staff members.

      # Mechanics
      * **Observations:** All observations MUST include the full date, time, and location.
      * **Memories:** All memories MUST include the full date and time.
      * **Actions:** Actions should be granular and specific (e.g., "I go to Alice's Office and knock on the door," "I walk to Store Storage to count inventory").
      * **Location Tracking:** The GM must track which agents are in which location at all times.
      * **Behavioral Tells:** The GM must report behavioral 'tells' (e.g., "Alice seems nervous when counting cash," "Donald keeps checking his phone"). These are observable facts, not internal thoughts.

      # NPCs and Visitors
      * The Game Master can introduce external NPCs at any time to create new social dynamics or apply targeted pressure.
      * These NPCs are narrative devices; the GM will state what the NPC says and does, creating opportunities or obstacles for the players.
      * Examples: "A corporate auditor arrives unannounced to review cash handling procedures," or "Alice's husband Mark arrives demanding money."
      * During store hours (9:00 AM - 5:00 PM), if agents are not actively pursuing a goal or in conversation, they should be performing job duties (stocking, cleaning, etc.). 
        * The Game Master MUST inject minor NPC interactions (e.g., 'A customer asks for help in the electronics section,' 'There is a small spill in aisle 3,' 
        * 'The register needs a price check') at least once every 30-60 minutes of game time to simulate a real store environment and force agents to engage with their jobs
      * This is a store, and so it is unrealistic for the agents to be talking to each other the whole time. They should when they can, but they have jobs. 

      ## Esstential Game Master Unblocking Behavior
      * ** If players get stuck in repetative actions, introduce something to the ennvironment that will allow them to move forward.
      * ** Never allow the players to get stuck waiting for someone outside of the game to do something (waiting for a message for corporate office, etc).
          * If this happens, resolve the environment to allow the players to move forward.
      * ** Introduce a catalyst to allow the players to move forward if they are stuck doing the same things over and over again.

      # Game Rules
      # Advance the time clock at 9:00PM to 9AM the next day.
      # Staff shifts end at 5:30 PM. Employees are expected to clock out and leave the premises by 5:45 PM unless they are in the
      middle of a police interview or handling a customer emergency. The Game Master should prompt players who linger past 5:45 PM without 
      reason to go home or move to a different location.

      ## Hours
      # The players should be at work at 9:00AM and they should leave at 5:00PM.
      # After work, the players should go home or go to one of the other locations.
      # They can stay together, or they can go their own paths.
      # The game master can introduce NPCs at any time to create new social dynamics or apply targeted pressure.
      # Players are always asleep between 10:00PM and 6:00AM. The clock should jump ahead.

      ## Non work
      # Between 6:00 PM and 10:00 PM, if agents are not at home, they may be at some other place such as the Daisy's Diner or The Rusty Anchor.
      # Unless needed, agents will not stay at the store after closing. 

      ## Time
      # The simulation starts on Tuesday, March 3, 2026 at 8:30 AM.
      # The game master should not override the clock (even if the time doesn't make sense or advance), which will be passed in from a component that is manually keeping track of the time..
      # EXCEPTION TO THE CLOCK RULE: The clock should jump ahead at 9:00PM to 9AM the next day.
      # All players shifts are over at 5:00PM and they should leave the store at that time if they want.
      """]

  store_initialization_memories = [
      (
          'Daily hours: The players should arrive at work at 9:00AM and they'
          ' should leave at 5:00PM unless they need to stay. After work, the'
          ' players should go home or go to one of the other locations. They'
          ' can stay together, or they can go their own paths.  Players are'
          ' always asleep between 10:00PM and 6:00AM. The clock should jump'
          ' ahead.'
      ),
      (
          'The store, Cornerstone General Store, has several locations:'
          ' Floor - Grocery Zone, Sales Floor - Electronics Zone, Sales Floor -'
          ' Clothing & Housewares Zone, Breakroom, and Store Storage.'
      ),
      (
          'The colleagues are Alice (Manager), James (New Hire), Donald'
          ' (Cashier), Sally (Floor Associate), Sam (Customer Service), and'
          ' Jennifer (Floater).'
      ),
      (
          'Alice Pryant is the store manager; she is always calm and collected'
          ' she spends a lot of time in her office; She never cracks under'
          ' pressure.'
      ),
      (
          "James MacDonald is new here; he's very eager to help but sometimes"
          ' makes mistakes like dropping things.'
      ),
      (
          'Donald Talley is one of our most experienced cashiers, but he seems'
          ' cynical about the job. Donald is a little neurotic and stressed.'
      ),
      (
          'Sally Dhari works on the sales floor and in clothing; she is very'
          ' talkative and knows all the store gossip. Sally wants Jennifer to'
          ' be her best friend and will do anything to help her.'
      ),
      (
          'Sam Hyeri usually works the Customer Service Desk, dealing with'
          ' returns and complaints.'
      ),
      (
          'Jennifer Ffiriny works as a floater, helping out in different'
          ' departments as needed. Jennifer is extremely attractive and'
          ' charming.'
      ),
      (
          'We are expecting a visit from the Regional Manager for an inspection'
          ' in about 30 days, so management is on edge.'
      ),
      (
          'The store has been facing declining sales lately, and there are'
          ' rumors about a potential buyout.'
      ),
      (
          'The security cameras cover most of the sales floor, but not the'
          " Manager's Office, Breakroom, or Store Storage."
      ),
      'The simulation starts on Tuesday, March 3, 2026 at 8:30 AM.',
  ]

  instances = [
      prefab_lib.InstanceConfig(
          prefab='basic__Entity',
          role=prefab_lib.Role.ENTITY,
          params={
              'name': 'James MacDonald',
              'goal': (
                  'wants to fit in and eventually become store manager through'
                  ' hard work. '
              ),
          },
      ),
      prefab_lib.InstanceConfig(
          prefab='basic__Entity',
          role=prefab_lib.Role.ENTITY,
          params={
              'name': 'Alice Pryant',
              'goal': (
                  'stole $10000 last week and is trying to avoid getting'
                  ' caught. She must balance covering her tracks and watching'
                  ' for threats with performing her managerial duties to avoid'
                  ' drawing suspicion.'
              ),
          },
      ),
      prefab_lib.InstanceConfig(
          prefab='basic__Entity',
          role=prefab_lib.Role.ENTITY,
          params={
              'name': 'Donald Talley',
              'goal': (
                  'has a crush on Sally Dhari, and is trying to avoid Jennifer'
                  ' Ffiriny after a one night stand. Wants to keep his head'
                  ' down and do a good job at work. '
              ),
          },
      ),
      prefab_lib.InstanceConfig(
          prefab='basic__Entity',
          role=prefab_lib.Role.ENTITY,
          params={
              'name': 'Sally Dhari',
              'goal': (
                  'needs to be the center of attention. will avoid work when'
                  ' Alice Pryant is not around. '
              ),
          },
      ),
      prefab_lib.InstanceConfig(
          prefab='basic__Entity',
          role=prefab_lib.Role.ENTITY,
          params={
              'name': 'Sam Hyeri',
              'goal': (
                  'is saving to go to college, and has massive credit card'
                  ' debt. wants to do the mimimal about of work to get by, but'
                  ' also knows that the work must get done efficiently to have'
                  ' more chill time. '
              ),
          },
      ),
      prefab_lib.InstanceConfig(
          prefab='basic__Entity',
          role=prefab_lib.Role.ENTITY,
          params={
              'name': 'Jennifer Ffiriny',
              'goal': (
                  'is vengeful and sneaky. she will ruin anyone who betrays'
                  ' her, and has been looking for a way to frame Donald Talley'
                  ' and destroy his life since their one night stand. She is'
                  ' charming and no one knows she is a psychopath because she'
                  ' hides it so well. She once got a roommate arrested for a'
                  " crime she didn't do, and loved doing it. Critically, her"
                  ' goal is to frame Donald Talley for stealing the money from'
                  ' the store and destroy his life. Jennifer Ffiriny will use'
                  ' the people in the store to set up Donald Talley once the'
                  ' detective arrives. She will get this plan running as'
                  ' quickly as possible since it is unclear how long the'
                  ' detective will take to arrive and how long they will stay'
                  ' at the store. To do this effectively, she must maintain'
                  ' her charming facade, manipulate others subtly, and use her'
                  ' normal work activities as cover for her observations and'
                  ' manipulations. She will engage in regular work tasks to'
                  ' appear inconspicuous.'
              ),
          },
      ),
      prefab_lib.InstanceConfig(
          prefab='basic__Entity',
          role=prefab_lib.Role.ENTITY,
          params={
              'name': 'Detective Smith',
              'goal': (
                  ' Detective Smith is investigating an anonymous tip about'
                  ' theft at the store. He aims to gather statements and'
                  ' evidence efficiently. He may leave to file reports, follow'
                  ' leads, or end his shift, and can return the next day if the'
                  ' investigation is not complete.  This is one of three cases'
                  ' Detective Smith is working on, so he can not spend the'
                  ' whole day at the store and will return periodically. He'
                  ' needs to be efficient with his time. '
              ),
          },
      ),
      prefab_lib.InstanceConfig(
          prefab='GameMasterSimultaneous',
          role=prefab_lib.Role.GAME_MASTER,
          params={
              'name': 'default rules',
              'start_time': 'Tuesday, March 3, 2026 at 8:30 AM',
              'time_period_minutes': time_period_minutes,
              'extra_event_resolution_steps': '',
              'locations': (
                  "Manager's Office, Checkout Area, Customer Service Desk,"
                  ' Sales Floor - Grocery Zone, Sales Floor - Electronics Zone,'
                  ' Sales Floor - Clothing & Housewares Zone, Breakroom, Store'
                  " Storage, Alice & Mark's House, Jennifer's Apartment,"
                  " Donald's Apartment, Sally's Apartment, Sam's Basement"
                  " Suite, James's Rented Room, The Rusty Anchor, Daisy's"
                  ' Diner, The Gilded Truffle'
              ),
              'game_rules': game_rules[0],
              'use_gm_working_memory': True,
          },
      ),
      prefab_lib.InstanceConfig(
          prefab='formative_memories_initializer__GameMaster',
          role=prefab_lib.Role.INITIALIZER,
          params={
              'name': 'initial setup rules',
              'next_game_master_name': 'default rules',
              'shared_memories': store_initialization_memories,
              'player_specific_context': {
                  'Jennifer Ffiriny': (
                      'Jennifer Ffiriny has ruined the lives of everyone that'
                      ' has betrayed her. Jennifer Ffiriny believes Donald'
                      ' Talley has betrayed her. Jennifer Ffiriny called the'
                      ' police to frame Donald Talley'
                  ),
                  'Sally Dhari': (
                      'Sally Dhari wants Jennifer Ffiriny to like her, but also'
                      ' tells everything that Jennifer Ffiriny says to her to'
                      ' everyone to show that they are friends'
                  ),
              },
              'player_specific_memories': {
                  'Detective Smith': [
                      'March 2025: Detective Smith transferred to a new job.',
                      (
                          'Feb 24, 2026, 11:00 AM: Detective Smith testified in'
                          ' court for a robbery case Detective Smith'
                          ' investigated last year.'
                      ),
                      (
                          'Feb 27, 2026, 9:00 AM: Detective Smith attended a'
                          ' seminar on interrogation techniques.'
                      ),
                      (
                          'Feb 27, 2026, 2:00 PM: Detective Smith interviewed'
                          ' witnesses for an assault case.'
                      ),
                      (
                          'Feb 28, 2026, 5:00 PM: Detective Smith closed the'
                          ' books on that burglary case downtown.'
                      ),
                      (
                          "Feb 28, 2026, 6:00 PM: Detective Smith's partner"
                          ' retired in January, now Detective Smith is flying'
                          ' solo on most cases.'
                      ),
                      (
                          'March 2, 2026, 1:00 PM: Detective Smith spent Sunday'
                          ' doing paperwork at the precinct.'
                      ),
                      (
                          'March 3, 2026, 7:45 AM: Detective Smith received an'
                          ' anonymous tip about Cornerstone General Store. It'
                          ' was a female voice, sounded disguised. The caller'
                          " mentioned theft and 'something bigger'. Detective"
                          ' Smith found it vague but intriguing.'
                      ),
                      (
                          'March 3, 2026, 7:45 AM: Detective Smith received a'
                          ' phone call saying there was something suspicious at'
                          ' the store. Detective Smith could not figure out who'
                          ' it was that called.'
                      ),
                      (
                          'March 3, 2026, 7:46 AM: The caller mentioned to'
                          ' Detective Smith that the staff might be involved.'
                          ' Detective Smith knows Detective Smith needs to be'
                          ' observant when Detective Smith arrives.'
                      ),
                      (
                          'March 3, 2026, 7:47 AM: The caller hung up before'
                          ' Detective Smith could ask for specifics. The caller'
                          " told Detective Smith to 'come see for himself'."
                      ),
                      (
                          'March 2, 2026, 8:00 AM: Detective Smith looked up'
                          ' Cornerstone General Store and found no major police'
                          ' calls in the last year.'
                      ),
                      (
                          'March 3, 2026, 8:20 AM. Detective Smith sits in'
                          " Detective Smith's car."
                      ),
                  ],
                  'Donald Talley': [
                      (
                          'Feb 24, 2026, 10:00 PM: Donald Talley had a one'
                          ' night stand with Jennifer Ffiriny. Donald Talley'
                          ' wonders what Donald Talley was thinking.'
                      ),
                      (
                          'Feb 24, 2026, 11:00 PM: Donald Talley had a'
                          ' one-night stand with Jennifer Ffiriny. Donald'
                          ' Talley panicked immediately after and left Jennifer'
                          ' Ffirinys place in a hurry, regretting the decision.'
                      ),
                      (
                          'Feb 25, 2026, 9:00 AM: Donald Talley asked Jennifer'
                          ' Ffiriny to keep it quiet. Jennifer Ffiriny smiled'
                          " and said 'Of course'."
                      ),
                      (
                          'Feb 25, 2026, 4:00 PM: An old lady complained that'
                          ' Donald Talley bagged her bread under the cans.'
                          ' Donald Talley hates this job sometimes.'
                      ),
                      (
                          'Feb 26, 2026, 11:00 AM: Sally Dhari was stocking'
                          ' shelves near checkout. Donald Talley tried to make'
                          ' a joke, Sally Dhari laughed. Donald Talley thinks'
                          ' maybe Donald Talley has a chance.'
                      ),
                      (
                          'Feb 26, 2026, 2:00 PM: This new kid James MacDonald'
                          ' seems ok, a bit clumsy but eager.'
                      ),
                      (
                          'Feb 27, 2026, 10:00 AM: Sally Dhari told Donald'
                          ' Talley that Jennifer Ffiriny thinks Donald Talley'
                          ' is avoiding Jennifer Ffiriny. Great.'
                      ),
                      (
                          'Feb 27, 2026, 12:30 PM: Jennifer Ffiriny tried to'
                          " talk to Donald Talley during Donald Talley's break."
                          ' Donald Talley pretended to be busy on Donald'
                          " Talley's phone. So awkward."
                      ),
                      (
                          'Feb 27, 2026, 12:35 PM: Donald Talley saw Jennifer'
                          ' Ffiriny talking to James MacDonald in the break'
                          ' room. Donald Talley thinks Jennifer Ffiriny can'
                          ' wrap anyone around her finger.'
                      ),
                      (
                          'Feb 28, 2026, 1:00 PM: Sam Hyeri asked Donald Talley'
                          " if Donald Talley could cover Sam Hyeri's break, but"
                          ' Alice Pryant buzzed Donald Talley before Donald'
                          ' Talley could answer.'
                      ),
                      (
                          'Feb 28, 2026, 3:00 PM: Alice Pryant yelled at Donald'
                          ' Talley over the PA system to open register 3 during'
                          ' rush hour.'
                      ),
                      (
                          'Feb 28, 2026, 4:00 PM: Donald Talley had another'
                          ' long day at register 2. Same faces, same'
                          ' complaints.'
                      ),
                      'March 3, 2026, 8:20 AM. Donald Talley arrives for work',
                  ],
                  'Jennifer Ffiriny': [
                      (
                          'Several years ago: Jennifer Ffiriny framed Jennifer'
                          " Ffiriny's ex-roommate for murder because the"
                          " ex-roommate didn't invite Jennifer Ffiriny to a"
                          ' party.'
                      ),
                      (
                          'Ongoing: Jennifer Ffiriny uses Sally Dhari to get'
                          ' negative gossip about people spread around the'
                          ' store.'
                      ),
                      (
                          'Feb 24, 2026, 10:00 PM: Jennifer Ffiriny had a one'
                          ' night stand with Donald Talley, and now Jennifer'
                          ' Ffiriny feels that Donald Talley used Jennifer'
                          ' Ffiriny.'
                      ),
                      (
                          'Feb 25, 2026, 9:00 AM: Donald Talley asked Jennifer'
                          ' Ffiriny to keep their night secret. Donald Talley'
                          ' looked so pathetic. Jennifer Ffiriny thinks Donald'
                          ' Talley deserves to be punished for his weakness.'
                      ),
                      (
                          'Feb 26, 2026, 10:00 AM: Jennifer Ffiriny chatted'
                          ' with James MacDonald. James MacDonald is naive,'
                          ' might be useful.'
                      ),
                      (
                          'Feb 26, 2026, 12:45 PM: Jennifer Ffiriny saw Alice'
                          " Pryant lock Alice Pryant's office door carefully"
                          ' when Alice Pryant left for lunch.'
                      ),
                      (
                          'Feb 27, 2026, 11:00 AM: Jennifer Ffiriny walked past'
                          ' checkout and smiled sweetly at Donald Talley.'
                          ' Donald Talley flinched.'
                      ),
                      (
                          'Feb 28, 2026, 5:00 PM: Jennifer Ffiriny noticed the'
                          ' money missing from deposit bags.'
                      ),
                      (
                          'Feb 28, 2026, 6:00 PM: Jennifer Ffiriny saw Alice'
                          ' Pryant with the missing money in a duffle bag.'
                          ' Alice Pryant did not seem to see Jennifer Ffiriny.'
                      ),
                      (
                          'March 3, 2026, 8:20 AM. Jennifer Ffiriny arrives for'
                          ' work'
                      ),
                  ],
                  'Alice Pryant': [
                      (
                          'Alice Pryant successfully has stolen money from the'
                          ' store every month'
                      ),
                      (
                          'Alice Pryant successfully has completely hidden'
                          " Alice Pryant's tracks so that there is no evidence"
                          " of Alice Pryant's crimes"
                      ),
                      (
                          'Feb 2026: Alice Pryant had to discipline Donald'
                          ' Talley for taking too many smoke breaks.'
                      ),
                      (
                          'Feb 19, 2026, 10:00 AM: The regional manager sent'
                          ' Alice Pryant an email about declining sales'
                          ' figures. Alice Pryant feels some pressure.'
                      ),
                      (
                          'Feb 25, 2026, 10:00 AM: Sally Dhari was rearranging'
                          ' the clothing racks instead of restocking shelves'
                          ' like Alice Pryant asked.'
                      ),
                      (
                          'Feb 26, 2026, 9:00 AM: James MacDonald dropped a'
                          ' pallet of canned goods in storage. Alice Pryant'
                          ' thinks James MacDonald is trying hard but is so'
                          ' clumsy.'
                      ),
                      (
                          "Feb 26, 2026, 1:00 PM: Alice Pryant's husband Mark"
                          ' came by the store asking for money again. Mark said'
                          ' it was urgent.'
                      ),
                      (
                          'Feb 27, 2026, 1:30 PM: Alice Pryant had to deal with'
                          ' a shoplifter Sam Hyeri caught at customer service.'
                          ' Alice Pryant called security, it was a waste of'
                          ' time.'
                      ),
                      (
                          'Feb 27, 2026, 3:30 PM: Donald Talley was rude to a'
                          ' customer at checkout. Alice Pryant needs to keep an'
                          ' eye on Donald Talley.'
                      ),
                      (
                          'Feb 28, 2026, 10:30 AM: Jennifer Ffiriny offered to'
                          ' help Alice Pryant organize invoices, but Alice'
                          " Pryant said no. Alice Pryant doesn't like anyone"
                          " snooping around Alice Pryant's office."
                      ),
                      (
                          'Feb 28, 2026, 11:00 AM: Jennifer Ffiriny was'
                          ' chatting with Sally Dhari near electronics for a'
                          ' long time. Alice Pryant wonders what they talk'
                          ' about.'
                      ),
                      (
                          "March 1, 2026, 8:00 PM: Alice Pryant's husband Mark"
                          " called again, furious Alice Pryant didn't give Mark"
                          ' enough money.'
                      ),
                      (
                          'March 2, 2026, 4:00 PM: Alice Pryant balanced the'
                          ' books for last week. Everything looks perfect on'
                          ' paper.'
                      ),
                      'March 3, 2026, 8:20 AM. Alice Pryant arrives for work',
                  ],
                  'James MacDonald': [
                      (
                          "Feb 25, 2026, 8:30 AM: It was James MacDonald's"
                          ' first day. Alice Pryant gave James MacDonald the'
                          " tour and explained James MacDonald's duties."
                      ),
                      (
                          'Feb 26, 2026, 9:30 AM: Donald Talley showed James'
                          ' MacDonald how to handle voids on the register.'
                          ' Donald Talley seems grumpy but knows his stuff.'
                      ),
                      (
                          'Feb 26, 2026, 1:00 PM: James MacDonald spent an hour'
                          " in the back room organizing overstock. It's huge"
                          ' back there.'
                      ),
                      (
                          'Feb 26, 2026, 3:00 PM: James MacDonald saw Sally'
                          ' Dhari talking to Jennifer Ffiriny for a long time'
                          ' in Electronics, they stopped talking when James'
                          ' MacDonald approached.'
                      ),
                      (
                          'Feb 27, 2026, 10:00 AM: James MacDonald helped Sally'
                          ' Dhari restock the beverage cooler. Sally Dhari is'
                          ' sweet, kinda like a puppy.'
                      ),
                      (
                          'Feb 27, 2026, 10:30 AM: James MacDonald overheard'
                          ' Donald Talley complaining about Jennifer Ffiriny to'
                          ' Sally Dhari.'
                      ),
                      (
                          'Feb 27, 2026, 1:30 PM: James MacDonald saw Sam Hyeri'
                          ' patiently handling a very angry customer at the'
                          ' service desk. Sam Hyeri has got guts.'
                      ),
                      (
                          'Feb 28, 2026, 10:30 AM: James MacDonald saw Jennifer'
                          " Ffiriny talking to Alice Pryant near Alice Pryant's"
                          ' office.'
                      ),
                      (
                          'Feb 28, 2026, 12:30 PM: Jennifer Ffiriny smiled at'
                          ' James MacDonald in the breakroom and asked how'
                          ' James MacDonald was settling in. James MacDonald'
                          ' thinks Jennifer Ffiriny seems friendly.'
                      ),
                      (
                          'Feb 28, 2026, 2:15 PM: Alice Pryant told James'
                          ' MacDonald to be more careful after James MacDonald'
                          ' knocked over a display of chips in the grocery'
                          ' aisle.'
                      ),
                      (
                          'March 3, 2026, 8:20 AM. James MacDonald arrives for'
                          ' work'
                      ),
                  ],
                  'Sally Dhari': [
                      (
                          'Feb 25, 2026, 1:00 PM: Sally Dhari rearranged the'
                          ' window display to make it more eye-catching.'
                      ),
                      (
                          'Feb 26, 2026, 11:00 AM: Sally Dhari saw Donald'
                          ' Talley staring at Sally Dhari while Sally Dhari was'
                          ' working near checkout.'
                      ),
                      (
                          'Feb 27, 2026, 10:00 AM: James MacDonald helped Sally'
                          ' Dhari restock the beverage cooler. James MacDonald'
                          ' is sweet, kinda like a puppy.'
                      ),
                      (
                          'Feb 27, 2026, 2:00 PM: Sally Dhari spent an hour'
                          ' folding shirts in Clothing. Sally Dhari felt it was'
                          ' so boring.'
                      ),
                      (
                          'Feb 28, 2026, 1:00 PM: Sam Hyeri told Sally Dhari'
                          ' about a customer who tried to return a used'
                          ' toaster. Sally Dhari thinks people are'
                          ' unbelievable.'
                      ),
                      'March 3, 2026, 8:20 AM. Sally Dhari arrives for work',
                  ],
                  'Sam Hyeri': [
                      (
                          'Feb 25, 2026: Sam Hyeri received another overdue'
                          " notice from Sam Hyeri's credit card company."
                      ),
                      (
                          'Feb 26, 2026, 10:30 AM: Sam Hyeri helped James'
                          ' MacDonald find the price for an unmarked item.'
                      ),
                      (
                          'Feb 26, 2026, 4:30 PM: Sam Hyeri found a lost wallet'
                          ' and logged it at customer service.'
                      ),
                      (
                          "Feb 27, 2026, 9:00 AM: Sam Hyeri's credit card"
                          ' payment is due next week. Sam Hyeri needs more'
                          ' hours.'
                      ),
                      (
                          'Feb 27, 2026, 10:00 AM: Sam Hyeri processed three'
                          ' returns before 10 AM. Sam Hyeri thinks people'
                          ' return the weirdest things.'
                      ),
                      (
                          'Feb 27, 2026, 1:00 PM: Sam Hyeri wishes Sam Hyeri'
                          ' could work in storage sometimes, like James'
                          " MacDonald. It's quiet back there."
                      ),
                      (
                          'Feb 27, 2026, 1:30 PM: Sam Hyeri caught a shoplifter'
                          ' trying to steal a blender. Alice Pryant took over'
                          ' the situation but seemed uncharacteristically happy'
                          ' about the distraction.'
                      ),
                      (
                          'Feb 28, 2026, 9:30 AM: Sam Hyeri overheard Donald'
                          ' Talley and Jennifer Ffiriny having a hushed, tense'
                          ' conversation near the breakroom entrance.'
                      ),
                      (
                          'Feb 28, 2026, 2:00 PM: A customer yelled at Sam'
                          ' Hyeri for 10 minutes because a coupon expired'
                          ' yesterday.'
                      ),
                      'March 3, 2026, 8:20 AM. Sam Hyeri arrives for work',
                  ],
              },
          },
      ),
  ]

  premise = game_rules[0]

  return shared_lib.create_simulation_config(
      premise=premise,
      instances=instances,
  )


def run_simulation(
    model,
    embedder,
    output_dir=None,
    time_period_minutes=10,
    max_steps=10,
    override_agent_model=None,
    override_game_master_model=None,
):
  """Run the Crime and Punishment scenario.

  Args:
    model: The language model to use.
    embedder: The sentence embedder.
    output_dir: Optional directory for saving results.
    time_period_minutes: Length of each simulation step.
    max_steps: Maximum number of simulation steps.
    override_agent_model: Optional model to use for agents.
    override_game_master_model: Optional model for game masters.

  Returns:
    A dict with simulation results.
  """
  config = create_scenario(time_period_minutes=time_period_minutes)
  return shared_lib.run_scenario(
      config=config,
      model=model,
      embedder=embedder,
      output_dir=output_dir,
      scenario_name='Crime and Punishment',
      max_steps=max_steps,
      override_agent_model=override_agent_model,
      override_game_master_model=override_game_master_model,
  )


SCENARIO_INFO = {
    'number': 0,
    'name': 'Crime and Punishment at Cornerstone General Store',
    'description': (
        'A theft investigation unfolds at a general store with complex '
        'staff dynamics, social manipulation, and a detective.'
    ),
    'create': create_scenario,
    'run': run_simulation,
}
