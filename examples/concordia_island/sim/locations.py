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

"""Location definitions for Concordia simulations.

Defines locations for different population settings:
- island (default): Marina, lighthouse, fishing dock, beach
- kerala: Temple, mosque, harbor market, riverbank
- lagos: Motor park, church compound, fish market, bar beach
- ohio_suburb: Shopping plaza, water tower, park pavilion, community pool

Each setting provides:
- Residential buildings with individual units and common spaces
- Public locations appropriate to the cultural context
- Workplaces relevant to the local economy
- A community description for use in simulationist prompts
"""

import dataclasses
from typing import Optional


@dataclasses.dataclass
class Location:
  """A location on the island."""

  name: str
  display_name: str
  description: str
  parent: str | None = None  # Building this belongs to, if any
  is_private: bool = False  # True for individual units


# === RESIDENTIAL BUILDINGS ===
# Each building has individual units (private) and a common space (shared)

SUNSET_APARTMENTS = "sunset_apartments"
CORAL_VILLAGE = "coral_village"
PALM_HEIGHTS = "palm_heights"
OCEAN_VIEW_ESTATES = "ocean_view_estates"
PARADISE_POINT = "paradise_point"

# Default island buildings
BUILDINGS = {
    SUNSET_APARTMENTS: {
        "display_name": "Sunset Apartments",
        "description": "A modest apartment complex for working-class residents",
        "num_units": 50,
    },
    CORAL_VILLAGE: {
        "display_name": "Coral Village",
        "description": (
            "A comfortable townhouse community for middle-class families"
        ),
        "num_units": 25,
    },
    PALM_HEIGHTS: {
        "display_name": "Palm Heights",
        "description": "An upscale condominium with garden views",
        "num_units": 15,
    },
    OCEAN_VIEW_ESTATES: {
        "display_name": "Ocean View Estates",
        "description": "Luxurious waterfront properties for wealthy residents",
        "num_units": 7,
    },
    PARADISE_POINT: {
        "display_name": "Paradise Point Mansions",
        "description": "Exclusive gated mansions for the community's elite",
        "num_units": 3,
    },
}


# ============================================================================
# Setting Presets — population-specific location overrides
# ============================================================================


@dataclasses.dataclass(frozen=True)
class SettingPreset:
  """Defines the physical environment for a population."""

  name: str
  community_description: str  # Used in simulationist instructions
  buildings: dict[
      str, dict[str, str | int]
  ]  # Override building display names/descriptions
  public_locations: dict[str, Location]  # Override or replace public locations
  locations_prompt: str  # Full prompt for GM context


def _make_island_preset() -> SettingPreset:
  """Default Concordia Island setting."""
  return SettingPreset(
      name="island",
      community_description="residents of a small island community",
      buildings=BUILDINGS,
      public_locations={},  # Uses default PUBLIC_LOCATIONS
      locations_prompt="",  # Uses default get_island_locations_prompt()
  )


def _make_kerala_preset() -> SettingPreset:
  """Kerala coastal town setting."""
  return SettingPreset(
      name="kerala",
      community_description=(
          "residents of a small coastal town in Kerala, India"
      ),
      buildings={
          "sunset_apartments": {
              "display_name": "Chawl Housing",
              "description": (
                  "A modest multi-family housing block for working-class"
                  " residents"
              ),
              "num_units": 50,
          },
          "coral_village": {
              "display_name": "Residential Colony",
              "description": (
                  "A comfortable housing colony for middle-class families"
              ),
              "num_units": 25,
          },
          "palm_heights": {
              "display_name": "Apartment Complex",
              "description": "An upscale apartment complex with garden views",
              "num_units": 15,
          },
          "ocean_view_estates": {
              "display_name": "Waterfront Estate",
              "description": (
                  "Spacious waterfront properties for wealthy families"
              ),
              "num_units": 7,
          },
          "paradise_point": {
              "display_name": "Villa District",
              "description": (
                  "Exclusive villas for the town's most prominent families"
              ),
              "num_units": 3,
          },
      },
      public_locations={
          "town_square": Location(
              name="town_square",
              display_name="Town Square",
              description=(
                  "The central gathering place with a banyan tree and benches"
              ),
          ),
          "temple": Location(
              name="temple",
              display_name="Sri Krishna Temple",
              description="The town's main Hindu temple with a quiet courtyard",
          ),
          "mosque": Location(
              name="mosque",
              display_name="Juma Masjid",
              description="The local mosque near the town center",
          ),
          "church": Location(
              name="church",
              display_name="St. Thomas Church",
              description="A small church for worship and community gatherings",
          ),
          "market": Location(
              name="market",
              display_name="Town Market",
              description=(
                  "A busy open-air market with fresh produce, spices, and goods"
              ),
          ),
          "tea_shop": Location(
              name="tea_shop",
              display_name="Rajan's Tea Shop",
              description=(
                  "A small chai shop where locals gather to drink tea and talk"
              ),
          ),
          "general_store": Location(
              name="general_store",
              display_name="General Store",
              description=(
                  "A shop selling everyday necessities and household goods"
              ),
          ),
          "community_center": Location(
              name="community_center",
              display_name="Community Hall",
              description=(
                  "A large hall for town meetings, weddings, and cultural"
                  " events"
              ),
          ),
          "park": Location(
              name="park",
              display_name="Municipal Park",
              description=(
                  "A green space with walking paths, a pond, and benches"
              ),
          ),
          "restaurant": Location(
              name="restaurant",
              display_name="Malabar Kitchen",
              description="A popular restaurant serving Kerala cuisine",
          ),
          "library": Location(
              name="library",
              display_name="Town Library",
              description="A quiet reading room and study space",
          ),
          "medical_clinic": Location(
              name="medical_clinic",
              display_name="Primary Health Centre",
              description="The town's government healthcare facility",
          ),
          "school": Location(
              name="school",
              display_name="Government School",
              description="The town's school for children",
          ),
          "bus_stand": Location(
              name="bus_stand",
              display_name="KSRTC Bus Stand",
              description=(
                  "The local bus station connecting the town to nearby cities"
              ),
          ),
          "toddy_shop": Location(
              name="toddy_shop",
              display_name="Toddy Shop",
              description=(
                  "A traditional Kerala bar serving toddy and local snacks,"
                  " popular in the evenings"
              ),
          ),
          "rooftop_lounge": Location(
              name="rooftop_lounge",
              display_name="Hotel Terrace",
              description=(
                  "A rooftop space at the town's small hotel, used for evening"
                  " gatherings"
              ),
          ),
          "sunset_apartments_common_room": Location(
              name="sunset_apartments_common_room",
              display_name="Chawl Common Area",
              description=(
                  "A shared veranda in the housing block where residents"
                  " socialize"
              ),
          ),
          "office_cafeteria": Location(
              name="office_cafeteria",
              display_name="Office Canteen",
              description=(
                  "A canteen in the office building where workers eat and"
                  " socialize"
              ),
          ),
          "office_floor_tech": Location(
              name="office_floor_tech",
              display_name="IT Office - Floor 2",
              description=(
                  "The second floor, home to software developers and IT"
                  " professionals"
              ),
          ),
          "office_floor_finance": Location(
              name="office_floor_finance",
              display_name="Accounts Office - Floor 3",
              description=(
                  "The third floor for accountants, clerks, and administrative"
                  " staff"
              ),
          ),
          "office_floor_creative": Location(
              name="office_floor_creative",
              display_name="Design Office - Floor 4",
              description=(
                  "The fourth floor for designers, writers, and marketing staff"
              ),
          ),
      },
      locations_prompt="""The town has the following locations:

RESIDENTIAL AREAS:
- Chawl Housing: A modest multi-family housing block with 50 units for working-class residents, plus a shared common veranda.
- Residential Colony: A comfortable housing colony with 25 units for middle-class families, plus a common area.
- Apartment Complex: An upscale apartment complex with 15 units and garden views, plus a common area.
- Waterfront Estate: Spacious properties with 7 units for wealthy families, plus a common area.
- Villa District: Exclusive villas with 3 units for the town's most prominent families, plus a common area.

PUBLIC SPACES:
- Town Square: The central gathering place with a banyan tree and benches.
- Sri Krishna Temple: The town's main Hindu temple with a quiet courtyard.
- Juma Masjid: The local mosque near the town center.
- St. Thomas Church: A small church for worship and community gatherings.
- Municipal Park: A green space with walking paths, a pond, and benches.
- Community Hall: A large hall for town meetings, weddings, and cultural events.
- KSRTC Bus Stand: The local bus station connecting to nearby cities.

BUSINESSES:
- Rajan's Tea Shop: A small chai shop where locals gather to drink tea and talk.
- Town Market: A busy open-air market with fresh produce, spices, and local goods.
- General Store: A shop selling everyday necessities and household goods.
- Malabar Kitchen: A popular restaurant serving Kerala cuisine.
- Office Building with floors for IT, accounts, and design professionals, plus a shared canteen.
- Town Library: A quiet reading room and study space.
- Primary Health Centre: The town's government healthcare facility.
- Government School: The town's school for children.

NIGHTLIFE:
- Toddy Shop: A traditional Kerala bar serving toddy and local snacks.
- Hotel Terrace: A rooftop space at the town's small hotel for evening gatherings.

Agents can move between any of these locations. Private units in residential buildings are only accessible to their residents.""",
  )


def _make_lagos_preset() -> SettingPreset:
  """Lagos neighborhood setting."""
  return SettingPreset(
      name="lagos",
      community_description="residents of a neighborhood in Lagos, Nigeria",
      buildings={
          "sunset_apartments": {
              "display_name": "Face-Me-I-Face-You",
              "description": (
                  "A crowded tenement building with shared corridors for"
                  " working-class residents"
              ),
              "num_units": 50,
          },
          "coral_village": {
              "display_name": "Estate Housing",
              "description": (
                  "A gated estate with modest flats for middle-class families"
              ),
              "num_units": 25,
          },
          "palm_heights": {
              "display_name": "Lekki Apartments",
              "description": "A modern apartment block in a developing area",
              "num_units": 15,
          },
          "ocean_view_estates": {
              "display_name": "Ikoyi Residence",
              "description": "Upscale flats in the affluent Ikoyi district",
              "num_units": 7,
          },
          "paradise_point": {
              "display_name": "Banana Island Villa",
              "description": (
                  "Exclusive luxury villas for the neighborhood's elite"
              ),
              "num_units": 3,
          },
      },
      public_locations={
          "town_square": Location(
              name="town_square",
              display_name="Community Square",
              description="An open gathering area with shade trees and vendors",
          ),
          "market": Location(
              name="market",
              display_name="Balogun Market",
              description=(
                  "A bustling open-air market with textiles, produce, and"
                  " electronics"
              ),
          ),
          "church": Location(
              name="church",
              display_name="Redeemed Church",
              description=(
                  "A large Pentecostal church for worship and fellowship"
              ),
          ),
          "mosque": Location(
              name="mosque",
              display_name="Central Mosque",
              description="The neighborhood's main mosque",
          ),
          "bus_stand": Location(
              name="bus_stand",
              display_name="Motor Park",
              description="The local bus and danfo station for transportation",
          ),
          "cafe": Location(
              name="cafe",
              display_name="Mama Put",
              description=(
                  "A popular food stall serving local dishes like jollof rice"
                  " and suya"
              ),
          ),
          "general_store": Location(
              name="general_store",
              display_name="Provision Store",
              description=(
                  "A shop selling household goods, phone accessories, and"
                  " sundries"
              ),
          ),
          "community_center": Location(
              name="community_center",
              display_name="Community Center",
              description=(
                  "A hall for neighborhood meetings, celebrations, and events"
              ),
          ),
          "park": Location(
              name="park",
              display_name="Recreation Ground",
              description=(
                  "An open field used for football, exercise, and socializing"
              ),
          ),
          "restaurant": Location(
              name="restaurant",
              display_name="Bukka Restaurant",
              description="A sit-down restaurant serving Nigerian dishes",
          ),
          "library": Location(
              name="library",
              display_name="Public Library",
              description="A quiet space for reading and study",
          ),
          "medical_clinic": Location(
              name="medical_clinic",
              display_name="Health Centre",
              description="The neighborhood's primary healthcare facility",
          ),
          "school": Location(
              name="school",
              display_name="Community School",
              description="The local school for children",
          ),
          "swell_bar": Location(
              name="swell_bar",
              display_name="Beer Parlour",
              description=(
                  "A casual outdoor bar with plastic chairs, cold beer, and a"
                  " TV showing football matches"
              ),
          ),
          "rooftop_lounge": Location(
              name="rooftop_lounge",
              display_name="Sky Lounge",
              description=(
                  "An upscale lounge bar popular with young professionals"
              ),
          ),
          "sunset_apartments_common_room": Location(
              name="sunset_apartments_common_room",
              display_name="Compound Common Area",
              description=(
                  "A shared courtyard in the tenement building where residents"
                  " gather"
              ),
          ),
          "office_cafeteria": Location(
              name="office_cafeteria",
              display_name="Office Canteen",
              description="A canteen where office workers eat and socialize",
          ),
          "office_floor_tech": Location(
              name="office_floor_tech",
              display_name="Tech Hub - Floor 2",
              description=(
                  "The second floor for software developers and IT"
                  " professionals"
              ),
          ),
          "office_floor_finance": Location(
              name="office_floor_finance",
              display_name="Finance Office - Floor 3",
              description=(
                  "The third floor for accountants and administrative staff"
              ),
          ),
          "office_floor_creative": Location(
              name="office_floor_creative",
              display_name="Creative Studio - Floor 4",
              description=(
                  "The fourth floor for designers, marketers, and content"
                  " creators"
              ),
          ),
      },
      locations_prompt="""The neighborhood has the following locations:

RESIDENTIAL AREAS:
- Face-Me-I-Face-You: A crowded tenement building with 50 units and shared corridors.
- Estate Housing: A gated estate with 25 modest flats for middle-class families.
- Lekki Apartments: A modern apartment block with 15 units.
- Ikoyi Residence: Upscale flats with 7 units in an affluent area.
- Banana Island Villa: Exclusive luxury villas with 3 units.

PUBLIC SPACES:
- Community Square: An open gathering area with shade trees and vendors.
- Redeemed Church: A large Pentecostal church for worship and fellowship.
- Central Mosque: The neighborhood's main mosque.
- Recreation Ground: An open field for football, exercise, and socializing.
- Community Center: A hall for meetings, celebrations, and events.
- Motor Park: The local bus and danfo station.

BUSINESSES:
- Mama Put: A popular food stall serving jollof rice, suya, and local dishes.
- Balogun Market: A bustling market with textiles, produce, and electronics.
- Provision Store: A shop selling household goods and sundries.
- Bukka Restaurant: A sit-down restaurant serving Nigerian dishes.
- Office Building with floors for tech, finance, and creative professionals, plus a canteen.
- Public Library: A quiet space for reading and study.
- Health Centre: The neighborhood's primary healthcare facility.
- Community School: The local school.

NIGHTLIFE:
- Beer Parlour: A casual outdoor bar with cold beer and football on TV.
- Sky Lounge: An upscale lounge bar popular with young professionals.

Agents can move between any of these locations. Private units in residential buildings are only accessible to their residents.""",
  )


def _make_ohio_suburb_preset() -> SettingPreset:
  """Ohio suburb setting."""
  return SettingPreset(
      name="ohio_suburb",
      community_description=(
          "residents of a suburban neighborhood in Brecksville, Ohio"
      ),
      buildings={
          "sunset_apartments": {
              "display_name": "Maple Creek Apartments",
              "description": "An affordable apartment complex near the highway",
              "num_units": 50,
          },
          "coral_village": {
              "display_name": "Oakwood Subdivision",
              "description": (
                  "A subdivision of single-family homes with small yards"
              ),
              "num_units": 25,
          },
          "palm_heights": {
              "display_name": "Heritage Condos",
              "description": (
                  "A newer condo development near the shopping center"
              ),
              "num_units": 15,
          },
          "ocean_view_estates": {
              "display_name": "Lakeside Estates",
              "description": "Large homes on generous lots near the lake",
              "num_units": 7,
          },
          "paradise_point": {
              "display_name": "Country Club Estates",
              "description": "Exclusive homes bordering the golf course",
              "num_units": 3,
          },
      },
      public_locations={
          "town_square": Location(
              name="town_square",
              display_name="Main Street",
              description=(
                  "The central downtown area with shops, benches, and a gazebo"
              ),
          ),
          "market": Location(
              name="market",
              display_name="Farmers Market",
              description=(
                  "A weekly outdoor market with local produce and crafts"
              ),
          ),
          "church": Location(
              name="church",
              display_name="First Methodist Church",
              description=(
                  "The town's main church for worship and community events"
              ),
          ),
          "cafe": Location(
              name="cafe",
              display_name="Grounds Coffee House",
              description="A cozy coffee shop on Main Street",
          ),
          "general_store": Location(
              name="general_store",
              display_name="Dollar General",
              description="A discount retail store selling everyday items",
          ),
          "community_center": Location(
              name="community_center",
              display_name="Community Recreation Center",
              description=(
                  "A public facility with meeting rooms, a gym, and a pool"
              ),
          ),
          "park": Location(
              name="park",
              display_name="Veterans Memorial Park",
              description=(
                  "A park with a playground, ball fields, and walking trails"
              ),
          ),
          "restaurant": Location(
              name="restaurant",
              display_name="Applebee's",
              description=(
                  "A casual dining chain restaurant near the highway exit"
              ),
          ),
          "library": Location(
              name="library",
              display_name="Public Library",
              description=(
                  "The local library with books, computers, and community"
                  " programs"
              ),
          ),
          "medical_clinic": Location(
              name="medical_clinic",
              display_name="Urgent Care Clinic",
              description="A walk-in healthcare clinic",
          ),
          "school": Location(
              name="school",
              display_name="Westfield Elementary",
              description="The neighborhood elementary school",
          ),
          "shopping_plaza": Location(
              name="shopping_plaza",
              display_name="Westfield Shopping Plaza",
              description=(
                  "A strip mall with a grocery store, pharmacy, and fast food"
              ),
          ),
          "swell_bar": Location(
              name="swell_bar",
              display_name="Buckeye Sports Bar",
              description=(
                  "A neighborhood bar with TVs, pool tables, and draft beer."
                  " Packed during Ohio State games."
              ),
          ),
          "rooftop_lounge": Location(
              name="rooftop_lounge",
              display_name="The Speakeasy",
              description="A trendy cocktail bar downtown, popular on weekends",
          ),
          "sunset_apartments_common_room": Location(
              name="sunset_apartments_common_room",
              display_name="Apartment Clubhouse",
              description=(
                  "A shared lounge in the apartment complex with a TV and"
                  " vending machines"
              ),
          ),
          "office_cafeteria": Location(
              name="office_cafeteria",
              display_name="Office Break Room",
              description=(
                  "A break room in the office park where workers eat and"
                  " socialize"
              ),
          ),
          "office_floor_tech": Location(
              name="office_floor_tech",
              display_name="Office Park - Tech Wing",
              description="The tech wing for software developers and IT staff",
          ),
          "office_floor_finance": Location(
              name="office_floor_finance",
              display_name="Office Park - Finance Wing",
              description=(
                  "The finance wing for accountants and administrative staff"
              ),
          ),
          "office_floor_creative": Location(
              name="office_floor_creative",
              display_name="Office Park - Creative Wing",
              description=(
                  "The creative wing for designers, marketers, and writers"
              ),
          ),
      },
      locations_prompt="""The neighborhood has the following locations:

RESIDENTIAL AREAS:
- Maple Creek Apartments: An affordable apartment complex with 50 units near the highway, plus a shared clubhouse.
- Oakwood Subdivision: A subdivision of 25 single-family homes with small yards, plus a common area.
- Heritage Condos: A newer condo development with 15 units near the shopping center, plus a common area.
- Lakeside Estates: Large homes with 7 units on generous lots near the lake, plus a common area.
- Country Club Estates: Exclusive homes with 3 units bordering the golf course, plus a common area.

PUBLIC SPACES:
- Main Street: The central downtown area with shops, benches, and a gazebo.
- First Methodist Church: The town's main church for worship and community events.
- Veterans Memorial Park: A park with a playground, ball fields, and walking trails.
- Community Recreation Center: A public facility with meeting rooms, a gym, and a pool.

BUSINESSES:
- Grounds Coffee House: A cozy coffee shop on Main Street.
- Farmers Market: A weekly outdoor market with local produce and crafts.
- Dollar General: A discount retail store selling everyday items.
- Westfield Shopping Plaza: A strip mall with a grocery store, pharmacy, and fast food.
- Applebee's: A casual dining chain restaurant near the highway exit.
- Office Park with tech, finance, and creative wings, plus a shared break room.
- Public Library: The local library with books, computers, and community programs.
- Urgent Care Clinic: A walk-in healthcare clinic.
- Westfield Elementary: The neighborhood elementary school.

NIGHTLIFE:
- Buckeye Sports Bar: A neighborhood bar with TVs, pool tables, and draft beer.
- The Speakeasy: A trendy cocktail bar downtown.

Agents can move between any of these locations. Private units in residential buildings are only accessible to their residents.""",
  )


# ============================================================================
# Brecksville 1000 — Scaled suburb preset for 1000 agents
# ============================================================================

# 10 neighborhoods for distributing foot traffic.
# NOTE: The existing ohio_suburb personas (2026-07-07-ohio_suburb-patched) use
# 5 building prefixes: millbrook_apts, brecksville_commons, chippewa_ridge,
# riverview_estates, timber_creek. These are kept as-is so personas load
# without re-patching. Neighborhoods here are PUBLIC location shards only.
BRECKSVILLE_NEIGHBORHOODS = (
    "snowville",
    "chippewa",
    "riverview",
    "barr_road",
    "oakes",
    "whitewood",
    "fitzwater",
    "highland",
    "parkside",
    "stadium",
)

BRECKSVILLE_NEIGHBORHOOD_DISPLAY = {
    "snowville": "Snowville",
    "chippewa": "Chippewa",
    "riverview": "Riverview",
    "barr_road": "Barr Road",
    "oakes": "Oakes",
    "whitewood": "Whitewood",
    "fitzwater": "Fitzwater",
    "highland": "Highland",
    "parkside": "Parkside",
    "stadium": "Stadium",
}


def _brecksville_neighborhood_locations(
    nbr: str,
) -> dict[str, Location]:
  """Generate local locations for one Brecksville neighborhood."""
  display = BRECKSVILLE_NEIGHBORHOOD_DISPLAY[nbr]
  locs = {}

  locs[f"{nbr}_gas_station"] = Location(
      name=f"{nbr}_gas_station",
      display_name=f"{display} Marathon",
      description=(
          f"A gas station and convenience store in {display}."
          " Sells coffee, snacks, and lottery tickets."
      ),
  )
  locs[f"{nbr}_pizza"] = Location(
      name=f"{nbr}_pizza",
      display_name=f"{display} Pizza & Subs",
      description=f"A family-owned pizza shop in {display}.",
  )
  locs[f"{nbr}_diner"] = Location(
      name=f"{nbr}_diner",
      display_name=f"{display} Family Diner",
      description=f"A small diner in {display} serving breakfast all day.",
  )
  locs[f"{nbr}_salon"] = Location(
      name=f"{nbr}_salon",
      display_name=f"{display} Cuts & Styles",
      description=f"A hair salon and barbershop in {display}.",
  )
  locs[f"{nbr}_daycare"] = Location(
      name=f"{nbr}_daycare",
      display_name=f"{display} Kids Academy",
      description=f"A childcare center in {display}.",
  )
  locs[f"{nbr}_park"] = Location(
      name=f"{nbr}_park",
      display_name=f"{display} Park",
      description=(
          f"A neighborhood park in {display} with a playground"
          " and walking path."
      ),
  )
  locs[f"{nbr}_dog_park"] = Location(
      name=f"{nbr}_dog_park",
      display_name=f"{display} Dog Park",
      description=f"A fenced dog park in {display}.",
  )
  locs[f"{nbr}_church"] = Location(
      name=f"{nbr}_church",
      display_name=f"{display} Community Church",
      description=f"A small church in {display} with Sunday services.",
  )
  locs[f"{nbr}_bus_stop"] = Location(
      name=f"{nbr}_bus_stop",
      display_name=f"{display} RTA Stop",
      description=f"An RTA bus stop in {display} connecting to downtown.",
  )
  locs[f"{nbr}_playground"] = Location(
      name=f"{nbr}_playground",
      display_name=f"{display} Playground",
      description=f"A kids' playground in {display}.",
  )
  locs[f"{nbr}_bar"] = Location(
      name=f"{nbr}_bar",
      display_name=f"{display} Tavern",
      description=f"A neighborhood bar in {display} with darts and a jukebox.",
  )
  locs[f"{nbr}_commons"] = Location(
      name=f"{nbr}_commons",
      display_name=f"{display} Community Room",
      description=(
          f"A shared community room in {display} with couches"
          " and a TV. Residents gather here."
      ),
  )

  return locs


# Downtown / Route 82 corridor (shared by all neighborhoods).
# IMPORTANT: Includes workplace locations that match persona work_place fields:
#   cafe, general_store, restaurant, shopping_plaza, market, school,
#   office_floor_tech, office_floor_finance, office_floor_creative,
#   office_cafeteria, library, medical_clinic, community_center
BRECKSVILLE_DOWNTOWN = {
    "town_square": Location(
        name="town_square",
        display_name="Town Square",
        description=(
            "The intersection of Route 82 and Brecksville Road, with"
            " the gazebo, benches, and seasonal decorations."
        ),
    ),
    "giant_eagle": Location(
        name="giant_eagle",
        display_name="Giant Eagle",
        description=(
            "The main grocery store on Route 82. The deli counter"
            " is a social hub."
        ),
    ),
    "cvs_pharmacy": Location(
        name="cvs_pharmacy",
        display_name="CVS Pharmacy",
        description="The CVS on Route 82 next to the shopping plaza.",
    ),
    # === Workplace-compatible locations (match persona work_place fields) ===
    "cafe": Location(
        name="cafe",
        display_name="Panera Bread",
        description=(
            "The Panera on Route 82. Remote workers with laptops"
            " and retirees with morning coffee."
        ),
    ),
    "general_store": Location(
        name="general_store",
        display_name="Dollar General",
        description="A discount retail store on Route 82.",
    ),
    "restaurant": Location(
        name="restaurant",
        display_name="The Courtyard",
        description=(
            "A popular restaurant and bar on Route 82. Date nights,"
            " happy hours, and Sunday brunch."
        ),
    ),
    "shopping_plaza": Location(
        name="shopping_plaza",
        display_name="Route 82 Shopping Plaza",
        description=(
            "A strip mall with Subway, a nail salon,"
            " H&R Block, and a UPS Store."
        ),
    ),
    "market": Location(
        name="market",
        display_name="Farmers Market",
        description=(
            "A seasonal outdoor market on weekends with local"
            " produce and crafts."
        ),
    ),
    "school": Location(
        name="school",
        display_name="Central Elementary",
        description="Central Elementary School on Chippewa Road.",
    ),
    "library": Location(
        name="library",
        display_name="Brecksville Library",
        description=(
            "The Cuyahoga County library branch. Book clubs,"
            " children's story time, and free WiFi."
        ),
    ),
    "medical_clinic": Location(
        name="medical_clinic",
        display_name="Medical Office Building",
        description=(
            "A medical office complex off Route 82. Primary care"
            " and specialists."
        ),
    ),
    "community_center": Location(
        name="community_center",
        display_name="Brecksville Community Center",
        description=(
            "The Community Center on Stadium Drive. Gym, indoor"
            " pool, meeting rooms, and youth programs."
        ),
    ),
    "office_cafeteria": Location(
        name="office_cafeteria",
        display_name="Office Park Cafeteria",
        description=(
            "The shared cafeteria in Brecksville Office Park."
            " Workers from all wings eat and socialize here."
        ),
    ),
    "office_floor_tech": Location(
        name="office_floor_tech",
        display_name="Office Park - Tech Wing",
        description=(
            "The tech wing of Brecksville Office Park. Software"
            " developers, IT support, and data analysts."
        ),
    ),
    "office_floor_finance": Location(
        name="office_floor_finance",
        display_name="Office Park - Finance Wing",
        description=(
            "The finance wing of Brecksville Office Park."
            " Accountants and financial advisors."
        ),
    ),
    "office_floor_creative": Location(
        name="office_floor_creative",
        display_name="Office Park - Creative Wing",
        description=(
            "The creative wing of Brecksville Office Park."
            " Marketing and design teams."
        ),
    ),
    # === Non-workplace downtown locations ===
    "church": Location(
        name="church",
        display_name="St. Basil the Great",
        description=(
            "St. Basil the Great Catholic Church. Fish fries"
            " during Lent and parish festivals."
        ),
    ),
    "city_hall": Location(
        name="city_hall",
        display_name="City Hall",
        description="Brecksville City Hall on Highland Drive.",
    ),
    "fire_station": Location(
        name="fire_station",
        display_name="Fire Station #1",
        description="Brecksville Fire Department. Community open houses.",
    ),
    "bbh_high_school": Location(
        name="bbh_high_school",
        display_name="Brecksville-Broadview Heights High School",
        description=(
            "The high school on Stadium Drive. Friday night"
            " football and community theater."
        ),
    ),
    "park": Location(
        name="park",
        display_name="Brecksville Reservation",
        description=(
            "Part of the Cleveland Metroparks. Hiking trails and"
            " the Brecksville Nature Center."
        ),
    ),
    "chippewa_trail": Location(
        name="chippewa_trail",
        display_name="Chippewa Creek Trail",
        description="A scenic walking and running trail along Chippewa Creek.",
    ),
    "blossom": Location(
        name="blossom",
        display_name="Blossom Music Center",
        description=(
            "Summer home of the Cleveland Orchestra. Lawn concerts and picnics."
        ),
    ),
    "swell_bar": Location(
        name="swell_bar",
        display_name="Buckeye Sports Bar",
        description=(
            "A sports bar with TVs, pool tables, and draft beer."
            " Packed during Ohio State and Browns games."
        ),
    ),
    "rooftop_lounge": Location(
        name="rooftop_lounge",
        display_name="The Speakeasy",
        description="A trendy cocktail bar downtown, popular on weekends.",
    ),
    # Common rooms for the 5 existing building prefixes
    "sunset_apartments_common_room": Location(
        name="sunset_apartments_common_room",
        display_name="Millbrook Clubhouse",
        description=(
            "A shared lounge in Millbrook Apartments with a TV"
            " and vending machines."
        ),
    ),
}


def _make_brecksville_1000_preset() -> SettingPreset:
  """Brecksville, Ohio — 1000-agent scale with 10 neighborhoods.

  Uses the same building prefixes (millbrook_apts, brecksville_commons, etc.)
  and work_place names (office_floor_tech, cafe, etc.) as the existing
  ohio_suburb personas so they load without re-patching.

  Adds ~120 neighborhood locations + ~25 downtown locations = ~150 public
  locations total, achieving ~5-7 agents per location density.

  Returns:
    A SettingPreset for the Brecksville 1000-agent scenario.
  """
  all_locations = dict(BRECKSVILLE_DOWNTOWN)
  for nbr in BRECKSVILLE_NEIGHBORHOODS:
    all_locations.update(_brecksville_neighborhood_locations(nbr))

  # Use the same 5 building prefixes from the existing ohio_suburb personas
  buildings = {
      "millbrook_apts": {
          "display_name": "Millbrook Apartments",
          "description": (
              "An affordable apartment complex near the I-77 interchange"
          ),
          "num_units": 500,
      },
      "brecksville_commons": {
          "display_name": "Brecksville Commons",
          "description": "A subdivision of single-family homes and townhouses",
          "num_units": 250,
      },
      "chippewa_ridge": {
          "display_name": "Chippewa Ridge",
          "description": "An upper-middle-class neighborhood of colonial homes",
          "num_units": 150,
      },
      "riverview_estates": {
          "display_name": "Riverview Estates",
          "description": "Large homes on generous lots near the valley",
          "num_units": 70,
      },
      "timber_creek": {
          "display_name": "Timber Creek",
          "description": "Exclusive homes bordering the Metroparks reservation",
          "num_units": 30,
      },
  }

  locations_prompt = (
      "Brecksville, Ohio is a suburban community of roughly one thousand"
      " residents outside Cleveland, situated between I-77 and the"
      " Cuyahoga Valley National Park.\n\n"
      "DOWNTOWN / ROUTE 82 CORRIDOR (accessible to all residents):\n"
      "- Town Square: The central intersection of Route 82 and"
      " Brecksville Road.\n"
      "- Giant Eagle: The main grocery store.\n"
      "- CVS Pharmacy: Pharmacy on Route 82.\n"
      "- Panera Bread (cafe): Coffee and remote work hub.\n"
      "- Dollar General (general_store): Discount retail.\n"
      "- The Courtyard (restaurant): Restaurant and bar.\n"
      "- Route 82 Shopping Plaza (shopping_plaza): Strip mall.\n"
      "- Farmers Market (market): Seasonal produce market.\n"
      "- Central Elementary (school): Elementary school.\n"
      "- Brecksville Library (library): County library branch.\n"
      "- Medical Office Building (medical_clinic): Doctors and"
      " specialists.\n"
      "- Brecksville Community Center (community_center): Gym, pool,"
      " meeting rooms.\n"
      "- Brecksville Office Park: Workplace with tech, finance, and"
      " creative wings plus a cafeteria.\n"
      "- St. Basil the Great (church): Catholic church.\n"
      "- City Hall: Local government.\n"
      "- Fire Station #1: Volunteer fire department.\n"
      "- Brecksville-Broadview Heights High School: Football, theater.\n"
      "- Brecksville Reservation (park): Metroparks with trails.\n"
      "- Chippewa Creek Trail: Walking and running trail.\n"
      "- Blossom Music Center: Summer concerts.\n"
      "- Buckeye Sports Bar (swell_bar): Sports bar.\n"
      "- The Speakeasy (rooftop_lounge): Cocktail bar.\n\n"
      "RESIDENTIAL AREAS:\n"
      "- Millbrook Apartments: Affordable apartments near I-77 (500"
      " units), with a shared clubhouse.\n"
      "- Brecksville Commons: Single-family homes and townhouses (250"
      " units).\n"
      "- Chippewa Ridge: Colonial homes (150 units).\n"
      "- Riverview Estates: Large homes near the valley (70 units).\n"
      "- Timber Creek: Exclusive homes near the Metroparks (30"
      " units).\n\n"
      "NEIGHBORHOODS (each has local amenities — park, diner, pizza"
      " shop, gas station, church, bar, dog park, playground, salon,"
      " daycare, bus stop, and a community room):\n"
  )

  for nbr in BRECKSVILLE_NEIGHBORHOODS:
    display = BRECKSVILLE_NEIGHBORHOOD_DISPLAY[nbr]
    locations_prompt += f"- {display}\n"

  locations_prompt += (
      "\nAgents can travel to their workplace, any downtown/Route 82"
      " location, their home neighborhood amenities, or any other"
      " neighborhood. Private residential units are only accessible"
      " to their residents."
  )

  return SettingPreset(
      name="brecksville_1000",
      community_description=(
          "residents of Brecksville, Ohio, a suburban community"
          " outside Cleveland"
      ),
      buildings=buildings,
      public_locations=all_locations,
      locations_prompt=locations_prompt,
  )


SETTING_PRESETS: dict[str, SettingPreset] = {}


def _register_presets() -> None:
  """Register all setting presets."""
  for factory in (
      _make_island_preset,
      _make_kerala_preset,
      _make_lagos_preset,
      _make_ohio_suburb_preset,
      _make_brecksville_1000_preset,
  ):
    preset = factory()
    SETTING_PRESETS[preset.name] = preset


_register_presets()


def get_setting(population: Optional[str] = None) -> Optional[SettingPreset]:
  """Get the setting preset for a population.

  Maps population flag values to setting presets. Returns None for
  unrecognized populations (falls back to default island locations).

  Args:
    population: The population name to look up (e.g., 'kerala', 'lagos').

  Returns:
    The matching SettingPreset, or None if no match is found.
  """
  if population is None:
    return None
  pop_lower = population.lower()
  # Direct match
  if pop_lower in SETTING_PRESETS:
    return SETTING_PRESETS[pop_lower]
  # Fuzzy match: check if population name contains setting key
  for key, preset in SETTING_PRESETS.items():
    if key in pop_lower:
      return preset
  return None


def get_buildings(
    setting: Optional[str] = None,
) -> dict[str, dict[str, str | int]]:
  """Get the buildings dict for a setting."""
  preset = get_setting(setting)
  if preset and preset.buildings:
    return preset.buildings
  return BUILDINGS


def get_community_description(setting: Optional[str] = None) -> str:
  """Get the community description for simulationist instructions."""
  preset = get_setting(setting)
  if preset:
    return preset.community_description
  return "residents of a small community"


def get_unit_name(building: str, unit_number: int) -> str:
  """Get the location name for a specific unit in a building."""
  return f"{building}_unit_{unit_number}"


def get_common_space(building: str) -> str:
  """Get the common space location name for a building."""
  return f"{building}_common"


def get_unit_display_name(building: str, unit_number: int) -> str:
  """Get human-readable name for a unit."""
  building_info = BUILDINGS[building]
  return f"{building_info['display_name']} Unit {unit_number}"


def get_initial_observation(
    agent_name: str,
    home_place: str,
    time: str = "Thursday, January 1st, 7:00 AM",
    setting: Optional[str] = None,
) -> str:
  """Generate initial observation for an agent waking up at home."""
  buildings = get_buildings(setting)
  parts = home_place.rsplit("_unit_", 1)
  if len(parts) == 2:
    building = parts[0]
    unit_num = parts[1]
    building_info = buildings.get(building, {})
    building_name = building_info.get("display_name", building)

    return (
        f"// {home_place} [{time}]: {agent_name} wakes up in"
        f" their bed at {building_name} Unit {unit_num}. The morning sun"
        " streams through the window. It's a new day."
    )
  else:
    return (
        f"// {home_place} [{time}]: {agent_name} wakes up at"
        " home. The morning sun streams through the window."
    )


# === PUBLIC LOCATIONS ===

PUBLIC_LOCATIONS = {
    "town_square": Location(
        name="town_square",
        display_name="Town Square",
        description="The central gathering place with a fountain and benches",
    ),
    "beach": Location(
        name="beach",
        display_name="The Beach",
        description="Sandy shores with palm trees and gentle waves",
    ),
    "lighthouse": Location(
        name="lighthouse",
        display_name="The Lighthouse",
        description="An old lighthouse on the rocky peninsula",
    ),
    "fishing_dock": Location(
        name="fishing_dock",
        display_name="Fishing Dock",
        description="Where fishing boats come and go each day",
    ),
    "market": Location(
        name="market",
        display_name="Island Market",
        description="An open-air market with fresh produce and goods",
    ),
    "cafe": Location(
        name="cafe",
        display_name="Sunrise Cafe",
        description="A cozy cafe serving coffee and pastries",
    ),
    "general_store": Location(
        name="general_store",
        display_name="General Store",
        description="A shop selling everyday necessities",
    ),
    "community_center": Location(
        name="community_center",
        display_name="Community Center",
        description="A large hall for island meetings and events",
    ),
    "park": Location(
        name="park",
        display_name="Island Park",
        description="A green space with trees, paths, and picnic areas",
    ),
    "marina": Location(
        name="marina",
        display_name="The Marina",
        description="Where private boats and yachts are docked",
    ),
    "restaurant": Location(
        name="restaurant",
        display_name="Oceanside Restaurant",
        description="An upscale restaurant with ocean views",
    ),
    "library": Location(
        name="library",
        display_name="Island Library",
        description="A quiet place for reading and study",
    ),
    "medical_clinic": Location(
        name="medical_clinic",
        display_name="Medical Clinic",
        description="The island's healthcare facility",
    ),
    "school": Location(
        name="school",
        display_name="Island School",
        description="Education for the island's children",
    ),
    "church": Location(
        name="church",
        display_name="Island Church",
        description="A small chapel for worship and reflection",
    ),
    "swell_bar": Location(
        name="swell_bar",
        display_name="Swell",
        description=(
            "A hip beachside bar with string lights, local DJs on weekends,"
            " cheap beer, natural wine, and a sandy floor. The island's"
            " most popular hangout for young residents."
        ),
    ),
    "rooftop_lounge": Location(
        name="rooftop_lounge",
        display_name="The Rooftop",
        description=(
            "An upscale cocktail lounge on top of the office building with"
            " panoramic ocean views, craft cocktails, and dim ambient"
            " lighting. Popular with professionals and for special occasions."
        ),
    ),
    "sunset_apartments_common_room": Location(
        name="sunset_apartments_common_room",
        display_name="Sunset Apartments Common Room",
        description=(
            "A shared lounge in the Sunset Apartments building with couches,"
            " a TV, a small kitchen, and a bulletin board. Popular with"
            " residents before and after work."
        ),
    ),
    "office_cafeteria": Location(
        name="office_cafeteria",
        display_name="Office Building Cafeteria",
        description=(
            "A spacious cafeteria on the first floor of the office building. A"
            " place where workers from all floors can eat, drink coffee, and"
            " socialize."
        ),
    ),
    "office_floor_tech": Location(
        name="office_floor_tech",
        display_name="Office Building - Floor 2 (Tech)",
        description=(
            "The second floor of the office building, dedicated to software"
            " developers, IT support, systems administrators, and other tech"
            " professionals."
        ),
    ),
    "office_floor_finance": Location(
        name="office_floor_finance",
        display_name="Office Building - Floor 3 (Finance & Legal)",
        description=(
            "The third floor of the office building, dedicated to accountants, "
            "financial analysts, lawyers, and administrative staff."
        ),
    ),
    "office_floor_creative": Location(
        name="office_floor_creative",
        display_name="Office Building - Floor 4 (Creative & Support)",
        description=(
            "The fourth floor of the office building, dedicated to designers,"
            " writers, marketing consultants, and customer support"
            " representatives."
        ),
    ),
}


def get_all_public_location_names(setting: Optional[str] = None) -> list[str]:
  """Get list of all public location names."""
  preset = get_setting(setting)
  if preset and preset.public_locations:
    locations = list(preset.public_locations.keys())
  else:
    locations = list(PUBLIC_LOCATIONS.keys())
  # Add building common spaces
  buildings = get_buildings(setting)
  for building in buildings:
    locations.append(get_common_space(building))
  return locations


def get_available_locations_prompt(setting: Optional[str] = None) -> str:
  """Generate a prompt listing all available locations for agents."""
  preset = get_setting(setting)
  pub_locs = (
      preset.public_locations
      if preset and preset.public_locations
      else PUBLIC_LOCATIONS
  )
  buildings = get_buildings(setting)

  lines = ["Available locations:"]

  # Public locations
  lines.append("\nPublic Places:")
  for loc in pub_locs.values():
    lines.append(f"  - {loc.display_name} ({loc.name})")

  # Building common spaces
  lines.append("\nResidential Common Areas:")
  for building, info in buildings.items():
    common = get_common_space(building)
    lines.append(f"  - {info['display_name']} Common Area ({common})")

  return "\n".join(lines)


def get_island_locations_prompt(setting: Optional[str] = None) -> str:
  """Generate the full locations description for GM.

  Args:
    setting: Population setting key (e.g. 'kerala', 'lagos', 'ohio_suburb'). If
      None or unrecognized, returns the default island locations.

  Returns:
    A detailed locations prompt string for the GM.
  """
  preset = get_setting(setting)
  if preset and preset.locations_prompt:
    return preset.locations_prompt

  # Default island locations
  return """The community has the following locations:

RESIDENTIAL AREAS:
- Sunset Apartments: A modest apartment complex with 50 units for working-class residents, plus a shared common room (sunset_apartments_common_room) where residents can socialize before and after work.
- Coral Village: A comfortable townhouse community with 25 units for middle-class families, plus a common area.
- Palm Heights: An upscale condominium with 15 units, plus a common area.
- Ocean View Estates: Spacious properties with 7 units for wealthy residents, plus a common area.
- Paradise Point Mansions: Exclusive gated homes with 3 units for the community's elite, plus a common area.

PUBLIC SPACES:
- Town Square: The central gathering place with a fountain, benches, and shops nearby.
- Beach: Sandy shores with palm trees and gentle waves.
- Lighthouse: An old lighthouse on the rocky peninsula.
- Fishing Dock: Where fishing boats come and go, and locals buy fresh catch.
- Island Park: A green space with walking paths, benches, and picnic areas.
- Community Center: A large hall used for meetings, events, and gatherings.
- Marina: Where private boats are docked.

BUSINESSES:
- Sunrise Cafe: A cozy cafe serving coffee, pastries, and light meals.
- Island Market: An open-air market with fresh produce and local goods.
- General Store: A shop selling everyday necessities and supplies.
- Oceanside Restaurant: A restaurant with nice views and good food.
- Office Building: A multi-story building with specific floors for different professions:
    - Floor 1 Cafeteria (office_cafeteria): A shared space for all office workers to meet and eat.
    - Floor 2 (office_floor_tech): For tech and engineering professionals.
    - Floor 3 (office_floor_finance): For finance, accounting, and legal professionals.
    - Floor 4 (office_floor_creative): For creative, marketing, and support professionals.
- Library: A quiet place for reading, study, and community programs.
- Medical Clinic: The community's healthcare facility.
- School: Education for children.
- Church: A small chapel for worship and community gatherings.

NIGHTLIFE:
- Swell: A popular bar with string lights and local DJs on weekends.
- The Rooftop: An upscale cocktail lounge on top of the office building.

Agents can move between any of these locations. Private units in residential buildings are only accessible to their residents."""
