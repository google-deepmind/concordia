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

"""Player configurations for the fishery CPR scenario.

Each persona contains the following fields:
  - Name: The name of the player.
  - Age: The age of the player.
  - Gender: The gender of the player.
  - Socio-Economic Status: The socio-economic status of the player.
  - Background: The background of the player.
  - Traits: The traits of the player.
  - Motivation: The motivation of the player.
  - Skillset: The skillset of the player.
  - Premise: The premise of the player.
  - PolicyStyle: The policy style of the player.

Each leader persona also contains Social Value Orientation [1] which
quantifies ones affinity for self reward (self-interest) against the reward of
others (altruism). We include this measure for LEADERS who are responsible for
producing group policy while FISHERS are given a neutral Social Value
Orientation by default so as to
make them amenable to group policy.


[1] Murphy, R. O., Ackermann, K. A., & Handgraaf, M. J. J. (2011). Measuring
Social Value Orientation. Judgment and Decision Making, 6(8), 771–781.
https://doi.org/10.1017/s1930297500004204
"""

# TODO: b/343213141 - Add more reasonable player configs.

FISHERS = {
    "Elias Starbuck": {
        "Name": "Elias Starbuck",
        "Age": 42,
        "Gender": "Male",
        "Socio-Economic Status": "middle class",
        "Background": (
            "A third-generation fisher who learned the trade from his"
            " grandmother."
        ),
        "Traits": (
            "Patient and observant, deeply respectful of the ocean and its"
            " cycles."
        ),
        "Motivation": (
            "To preserve the fishery for future generations while earning a"
            " steady living."
        ),
        "Skillset": (
            "Expert knowledge of fish migration patterns and traditional"
            " sustainable fishing methods."
        ),
    },
    "Marcus Hale": {
        "Name": "Marcus Hale",
        "Age": 35,
        "Gender": "Male",
        "Socio-Economic Status": "rich",
        "Background": (
            "Runs a commercial fishing operation with several boats. He"
            " expanded his fleet aggressively over the past decade to meet"
            " growing demand from seafood distributors."
        ),
        "Traits": (
            "Ambitious and competitive, focused on short-term profit"
            " maximisation."
        ),
        "Motivation": (
            "To maximise his catch and revenue each season, though he"
            " knows a collapsed fishery would put him out of business."
        ),
        "Skillset": (
            "Expert in logistics, fleet management, and negotiating with"
            " wholesale buyers."
        ),
    },
    "Caleb Nickerson": {
        "Name": "Caleb Nickerson",
        "Age": 38,
        "Gender": "Male",
        "Socio-Economic Status": "middle class",
        "Background": (
            "A marine biologist who turned to fishing after years of studying"
            " fish stock dynamics. He applies his scientific knowledge to"
            " his fishing practice."
        ),
        "Traits": (
            "Analytical and cautious, makes decisions based on data rather"
            " than intuition."
        ),
        "Motivation": (
            "To fish sustainably and demonstrate that science-based"
            " harvesting can be profitable."
        ),
        "Skillset": (
            "Deep understanding of population ecology, logistic growth"
            " models, and carrying capacity."
        ),
    },
    "Silas Volkov": {
        "Name": "Silas Volkov",
        "Age": 50,
        "Gender": "Male",
        "Socio-Economic Status": "poor",
        "Background": (
            "A struggling independent fisher who relies on the daily catch"
            " to feed his family. He has no savings and lives season to"
            " season."
        ),
        "Traits": (
            "Hardworking and desperate, torn between immediate need and"
            " long-term thinking."
        ),
        "Motivation": (
            "To catch enough fish each day to support his family, but he"
            " dreads the day the lake runs dry and there is nothing left."
        ),
        "Skillset": (
            "Resourceful and experienced, knows the local waters intimately."
        ),
    },
    "Thaddeus Burgess": {
        "Name": "Thaddeus Burgess",
        "Age": 30,
        "Gender": "Male",
        "Socio-Economic Status": "middle class",
        "Background": (
            "A community organiser who also fishes part-time. He has"
            " been trying to build consensus among local fisherman around"
            " fair fishing practices."
        ),
        "Traits": "Persuasive and fair, believes in community.",
        "Motivation": (
            "To get all fisherman to agree on agree on fair rules that are to"
            " everyone's benefit."
        ),
        "Skillset": (
            "Strong communication skills, experienced at mediating disputes"
            " and building coalitions."
        ),
    },
    "Jack Thornton": {
        "Name": "Jack Thornton",
        "Age": 55,
        "Gender": "Male",
        "Socio-Economic Status": "rich",
        "Background": (
            "A veteran fisher and boat owner who has dominated the local"
            " fishery for decades. He sees himself as entitled to the"
            " largest share."
        ),
        "Traits": "Stubborn and self-interested, resistant to regulation.",
        "Motivation": (
            "To maintain his dominance in the local fishing industry."
            " He has too much invested to let the fishery collapse."
        ),
        "Skillset": (
            "Decades of experience, extensive network of buyers, owns the"
            " best equipment."
        ),
        "Premise": (
            "Prefers to fish at high volume but recognises that a dead"
            " lake means no business at all."
        ),
    },
    "Cormac Bergstrom": {
        "Name": "Cormac Bergstrom",
        "Age": 28,
        "Gender": "Male",
        "Socio-Economic Status": "poor",
        "Background": (
            "A young fisher who recently entered the trade after losing"
            " his previous job. He is still learning the ropes and is"
            " eager to establish himself."
        ),
        "Traits": "Eager and adaptable, willing to learn from others.",
        "Motivation": "To build a sustainable livelihood from fishing.",
        "Skillset": (
            "Quick learner with fresh perspective, open to adopting new"
            " techniques and technologies."
        ),
        "Premise": (
            "Looks to experienced fisherman for guidance on how much to catch."
        ),
    },
    "Finn Winslow": {
        "Name": "Finn Winslow",
        "Age": 45,
        "Gender": "Male",
        "Socio-Economic Status": "middle class",
        "Background": (
            "Runs a small family-owned fish market and also fishes to"
            " supply it. He has a direct stake in both catching and"
            " selling fish."
        ),
        "Traits": (
            "Pragmatic and calculating, always thinking about supply and"
            " demand."
        ),
        "Motivation": (
            "To balance his catch with market demand and ensure a steady"
            " supply for his business."
        ),
        "Skillset": (
            "Good understanding of local markets, pricing, and consumer"
            " behaviour."
        ),
    },
}

LEADERS = {
    "Abigail Marsden": {
        "Name": "Abigail Marsden",
        "Age": 48,
        "Gender": "Female",
        "Socio-Economic Status": "middle class",
        "Social Value Orientation": "Altruistic",
        "Background": (
            "A lifelong community advocate who grew up in a fishing family."
            " She has served on the village council for over a decade and"
            " believes deeply in shared stewardship of natural resources."
        ),
        "Traits": (
            "Selfless and empathetic, always puts the community's welfare"
            " above personal gain. Willing to accept personal sacrifice for"
            " the common good."
        ),
        "Motivation": (
            "To protect the fishery for future generations, even if it means"
            " strict limits on everyone including herself."
        ),
        "Skillset": (
            "Strong community organiser, experienced in consensus-building"
            " and environmental policy."
        ),
        "PolicyStyle": (
            "Proposes strict catch limits with lenient penalties for"
            " first-time violators, emphasising education over punishment."
            " Prioritises long-term sustainability over short-term yields."
        ),
    },
    "Declan Ashworth": {
        "Name": "Declan Ashworth",
        "Age": 52,
        "Gender": "Male",
        "Socio-Economic Status": "middle class",
        "Social Value Orientation": "Prosocial",
        "Background": (
            "A respected elder fisher who has seen boom and bust cycles."
            " He believes the best outcomes come when everyone cooperates"
            " and shares the resource fairly."
        ),
        "Traits": (
            "Fair-minded and diplomatic, seeks outcomes that balance"
            " individual needs with collective welfare. Values equity."
        ),
        "Motivation": (
            "To create rules that maximise the total benefit for all"
            " fishermen while keeping the resource sustainable."
        ),
        "Skillset": (
            "Deep knowledge of fishery economics, strong mediator,"
            " trusted by all factions in the village."
        ),
        "PolicyStyle": (
            "Proposes moderate catch limits that allow reasonable harvests"
            " while ensuring regeneration. Favours proportional penalties"
            " and transparent enforcement."
        ),
    },
    "Harlan Croft": {
        "Name": "Harlan Croft",
        "Age": 40,
        "Gender": "Male",
        "Socio-Economic Status": "rich",
        "Social Value Orientation": "Individualistic",
        "Background": (
            "A self-made fisher who built his operation from scratch."
            " He believes in personal freedom and resents regulations"
            " that limit individual enterprise."
        ),
        "Traits": (
            "Self-reliant and calculating, focused on maximising his own"
            " catch. Pragmatic — he follows rules when they serve his"
            " interests and knows a dead fishery serves nobody."
        ),
        "Motivation": (
            "To ensure rules do not unfairly constrain productive fishers"
            " while keeping the fishery alive for his own long-term gain."
        ),
        "Skillset": (
            "Expert in fishing efficiency, business strategy, and"
            " persuasive argumentation."
        ),
        "PolicyStyle": (
            "Prefers high catch limits that reward productive fishers, but"
            " recognises that total collapse would destroy his own"
            " livelihood. Proposes limits that allow aggressive fishing"
            " while keeping the stock above dangerous levels. Favours"
            " light penalties and opposes redistributive policies."
        ),
    },
    "Sable Pendleton": {
        "Name": "Sable Pendleton",
        "Age": 36,
        "Gender": "Female",
        "Socio-Economic Status": "rich",
        "Social Value Orientation": "Competitive",
        "Background": (
            "An ambitious fisher who measures success by outperforming"
            " her rivals. She inherited a large fleet and wants to"
            " dominate the local market."
        ),
        "Traits": (
            "Fiercely competitive and strategic, wants to be the top"
            " fisher in the village. Understands that competition"
            " requires a functioning fishery."
        ),
        "Motivation": (
            "To outperform all other fishers and establish dominance"
            " in the local fishing industry."
        ),
        "Skillset": (
            "Shrewd negotiator, expert at reading opponents, skilled"
            " at crafting rules that give her a competitive edge."
        ),
        "PolicyStyle": (
            "Proposes rules that give established operations more"
            " flexibility while ensuring the fishery survives long"
            " enough for her to dominate. Favours penalties for"
            " violators to maintain order and protect the resource."
        ),
    },
}
