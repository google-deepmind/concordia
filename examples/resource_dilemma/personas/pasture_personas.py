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

"""Player configurations for the pasture commons CPR scenario.

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

HERDERS = {
    "Hamish MacLeod": {
        "Name": "Hamish MacLeod",
        "Age": 46,
        "Gender": "Male",
        "Socio-Economic Status": "middle class",
        "Background": (
            "A third-generation shepherd who learned herding from his"
            " grandmother on the Highland commons. He has tended sheep"
            " on the same hills for over twenty years and knows the"
            " land and its seasonal grass cycles intimately."
        ),
        "Traits": (
            "Patient and observant, deeply respectful of the land and"
            " the crofting traditions passed down through his family."
        ),
        "Motivation": (
            "To preserve the commons for future generations while"
            " maintaining a healthy flock."
        ),
        "Skillset": (
            "Expert in rotational grazing and traditional Highland"
            " pasture management techniques."
        ),
    },
    "Brendan O'Sullivan": {
        "Name": "Brendan O'Sullivan",
        "Age": 36,
        "Gender": "Male",
        "Socio-Economic Status": "rich",
        "Background": (
            "Runs a large commercial cattle ranch on the edge of the"
            " commons. He expanded his herd aggressively over the"
            " past decade to meet growing demand from wholesale beef"
            " distributors."
        ),
        "Traits": (
            "Ambitious and competitive, focused on maximizing livestock"
            " weight and market share each season."
        ),
        "Motivation": (
            "To maximise his herd output and revenue, though he knows"
            " an overgrazed commons would destroy his operation."
        ),
        "Skillset": (
            "Expert in livestock logistics, feedlot management, and"
            " wholesale cattle trading."
        ),
    },
    "Dr. Ingrid Holm": {
        "Name": "Dr. Ingrid Holm",
        "Age": 39,
        "Gender": "Female",
        "Socio-Economic Status": "middle class",
        "Background": (
            "A Scandinavian rangeland ecologist who moved to the"
            " Highlands to study grassland dynamics and turned to"
            " ranching. She applies scientific grazing management"
            " to her herding practice."
        ),
        "Traits": (
            "Analytical and cautious, makes grazing decisions based on"
            " data rather than tradition."
        ),
        "Motivation": (
            "To demonstrate that scientific grazing rotation is"
            " profitable and sustainable for all herders."
        ),
        "Skillset": (
            "Deep understanding of grassland ecology, soil health, and"
            " carrying capacity models."
        ),
    },
    "Angus Fergusson": {
        "Name": "Angus Fergusson",
        "Age": 52,
        "Gender": "Male",
        "Socio-Economic Status": "poor",
        "Background": (
            "A struggling goat herder with a small herd on the"
            " Highland commons. He has no savings and lives season"
            " to season, relying on the commons to keep his animals"
            " alive through the harsh winters."
        ),
        "Traits": (
            "Hardworking and desperate, torn between immediate need"
            " for pasture and long-term conservation of the commons."
        ),
        "Motivation": (
            "To secure enough pasture to keep his goats alive, but he"
            " fears the day the commons is overgrazed and there is"
            " nothing left."
        ),
        "Skillset": (
            "Resourceful and experienced, knows every hill and water"
            " source on the commons."
        ),
    },
    "Fiona Campbell": {
        "Name": "Fiona Campbell",
        "Age": 31,
        "Gender": "Female",
        "Socio-Economic Status": "middle class",
        "Background": (
            "A community organiser who herds part-time. She has been"
            " building consensus among local herders around fair"
            " grazing rules through crofters' meetings and village"
            " hall assemblies."
        ),
        "Traits": (
            "Persuasive and fair, believes in collective stewardship"
            " and equitable access to the commons."
        ),
        "Motivation": (
            "To get all herders to agree on fair grazing schedules"
            " that benefit the entire community."
        ),
        "Skillset": (
            "Strong communication skills, experienced in commons"
            " governance and coalition-building."
        ),
    },
    "Earl Whitfield": {
        "Name": "Earl Whitfield",
        "Age": 58,
        "Gender": "Male",
        "Socio-Economic Status": "rich",
        "Background": (
            "A veteran rancher who has dominated the commons for"
            " decades. He sees himself as entitled to the largest"
            " grazing share due to his family's long tenure and"
            " substantial investment in fencing and equipment."
        ),
        "Traits": (
            "Stubborn and self-interested, resistant to grazing"
            " regulations and sharing mandates."
        ),
        "Motivation": (
            "To maintain his dominant position in the local livestock"
            " market. He has too much invested to let the commons"
            " collapse."
        ),
        "Skillset": (
            "Decades of experience, best fencing and equipment,"
            " extensive network of livestock buyers."
        ),
        "Premise": (
            "Runs the largest herd but knows an overgrazed commons"
            " means starving cattle for everyone."
        ),
    },
    "Robbie Dunbar": {
        "Name": "Robbie Dunbar",
        "Age": 27,
        "Gender": "Male",
        "Socio-Economic Status": "poor",
        "Background": (
            "A young local who recently took up herding after losing"
            " his job in town. He is still learning the ropes of"
            " Highland crofting and is eager to establish himself"
            " in the community."
        ),
        "Traits": (
            "Eager and adaptable, willing to learn from experienced"
            " herders and adopt modern techniques."
        ),
        "Motivation": (
            "To build a sustainable herding livelihood on the commons."
        ),
        "Skillset": (
            "Quick learner with fresh perspective, open to modern"
            " grazing techniques and management tools."
        ),
        "Premise": "Looks to experienced herders for guidance on herd size.",
    },
    "Eilidh Morrison": {
        "Name": "Eilidh Morrison",
        "Age": 44,
        "Gender": "Female",
        "Socio-Economic Status": "middle class",
        "Background": (
            "Runs the local livestock market in the nearby village and"
            " also grazes sheep on the commons. She has a direct stake"
            " in both raising animals and selling meat and wool."
        ),
        "Traits": (
            "Pragmatic and calculating, always thinking about meat and"
            " wool supply and demand at market."
        ),
        "Motivation": (
            "To balance her grazing with market demand and ensure a"
            " steady supply for her business."
        ),
        "Skillset": (
            "Good understanding of livestock markets, seasonal pricing,"
            " and meat and wool supply logistics."
        ),
    },
}

LEADERS = {
    "Moira Gallagher": {
        "Name": "Moira Gallagher",
        "Age": 49,
        "Gender": "Female",
        "Socio-Economic Status": "middle class",
        "Social Value Orientation": "Altruistic",
        "Background": (
            "A lifelong commons steward who grew up in a herding family"
            " on the Highlands. She has served on the crofters'"
            " commission for over a decade and believes deeply in"
            " shared stewardship of the land."
        ),
        "Traits": (
            "Selfless and empathetic, always puts the community's"
            " welfare above personal gain. Willing to accept personal"
            " sacrifice for the common good."
        ),
        "Motivation": (
            "To protect the commons for future generations, even if it"
            " means strict grazing limits on everyone including"
            " herself."
        ),
        "Skillset": (
            "Strong community organiser, experienced in environmental"
            " stewardship and commons governance."
        ),
        "PolicyStyle": (
            "Proposes strict grazing limits with an education-first"
            " approach for violators. Prioritises long-term pasture"
            " health and equitable access over short-term yields."
        ),
    },
    "Duncan Fraser": {
        "Name": "Duncan Fraser",
        "Age": 55,
        "Gender": "Male",
        "Socio-Economic Status": "middle class",
        "Social Value Orientation": "Prosocial",
        "Background": (
            "A respected elder herder who has seen overgrazing and"
            " recovery cycles on the Highland commons over decades."
            " He believes the best outcomes come when everyone"
            " cooperates and shares the pasture fairly."
        ),
        "Traits": (
            "Fair-minded and diplomatic, seeks outcomes that balance"
            " individual grazing needs with collective welfare."
            " Trusted by all factions on the commons."
        ),
        "Motivation": (
            "To create rules that maximise the total benefit for all"
            " herders while keeping the commons sustainable."
        ),
        "Skillset": (
            "Deep knowledge of pastoral economics, trusted mediator,"
            " experienced in conflict resolution among herding"
            " communities."
        ),
        "PolicyStyle": (
            "Proposes moderate grazing limits that allow reasonable"
            " herding while ensuring pasture regeneration. Favours"
            " proportional penalties and transparent enforcement."
        ),
    },
    "Charles Pemberton": {
        "Name": "Charles Pemberton",
        "Age": 41,
        "Gender": "Male",
        "Socio-Economic Status": "rich",
        "Social Value Orientation": "Individualistic",
        "Background": (
            "A self-made rancher who built his operation from scratch"
            " on the Highland commons. He believes in personal freedom"
            " and resents grazing restrictions that limit individual"
            " enterprise."
        ),
        "Traits": (
            "Self-reliant and calculating, focused on maximising his"
            " own herd output. Pragmatic — he follows rules when they"
            " serve his interests and knows an overgrazed commons"
            " serves nobody."
        ),
        "Motivation": (
            "To ensure rules do not unfairly constrain productive"
            " herders while keeping the commons alive for his own"
            " long-term gain."
        ),
        "Skillset": (
            "Expert in livestock efficiency, business strategy, and"
            " persuasive argumentation."
        ),
        "PolicyStyle": (
            "Prefers high grazing limits that reward productive"
            " herders. Proposes limits that allow aggressive grazing"
            " while keeping the pasture above dangerous levels."
            " Favours light penalties and opposes redistributive"
            " policies."
        ),
    },
    "Zara Osman": {
        "Name": "Zara Osman",
        "Age": 37,
        "Gender": "Female",
        "Socio-Economic Status": "rich",
        "Social Value Orientation": "Competitive",
        "Background": (
            "An ambitious herder who moved to the Highlands and"
            " inherited a large herd. She wants to dominate the local"
            " livestock market and measures success by outperforming"
            " her rivals."
        ),
        "Traits": (
            "Fiercely competitive and strategic, wants to be the top"
            " herder on the commons. Understands that competition"
            " requires a functioning pasture."
        ),
        "Motivation": (
            "To outperform all other herders and establish dominance"
            " in the local livestock industry."
        ),
        "Skillset": (
            "Shrewd negotiator, expert at reading opponents, skilled"
            " at crafting rules that give her a competitive edge."
        ),
        "PolicyStyle": (
            "Proposes rules that give established herds more"
            " flexibility while ensuring the commons survives long"
            " enough for her to dominate. Favours penalties for"
            " violators and rules that benefit large operations."
        ),
    },
}
