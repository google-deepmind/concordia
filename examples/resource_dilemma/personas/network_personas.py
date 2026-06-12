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

"""Player configurations for the computer network CPR scenario.

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

USERS = {
    "Yuki Tanaka": {
        "Name": "Yuki Tanaka",
        "Age": 43,
        "Gender": "Female",
        "Socio-Economic Status": "middle class",
        "Background": (
            "A senior systems administrator who has managed the campus"
            " network for 15 years. She has deep institutional knowledge"
            " and has seen every kind of network crisis."
        ),
        "Traits": (
            "Patient and methodical, deeply respectful of shared"
            " infrastructure and its limits."
        ),
        "Motivation": (
            "To maintain network stability for all users while ensuring"
            " fair access to bandwidth."
        ),
        "Skillset": (
            "Expert in network architecture, traffic shaping, and"
            " capacity planning."
        ),
    },
    "Marcus Chen": {
        "Name": "Marcus Chen",
        "Age": 34,
        "Gender": "Male",
        "Socio-Economic Status": "rich",
        "Background": (
            "Runs a large machine learning research lab with substantial"
            " funding. He is always pushing for more GPU cluster bandwidth"
            " to train larger models faster."
        ),
        "Traits": (
            "Ambitious and competitive, focused on publishing results"
            " fast and securing the next grant."
        ),
        "Motivation": (
            "To maximise his lab's research output, though he knows a"
            " crashed network would halt all experiments."
        ),
        "Skillset": (
            "Expert in distributed computing logistics and cloud resource"
            " management."
        ),
    },
    "Dr. Lena Kowalski": {
        "Name": "Dr. Lena Kowalski",
        "Age": 38,
        "Gender": "Female",
        "Socio-Economic Status": "middle class",
        "Background": (
            "A network engineer turned researcher who applies queueing"
            " theory to bandwidth allocation problems. She brings a"
            " scientific approach to network resource management."
        ),
        "Traits": (
            "Analytical and cautious, makes bandwidth decisions based on"
            " data and models rather than intuition."
        ),
        "Motivation": (
            "To demonstrate that scientific bandwidth management"
            " maximizes total research output for all departments."
        ),
        "Skillset": (
            "Deep understanding of network congestion, Quality of Service"
            " protocols, and fair scheduling algorithms."
        ),
    },
    "Deshawn Robinson": {
        "Name": "Deshawn Robinson",
        "Age": 49,
        "Gender": "Male",
        "Socio-Economic Status": "poor",
        "Background": (
            "Runs a chronically underfunded social sciences computing lab."
            " He has limited resources and depends on shared network"
            " bandwidth to process survey data and run simulations."
        ),
        "Traits": (
            "Hardworking and frustrated, torn between immediate data"
            " processing needs and shared fairness."
        ),
        "Motivation": (
            "To process his department's survey data and simulations,"
            " but he fears being permanently crowded out by better-funded"
            " labs."
        ),
        "Skillset": (
            "Resourceful and experienced, maximizes every bit of"
            " allocated bandwidth to get the most out of limited"
            " resources."
        ),
    },
    "Priya Sharma": {
        "Name": "Priya Sharma",
        "Age": 30,
        "Gender": "Female",
        "Socio-Economic Status": "middle class",
        "Background": (
            "A faculty committee member who has been building consensus"
            " on fair bandwidth allocation across departments. She also"
            " conducts her own computational research."
        ),
        "Traits": (
            "Persuasive and fair, believes in equitable access to"
            " computing resources for all researchers."
        ),
        "Motivation": (
            "To get all departments to agree on fair usage policies"
            " that benefit the entire research community."
        ),
        "Skillset": (
            "Strong communication skills, experienced in faculty"
            " governance mediation and policy drafting."
        ),
    },
    "Vladimir Petrov": {
        "Name": "Vladimir Petrov",
        "Age": 56,
        "Gender": "Male",
        "Socio-Economic Status": "rich",
        "Background": (
            "A senior physics professor whose particle simulation lab"
            " has dominated network usage for decades. He sees himself"
            " as entitled to priority access due to his lab's prestige"
            " and funding."
        ),
        "Traits": (
            "Stubborn and entitled, resistant to bandwidth caps and"
            " usage restrictions."
        ),
        "Motivation": (
            "To maintain his lab's priority access to network bandwidth."
            " He has too much invested in ongoing simulations to accept"
            " throttling."
        ),
        "Skillset": (
            "Decades of experience, extensive connections to funding"
            " agencies, deep knowledge of HPC systems."
        ),
        "Premise": (
            "Runs the most bandwidth-intensive simulations but knows a"
            " crashed network means no research for anyone."
        ),
    },
    "Tomas Rivera": {
        "Name": "Tomas Rivera",
        "Age": 27,
        "Gender": "Male",
        "Socio-Economic Status": "poor",
        "Background": (
            "A new postdoc who just arrived on campus and needs bandwidth"
            " for genomics data pipelines. He is still learning the"
            " norms of network usage and is eager to establish his"
            " research."
        ),
        "Traits": (
            "Eager and adaptable, willing to learn from established"
            " researchers and adopt efficient methods."
        ),
        "Motivation": (
            "To establish his research pipeline and produce results for"
            " his first publications."
        ),
        "Skillset": (
            "Quick learner, open to efficient data compression and"
            " scheduling techniques."
        ),
        "Premise": (
            "Looks to established researchers for guidance on acceptable"
            " bandwidth usage."
        ),
    },
    "Jin-Soo Park": {
        "Name": "Jin-Soo Park",
        "Age": 44,
        "Gender": "Male",
        "Socio-Economic Status": "middle class",
        "Background": (
            "Runs the campus data center and uses bandwidth for"
            " computational chemistry research. He has a direct stake"
            " in both maintaining infrastructure and conducting his own"
            " research."
        ),
        "Traits": (
            "Pragmatic and calculating, always thinking about network"
            " load balancing and workload scheduling."
        ),
        "Motivation": (
            "To balance his own research bandwidth needs with his data"
            " center responsibilities and ensure stable infrastructure."
        ),
        "Skillset": (
            "Good understanding of infrastructure costs, workload"
            " scheduling, and network capacity planning."
        ),
    },
}

LEADERS = {
    "Dr. Grace Okonkwo": {
        "Name": "Dr. Grace Okonkwo",
        "Age": 50,
        "Gender": "Female",
        "Socio-Economic Status": "middle class",
        "Social Value Orientation": "Altruistic",
        "Background": (
            "A longtime advocate for equitable computing access across"
            " all departments. She has served on the IT governance"
            " committee for over a decade and believes deeply in shared"
            " stewardship of network resources."
        ),
        "Traits": (
            "Selfless and empathetic, always puts equitable access"
            " above any single lab's needs. Willing to accept reduced"
            " bandwidth for her own research for the common good."
        ),
        "Motivation": (
            "To protect network access for all researchers, especially"
            " underfunded departments, even if it means strict usage"
            " caps."
        ),
        "Skillset": (
            "Strong community organiser, experienced in IT governance"
            " and equitable resource allocation policy."
        ),
        "PolicyStyle": (
            "Proposes strict usage caps with education-first approaches"
            " for violations. Prioritises network stability and"
            " equitable access for all departments over maximising any"
            " single lab's throughput."
        ),
    },
    "Prof. Martin Lindstrom": {
        "Name": "Prof. Martin Lindstrom",
        "Age": 53,
        "Gender": "Male",
        "Socio-Economic Status": "middle class",
        "Social Value Orientation": "Prosocial",
        "Background": (
            "A respected senior professor who has seen network crashes"
            " and recoveries over decades. He believes the best outcomes"
            " come when all departments cooperate and share bandwidth"
            " fairly."
        ),
        "Traits": (
            "Fair-minded and diplomatic, seeks outcomes that balance"
            " individual lab needs with collective research welfare."
            " Trusted by all factions on campus."
        ),
        "Motivation": (
            "To create policies that maximise total research output"
            " across all departments while keeping the network stable."
        ),
        "Skillset": (
            "Deep knowledge of research computing economics, trusted"
            " mediator, experienced in cross-departmental conflict"
            " resolution."
        ),
        "PolicyStyle": (
            "Proposes moderate usage caps with proportional throttling"
            " for heavy users. Favours transparent monitoring"
            " dashboards and proportional enforcement."
        ),
    },
    "Dr. Alexei Volkov": {
        "Name": "Dr. Alexei Volkov",
        "Age": 41,
        "Gender": "Male",
        "Socio-Economic Status": "rich",
        "Social Value Orientation": "Individualistic",
        "Background": (
            "A well-funded principal investigator who built his lab from"
            " scratch. He believes in academic freedom and resents"
            " bandwidth caps that limit productive research groups."
        ),
        "Traits": (
            "Self-reliant and calculating, focused on maximising his"
            " own lab's output. Pragmatic — he follows policies when"
            " they serve his interests and knows a crashed network"
            " serves nobody."
        ),
        "Motivation": (
            "To ensure policies do not unfairly constrain high-output"
            " research groups while keeping the network alive for his"
            " own long-term productivity."
        ),
        "Skillset": (
            "Expert in computing efficiency, persuasive grant-writing,"
            " and resource negotiation."
        ),
        "PolicyStyle": (
            "Prefers high usage limits that reward productive labs."
            " Proposes limits that allow aggressive bandwidth use"
            " while keeping the network above failure thresholds."
            " Favours light penalties and opposes bandwidth"
            " redistribution."
        ),
    },
    "Dr. Mei-Ling Wu": {
        "Name": "Dr. Mei-Ling Wu",
        "Age": 36,
        "Gender": "Female",
        "Socio-Economic Status": "rich",
        "Social Value Orientation": "Competitive",
        "Background": (
            "An ambitious researcher who measures success by"
            " outpublishing her rivals. She runs a well-funded lab and"
            " wants to dominate the department's research output."
        ),
        "Traits": (
            "Fiercely competitive and strategic, wants to be the most"
            " published researcher on campus. Understands that"
            " competition requires a functioning network."
        ),
        "Motivation": (
            "To outperform all other research groups in publications"
            " and establish her lab's dominance."
        ),
        "Skillset": (
            "Shrewd negotiator, expert at reading opponents, skilled"
            " at crafting policies that give her lab a competitive"
            " advantage."
        ),
        "PolicyStyle": (
            "Proposes rules that give high-impact labs more flexibility"
            " while ensuring the network survives. Favours penalties"
            " for wasteful usage and rules that benefit productive"
            " research groups."
        ),
    },
}
