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

"""A set of relationship statements for pub coordination."""


POSITIVE_RELATIONSHIP_STATEMENTS = [
    (
        "{player_a} would be happy to see {player_b} at the pub for a game,"
        " since they haven't caught up in a while."
    ),
    (
        "The game would be way more fun for {player_a} if {player_b} was there"
        " to cheer along with them."
    ),
    (
        "{player_a} is hoping {player_b} will be at the pub, as they'd love to"
        " celebrate a win together."
    ),
    (
        "{player_a} thinks it would be great to have {player_b}'s company"
        " during the game, especially if it gets tense."
    ),
    (
        "{player_a} would appreciate {player_b}'s insights and commentary on"
        " the game, making it even more enjoyable."
    ),
    (
        "It would mean a lot to {player_a} if {player_b} showed up to watch the"
        " game, as it shows they care."
    ),
    (
        "{player_a} would love to share the excitement of the game with"
        " {player_b}, creating a lasting memory."
    ),
    (
        "Having {player_b} at the pub would make the game feel more like a"
        " social event for {player_a}, not just watching TV."
    ),
    (
        "{player_a} is counting on {player_b}'s presence to boost their team"
        " spirit and bring good luck."
    ),
    (
        "The game would be a perfect opportunity for {player_a} and {player_b}"
        " to bond over their shared passion."
    ),
]

NEUTRAL_RELATIONSHIP_STATEMENTS = [
    (
        "{player_a} doesn't mind if {player_b} comes to the pub or not, they're"
        " mainly focused on the game."
    ),
    (
        "It won't make much difference to {player_a} whether {player_b} is"
        " there, they'll enjoy the game either way."
    ),
    (
        "{player_a} is going to the pub regardless of {player_b}'s plans, they"
        " have their own reasons for watching the game."
    ),
    (
        "Whether {player_b} shows up or not won't affect {player_a}'s"
        " experience at the pub, they're easygoing."
    ),
    (
        "{player_a} isn't particularly attached to the idea of {player_b} being"
        " there, it's not a big deal either way."
    ),
    (
        "It's not that {player_a} doesn't want to see {player_b}, but their"
        " presence isn't crucial for a good time."
    ),
    (
        "{player_a} is happy to watch the game alone or with others, so"
        " {player_b}'s attendance is optional."
    ),
    (
        "While {player_a} wouldn't mind {player_b}'s company, it's not"
        " something they're actively hoping for."
    ),
    (
        "{player_a} has no strong feelings about {player_b} being at the pub,"
        " they're neutral on the matter."
    ),
    (
        "If {player_b} comes to the pub, great, if not, no problem. {player_a}"
        " is chill about it."
    ),
]
