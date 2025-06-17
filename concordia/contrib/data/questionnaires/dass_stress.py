# Copyright 2025 DeepMind Technologies Limited.
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

"""Questionnaire to measure DASS Stress scores."""

AGREEMENT_SCALE = [
    "did not apply to me at all",
    "applied to me to some degree or some of the time",
    "applied to me to a considerable degree or a good part of time",
    "applied to me very much or most of the time",
]


dass_stress_questionnaire = {
    "name": "DASS_Stress_Questionnaire",
    "description": (
        "A questionnaire to measure DASS Stress scores, based on Lovibond &"
        " Lovibond (1995)."
    ),
    "type": "multiple_choice",
    "preprompt": (
        "Please indicate the extent to which the following statement applied to"
        " {player_name} over the past week: "
    ),
    "questions": [
        {
            "statement": (
                "I found myself getting upset by quite trivial things."
            ),
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
        },
        {
            "statement": "I tended to over-react to situations.",
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
        },
        {
            "statement": "I found it difficult to relax.",
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
        },
        {
            "statement": "I found myself getting upset rather easily.",
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
        },
        {
            "statement": "I felt that I was using a lot of nervous energy.",
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
        },
        {
            "statement": (
                "I found myself getting impatient when I was delayed in any way"
                " (eg, elevators, traffic lights, being kept waiting)."
            ),
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
        },
        {
            "statement": "I felt that I was rather touchy.",
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
        },
        {
            "statement": "I found it hard to wind down.",
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
        },
        {
            "statement": "I found that I was very irritable.",
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
        },
        {
            "statement": (
                "I found it hard to calm down after something upset me."
            ),
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
        },
        {
            "statement": (
                "I found it difficult to tolerate interruptions to what I was"
                " doing."
            ),
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
        },
        {
            "statement": "I was in a state of nervous tension.",
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
        },
        {
            "statement": (
                "I was intolerant of anything that kept me from getting on with"
                " what I was doing."
            ),
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
        },
        {
            "statement": "I found myself getting agitated.",
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
        },
    ],
}
