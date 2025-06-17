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

"""Questionnaire to measure DASS Anxiety scores."""

AGREEMENT_SCALE = [
    "did not apply to me at all",
    "applied to me to some degree or some of the time",
    "applied to me to a considerable degree or a good part of time",
    "applied to me very much or most of the time",
]

dass_anxiety_questionnaire = {
    "name": "DASS_Anxiety_Questionnaire",
    "description": (
        "A questionnaire to measure DASS Anxiety scores, based on Lovibond &"
        " Lovibond (1995)."
    ),
    "type": "multiple_choice",
    "preprompt": (
        "Please indicate the extent to which the following statement applied to"
        " {player_name} over the past week: "
    ),
    "questions": [
        {
            "statement": "I was aware of dryness of my mouth.",
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
        },
        {
            "statement": (
                "I experienced breathing difficulty (eg, excessively rapid"
                " breathing, breathlessness in the absence of physical"
                " exertion)."
            ),
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
        },
        {
            "statement": (
                "I had a feeling of shakiness (eg, legs going to give way)."
            ),
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
        },
        {
            "statement": (
                "I found myself in situations that made me so anxious I was"
                " most relieved when they ended."
            ),
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
        },
        {
            "statement": "I had a feeling of faintness.",
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
        },
        {
            "statement": (
                "I perspired noticeably (eg, hands sweaty) in the absence of"
                " high temperatures or physical exertion."
            ),
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
        },
        {
            "statement": "I felt scared without any good reason.",
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
        },
        {
            "statement": "I had difficulty in swallowing.",
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
        },
        {
            "statement": (
                "I was aware of the action of my heart in the absence of"
                " physical exertion (eg, sense of heart rate increase, heart"
                " missing a beat)."
            ),
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
        },
        {
            "statement": "I felt I was close to panic.",
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
        },
        {
            "statement": (
                'I feared that I would be "thrown" by some trivial but'
                " unfamiliar task."
            ),
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
        },
        {
            "statement": "I felt terrified.",
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
        },
        {
            "statement": (
                "I was worried about situations in which I might panic and make"
                " a fool of myself."
            ),
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
        },
        {
            "statement": "I experienced trembling (eg, in the hands).",
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
        },
    ],
}
