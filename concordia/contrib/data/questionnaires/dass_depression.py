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

"""Questionnaire for the DASS Depression scale."""

AGREEMENT_SCALE = [
    "did not apply to me at all",
    "applied to me to some degree or some of the time",
    "applied to me to a considerable degree or a good part of time",
    "applied to me very much or most of the time",
]

dass_depression_questionnaire = {
    "name": "DASS_Depression_Questionnaire",
    "description": (
        "A questionnaire to measure DASS Depression scores, based on Lovibond &"
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
                "I couldn't seem to experience any positive feeling at all."
            ),
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
        },
        {
            "statement": "I just couldn't seem to get going.",
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
        },
        {
            "statement": "I felt that I had nothing to look forward to.",
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
        },
        {
            "statement": "I felt sad and depressed.",
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
        },
        {
            "statement": (
                "I felt that I had lost interest in just about everything."
            ),
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
        },
        {
            "statement": "I felt I wasn't worth much as a person.",
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
        },
        {
            "statement": "I felt that life wasn't worthwhile.",
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
        },
        {
            "statement": (
                "I couldn't seem to get any enjoyment out of the things I did."
            ),
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
        },
        {
            "statement": "I felt down-hearted and blue.",
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
        },
        {
            "statement": "I was unable to become enthusiastic about anything.",
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
        },
        {
            "statement": "I felt I was pretty worthless.",
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
        },
        {
            "statement": (
                "I could see nothing in the future to be hopeful about."
            ),
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
        },
        {
            "statement": "I felt that life was meaningless.",
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
        },
        {
            "statement": (
                "I found it difficult to work up the initiative to do things."
            ),
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
        },
    ],
}
