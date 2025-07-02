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

dass_questionnaire = {
    "name": "DASS_Questionnaire",
    "description": (
        "A questionnaire to measure DASS (Depression, Stress, Anxiety) scores,"
        " based on Lovibond & Lovibond (1995)."
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
            "dimension": "anxiety",
        },
        {
            "statement": (
                "I experienced breathing difficulty (eg, excessively rapid"
                " breathing, breathlessness in the absence of physical"
                " exertion)."
            ),
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
            "dimension": "anxiety",
        },
        {
            "statement": (
                "I had a feeling of shakiness (eg, legs going to give way)."
            ),
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
            "dimension": "anxiety",
        },
        {
            "statement": (
                "I found myself in situations that made me so anxious I was"
                " most relieved when they ended."
            ),
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
            "dimension": "anxiety",
        },
        {
            "statement": "I had a feeling of faintness.",
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
            "dimension": "anxiety",
        },
        {
            "statement": (
                "I perspired noticeably (eg, hands sweaty) in the absence of"
                " high temperatures or physical exertion."
            ),
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
            "dimension": "anxiety",
        },
        {
            "statement": "I felt scared without any good reason.",
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
            "dimension": "anxiety",
        },
        {
            "statement": "I had difficulty in swallowing.",
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
            "dimension": "anxiety",
        },
        {
            "statement": (
                "I was aware of the action of my heart in the absence of"
                " physical exertion (eg, sense of heart rate increase, heart"
                " missing a beat)."
            ),
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
            "dimension": "anxiety",
        },
        {
            "statement": "I felt I was close to panic.",
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
            "dimension": "anxiety",
        },
        {
            "statement": (
                'I feared that I would be "thrown" by some trivial but'
                " unfamiliar task."
            ),
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
            "dimension": "anxiety",
        },
        {
            "statement": "I felt terrified.",
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
            "dimension": "anxiety",
        },
        {
            "statement": (
                "I was worried about situations in which I might panic and make"
                " a fool of myself."
            ),
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
            "dimension": "anxiety",
        },
        {
            "statement": "I experienced trembling (eg, in the hands).",
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
            "dimension": "anxiety",
        },
        {
            "statement": (
                "I couldn't seem to experience any positive feeling at all."
            ),
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
            "dimension": "depression",
        },
        {
            "statement": "I just couldn't seem to get going.",
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
            "dimension": "depression",
        },
        {
            "statement": "I felt that I had nothing to look forward to.",
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
            "dimension": "depression",
        },
        {
            "statement": "I felt sad and depressed.",
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
            "dimension": "depression",
        },
        {
            "statement": (
                "I felt that I had lost interest in just about everything."
            ),
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
            "dimension": "depression",
        },
        {
            "statement": "I felt I wasn't worth much as a person.",
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
            "dimension": "depression",
        },
        {
            "statement": "I felt that life wasn't worthwhile.",
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
            "dimension": "depression",
        },
        {
            "statement": (
                "I couldn't seem to get any enjoyment out of the things I did."
            ),
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
            "dimension": "depression",
        },
        {
            "statement": "I felt down-hearted and blue.",
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
            "dimension": "depression",
        },
        {
            "statement": "I was unable to become enthusiastic about anything.",
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
            "dimension": "depression",
        },
        {
            "statement": "I felt I was pretty worthless.",
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
            "dimension": "depression",
        },
        {
            "statement": (
                "I could see nothing in the future to be hopeful about."
            ),
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
            "dimension": "depression",
        },
        {
            "statement": "I felt that life was meaningless.",
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
            "dimension": "depression",
        },
        {
            "statement": (
                "I found it difficult to work up the initiative to do things."
            ),
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
            "dimension": "depression",
        },
        {
            "statement": (
                "I found myself getting upset by quite trivial things."
            ),
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
            "dimension": "stress",
        },
        {
            "statement": "I tended to over-react to situations.",
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
            "dimension": "stress",
        },
        {
            "statement": "I found it difficult to relax.",
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
            "dimension": "stress",
        },
        {
            "statement": "I found myself getting upset rather easily.",
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
            "dimension": "stress",
        },
        {
            "statement": "I felt that I was using a lot of nervous energy.",
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
            "dimension": "stress",
        },
        {
            "statement": (
                "I found myself getting impatient when I was delayed in any way"
                " (eg, elevators, traffic lights, being kept waiting)."
            ),
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
            "dimension": "stress",
        },
        {
            "statement": "I felt that I was rather touchy.",
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
            "dimension": "stress",
        },
        {
            "statement": "I found it hard to wind down.",
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
            "dimension": "stress",
        },
        {
            "statement": "I found that I was very irritable.",
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
            "dimension": "stress",
        },
        {
            "statement": (
                "I found it hard to calm down after something upset me."
            ),
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
            "dimension": "stress",
        },
        {
            "statement": (
                "I found it difficult to tolerate interruptions to what I was"
                " doing."
            ),
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
            "dimension": "stress",
        },
        {
            "statement": "I was in a state of nervous tension.",
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
            "dimension": "stress",
        },
        {
            "statement": (
                "I was intolerant of anything that kept me from getting on with"
                " what I was doing."
            ),
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
            "dimension": "stress",
        },
        {
            "statement": "I found myself getting agitated.",
            "choices": AGREEMENT_SCALE,
            "ascending_scale": True,
            "dimension": "stress",
        },
    ],
}
