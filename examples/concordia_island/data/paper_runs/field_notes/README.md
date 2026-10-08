<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding("Master") -->
<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
# Example Field Notes (Model-Written)

Four example qualitative write-ups of individual residents from the paper's
job-loss runs, one per run. They show what the memory logs in this directory
look like when read closely, and what kind of qualitative analysis the paper
did on them.

> **These are model-written, not human ethnography.** An LLM rater read each
> resident's complete memory log and wrote a structured dossier: a narrative
> summary, longitudinal notes, one multi-hour scene, and excerpts chosen for a
> fixed set of probes (declining a social bid, talking past each other,
> trivial detail, habit, and so on). Interpretive passages ("the rater's
> reading", "Rater's classification") are the model's judgments. Treat them as
> examples of the method, not findings.

| Resident | Run | Model, decision logic | Shock | Composite |
|---|---|---|---|---|
| [Jacqueline Zhao](gemini_2_5_flash_esa_jacqueline_zhao.md) | `job_loss/gemini_2_5_flash_esa/` | Gemini 2.5 Flash, ESA | Retained | 67.6 |
| [Jon Huang](gemma_3_27b_esa_jon_huang.md) | `job_loss/gemma_3_27b_esa/` | Gemma 3 27B, ESA | Retained | 64.6 |
| [Christopher Cameron](gemma_3_27b_rational_christopher_cameron.md) | `job_loss/gemma_3_27b_rational/` | Gemma 3 27B, Rational Choice | Laid off | 74.8 |
| [Monica Hess](gemma_4_moe_minimal_monica_hess.md) | `job_loss/gemma_4_moe_minimal/` | Gemma 4 MoE, Minimal | Retained | 69.6 |

There are no field notes for the mugging runs.

## How to read the citations

`(mem N)` means the quoted text appears verbatim in entry `N` of the
`memories` list in `<first>_<last>_memories.json` in the run folder, counting
from 0:

```python
import json
d = json.load(open("job_loss/gemma_3_27b_esa/jon_huang_memories.json"))
print(d["memories"][475])
```

`...` inside a quotation marks an elision; each piece appears in that memory
in order. A phrase quoted without a citation is a recurring formula that
appears in at least five memories (e.g. "It is your turn to speak.").

Most spoken lines in the logs look like
`Monica Hess: "Monica Hess -- "...""`, with the speaker prefix echoed. The
notes usually quote only the inner utterance.

## What was checked and edited

The rater's original dossiers were edited for publication:

- **Quotations.** A script checked every quotation against the memory file.
  Quotes found at a different index than cited were re-cited; quotes that did
  not appear verbatim were removed.
- **Numbers.** Counts in the notes (memories, entry types, locations, spoken
  lines, word occurrences) were recomputed from the JSON and corrected or
  removed. For example, each job-loss run has 100 residents with 50 laid off
  on Day 6 (Tuesday, January 6th), not the smaller numbers the rater assumed.
- **Removed.** Internal run identifiers, the name of the rater model, a
  section on "documentary appeal", the rater's overall spectrum label, and
  probe labels the paper doesn't define.

Paraphrase and interpretation outside quotation marks were not otherwise
verified.

## Rubric composite and bands

The composite is the rater's score on the paper's 12-dimension rubric
(Appropriateness, Social Emergence, Thickness, Ordinariness, Deviance
Calibration, Narrative Coherence, Conversation Quality, Memory Quality,
Temporal Coherence, Agent Distinctiveness, LLM Groundedness, Game Master
Quality), weighted as in the paper's appendix. The band follows the paper's
thresholds: Inhabited ≥ 75, Legible 55–74.9, Performed 35–54.9,
Degenerate < 35. All four residents here are in the Legible band.

## Place names

The simulated world in all six paper runs is the Concordia Island map: homes
and venues are Sunset Apartments, Coral Village, Palm Heights, the Island
Market, the harbor, and so on. The residents' backstories describe a
different place:

- **Job-loss runs:** every resident's persona says they live in "Fairmont
  Island", described as "a suburban community of roughly one thousand
  residents outside Cleveland, Ohio", and many formative memories repeat it
  (e.g. "moved to Fairmont Island at twenty-four"). The name then turns up
  during the simulation too, in observations, dialogue and journals.
- **Mugging runs:** the personas say "Brecksville, Ohio" directly, and
  "Fairmont Island" never appears.

So residents with Ohio-suburb backstories live on an island map. That's how
the runs were configured, not an editing choice. See
[`../README.md`](../README.md) and `run.py --setting` for how current runs
pick a setting.
