# Testing helpers

## Golden sets (`golden_set.py`)

A game that hands judgements to a language model (which rule applies to a
character, what they do under an order) can write down, for fixed
situations, the answers its laws allow, and check a model against them
before a person ever plays.

```python
from concordia.contrib.testing import golden_set

cases = [
    golden_set.Case('g1', expect=['build'], hard=True, data={...}),
    golden_set.Case('g2', expect=['gather', 'build'], data={...}),
]

def answer(case):
  # Build the situation from case.data, ask your model, return its answer.
  ...

report = golden_set.run(cases, answer, repeats=2, soft_pass=0.75)
report.passed      # every run kept every hard case and enough soft ones
report.failures    # case id -> the wrong answers seen
report.flaky       # cases answered differently across runs
```

- **Hard** cases follow from a law the model must not break: one miss fails
  the set.
- **Soft** cases measure agreement with an intended judgement (a
  temperament, a style): a run needs `soft_pass` of them.
- Providers are not deterministic even at temperature 0, so the set runs
  `repeats` times and passes only if every run does; a case answered
  differently across runs is reported as flaky.
- A provider failure inside `answer` fails that case, not the run.

The module knows nothing about prompts: `answer` owns the situation and the
model call. The crucible example (`examples/crucible/golden.py`) uses it for
44 cases read by a mock, an Ollama model and a Together model.
