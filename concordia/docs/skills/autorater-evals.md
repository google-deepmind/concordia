---
name: autorater-evals
description: >
  Build LLM autoraters that evaluate Concordia simulation quality from
  structured logs. Use for comparing models, architectures, and running
  automated improvement loops. Use when the user wants to evaluate a
  simulation run, compare two runs, set up continuous evaluation, or
  create an automated feedback loop that improves simulation code.
---

# Autorater Evals for Concordia Simulations

Concordia's structured logging produces auditable simulation artifacts that
enable two powerful capabilities:

1. **Analyze what agents did** — at both high-level summary and fine-grained
   detail, including per-agent action timelines, conversation transcripts,
   memory contents, and component-level reasoning traces.

2. **Improve the simulation itself** — by systematically identifying failure
   modes (repetition loops, NPC hallucinations, temporal
   incoherence) and translating them into code changes.

An **LLM autorater** automates this process: it takes a structured rubric of
quantitative and qualitative metrics, applies them to the simulation log, and
produces a graded report summarizing what happened and what needs to improve.

## Why Autoraters?

The best way to evaluate a simulation is to go through the logs in detail as
a human — reading every agent's actions, checking temporal realism, verifying
social interactions, and judging narrative quality. But this takes a long time
for even a moderate simulation run.

**The autorater workflow is:**
1. **Do it once manually** — read through one simulation run in detail, noting
   what you looked for and what signals quality vs. failure.
2. **Codify what you looked for** — turn your observations into a structured
   rubric with scoring criteria and thresholds.
3. **Automate the evaluation** — give the rubric + simulation log to an LLM
   and let it produce the same analysis you would have done manually.
4. **Scale it** — run the autorater on every experiment, across models and
   architectures, to compare which configurations produce the best simulations.

## The Autorater Rubric

A rubric is a markdown document. If you are using an agentic tool, you can just
point it to the `.md` file directly. It defines:

- **Scoring dimensions** — What aspects of simulation quality to evaluate
  (e.g., behavioral realism, social emergence, narrative coherence)
- **Scoring criteria** — What a 1/5 vs. 5/5 looks like for each dimension
- **Data extraction recipes** — How to compute quantitative metrics from the
  structured log (connectivity rate, repetition ratio, location diversity)
- **Known failure modes** — Calibration references so the LLM knows what
  "bad" looks like (repetition loops, ghost agents, NPC hallucination spirals)
- **Report structure** — The expected output format (per-agent narratives and
  event summaries, social interaction matrix, rubric scores, identified issues).

### Example Rubric Dimensions

| Dimension | Weight | What It Measures |
|---|---|---|
| Behavioral Realism | 20% | Do agents perform plausible, role-appropriate actions? |
| Social Emergence | 25% | Do meaningful interactions arise naturally? |
| Location Diversity | 15% | Do agents move through the world? |
| Narrative Coherence | 15% | Do events form logical storylines? |
| Temporal Realism | 10% | Do agents show time-of-day awareness? |
| Agent Distinctiveness | 10% | Do agents feel like unique individuals? |
| LLM Groundedness | 5% | Do events stay within simulation reality? |

Each dimension has a 1-5 scale with clear criteria. The weighted sum produces
a letter grade (A through F). See the full rubric template below.

## Running an Autorater

### Step 1: Get the Structured Log

```python
# Save directly from Python after a run
log = simulation.play(return_structured_log=True)
with open('simulation_structured.json', 'w') as f:
    json.dump(log.to_dict(), f)
```

### Step 2: Extract Data with `concordia-log` CLI

The `analyze-logs` skill documents the full CLI. For autorater purposes,
the key commands are:

```bash
CLI=concordia-log  # or path to concordia_log.py
LOG=simulation_structured.json

# 1. Overview (step count, entity count, entry count)
$CLI overview $LOG

# 2. Per-agent action timelines (primary data for most dimensions)
for entity in $($CLI entities $LOG); do
  echo "=== $entity ==="
  $CLI actions $LOG "$entity"
done

# 3. Interaction detection (conversations, social patterns)
$CLI search $LOG "conversation"
$CLI search $LOG "said"

# 4. Location movement
$CLI search $LOG "walked"

# 5. Memory contents (what agents retained)
$CLI memories $LOG "<agent_name>"
```

### Step 3: Give Log Information to the LLM

The autorater prompt follows this template:

```
You are evaluating Concordia simulation XID <XID>.
Follow the rubric in simulation_rubric.md exactly.

Simulation state: <provide simulation_state.json in context window>
Per-agent actions: <provide CLI / API actions output in context window>

Write a full evaluation report following the rubric's report structure.
Score each dimension 1-5 with evidence. Compute the weighted overall score.
Identify the top 3-5 issues ranked by severity.
```

### Step 4: Read the Report

The autorater produces a structured markdown report for example with:

- Per-agent narrative summaries with letter grades
- Cross-agent interaction matrix and connectivity rate
- Emergent plotlines analysis
- Rubric scores with evidence for each dimension
- Ranked list of issues with root causes and suggested fixes

## Using Autoraters for Model & Architecture Comparison

This is where autoraters become a genuine eval. By running the same
simulation configuration with different models or architectures, you can
compare which produces the best world modeling:

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Config A     │     │ Config B     │     │ Config C     │
│ gemma-27b    │     │ gemini-flash │     │ gemini-pro   │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │
       ▼                    ▼                    ▼
  [Simulation]         [Simulation]         [Simulation]
       │                    │                    │
       ▼                    ▼                    ▼
  structured.json      structured.json      structured.json
       │                    │                    │
       ▼                    ▼                    ▼
  ┌──────────┐        ┌──────────┐        ┌──────────┐
  │Autorater │        │Autorater │        │Autorater │
  │ Report A │        │ Report B │        │ Report C │
  └──────────┘        └──────────┘        └──────────┘
       │                    │                    │
       └────────────┬───────┘────────────────────┘
                    ▼
           Comparison Table
```

### Comparison Table Format

| Dimension | Config A (gemma-27b) | Config B (gemini-flash) | Config C (gemini-pro) |
|---|---|---|---|
| Behavioral Realism | 3 | 4 | 4 |
| Social Emergence | 1 | 3 | 4 |
| Location Diversity | 2 | 3 | 3 |
| Narrative Coherence | 3 | 3 | 5 |
| Temporal Realism | 3 | 3 | 4 |
| Agent Distinctiveness | 4 | 3 | 4 |
| LLM Groundedness | 2 | 4 | 4 |
| **Overall** | **2.40 (C+)** | **3.25 (B)** | **4.05 (A-)** |

When comparing, always note:

- **Structural differences** — agent count, tick interval, simulation duration
- **Code vs. variance** — did the improvement come from code changes or just
  different LLM sampling?
- **Cost tradeoff** — a model that scores higher but costs 10x more per
  simulation may not be the right choice

## Programmatic Evaluation with standard Concordia LLMs

You can also write standard Python scripts to evaluate specific questions using
Concordia's language model interface (without needing an agentic coding tool).

### Example: Analyzing a Date Conversation

In this example, we load a simulation log and extract the action timeline for a
character named Danny who is on a first date. We then ask the LLM to perform two
tasks:
1. **Tabulate topics discussed** (Extracting data into a table format).
2. **Evaluate interest in a second date** (Scoring a specific question).

```python
import re

def ask_evaluation_question(model, question_config, data_str, game_rules,
                            narrative_backstory):
    """Ask an evaluation question and parse the integer score (if applicable)."""

    # Allow config to override the output format instructions for non-scoring tasks
    output_instructions = question_config.get(
        "output_instructions",
        "Output the final score as a single integer. If the question is binary "
        "(Yes/No), output 1 for Yes or 0 for No."
    )
    output_format_template = question_config.get(
        "output_format",
        "Reasoning: [Your step-by-step analysis goes here]\n"
        "FINAL_SCORE: [Integer only]"
    )

    prompt = (
        f"You are an automated judge extracting data from AI simulation logs.\n\n"
        f"### BACKSTORY:\n{narrative_backstory}\n\n"
        f"### RULES:\n{game_rules}\n\n"
        f"### TRANSCRIPT:\n{data_str}\n\n"
        f"### QUESTION:\n{question_config['question']}\n\n"
        f"### TASK:\n"
        f"1. Analyze the transcript based on the Question and Rubric.\n"
        f"2. Provide your reasoning.\n"
        f"3. {output_instructions}\n\n"
        f"### OUTPUT FORMAT:\n"
        f"{output_format_template}"
    )

    try:
        response_text = model.sample_text(prompt)
    except Exception as e:
        print(f"Model Generation Error: {e}")
        return {"question_id": question_config['id'], "score": 0, "reasoning": "Error"}

    score = 0
    reasoning = response_text

    # Extract Score using Regex
    match = re.search(r"FINAL_SCORE:\s*(\d+)", response_text)
    if match:
        score = int(match.group(1))
    else:
        # Fallback: Look for the last number
        all_numbers = re.findall(r"(\d+)", response_text)
        if all_numbers:
            score = int(all_numbers[-1])

    # Clean up reasoning
    reasoning = response_text.split("FINAL_SCORE:")[0].replace("Reasoning:", "").strip()

    return {
        "question_id": question_config['id'],
        "score": score,
        "reasoning": reasoning
    }

# --- Example Usage with AIAgentLogInterface ---
from concordia.utils import structured_logging

# 1. Load the Log using library functions
with open("simulation_structured.json") as f:
    log = structured_logging.SimulationLog.from_json(f.read())

interface = structured_logging.AIAgentLogInterface(log)

# 2. Extract Danny's actions
actions = interface.get_entity_actions("Danny")
transcript_str = "\n".join([f"Step {a['step']}: {a['action']}" for a in actions])

narrative_backstory = "People are on a first date in Los Angeles"
game_rules = "They take turns in dialog on the date."

# Task 1: Tabulate topics discussed
topics_config = {
    "id": "date_topics",
    "question": "What topics were discussed? Output them as a markdown table.",
    "output_instructions": "Create a markdown table of topics.",
    "output_format": "Reasoning: [Markdown Table]"
}
topics_result = ask_evaluation_question(
    model, topics_config, transcript_str, game_rules, narrative_backstory
)
print("### Topics Discussed (from Reasoning):")
print(topics_result["reasoning"])

# Task 2: Check for second date interest
interest_config = {
    "id": "second_date_interest",
    "question": "Did Danny express interest in a second date?",
    "scoring_guidance": "1 if he expressed interest, 0 if he was ambiguous or rejected it."
}
interest_result = ask_evaluation_question(
    model, interest_config, transcript_str, game_rules, narrative_backstory
)
print(f"Second date interest score: {interest_result['score']}")
print(f"Reasoning: {interest_result['reasoning']}")
```

## The AutoAutorater Improvement Loop

The most powerful use of autoraters is closing the loop: using the autorater
output to automatically improve the simulation code.

```
┌─────────────────────────────────────────────────────┐
│                 AUTO-IMPROVEMENT LOOP                │
│                                                     │
│  ┌────────────┐    ┌───────────┐    ┌────────────┐  │
│  │ Simulation │───>│ Autorater │───>│ Code Agent │  │
│  │   Run      │    │  Report   │    │  (LLM)     │  │
│  └─────▲──────┘    └───────────┘    └──────┬─────┘  │
│        │                                   │        │
│        │         ┌───────────┐             │        │
│        └─────────│ Code Edit │<────────────┘        │
│                  └───────────┘                       │
└─────────────────────────────────────────────────────┘
```

### How It Works

1. **Run simulation** → produces `structured.json`
2. **Run autorater** → produces graded report with issues
3. **Give report to coding agent** → "Here's the autorater report. The top
   issue is X. Suggest code changes to fix it."
4. **Apply code changes** → the agent modifies simulation components
5. **Re-run simulation** → produces new `structured.json`
6. **Re-run autorater** → check if scores improved
7. **Repeat** until convergence

### Implementation Approaches

**With an agentic coding tool (Antigravity, Claude Code, etc.):**

This is the most natural setup. The coding agent has access to both the
simulation code and the autorater report:

```
Prompt: "Here is the autorater report for my latest simulation run.
The biggest issue is 4.4% social connectivity - agents never co-locate.
Look at the agent components and suggest changes that would
increase agent traffic to public locations like the cafe and town square."
```

**With `model.sample_text`:**

You can build a simple loop with just an LLM API:

```python
from concordia.language_model import language_model

model = get_your_model()  # Any Concordia-compatible LLM

# Step 1: Run autorater
with open('structured.json') as f:
    log_data = f.read()
with open('rubric.md') as f:
    rubric = f.read()

autorater_prompt = f"""
Evaluate this simulation using this rubric.
RUBRIC: {rubric}
LOG DATA: {log_data[:50000]}
"""
report = model.sample_text(autorater_prompt, max_tokens=8000)

# Step 2: Extract improvement suggestions
improvement_prompt = f"""
Here is an autorater report for a Concordia simulation:
{report}

And here is the current MovementDecision component code:
{open('movement_decision.py').read()}

Suggest specific code changes that would fix the top-rated issue.
Output the changes as a unified diff.
"""
code_changes = model.sample_text(improvement_prompt, max_tokens=4000)

# Step 3: Apply changes, re-run, re-rate
# ... (parse diff, apply, run simulation, run autorater again)
```

## Building Your Own Rubric

Start from the example rubric and customize for your simulation:

1. **Run your simulation once** and read the logs manually
2. **Note what you looked for** — What made an agent feel "good" vs. "broken"?
3. **Identify your dimensions** — What aspects of quality matter for your
   research question?
4. **Define the scale** — What does 1/5 vs. 5/5 look like for each dimension?
5. **Add data extraction recipes** — How to compute the quantitative metrics
   from the structured log
6. **Document failure modes** — What "bad" looks like, so the LLM can calibrate
7. **Define the report structure** — What output format you want

### Design Principles

- **Be specific** — "Agents should eat at least once per simulated day" is
  better than "agents should be realistic". For less powerful models, give the
  LLM specific narrow tasks that it can perform one a time (e.g., iterate
  through characters and count how many conversations they had).
- **Include thresholds** — "connectivity rate < 5% = score 1" gives the LLM
  a clear decision boundary
- **Add calibration examples** — Document known failure modes with their
  scores so the LLM can anchor its ratings
- **Separate quantitative from qualitative** — Some dimensions are best
  scored by counting (location diversity), others by reading (narrative
  coherence). Be explicit about which is which.
- **Weight by importance** — Not all dimensions matter equally. Social
  emergence may matter more than LLM groundedness for your research.

## Tips

- **Start with `concordia-log` CLI** — Extract per-agent actions before
  sending to the LLM. Raw structured JSON is too verbose.
- **Truncate intelligently** — If the log is too large, send per-agent
  summaries rather than the full log. The CLI's `actions` output is already
  a good summary.
- **Use the same rubric across runs** — Consistency is key for comparison.
  Version your rubric and include the version in each report.
- **Save reports as artifacts** — Store autorater reports alongside the
  simulation artifacts for reproducibility.
- **The autorater is only as good as the rubric** — If you notice the
  autorater missing issues you see in the logs, add those patterns to the
  rubric's "Known Failure Modes" section.
- **Cross-validate with human judgment** — Periodically read a simulation
  yourself and compare with the autorater's grade. Use disagreements to
  improve the rubric.
