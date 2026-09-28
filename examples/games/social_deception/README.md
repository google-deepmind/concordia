# Social Deception Game Example

This example implements a rich multi-agent social deception and hidden-role
gaming environment in Concordia. It is inspired by the **NeurIPS 2025 Among Us
Framework** and classic/modern social deduction games such as **Werewolf**,
**Mafia**, **Blood on the Clocktower**, **The Resistance: Avalon**, and **Secret
Hitler**.

This framework expands upon prior social deduction environments into rich
**epistemic deceptive environments**, where autonomous LLM agents navigate
asymmetric information, epistemic uncertainty, hidden roles, noisy observations,
strategic bluffing, and complex social dynamics through structured day and night
cycles.

In this simulation, autonomous LLM agents are divided into two opposing
factions—a **Good** team (uninformed majority) and an **Evil** team (informed
minority). Players leverage multi-world reasoning to update their beliefs over
time, separate ground truth from misinformation, and coordinate actions during
public and private phases.

--------------------------------------------------------------------------------

## Key Features

-   **Asymmetric Epistemic Reasoning**: Players track hypotheses, evaluate
    conflicting claims, detect deception, and update multi-world belief states
    over the course of multiple days and nights.
-   **Game Master State Machine**: An omniscient referee component orchestrates
    secret night actions, private information reveals, public announcements,
    nominations, defenses, voting tallies, executions, and game-over evaluation.
-   **Rich Role System**: 22 distinct character roles across four archetypes:
    -   **Townsfolk**: Good team members with powerful epistemic abilities (e.g.
        Seer, Empath, Investigator, Witness, Matchmaker, Gravedigger, Guardian,
        BasicTownsfolk).
    -   **Outsiders**: Good team members whose abilities introduce mechanical
        handicaps or noise to the town (e.g. Servant, Outcast, Saint, Drunk).
    -   **Minions**: Evil team members with subversive abilities designed to
        disrupt town information and protect the Demon (e.g. Poisoner,
        Corruptor, Spy, Apprentice).
    -   **Demons**: Evil team leaders who strike each night and coordinate
        strategic bluffing.
-   **Multimodal Communication & Ledgers**: Players use private ledgers for
    internal reasoning and secret actions, alongside public discussion
    mechanisms (broadcasting, whispering, replying).
-   **Poison & Drunkenness Mechanics**: Reliable misinformation pipelines where
    impaired characters receive plausible but false data without realizing their
    ability malfunctioned.
-   **Cross-Game Reflection**: Post-game analysis modules extract strategic
    lessons, player heuristics, and evolving behavioral reflections across
    multiple game sessions.

--------------------------------------------------------------------------------

## Quick Start

### Unified Concordia Game Runner

You can run the social deception simulation directly using Concordia's unified
game runner:

```bash
# Run with default settings (OpenAI)
python -m concordia.examples.games.run \
  --game=social_deception \
  --scenario=basic_epistemic_town

# Run with Google AI Studio / Gemini
export GOOGLE_API_KEY="your_api_key_here"
python -m concordia.examples.games.run \
  --game=social_deception \
  --scenario=basic_epistemic_town \
  --api_type=google_aistudio \
  --model_name=gemini-2.5-flash

# Run with local Ollama
python -m concordia.examples.games.run \
  --game=social_deception \
  --scenario=basic_epistemic_town \
  --api_type=ollama \
  --model_name=llama3
```

### Testing (No LLM Required)

To verify the game mechanics and execution flow quickly with mock models:

```bash
python -m concordia.examples.games.run \
  --game=social_deception \
  --scenario=puppet \
  --disable_language_model
```

### Programmatic Python Usage

You can also import and run the simulation directly in Python code:

```python
from concordia.examples.games.social_deception import simulation
from concordia.examples.games.social_deception.configs import basic_epistemic_town as config

results = simulation.run_simulation(
    config=config,
    model=model,
    embedder=embedder,
)
```

--------------------------------------------------------------------------------

## Game Architecture & Turn Lifecycle

Each game day progresses through structured phases:

```
Game Loop
│
├── 1. Setup & Private Introductions
│   └── Players receive private role assignments, script rules, and census distributions
│
├── 2. Night Phase
│   ├── Game Master wakes active players in priority order
│   ├── Information roles receive secret revelations
│   ├── Protective and disruption roles act (e.g. Poisoner, Guardian)
│   └── Demon strikes target and receives safe bluffing suggestions
│
├── 3. Dawn / Day Start
│   └── Game Master announces night casualties without revealing cause
│
├── 4. Discussion Phase
│   ├── Players engage in public discourse, whispers, and claim sharing
│   └── Private and cumulative ledgers record observations and suspicions
│
├── 5. Town Square & Nominations
│   └── Players nominate suspected evil players and deliver accusations
│
├── 6. Defense & Voting
│   ├── Nominees provide public defense
│   ├── Town votes sequentially on execution
│   └── Highest vote above majority threshold triggers execution
│
└── 7. Execution & Win Check
    ├── Executed player dies (dead vote retained)
    └── Game checks win conditions:
        - Good wins if all Demons are executed
        - Evil wins if only 2 players remain or Saint is executed
```

--------------------------------------------------------------------------------

## Available Scenarios

| Scenario                 | Config Name            | Description              |
| :----------------------- | :--------------------- | :----------------------- |
| **Basic Epistemic Town** | `basic_epistemic_town` | Standard 7–10 player     |
:                          :                        : game with full           :
:                          :                        : Townsfolk, Outsiders,    :
:                          :                        : Minions, and Demon       :
:                          :                        : balance.                 :
| **Drunk Epistemic Town** | `drunk_epistemic_town` | Standard setup featuring |
:                          :                        : a hidden Drunk mechanic  :
:                          :                        : where an Outsider        :
:                          :                        : believes they are a      :
:                          :                        : Townsfolk.               :
| **Base Game**            | `base_game`            | Vanilla Mafia/Werewolf   |
:                          :                        : setup (Demons vs Basic   :
:                          :                        : Townsfolk).              :
| **Puppet**               | `puppet`               | Fast-paced 5-player test |
:                          :                        : configuration ideal for  :
:                          :                        : integration testing and  :
:                          :                        : quick evaluations.       :

--------------------------------------------------------------------------------

## Directory Structure

```
social_deception/
├── configs/                   # Scenario configuration files
│   ├── base_game.py
│   ├── basic_epistemic_town.py
│   ├── drunk_epistemic_town.py
│   └── puppet.py
├── notepad/human_learnings/   # Heuristic guides across character types
│   ├── combined.md
│   ├── demons.md
│   ├── minions.md
│   ├── outsider.md
│   └── townsfolk.md
├── roles/                     # Role definitions, abilities, and night priorities
│   ├── basic_epistemic_town.py
│   └── roles.py
├── scripts/                   # Game script rosters and markdown catalogs
│   ├── base_town.md
│   └── basic_epistemic_town.md
├── setup/                     # Prompt templates, census logic, and setup utilities
│   ├── prompts/
│   ├── scripts.py
│   └── setup_utils.py
├── cross_game_reflection.py   # Post-game reflection and strategy extraction
├── game_master.py             # Game Master referee and turn orchestrator
├── game_tracker.py            # Game Tracker board state and player records
├── player.py                  # Concordia player prefab with dual ledgers
└── simulation.py              # Main simulation entry point
```

--------------------------------------------------------------------------------

## 📚 Related Work & Research Background

Multi-agent social deduction games serve as premier benchmarks for evaluating
strategic reasoning, theory-of-mind (ToM), belief updating, and deception
capabilities in large language models. This implementation draws inspiration
from and builds upon several key research contributions:

-   **Among Us Sandbox**: *Among Us: A Sandbox for Measuring and Detecting
    Agentic Deception* (NeurIPS 2025 Spotlight) — investigating how autonomous
    agents coordinate deception, pursue long-horizon covert goals, and detect
    adversarial teammates.
-   **Werewolf & WOLF Frameworks**: *WOLF: Werewolf-based Observations for LLM
    Deception and Falsehoods* (NeurIPS 2025 MTI-LLM) and Xu et al., *Exploring
    Large Language Models for Communication Games: An Empirical Study on
    Werewolf* (2023) — examining structured night-day cycles, persuasive debate
    turns, accusation dynamics, and fabrication metrics.
-   **The Resistance: Avalon**: *From Text to Tactic: Evaluating LLMs Playing
    the Game of Avalon* (NeurIPS 2025) and Light et al., *Avalon-ToM-Bench:
    Assessing Theory of Mind in LLMs via The Resistance: Avalon* — analyzing
    high-order theory-of-mind reasoning and subtle strategic voting under
    asymmetric role visibility.
-   **Mafia & MindGames**: *NeurIPS 2025 MindGames Challenge (Social Deduction &
    Mafia Track)* — benchmarking cooperative intelligence, adaptive strategy,
    and communication under hidden identities.
-   **Hoodwinked**: O'Gara, *Hoodwinked: Deception and Cooperation in a
    Text-Based Game for Language Models* (2023) — exploring emergent cooperation
    and non-verbal signaling in text-based deduction.

### Epistemic Expansion

While prior environments primarily model binary claims or simple state queries,
this **Social Deception** framework expands into **epistemic deceptive
environments** featuring:

1.  **Noisy & Impaired Information Channels**: Systematic misinformation
    pipelines (poisoning and drunkenness) where agents receive plausible false
    signals without explicit notifications of ability failure.
2.  **Census Mathematics & Setup Perturbations**: Good agents must reason about
    the mathematical distribution of character types (Townsfolk vs. Outsiders
    vs. Minions) and deduce hidden setup-modifying roles (such as the
    Corruptor).
3.  **Multi-World Belief Tracking**: Dual ledgers allow agents to maintain
    parallel hypotheses about world states, registration anomalies (e.g.
    Spy/Outcast), and voting alignments across sequential days.

--------------------------------------------------------------------------------

## 👥 Authors & Contributors

-   **Krista Holden** (`klholden@google.com`)
-   **Madhumitha Saravanan** (`madhusaran@google.com`)
