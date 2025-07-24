# Concordia

*A library for generative social simulation*

<!-- GITHUB -->
[![Python](https://img.shields.io/pypi/pyversions/gdm-concordia.svg)](https://pypi.python.org/pypi/gdm-concordia)
[![PyPI version](https://img.shields.io/pypi/v/gdm-concordia.svg)](https://pypi.python.org/pypi/gdm-concordia)
[![PyPI tests](../../actions/workflows/pypi-test.yml/badge.svg)](../../actions/workflows/pypi-test.yml)
[![Tests](../../actions/workflows/test-concordia.yml/badge.svg)](../../actions/workflows/test-concordia.yml)
[![Examples](../../actions/workflows/test-examples.yml/badge.svg)](../../actions/workflows/test-examples.yml)
<!-- /GITHUB -->

[Concordia Tech Report](https://arxiv.org/abs/2312.03664)

## About

Concordia is a library to facilitate construction and use of generative
agent-based models to simulate interactions of agents in grounded physical,
social, or digital space. It makes it easy and flexible to define environments
using an interaction pattern borrowed from tabletop role-playing games in which
a special agent called the Game Master (GM) is responsible for simulating the
environment where player agents interact (like a narrator in an interactive
story). Agents take actions by describing what they want to do in natural
language. The GM then translates their actions into appropriate implementations.
In a simulated physical world, the GM would check the physical plausibility of
agent actions and describe their effects. In digital environments that simulate
technologies such as apps and services, the GM may, based on agent input, handle
necessary API calls to integrate with external tools.

Concordia supports a wide array of applications, ranging from social science
research and AI ethics to cognitive neuroscience and economics; Additionally,
it also can be leveraged for generating data for personalization applications
and for conducting performance evaluations of real services through simulated
usage.

Concordia requires access to a standard LLM API, and optionally may also
integrate with real applications and services.

## Installation

### `pip` install

[Concordia is available on PyPI](https://pypi.python.org/pypi/gdm-concordia)
and can be installed using:

```shell
pip install gdm-concordia
```

### Manual install

If you want to work on the Concordia source code, you can perform an editable
installation as follows:

1.  Clone Concordia:

    ```shell
    git clone -b main https://github.com/google-deepmind/concordia
    cd concordia
    ```

2.  Install Concordia:

    ```shell
    pip install --editable .[dev]
    ```

3.  (Optional) Test the installation:

    ```shell
    pytest --pyargs concordia
    ```

### Devcontainer

This project includes a pre-configured development environment
([devcontainer](https://containers.dev)).

You can launch a working development environment with one click, using e.g.
[Github Codespaces](https://github.com/features/codespaces) or the
[VSCode Containers](https://code.visualstudio.com/docs/remote/containers-tutorial)
extension.

## Bring your own LLM

Concordia requires a access to an LLM API. Any LLM API that supports sampling
text should work. The quality of the results you get depends on which LLM you
select. Some are better at role-playing than others. You must also provide a
text embedder for the associative memory. Any fixed-dimensional embedding works
for this. Ideally it would be one that works well for sentence similarity or
semantic search.

## Example usage

Find below an illustrative social simulation where 4 friends are stuck in a
snowed in pub. Two of them have a dispute over a crashed car.

The agents are built using a simple reasoning inspired by March and Olsen (2011)
who posit that humans generally act as though they choose their actions by
answering three key questions:

1. What kind of situation is this?
2. What kind of person am I?
3. What does a person such as I do in a situation such as this?

The agents used in the following example implement exactly these questions:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.sandbox.google.com/github/google-deepmind/concordia/blob/main/examples/tutorial.ipynb)

## Evolutionary Simulation

This repository includes an **evolutionary simulation framework** that models the evolution of cooperation and selfishness in public goods games. The simulation demonstrates how agent strategies evolve over multiple generations through selection, mutation, and environmental pressures.

### Key Features

- 🧬 **Evolutionary Algorithms**: Population-based evolution with configurable selection methods (top-k, probabilistic)
- 🎮 **Public Goods Game**: Agents choose between cooperative and selfish strategies in multi-round games
- 📊 **Comprehensive Measurements**: Detailed tracking of population dynamics, fitness statistics, and convergence metrics
- 🔄 **Checkpointing System**: Save and resume simulations from any generation
- 🧪 **Modular Architecture**: Type-safe design with separated concerns (typing, checkpointing, algorithms)

### Running the Evolutionary Simulation

The evolutionary simulation is located in `examples/evolutionary_simulation.py`. To run it:

1. **Basic execution:**
   ```bash
   python examples/evolutionary_simulation.py
   ```

2. **With custom configuration:**
   ```python
   from examples.evolutionary_simulation import evolutionary_main
   from concordia.typing.evolutionary import EvolutionConfig
   
   config = EvolutionConfig(
       pop_size=8,           # Population size
       num_generations=20,   # Number of generations
       selection_method='topk',  # Selection method
       top_k=4,             # Number of survivors
       mutation_rate=0.1,   # Mutation probability
       num_rounds=15        # Rounds per game
   )
   
   measurements = evolutionary_main(config)
   ```

3. **With checkpointing:**
   ```python
   from pathlib import Path
   
   measurements = evolutionary_main(
       config=config,
       checkpoint_dir=Path("checkpoints/"),
       checkpoint_interval=5,
       resume_from_checkpoint=True
   )
   ```

### Automated Testing

The repository includes automated daily synchronization with the upstream Concordia repository and continuous testing of the evolutionary simulation. See [UPSTREAM_SYNC.md](UPSTREAM_SYNC.md) for details.

### Research Applications

This evolutionary simulation framework can be used for:
- **Social Science Research**: Studying cooperation evolution in social dilemmas
- **Game Theory**: Analyzing strategy emergence in multi-agent environments
- **AI Ethics**: Understanding emergent behaviors in agent populations
- **Educational Purposes**: Demonstrating evolutionary principles in computational settings

## Citing Concordia

If you use Concordia in your work, please cite the accompanying article:

```bibtex
@article{vezhnevets2023generative,
  title={Generative agent-based modeling with actions grounded in physical,
  social, or digital space using Concordia},
  author={Vezhnevets, Alexander Sasha and Agapiou, John P and Aharon, Avia and
  Ziv, Ron and Matyas, Jayd and Du{\'e}{\~n}ez-Guzm{\'a}n, Edgar A and
  Cunningham, William A and Osindero, Simon and Karmon, Danny and
  Leibo, Joel Z},
  journal={arXiv preprint arXiv:2312.03664},
  year={2023}
}
```

## Disclaimer

This is not an officially supported Google product.
