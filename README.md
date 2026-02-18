# Concordia

*A library for generative social simulation*

<!-- GITHUB -->
<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
[![Python](https://img.shields.io/pypi/pyversions/gdm-concordia.svg)](https://pypi.python.org/pypi/gdm-concordia)
[![PyPI version](https://img.shields.io/pypi/v/gdm-concordia.svg)](https://pypi.python.org/pypi/gdm-concordia)
[![PyPI tests](../../actions/workflows/pypi-test.yml/badge.svg)](../../actions/workflows/pypi-test.yml)
[![Tests](../../actions/workflows/test-concordia.yml/badge.svg)](../../actions/workflows/test-concordia.yml)
[![Examples](../../actions/workflows/test-examples.yml/badge.svg)](../../actions/workflows/test-examples.yml)
<!-- /GITHUB -->

[Concordia Tech Report](https://arxiv.org/abs/2312.03664) | [Concordia Design Pattern](https://arxiv.org/abs/2507.08892) | [Code Cheat Sheet](CHEATSHEET.md)

## About

Concordia is a library for constructing and running generative agent-based
models that simulate interactions among entities in grounded physical, social,
or digital environments. It uses an interaction pattern inspired by tabletop
role-playing games: a special entity called the **Game Master** (GM) simulates
the environment in which player entities interact. Entities describe their
intended actions in natural language, and the GM translates these into
appropriate outcomes e.g. checking physical plausibility in simulated worlds.

Concordia supports a broad range of applications, including social science
research, AI safety and ethics, cognitive neuroscience, economics, synthetic
data generation for personalization, and performance evaluation of real services
through simulated usage.

Concordia requires access to a standard LLM API and may optionally integrate
with external applications and services.

## How it Works

Concordia operates as a **game engine** for generative agents, built around
three core concepts:

*   **Entities**: The actors in the simulation—either player characters
    (Agents) or system controllers (Game Masters).
*   **Components**: Modular building blocks of an Entity. Entity
    behaviors e.g. logic, chains of thought, memory operations, etc are all
    implemented within components. Concordia comes with a core library of
    components and user-created components are also included in the main
    library under the contrib directory. It's easy to create your own components
    and add them to the library.
*   **Engine**: The simulation loop. It solicits actions from entities and
    delegates resolution to the Game Master.

This modular architecture enables complex behaviors to be assembled from simple,
reusable parts.

## Folder Structure

*   **[`concordia/prefabs`](concordia/prefabs/README.md)**: Pre-assembled
    recipes for common agents and Game Masters.
*   **[`concordia/components`](concordia/components/README.md)**: Modular
    building blocks for agents, including memory systems, reasoning chains, and
    sensory modules.
*   **[`concordia/environment`](concordia/environment/README.md)**: The "engine"
    of the simulation, containing the Game Master and the turn-taking loop.
*   **[`concordia/document`](concordia/document/README.md)**: Utilities for
    managing LLM prompts and context.
*   **[`concordia/language_model`](concordia/language_model/README.md)**: LLM
    integration and API wrappers.
*   **[`examples/`](examples/)**: Tutorials and example simulations to help you
    get started.

> [!TIP]
> The best way to learn is to watch the [Concordia: Building Generative Agent-Based Models](https://youtu.be/2FO5g65mu2I?si=TSk7XTk4gCaadEDs) tutorial on YouTube, run the
> **[`examples/tutorial.ipynb`](examples/tutorial.ipynb)** and then try
> modifying the **Prefabs** to see how agent behavior changes.

## Installation

[Concordia is available on PyPI](https://pypi.python.org/pypi/gdm-concordia)
and can be installed using:

```shell
pip install gdm-concordia
```

After doing this you can then `import concordia` in your own code.

## Development

### Codespace

The easiest way to work on the Concordia source code is to use our
pre-configured development environment via a
[GitHub Codespace](https://github.com/features/codespaces).

This provides a tested, reproducible development workflow that minimizes
dependency management. We strongly recommend preparing all pull requests for
Concordia via this workflow.

### Manual setup

If you want to work on the Concordia source code within your own development
environment you will have to handle installation and dependency management
yourself.

For example, you can perform an editable installation as follows:

1.  Clone Concordia:

    ```shell
    git clone -b main https://github.com/google-deepmind/concordia
    cd concordia
    ```

2.  Create and activate a virtual environment:

    ```shell
    python -m venv venv
    source venv/bin/activate
    ```

3.  Install Concordia:

    ```shell
    pip install --editable .[dev]
    ```

4.  Test the installation:

    ```shell
    pytest --pyargs concordia
    ```

5.  Install any additional language model dependencies you will need, e.g.:

    ```shell
    pip install .[google]
    pip install --requirement=examples/requirements.in
    ```

    Note that at this stage you may find that your development environment is
    not supported by some underlying dependencies and you will need to do some
    dependency management.

## Bring your own LLM

Concordia requires access to an LLM API. Any LLM API that supports sampling
text should work, though result quality depends on the capabilities of the
chosen model. You must also provide a text embedder for associative memory. Any
fixed-dimensional embedding works for this, ideally one suited to sentence
similarity or semantic search.

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

## Citing Concordia

If you use Concordia in your work, please cite the accompanying article:

<!-- disableFinding(SNIPPET_INVALID_LANGUAGE) -->
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
