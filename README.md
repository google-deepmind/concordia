# Concordia 

*A library for generative social simulation*

[![Python](https://img.shields.io/pypi/pyversions/gdm-concordia.svg)](https://pypi.python.org/pypi/gdm-concordia)
[![PyPI version](https://img.shields.io/pypi/v/gdm-concordia.svg)](https://pypi.python.org/pypi/gdm-concordia)
[![PyPI tests](../../actions/workflows/pypi-test.yml/badge.svg)](../../actions/workflows/pypi-test.yml)
[![Tests](../../actions/workflows/test-concordia.yml/badge.svg)](../../actions/workflows/test-concordia.yml)
[![Examples](../../actions/workflows/test-examples.yml/badge.svg)](../../actions/workflows/test-examples.yml)

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


## Bring your own LLM

To work, Concordia requires an access to an LLM API. Any LLM API that 
supports sampling text would work. We tested Concordia with a model with 340B
parameters. If using a custom LLM API, the user has to provide a text embedder
to be used by the associative memory. By default we use the Sentence-T5 for
this, but any fixed-dimensional embedding would work.

## Example usage

Find below an illustrative social simulation where 4 friends are stuck in a
snowed in pub. Two of them have a dispute over a crashed car.

The agents are built using a simple reasoning inspired by March and Olsen (2011)
who posit that humans generally act as though they choose their actions by
answering three key questions:

1. What kind of situation is this?
2. What kind of person am I?
3. What does a person such as I do in a situation such as this?

The agents used in this example implement exactly these components, and nothing
else.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/concordia/blob/main/examples/three_key_questions.ipynb)

## Citing Concordia

If you use Concordia in your work, please cite the accompanying article:

```bibtex
@misc{vezhnevets2023generative,
      title={Generative agent-based modeling with actions grounded in physical,
      social, or digital space using Concordia}, 
      author={Alexander Sasha Vezhnevets and John P. Agapiou and Avia Aharon and
      Ron Ziv and Jayd Matyas and Edgar A. Duéñez-Guzmán and
      William A. Cunningham and Simon Osindero
      and Danny Karmon and Joel Z. Leibo},
      year={2023},
      eprint={2312.03664},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```

## Disclaimer

This is not an officially supported Google product.
