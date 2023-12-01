# Concordia 

*A library for generative social simulation*

[![Python](https://img.shields.io/pypi/pyversions/dm-concordia.svg)](https://pypi.python.org/pypi/dm-concordia)
[![PyPI version](https://img.shields.io/pypi/v/dm-concordia.svg)](https://pypi.python.org/pypi/dm-concordia)
[![PyPI tests](../../actions/workflows/pypi-test.yml/badge.svg)](../../actions/workflows/pypi-test.yml)
[![Tests](../../actions/workflows/test-concordia.yml/badge.svg)](../../actions/workflows/test-concordia.yml)
[![Examples](../../actions/workflows/test-examples.yml/badge.svg)](../../actions/workflows/test-examples.yml)

<!-- TODO: b/311364310 - add link to the tech report once it is published -->
[Concordia Tech Report]()

## About

Concordia is a platform designed for constructing generative models that 
simulate social interactions within a digitally-grounded action space. This 
platform facilitates the emulation of agent behaviors and activities. The 
framework can cater and support a wide array of applications, ranging from 
social science research and AI ethics to cognitive neuroscience and economics; 
Additionally, it also can be leveraged for generating data for personalization 
applications and for conducting performance evaluations of real services through
simulated usage. Our system simply requires access to a standard LLM API, and 
possible integration to real applications and services. The rest is python for 
scaffolding, orchestration, prompt-templating, experiment design and analysis. 


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

To work, Concordia requires an access to an LLM API. The example below is 
written using [Saxml](https://github.com/google/saxml), but any LLM API that 
supports sampling text and calculating log-likelihood would work. We recommend
using large (>300B parameters) models. If using a custom LLM API, the user has
to provide a text embedder to be used by the associative memory. By default we
use the Sentence-T5 for this, but any fixed-dimensional embedding would work.

## Example usage

Find below an illustrative social simulation with 5 players which simulates the 
day of mayoral elections in an imaginary town caller Riverbend. First two 
players, Alice and Bob, are running for mayor. The third player, Charlie, 
is trying to ruin Alice's reputation with disinformation. The last two players 
have no specific agenda, apart from voting in the election.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/concordia/examples/village/riverbend_elections.ipynb)

## Citing Concordia

If you use Concordia in your work, please cite the accompanying article:

<!-- TODO: b/311364310 - update CITATION.bib and README.md once tech report published -->

```bibtex
@inproceedings{vezhnevets2023concordia,
    title={Concordia: a library for generative social simulation},
    author={Alexander Sasha Vezhnevets AND Joel Z. Leibo AND John P. Agapiou
    AND Danny Karmon AND Avia Aharon AND Ron Viz
    AND Jayd Matyas AND Edgar Du\'e\~nez-Guzm\'an AND Wil Cunnigham 
            AND Simon Osindero},
    year={2023},
}
```

## Disclaimer

This is not an officially supported Google product.
