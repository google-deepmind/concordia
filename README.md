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

The evolutionary simulation is located in `examples/evolutionary_simulation.py` and supports multiple language model backends.

#### 1. **Quick Start (Dummy Model)**
Run with no dependencies - uses a dummy language model for testing:
```bash
python examples/evolutionary_simulation.py
```

#### 2. **With Language Models**

The simulation supports real language models for more sophisticated agent reasoning:

**Setup virtual environment (recommended):**
```bash
python -m venv evolutionary_env
source evolutionary_env/bin/activate  # On Windows: evolutionary_env\Scripts\activate
pip install sentence-transformers torch
```

**Using Gemma (Local PyTorch Model):**
```python
from examples.evolutionary_simulation import evolutionary_main, GEMMA_CONFIG

# This will download google/gemma-2b-it model locally
measurements = evolutionary_main(config=GEMMA_CONFIG)
```

**Using OpenAI GPT Models:**
```python
import os
from examples.evolutionary_simulation import evolutionary_main, OPENAI_CONFIG

# Set your OpenAI API key
os.environ['OPENAI_API_KEY'] = 'your-api-key-here'
openai_config = OPENAI_CONFIG
openai_config.api_key = os.getenv('OPENAI_API_KEY')

measurements = evolutionary_main(config=openai_config)
```

**Custom Language Model Configuration:**
```python
from examples.evolutionary_simulation import evolutionary_main
from concordia.typing.evolutionary import EvolutionConfig

config = EvolutionConfig(
    # Evolutionary parameters
    pop_size=8,
    num_generations=20,
    selection_method='topk',
    top_k=4,
    mutation_rate=0.1,
    num_rounds=15,
    
    # Language model configuration
    api_type='pytorch_gemma',        # or 'openai', 'mistral', etc.
    model_name='google/gemma-7b-it', # Larger model for better reasoning
    embedder_name='all-mpnet-base-v2',
    device='cuda:0',                 # Use GPU if available
    disable_language_model=False
)

measurements = evolutionary_main(config)
```

#### 3. **Supported Language Model Types**

| API Type | Example Models | Requirements |
|----------|----------------|--------------|
| `pytorch_gemma` | `google/gemma-2b-it`, `google/gemma-7b-it` | `torch`, `transformers` |
| `openai` | `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo` | API key via `OPENAI_API_KEY` |
| `mistral` | `mistral-large-latest`, `mistral-medium` | API key |
| `amazon_bedrock` | Claude, Llama models | AWS credentials |
| `google_aistudio_model` | Gemini models | Google AI Studio API key |

#### 4. **With Checkpointing and Resumption**
```python
from pathlib import Path

measurements = evolutionary_main(
    config=config,
    checkpoint_dir=Path("evolutionary_checkpoints/"),
    checkpoint_interval=5,           # Save every 5 generations
    resume_from_checkpoint=True      # Resume from latest if available
)
```

#### 5. **Environment Setup for Different Models**

**For local models (Gemma, Llama):**
```bash
pip install torch transformers sentence-transformers
```

**For OpenAI models:**
```bash
pip install openai sentence-transformers
export OPENAI_API_KEY="your-api-key"
```

**For Google AI Studio:**
```bash
pip install google-generativeai sentence-transformers
export GOOGLE_AI_STUDIO_API_KEY="your-api-key"
```

#### 6. **Performance Considerations**

- **Dummy Model**: Fastest, no LLM dependencies, good for testing algorithms
- **Local Models**: Slower startup (model download), but no API costs
- **Cloud APIs**: Fast startup, but incur API costs per generation

**Recommended configurations:**
- **Development/Testing**: Use dummy model (`disable_language_model=True`)
- **Research**: Use Gemma 2B for balanced performance and quality
- **Production**: Use GPT-4o or Gemma 7B for highest quality reasoning

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
