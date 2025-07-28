# Run Evolutionary Simulation with Gemma 7B

You are tasked with running the evolutionary simulation using the Gemma 7B model. Follow these steps:

## Prerequisites
1. Ensure you're in the concordia project directory
2. Activate the evolutionary environment: `source evolutionary_env/bin/activate`
3. Verify the environment is active by checking `which python`

## Configuration
Use the Gemma 7B model configuration by creating a custom config:

```python
from concordia.typing import evolutionary as evolutionary_types

GEMMA_7B_CONFIG = evolutionary_types.EvolutionConfig(
    pop_size=6,
    num_generations=10,
    api_type='pytorch_gemma',
    model_name='google/gemma-7b-it',
    embedder_name='all-mpnet-base-v2',
    device='mps',  # Mac GPU acceleration via Metal Performance Shaders
    disable_language_model=False,
    # Optional: Add more configuration parameters
    tournament_size=3,
    mutation_rate=0.1,
    crossover_rate=0.8,
)
```

## Execution Steps
1. Run the simulation with the Gemma 7B configuration:
```bash
source evolutionary_env/bin/activate
PYTHONPATH=. python -c "
from examples.evolutionary_simulation import evolutionary_main
from concordia.typing import evolutionary as evolutionary_types

# Gemma 7B Configuration
GEMMA_7B_CONFIG = evolutionary_types.EvolutionConfig(
    pop_size=6,
    num_generations=10,
    selection_method='topk',
    top_k=3,
    mutation_rate=0.1,
    num_rounds=12,
    api_type='pytorch_gemma',
    model_name='google/gemma-7b-it',
    embedder_name='all-mpnet-base-v2',
    device='mps',  # Mac GPU acceleration via Metal Performance Shaders
    disable_language_model=False,
)

print('🚀 Starting evolutionary simulation with Gemma 7B...')
print(f'Population size: {GEMMA_7B_CONFIG.pop_size}')
print(f'Generations: {GEMMA_7B_CONFIG.num_generations}')
print(f'Model: {GEMMA_7B_CONFIG.model_name}')

measurements = evolutionary_main(config=GEMMA_7B_CONFIG)

print('✅ Simulation completed successfully!')
print('📊 Results have been saved to simulation_results/')
"
```

## Expected Output
- The simulation will run for the specified number of generations
- Progress will be logged showing cooperation rates and fitness statistics
- Results will be exported to `simulation_results/` directory
- Final cooperation rates and average scores will be displayed

## Troubleshooting
- If model download fails, ensure you have sufficient disk space
- Mac GPU acceleration is enabled with `device='mps'` for faster execution
- If memory issues occur, reduce `pop_size` or `num_generations`
- Ensure `evolutionary_env` is activated before running

## Performance Notes
- Gemma 7B requires significant memory (~14GB RAM recommended)
- First run will download the model (~13GB)
- **Mac GPU acceleration enabled**: Using Metal Performance Shaders (MPS) for faster inference
- **Mac-specific optimizations**: Model loading and inference will use your Mac's GPU
- Simulation time scales with population size and number of generations
- Download speed depends on internet connection, but inference will be significantly faster with MPS