# Tetris Heuristic Agent Testing Framework

This directory contains a testing framework for evaluating and tuning heuristic-based Tetris agents.

## Overview

The framework consists of the following components:

1. **Heuristic Agent** (`heuristic_agent.py`): A simulation-based agent that evaluates all possible final positions for each piece and selects the best move based on heuristic evaluation.

2. **Testing Framework** (`test_heuristic_agent.py`): A script to run tests with the agent and output detailed results to the terminal.

3. **Weight Tuning** (`tune_heuristic_weights.py`): A script to find the optimal weights for the heuristic agent by testing multiple weight combinations.

## Installation

Make sure you have all the required dependencies installed:

```bash
pip install -r requirements.txt
```

## Usage

### Running Tests

To run tests with the default heuristic weights:

```bash
python test_heuristic_agent.py
```

Options:
- `--episodes N`: Number of episodes to run (default: 3)
- `--delay D`: Delay between moves in seconds (default: 0.01)
- `--no-render`: Disable rendering
- `--verbose`: Print detailed move information
- `--test-name NAME`: Name for the test (default: 'heuristic_test')

You can also customize the weights:
- `--holes-weight W`: Weight for holes (default: -4.0)
- `--height-weight W`: Weight for height (default: -0.5)
- `--bumpiness-weight W`: Weight for bumpiness (default: -1.0)
- `--lines-weight W`: Weight for lines cleared (default: 3.0)
- `--well-weight W`: Weight for well depth (default: 0.5)

### Tuning Weights

To find the optimal weights for the heuristic agent:

```bash
python tune_heuristic_weights.py
```

Options:
- `--episodes N`: Number of episodes per weight combination (default: 2)
- `--render`: Enable rendering (slower)
- `--test-name NAME`: Name for the tuning test (default: 'weight_tuning')

You can also customize the weight ranges to test:
- `--holes-values`: Comma-separated values for holes weight (default: '-5.0,-4.0,-3.0,-2.0')
- `--height-values`: Comma-separated values for height weight (default: '-1.0,-0.5,-0.1')
- `--bumpiness-values`: Comma-separated values for bumpiness weight (default: '-2.0,-1.0,-0.5')
- `--lines-values`: Comma-separated values for lines cleared weight (default: '2.0,3.0,4.0')
- `--well-values`: Comma-separated values for well depth weight (default: '0.0,0.5,1.0')

### Visualizing Results

To visualize test results:

```bash
python visualize_results.py --latest
```

Options:
- `--file PATH`: Path to a specific results file
- `--latest`: Use the latest test results file
- `--latest-tuning`: Use the latest tuning results file
- `--save-dir DIR`: Directory to save plots
- `--x-weight WEIGHT`: Weight for x-axis in heatmap (default: 'holes')
- `--y-weight WEIGHT`: Weight for y-axis in heatmap (default: 'lines_cleared')

## Heuristics

The agent evaluates positions using the following heuristics:

1. **Holes**: Empty cells with filled cells above them (penalty)
2. **Height**: Maximum height of the board (penalty)
3. **Bumpiness**: Sum of absolute differences between adjacent column heights (penalty)
4. **Lines Cleared**: Number of lines cleared by the move (reward)
5. **Well Depth**: Depth of wells in the board (reward)

## Output

The testing framework outputs detailed results to the terminal, including:

- Weights used
- Aggregated metrics (average score, lines cleared, etc.)
- Action counts
- Episode results

Results are also saved to JSON files for later analysis.

## Visualization

The visualization script creates the following plots:

1. **Episode Metrics**: Bar charts showing scores and lines cleared for each episode
2. **Action Distribution**: Bar chart showing the distribution of actions taken
3. **Weight Comparison**: Bar charts comparing the performance of different weight combinations
4. **Weight Heatmap**: Heatmap showing how two weights affect the score

## Extending the Framework

To add new heuristics:

1. Add the new heuristic to the `_evaluate_position` method in `heuristic_agent.py`
2. Add a default weight for the new heuristic in the `__init__` method
3. Update the command-line arguments in `test_heuristic_agent.py` and `tune_heuristic_weights.py`
4. Update the visualization script if needed 