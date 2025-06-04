# MDP GridWorld Visualization for CS3600

This repository provides an interactive visualization tool for exploring Markov Decision Processes (MDPs) as part of the CS3600 (Introduction to Artificial Intelligence) course at Georgia Tech. It contains implementations of value iteration and policy iteration algorithms with step-by-step visualization capabilities.

Below is a demo of the step-by-step interactive visualization of the value iteration algorithm for the classic 4x3 gridworld environment:

![Value Iteration Interactive Visualization](assets/mdp_viz_1080p.gif)

## Overview

### MDP Algorithms

This repository includes implementations of two fundamental MDP algorithms:

1. **Value Iteration**: Iteratively computes optimal state values using the Bellman equation until convergence.

2. **Policy Iteration**: Alternates between policy evaluation and policy improvement to find the optimal policy.

### Bellman Equation Formulations

The implementation supports two different Bellman equation formulations:

- **AIMA**: `Q = Σ P(s'|s,a) * (R(s,a,s') + γ*V(s'))` (AI textbook standard)
- **BERKELEY**: `Q = R(s) + γ * Σ P(s'|s,a) * V(s')` (Berkeley CS188 style)

### Visualization Features

- **Environment Structure**: View terminal states, obstacles, and rewards
- **Interactive Widgets**: Step through iterations with play/pause controls
- **Q-Value Display**: Triangular indicators showing action values at cell edges
- **Policy Visualization**: Arrows indicating optimal actions

## How to Use This Repository

There are three ways to use this code:

### 1. Using Google Colab (Recommended for Quick Start)

The easiest way to use this code is to use this Google Colab that replicates the functionality in the gridworld MDP notebook. The Colab notebook is a good way to get started with the code and explore the algorithms.

**Colab Link**: [https://tinyurl.com/cs3600-colab-mdp](https://tinyurl.com/cs3600-colab-mdp)

### 2. Local Setup

To run the code locally:

1. Clone this repository:
   ```bash
   git clone https://github.com/karpekov/cs3600-mdp-gridworld.git
   cd cs3600-mdp-gridworld
   ```

2. Set up the environment:

   **Using Conda (Recommended)**:
   ```bash
   conda env create -f environment.yml
   conda activate cs3600-mdp-gridworld
   ```

   **Using pip**:
   ```bash
   pip install numpy matplotlib ipywidgets jupyter notebook
   ```

3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

4. Open `gridworld_mdp.py` or create a new notebook to explore the MDP algorithms.

### 3. Direct Python Script Usage

You can also run the code directly as a Python script without Jupyter:

```python
python gridworld_mdp.py
```

This will run a default example and display the visualization.

### Quick Start Example

```python
from gridworld_mdp import GridWorldMDP

# Create a simple 4x3 gridworld
mdp = GridWorldMDP(
    grid_size=(4, 3),
    terminal_states={(3, 2): 1.0, (3, 1): -1.0},
    obstacles={(1, 1)},
    step_cost=-0.04,
    gamma=0.9,
    noise=0.1
)

# Run value iteration and visualize
mdp.value_iteration(max_iterations=20)
mdp.create_visualization_widgets()
```

## Files Description

- `gridworld_mdp.py`: Main implementation with MDP algorithms and visualization
- `README.md`: This file
- `LICENSE`: MIT license file

## Debugging the Algorithms

For educational purposes, the code includes simple debugging functions:

### Debug Functions

```python
# Step through value iteration with breakpoints
debug_value_iteration()

# Step through policy iteration with breakpoints
debug_policy_iteration()

# Show detailed iteration information
show_iteration_details(iteration=5)
```

### Setting Breakpoints

1. Open `gridworld_mdp.py` in your IDE
2. Set breakpoints in the debug functions or main algorithm loops
3. Run the debug functions to step through the algorithms
4. Watch key variables like `values`, `policy`, and `q_values`

### Key Variables to Monitor

- `values`: Current state value estimates
- `policy`: Current policy (action for each state)
- `q_values`: Action-value estimates for each state-action pair
- `delta`: Maximum change in values between iterations

## Environment Examples

The code includes several pre-configured environments:

- **Simple 4x3**: Classic textbook example with goal and trap
- **Large 10x10**: Complex maze with multiple goals and obstacles
- **Custom**: Create your own gridworld configurations

## Author

[Alexander Karpekov](https://alexkarpekov.com) is the author of this repository. He is a PhD student at Georgia Tech and created this repository to support his teaching of MDP algorithms in the CS3600 course.

*Parts of this repository were co-developed with the assistance of AI tools, including Claude 4.0 Sonnet and Cursor. All content was reviewed and edited by the author.*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.