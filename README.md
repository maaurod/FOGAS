# FOGAS: Feature-Occupancy Gradient Ascent

Implementation of the FOGAS algorithm for offline reinforcement learning in linear MDPs.

## Repository Structure

```
fogas/
├── mdp/                      # Core MDP framework
│   ├── linear_mdp.py         # LinearMDP: Linear MDP implementation
│   └── policy_solver.py      # PolicySolver: MDP with policy evaluation
│
├── algorithm/                # FOGAS algorithm components
│   ├── fogas_dataset.py      # FOGASDataset: Load and manage offline datasets
│   ├── fogas_parameters.py   # FOGASParameters: Theoretical parameter computation
│   ├── fogas_solver.py       # FOGASSolver: Main FOGAS algorithm
│   ├── fogas_solver_vectorized.py  # FOGASSolverVectorized: Optimized version
│   ├── fogas_oraclesolver.py       # FOGASOracleSolver: Oracle variant with true dynamics
│   ├── fogas_oraclesolver_vectorized.py  # FOGASOracleSolverVectorized: Optimized oracle
│   ├── fogas_evaluator.py    # FOGASEvaluator: Metrics and evaluation tools
│   └── fogas_hyperoptimizer.py  # FOGASHyperOptimizer: Hyperparameter optimization
│
└── dataset_collection/       # Dataset generation utilities
    ├── linear_mdp_env.py     # LinearMDPEnv: Gymnasium-compatible environment
    └── env_data_collector.py # EnvDataCollector: Collect offline datasets
```

## Class Hierarchy and Usage Order

### 1. **MDP Definition** (Choose one)

#### Option A: Direct MDP Specification
```python
from fogas import LinearMDP

mdp = LinearMDP(
    states=states,
    actions=actions,
    phi=phi,           # Feature mapping function
    omega=omega,       # Reward feature weights
    gamma=gamma,       # Discount factor
    x0=initial_state
)
```

#### Option B: MDP with Policy Evaluation
```python
from fogas import PolicySolver

mdp = PolicySolver(
    states=states,
    actions=actions,
    phi=phi,
    omega=omega,
    gamma=gamma,
    x0=initial_state,
    psi=psi            # Transition feature mapping
)
```

### 2. **Dataset Collection** (Optional - if you need to generate data)

```python
from fogas import LinearMDPEnv, EnvDataCollector

# Create environment
env = LinearMDPEnv(mdp=mdp)

# Collect dataset
collector = EnvDataCollector(mdp=mdp, env_name="my_problem")
collector.collect_dataset(
    n_steps=1000,
    save_path="datasets/my_problem.csv",
    verbose=True
)
```

### 3. **FOGAS Solver** (Choose one variant)

#### Standard FOGAS (with offline dataset)
```python
from fogas import FOGASSolverVectorized, FOGASDataset

solver = FOGASSolverVectorized(
    mdp=mdp,
    csv_path="datasets/my_problem.csv",
    delta=0.05,        # Confidence level
    print_params=True
)

# Run the algorithm
policy = solver.run(T=1000)  # T = number of iterations
```

#### Oracle FOGAS (with true dynamics)
```python
from fogas import FOGASOracleSolverVectorized

solver = FOGASOracleSolverVectorized(
    mdp=mdp,
    delta=0.05,
    print_params=True
)

policy = solver.run(T=1000)
```

### 4. **Evaluation**

```python
from fogas import FOGASEvaluator

evaluator = FOGASEvaluator(solver)

# Compare with optimal policy
evaluator.compare_final_rewards()

# Analyze value functions
evaluator.compare_value_functions()

# Get performance metrics
reward_metric = evaluator.get_metric("reward")
gap_metric = evaluator.get_metric("expected_gap")
```

### 5. **Hyperparameter Optimization** (Optional)

```python
from fogas import FOGASHyperOptimizer

optimizer = FOGASHyperOptimizer(
    solver=solver,
    metric_callable=evaluator.get_metric("reward"),
    seed=42
)

# Optimize hyperparameters
results = optimizer.optimize_fogas_hyperparameters(
    order=("alpha", "rho", "eta"),
    search_method="bo",  # or "random"
    num_runs=3,
    plot=True
)

print(f"Optimized alpha: {results['alpha']}")
print(f"Optimized rho: {results['rho']}")
print(f"Optimized eta: {results['eta']}")
```

## Quick Start

See example notebooks in `testing_vectorized/`:
- `2State.ipynb`: Simple 2-state MDP example
- `3grid.ipynb`: 3x3 gridworld navigation
- `3grid_wall.ipynb`: 3x3 gridworld with obstacles

## Google Colab Usage

There are **two ways** to use this repository in Google Colab:

### Method 1: Clone and Run Notebooks Directly (Easiest) ⭐

The notebooks are already set up to work in Colab! Just clone and run:

```python
# 1. Clone the repository
!git clone https://github.com/maaurod/FOGAS.git
%cd FOGAS

# 2. Install dependencies
!pip install -r requirements.txt

# 3. Open any notebook from testing_vectorized/ and run it!
# The notebooks use PROJECT_ROOT which automatically works in Colab
```

**Then**: In Colab's file browser, navigate to `testing_vectorized/` and open any notebook (e.g., `2State.ipynb`). All paths will work automatically!

### Method 2: Install as Package (For Custom Scripts)

If you want to write your own code instead of using the notebooks:

```python
# 1. Install the package directly from GitHub
!pip install git+https://github.com/maaurod/FOGAS.git

# 2. Now you can import from anywhere
from fogas import FOGASSolverVectorized, PolicySolver

# 3. Write your own code
mdp = PolicySolver(...)
solver = FOGASSolverVectorized(mdp=mdp, csv_path="path/to/data.csv")
policy = solver.run(T=1000)
```

**Use Method 1 if**: You want to run the example notebooks  
**Use Method 2 if**: You want to write custom Python scripts in Colab

**Note**: The example notebooks already use relative paths that work seamlessly in Colab after cloning.

## PyTorch Version

A PyTorch implementation is available in `fogas_torch/` with the same API and class structure.

## Requirements

- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- scipy
- tqdm
- gymnasium (for environment simulation)

See `requirements.txt` for specific versions.

## License

MIT License - see [LICENSE](LICENSE) file for details.
