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

To use this repository in Google Colab:

```python
# Clone the repository
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
%cd YOUR_REPO_NAME

# Install dependencies
!pip install -r requirements.txt

# Import and use
from fogas import FOGASSolverVectorized, PolicySolver
```

**Note**: The repository uses relative paths (`PROJECT_ROOT / "datasets" / ...`) which work seamlessly in Colab after cloning.

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

[-]
