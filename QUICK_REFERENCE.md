# FOGAS Quick Reference

## Class Import Order

```python
from fogas import (
    # 1. MDP Definition
    LinearMDP,              # Basic linear MDP
    PolicySolver,           # MDP with policy evaluation
    
    # 2. Data Collection (Optional)
    LinearMDPEnv,           # Gymnasium environment
    EnvDataCollector,       # Dataset generator
    
    # 3. Algorithm Components
    FOGASDataset,           # Load offline data
    FOGASParameters,        # Compute theoretical params
    FOGASSolverVectorized,  # Main algorithm (vectorized)
    FOGASOracleSolverVectorized,  # Oracle variant
    
    # 4. Evaluation & Optimization
    FOGASEvaluator,         # Metrics and analysis
    FOGASHyperOptimizer,    # Hyperparameter tuning
)
```

## Typical Workflow

```
1. MDP → 2. Dataset → 3. Solver → 4. Evaluation → 5. Optimization
   ↓         ↓            ↓           ↓              ↓
LinearMDP   DATASET   FOGASSolverV  Evaluator   HyperOptim
           or Oracle
```

## Class Dependencies

```
LinearMDP (no dependencies)
    ↓
PolicySolver (extends LinearMDP)
    ↓
LinearMDPEnv (uses PolicySolver)
    ↓
EnvDataCollector (uses LinearMDPEnv)
    ↓
FOGASDataset (loads CSV)
    ↓
FOGASSolverVectorized (uses MDP + Dataset)
    ↓
FOGASEvaluator (uses Solver)
    ↓
FOGASHyperOptimizer (uses Solver + Evaluator)
```

## Minimal Example

```python
# 1. Define MDP
mdp = PolicySolver(states, actions, phi, omega, gamma, x0, psi)

# 2. Create solver
solver = FOGASSolverVectorized(mdp, csv_path="data.csv")

# 3. Run
policy = solver.run(T=1000)

# 4. Evaluate
evaluator = FOGASEvaluator(solver)
evaluator.compare_final_rewards()
```

## Available Metrics

- `"reward"` - Final policy return
- `"expected_gap"` - Average optimality gap
