# Quick Reference

## Core Imports

```python
from rl_methods import (
    LinearMDP,
    PolicySolver,
    DiscretizedLinearMDP,
    BoxStateDiscretizer,
    DiscreteActionDiscretizer,
    LinearMDPEnv,
    EnvDataCollector,
)

from rl_methods.fogas import (
    FOGASDataset,
    FOGASParameters,
    FOGASSolverVectorized,
    FOGASOracleSolverVectorized,
    FOGASEvaluator,
    FOGASHyperOptimizer,
)

from rl_methods.fogas_generalization import (
    FOGASSolverBeta,
    FOGASSolverBetaVectorized,
    FOGASSolverPolicy,
)

from rl_methods.fqi import FQISolver, FQIEvaluator
```

## Typical Workflow

```text
1. Define or discretize an MDP
2. Collect or load a dataset
3. Run FOGAS / generalized FOGAS / FQI
4. Evaluate
5. Save results under data/results/
```

## Important Paths

```python
PROJECT_ROOT / "src"
PROJECT_ROOT / "data" / "datasets"
PROJECT_ROOT / "data" / "results"
PROJECT_ROOT / "experiments"
```

## Method Families

- `rl_methods.fogas`: original FOGAS implementation
- `rl_methods.fogas_generalization`: generalized FOGAS line
- `rl_methods.fqi`: FQI implementation
- `rl_methods.sbeed`: placeholder for future SBEED work

## Notebook Split

- `experiments/fogas/notebooks/`: all FOGAS notebooks, including `10grid_*`, `20gird_opt`, `40grid_opt`, and the kept legacy FOGAS notebooks under `legacy/`
- `experiments/fogas_generalization/notebooks/`: only `2State_gen.ipynb`, `3grid_gen.ipynb`, and `3gridw_gen.ipynb`
- `experiments/fqi/notebooks/`: only `fqi_testing.ipynb`
