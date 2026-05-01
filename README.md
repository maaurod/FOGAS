# RL Methods Research Repository

Reordered research codebase for:

- FOGAS
- generalized FOGAS experiments
- FQI
- shared MDP and dataset tooling
- a reserved `sbeed/` package for future work

The previous repository layout is preserved under [`old/`](/home/mauro/Desktop/EMAI/Ljubljana/Thesis/Code/old).

## Repository Layout

```text
.
├── src/rl_methods/             # Reusable package code
│   ├── mdp/                    # Core MDP abstractions and discretization
│   ├── dataset_collection/     # Data collection, env wrappers, analysis
│   ├── fogas/                  # Original FOGAS implementation
│   ├── fogas_generalization/   # Generalized FOGAS implementation
│   ├── fqi/                    # FQI implementation
│   ├── q_learning/             # Q-learning utilities
│   └── sbeed/                  # Placeholder for future SBEED work
├── experiments/
│   ├── fogas/
│   ├── fogas_generalization/
│   ├── fqi/
│   ├── media/
│   └── shared/
├── data/
│   ├── datasets/               # Transition datasets
│   └── results/                # Grid-search and result tables
├── jobs/                       # Cluster / batch submission scripts
├── scripts/                    # Local helper scripts
├── docs/                       # Supporting notes and references
├── tests/
└── old/                        # Snapshot of the pre-reorganization repo
```

## Package Imports

Install in editable mode from the repo root:

```bash
pip install -e .
```

Then import from `rl_methods`:

```python
from rl_methods import PolicySolver, EnvDataCollector
from rl_methods.fogas import FOGASSolverVectorized, FOGASEvaluator
from rl_methods.fogas_generalization import FOGASSolverBetaVectorized
from rl_methods.fqi import FQISolver, FQIEvaluator
```

## MDP Layer

The MDP layer intentionally stays close to the previous design:

- `rl_methods.mdp.linear_mdp.LinearMDP`
- `rl_methods.mdp.policy_solver.PolicySolver`
- `rl_methods.mdp.abstract_mdp`

This keeps the current linear-MDP workflow intact while still supporting:

- tabular finite MDPs
- discretized continuous problems through `DiscretizedLinearMDP`
- feature-only abstractions used by the MountainCar workflow

So the repo is reorganized, but the MDP conceptual model is not replaced.

## Data and Results

- input datasets live in [`data/datasets/`](/home/mauro/Desktop/EMAI/Ljubljana/Thesis/Code/data/datasets)
- generated result tables live in [`data/results/`](/home/mauro/Desktop/EMAI/Ljubljana/Thesis/Code/data/results)

Notebook and script path helpers were updated to use:

- `PROJECT_ROOT / "data" / "datasets"`
- `PROJECT_ROOT / "data" / "results"`
- `PROJECT_ROOT / "src"`

## Experiments

The notebooks are grouped by algorithm family:

- [`experiments/fogas/`](/home/mauro/Desktop/EMAI/Ljubljana/Thesis/Code/experiments/fogas)
- [`experiments/fogas_generalization/`](/home/mauro/Desktop/EMAI/Ljubljana/Thesis/Code/experiments/fogas_generalization)
- [`experiments/fqi/`](/home/mauro/Desktop/EMAI/Ljubljana/Thesis/Code/experiments/fqi)

The notebook split now matches your classification:

- only `fqi_testing.ipynb` is under `experiments/fqi/notebooks/`
- only the 3 copied generalized notebooks are under `experiments/fogas_generalization/notebooks/`
- every other notebook is under `experiments/fogas/notebooks/`

The older non-vectorized FOGAS notebooks are still kept under `experiments/fogas/notebooks/legacy/`, so no notebook files were dropped.

The generalized notebooks that were previously in `testing copy/` now live under `experiments/fogas_generalization/notebooks/`, and the generalized algorithm code that was previously in `algorithm_gen/` now lives in `src/rl_methods/fogas_generalization/`.

## Old Snapshot

The `old/` folder contains the repository exactly as it existed before the reorganization pass, including the original folder names and paths. It is kept for reference and migration checks.

## Notes

- `src/rl_methods/sbeed/` is intentionally just a placeholder package for now.
- Notebook outputs were left intact where possible, so some rendered outputs may still display old absolute paths even though the executable source cells were updated.
