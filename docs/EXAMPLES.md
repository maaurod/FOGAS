# Examples

The active notebooks are now grouped by method family.

## FOGAS

Location: [`experiments/fogas/notebooks/`](/home/mauro/Desktop/EMAI/Ljubljana/Thesis/Code/experiments/fogas/notebooks)

Main notebooks:

- `2State.ipynb`
- `3grid.ipynb`
- `3grid_wall.ipynb`
- `10grid_tabular.ipynb`
- `10grid_RBF.ipynb`
- `20grid.ipynb`
- `20gird_opt.ipynb`
- `40grid.ipynb`
- `40grid_opt.ipynb`
- `100grid.ipynb`
- `MountainCar_rebuilt.ipynb`

Legacy non-vectorized FOGAS notebooks are also kept under:

- [`experiments/fogas/notebooks/legacy/`](/home/mauro/Desktop/EMAI/Ljubljana/Thesis/Code/experiments/fogas/notebooks/legacy)

## Generalized FOGAS

Location: [`experiments/fogas_generalization/notebooks/`](/home/mauro/Desktop/EMAI/Ljubljana/Thesis/Code/experiments/fogas_generalization/notebooks)

Main notebooks:

- `2State_gen.ipynb`
- `3grid_gen.ipynb`
- `3gridw_gen.ipynb`

## FQI

Location: [`experiments/fqi/notebooks/`](/home/mauro/Desktop/EMAI/Ljubljana/Thesis/Code/experiments/fqi/notebooks)

Main notebooks:

- `fqi_testing.ipynb`

## Running

From the repo root:

```bash
pip install -e .
```

The notebooks now resolve imports through `src/rl_methods` and load assets from `data/` and `experiments/shared/assets/`.
