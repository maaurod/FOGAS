"""Continuous-observation FOGAS RBF mini-batch grid search for MountainCar."""

from grid_search_continuous_mountaincar_fogas_rbf import (
    RESULTS_DIR,
    run_grid_search,
)
import grid_search_continuous_mountaincar_fogas_rbf as base


base.MINIBATCH_GRID = True
base.OUTPUT_CSV = RESULTS_DIR / "continuous_fogas_rbf_batches_grid_search.csv"
base.BEST_CSV = RESULTS_DIR / "continuous_fogas_rbf_batches_grid_search_best.csv"
base.CHECKPOINT_CSV = RESULTS_DIR / "continuous_fogas_rbf_batches_eval_checkpoints.csv"


if __name__ == "__main__":
    run_grid_search()
