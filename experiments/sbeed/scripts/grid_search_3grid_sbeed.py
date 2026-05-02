from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from tqdm import tqdm


def find_root(current_path: Path, marker: str = "setup.py") -> Path:
    current_path = current_path.resolve()
    for parent in [current_path] + list(current_path.parents):
        if (parent / marker).exists():
            return parent
    return current_path


PROJECT_ROOT = find_root(Path.cwd())
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "grids"
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_methods import (  # noqa: E402
    DiscreteMDPSpec,
    SBEEDSolver,
    SBEEDEvaluator,
    TabularStateActionFeatures,
    TabularStateFeatures,
)


STATES = torch.arange(9, dtype=torch.long)
ACTIONS = torch.arange(4, dtype=torch.long)
N = len(STATES)
A = len(ACTIONS)
GAMMA = 0.9
X0 = 0

GOAL_GRID = 8
PIT_GRID = 5
WALL_STATES = {4}
TERMINAL_STATES = {GOAL_GRID, PIT_GRID}


# Edit these lists to change the default grid search.
# The ranges are intentionally moderate for the 3x3 tabular problem: wide
# enough to catch unstable learning rates and entropy scales, but not so wide
# that a full search becomes unreasonable.
DEFAULT_GRID = {
    "seeds": [42],
    "lambda_entropies": [0.001, 0.01, 0.1],
    "etas": [0.001, 0.01, 0.1],
    "lr_values": [0.001, 0.01, 0.1, 1],
    "lr_policies": [0.001, 0.01, 0.1, 1],
    "taus": [1, 10, 100, 1_000.0, 10_000.0],
    "max_buffer_sizes": [3_000, 6_000, 12_000],
    "batch_sizes": [64, 128, 256],
    "episodes": [1_000, 2_000, 4000, 8000, 16000],
    "collect_per_episodes": [10, 20, 50, 100, 200, 500],
    "updates_per_episodes": [10, 20, 50, 100],
    "epsilons": [0.1, 0.2, 0.3],
}


def next_state(s: int, a: int) -> int:
    s = int(s)
    a = int(a)

    if s in TERMINAL_STATES:
        return s

    row, col = divmod(s, 3)

    if a == 0:
        new_row, new_col = row - 1, col
    elif a == 1:
        new_row, new_col = row + 1, col
    elif a == 2:
        new_row, new_col = row, col - 1
    elif a == 3:
        new_row, new_col = row, col + 1
    else:
        raise ValueError("action must be in {0, 1, 2, 3}")

    if not (0 <= new_row < 3 and 0 <= new_col < 3):
        return s

    sp = new_row * 3 + new_col
    if sp in WALL_STATES:
        return s
    return sp


def reward_fn(s: int, a: int, sp: int) -> float:
    del s, a
    if int(sp) == GOAL_GRID:
        return 1.0
    if int(sp) == PIT_GRID:
        return -1.0
    return -0.1


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def build_spec() -> DiscreteMDPSpec:
    value_features = TabularStateFeatures(n_states=N)
    rho_features = TabularStateActionFeatures(n_states=N, n_actions=A)
    return DiscreteMDPSpec(
        n_states=N,
        n_actions=A,
        gamma=GAMMA,
        value_features=value_features,
        rho_features=rho_features,
        x0=X0,
    )


def param_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["seed"],
        row["lambda_entropy"],
        row["eta"],
        row["lr_value"],
        row["lr_policy"],
        row["tau"],
        row["max_buffer_size"],
        row["batch_size"],
        row["episodes"],
        row["collect_per_episode"],
        row["updates_per_episode"],
        row["epsilon"],
    )


def completed_keys(path: Path) -> set[tuple[Any, ...]]:
    if not path.exists():
        return set()

    keys = set()
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("status") != "ok":
                continue
            try:
                keys.add(
                    (
                        int(row["seed"]),
                        float(row["lambda_entropy"]),
                        float(row["eta"]),
                        float(row["lr_value"]),
                        float(row["lr_policy"]),
                        float(row["tau"]),
                        int(row["max_buffer_size"]),
                        int(row["batch_size"]),
                        int(row["episodes"]),
                        int(row["collect_per_episode"]),
                        int(row["updates_per_episode"]),
                        float(row["epsilon"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    return keys


def build_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    names = [
        "seed",
        "lambda_entropy",
        "eta",
        "lr_value",
        "lr_policy",
        "tau",
        "max_buffer_size",
        "batch_size",
        "episodes",
        "collect_per_episode",
        "updates_per_episode",
        "epsilon",
    ]
    values: list[Iterable[Any]] = [
        args.seeds,
        args.lambda_entropies,
        args.etas,
        args.lr_values,
        args.lr_policies,
        args.taus,
        args.max_buffer_sizes,
        args.batch_sizes,
        args.episodes,
        args.collect_per_episodes,
        args.updates_per_episodes,
        args.epsilons,
    ]
    grid = [dict(zip(names, combo)) for combo in itertools.product(*values)]
    if args.shuffle:
        rng = random.Random(args.shuffle_seed)
        rng.shuffle(grid)
    if args.max_trials is not None:
        grid = grid[: args.max_trials]
    return grid


def run_trial(params: dict[str, Any], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    seed = int(params["seed"])
    seed_all(seed)

    solver = SBEEDSolver(
        spec=build_spec(),
        lambda_entropy=float(params["lambda_entropy"]),
        eta=float(params["eta"]),
        ridge=args.ridge,
        lr_value=float(params["lr_value"]),
        lr_policy=float(params["lr_policy"]),
        tau=float(params["tau"]),
        buffer_mode="fifo",
        max_buffer_size=int(params["max_buffer_size"]),
        batch_size=int(params["batch_size"]),
        seed=seed,
        device=device,
    )

    start_time = time.time()
    solver.run(
        transition_fn=next_state,
        reward_fn=reward_fn,
        episodes=int(params["episodes"]),
        collect_per_episode=int(params["collect_per_episode"]),
        updates_per_episode=int(params["updates_per_episode"]),
        initial_collect_steps=args.initial_collect_steps,
        start_state=X0,
        behavior="policy",
        epsilon=float(params["epsilon"]),
        terminal_states=TERMINAL_STATES,
        tqdm_print=args.tqdm_inner,
        verbose=args.verbose_solver,
        log_every=args.log_every,
        store_history=False,
    )
    elapsed_seconds = time.time() - start_time

    evaluator = SBEEDEvaluator(
        solver,
        next_state_fn=next_state,
        reward_fn=reward_fn,
        terminal_states=TERMINAL_STATES,
    )
    stats = evaluator.compare_to_optimal_values(print_each=False)

    objective = None
    if solver.n > 0:
        objective = float(solver.objective()["objective"])

    return {
        **params,
        "status": "ok",
        "error": "",
        "elapsed_seconds": elapsed_seconds,
        "buffer_size_final": int(solver.n),
        "objective": objective,
        "v_sbeed_minus_v_lambda_star_l2": stats["l2_to_soft"],
        "v_sbeed_minus_v_lambda_star_linf": stats["linf_to_soft"],
        "v_sbeed_minus_v_star_l2": stats["l2_to_hard"],
        "v_sbeed_minus_v_star_linf": stats["linf_to_hard"],
        "soft_bellman_residual_linf": stats["soft_residual"],
        "hard_bellman_residual_linf": stats["hard_residual"],
        "V_sbeed": json.dumps(stats["V_sbeed"].tolist()),
        "V_lambda_star": json.dumps(stats["V_lambda_star"].tolist()),
        "V_star": json.dumps(stats["V_star"].tolist()),
    }


def write_row(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid search SBEED hyperparameters on the 3x3 wall/pit/goal gridworld."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "sbeed_3grid_grid_search.csv",
        help="CSV path for trial results.",
    )
    parser.add_argument("--device", default=None, help="Torch device. Defaults to cuda if available, else cpu.")
    parser.add_argument("--seeds", type=parse_int_list, default=DEFAULT_GRID["seeds"])
    parser.add_argument("--lambda-entropies", type=parse_float_list, default=DEFAULT_GRID["lambda_entropies"])
    parser.add_argument("--etas", type=parse_float_list, default=DEFAULT_GRID["etas"])
    parser.add_argument("--lr-values", type=parse_float_list, default=DEFAULT_GRID["lr_values"])
    parser.add_argument("--lr-policies", type=parse_float_list, default=DEFAULT_GRID["lr_policies"])
    parser.add_argument("--taus", type=parse_float_list, default=DEFAULT_GRID["taus"])
    parser.add_argument("--max-buffer-sizes", type=parse_int_list, default=DEFAULT_GRID["max_buffer_sizes"])
    parser.add_argument("--batch-sizes", type=parse_int_list, default=DEFAULT_GRID["batch_sizes"])
    parser.add_argument("--episodes", type=parse_int_list, default=DEFAULT_GRID["episodes"])
    parser.add_argument("--collect-per-episodes", type=parse_int_list, default=DEFAULT_GRID["collect_per_episodes"])
    parser.add_argument("--updates-per-episodes", type=parse_int_list, default=DEFAULT_GRID["updates_per_episodes"])
    parser.add_argument("--epsilons", type=parse_float_list, default=DEFAULT_GRID["epsilons"])
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--initial-collect-steps", type=int, default=50)
    parser.add_argument("--max-trials", type=int, default=None, help="Optional cap after grid construction/shuffle.")
    parser.add_argument("--resume", action="store_true", help="Skip successful trials already present in the CSV.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle trial order before applying --max-trials.")
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Only print trial count and output path.")
    parser.add_argument("--tqdm-inner", action="store_true", help="Show SBEED episode progress bars inside each trial.")
    parser.add_argument("--verbose-solver", action="store_true", help="Print SBEED solver logs inside each trial.")
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    grid = build_grid(args)

    if args.resume:
        done = completed_keys(args.output)
        grid = [params for params in grid if param_key(params) not in done]

    print(f"Using device: {device}")
    print(f"Trials to run: {len(grid)}")
    print(f"Writing results to: {args.output}")
    if args.dry_run:
        return

    fieldnames = [
        "status",
        "error",
        "seed",
        "lambda_entropy",
        "eta",
        "lr_value",
        "lr_policy",
        "tau",
        "max_buffer_size",
        "batch_size",
        "episodes",
        "collect_per_episode",
        "updates_per_episode",
        "epsilon",
        "elapsed_seconds",
        "buffer_size_final",
        "objective",
        "v_sbeed_minus_v_lambda_star_l2",
        "v_sbeed_minus_v_lambda_star_linf",
        "v_sbeed_minus_v_star_l2",
        "v_sbeed_minus_v_star_linf",
        "soft_bellman_residual_linf",
        "hard_bellman_residual_linf",
        "V_sbeed",
        "V_lambda_star",
        "V_star",
    ]

    for params in tqdm(grid, desc="SBEED grid search"):
        try:
            row = run_trial(params, args, device)
        except Exception as exc:
            row = {
                **params,
                "status": "error",
                "error": repr(exc),
            }
        write_row(args.output, row, fieldnames)


if __name__ == "__main__":
    main()
