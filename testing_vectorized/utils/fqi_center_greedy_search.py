# =============================================================================
# GREEDY SEARCH: Minimum RBF Centers for FQI Convergence
# =============================================================================
# Strategy:
#   Start at n_centers = 72 (known working config), go down by 1.
#   For each n_centers, grid-search over (ridge, sigma_multiplier).
#   Convergence = trajectory from x0 reaches state 99 within max_steps.
#
#   If no config converges:
#     → Retry excluding pits from the KMeans pool (state_mode='no_pit')
#     → Retry also excluding goal              (state_mode='no_pit_no_goal')
#     → If still nothing → declare failure and stop outer loop.
# =============================================================================

import itertools
import numpy as np

# ── Hyperparameter grid ──────────────────────────────────────────────────────
RIDGES         = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]
SIGMA_MULTS    = [0.8, 1.0, 1.2, 1.5, 2.0]
K_ITERS        = 5000
TAU            = 0.1
MAX_PATH_STEPS = 24
GOAL_STATE     = 99
START_CENTERS  = 72     # inclusive upper bound
# ─────────────────────────────────────────────────────────────────────────────


def build_phi3_closure(n_centers, sigma_mult, state_mode='standard'):
    """
    Build (centers, phi_state3_fn, phi3_fn, d) for a given configuration.

    state_mode:
        'standard'      → exclude walls only (normal behaviour)
        'no_pit'        → exclude walls + pits
        'no_pit_no_goal'→ exclude walls + pits + goal
    """
    # 1. Decide which states are eligible for clustering
    excluded = set(walls)
    if state_mode in ('no_pit', 'no_pit_no_goal'):
        excluded |= set(pits)
    if state_mode == 'no_pit_no_goal':
        excluded.add(goal)

    valid_coords = []
    for s in range(100):
        if s not in excluded:
            r, c = divmod(s, 10)
            valid_coords.append([r / 9.0, c / 9.0])

    # Need at least n_centers non-excluded states
    if len(valid_coords) < n_centers:
        return None

    kmeans = KMeans(n_clusters=n_centers, n_init=10, random_state=42).fit(valid_coords)
    _centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float64)

    # 2. Compute base sigma and apply multiplier
    _base_sigma = calculate_local_sigma(_centers, k=2)
    _sigma      = _base_sigma * sigma_mult

    # 3. Closure (captures _centers and _sigma by value)
    def _phi_state(x, centers=_centers, sigma=_sigma):
        coords  = get_norm_coords(x)
        dist_sq = torch.sum((coords - centers) ** 2, dim=1)
        return torch.exp(-dist_sq / (2 * sigma ** 2))

    def _phi(x, a, centers=_centers, sigma=_sigma):
        s_feat      = _phi_state(x, centers=centers, sigma=sigma)
        e_a         = torch.zeros(A, dtype=torch.float64)
        e_a[int(a)] = 1.0
        return torch.kron(e_a, s_feat)

    d = int(_phi(int(states[0]), int(actions[0])).shape[0])
    return _centers, _sigma, _phi_state, _phi, d


def reaches_goal(solver_fqi, max_steps=MAX_PATH_STEPS, goal=GOAL_STATE):
    """Return True iff the simulated trajectory visits the goal state."""
    evaluator = FQIEvaluator(solver_fqi)
    traj = evaluator.simulate_trajectory(max_steps=max_steps)
    visited = {s for s, a, r, sn in traj} | {sn for s, a, r, sn in traj}
    return goal in visited


def try_config(n_centers, ridge, sigma_mult, state_mode):
    """
    Build MDP + run FQI for one (n_centers, ridge, sigma_mult, state_mode).
    Returns True if it converges (reaches goal), False otherwise.
    Catches all exceptions.
    """
    try:
        result = build_phi3_closure(n_centers, sigma_mult, state_mode)
        if result is None:
            return False

        _centers, _sigma, _phi_state_fn, _phi_fn, d_local = result

        # Build a fresh MDP (PolicySolver) with the new phi
        _mdp = PolicySolver(
            states=states, actions=actions, phi=_phi_fn,
            reward_fn=reward_fn, gamma=gamma, x0=0, P=P,
        )

        _solver = FQISolver(
            mdp=_mdp,
            csv_path=str(DATASET_PATH2),
            device=device,
            seed=seed,
            ridge=ridge,
        )
        _solver.run(K=K_ITERS, tau=TAU, verbose=False)

        return reaches_goal(_solver)

    except Exception as e:
        # Uncomment for debugging:
        # print(f"      [EXC] n={n_centers} ridge={ridge} sm={sigma_mult} mode={state_mode}: {e}")
        return False


# =============================================================================
# MAIN GREEDY SEARCH LOOP
# =============================================================================
print("=" * 70)
print("  GREEDY SEARCH — minimum RBF centers for FQI convergence")
print("=" * 70)
print(f"  Ridge candidates  : {RIDGES}")
print(f"  Sigma multipliers : {SIGMA_MULTS}")
print(f"  K={K_ITERS},  tau={TAU},  max_path_steps={MAX_PATH_STEPS}")
print(f"  Starting from n_centers = {START_CENTERS}, decreasing by 1")
print("=" * 70)

STATE_MODES   = ['standard', 'no_pit', 'no_pit_no_goal']
MODE_LABELS   = {
    'standard':       'exclude walls only',
    'no_pit':         'exclude walls + pits',
    'no_pit_no_goal': 'exclude walls + pits + goal',
}

search_log  = []   # list of dicts with full result per trial
min_centers = None # will hold the answer

for n_centers in range(START_CENTERS, 0, -1):
    print(f"\n{'─'*70}")
    print(f"  Testing n_centers = {n_centers}")
    print(f"{'─'*70}")

    found_for_n = False

    for mode in STATE_MODES:
        print(f"\n  [Mode] {MODE_LABELS[mode]}")
        found_for_mode = False

        for ridge, sm in itertools.product(RIDGES, SIGMA_MULTS):
            converged = try_config(n_centers, ridge, sm, mode)
            status    = "✅ CONVERGED" if converged else "   failed   "
            print(f"    ridge={ridge:.0e}  sigma_mult={sm:.2f}  → {status}")

            search_log.append({
                "n_centers"   : n_centers,
                "state_mode"  : mode,
                "ridge"       : ridge,
                "sigma_mult"  : sm,
                "converged"   : converged,
            })

            if converged:
                found_for_mode = True
                found_for_n    = True
                break   # ← stop inner loop immediately on first convergence

        if found_for_mode:
            print(f"\n  ✅ n_centers={n_centers} converges in mode '{mode}' "
                  f"(ridge={ridge:.0e}, sigma_mult={sm:.2f})")
            break   # ← stop mode loop, no need to try stricter exclusions

        else:
            print(f"\n  ✗  No convergence in mode '{mode}' — trying stricter exclusion...")

    if not found_for_n:
        print(f"\n  ✗  n_centers={n_centers} FAILED in ALL modes.")
        print(f"  → Minimum working n_centers = {n_centers + 1}")
        min_centers = n_centers + 1
        break

    # If we get here, n_centers worked → record it and try n_centers-1
    min_centers = n_centers
    print(f"\n  ✅ n_centers={n_centers} works — continuing search for fewer centers...")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("  GREEDY SEARCH COMPLETE")
print("=" * 70)

if min_centers is not None:
    print(f"\n  🏆  Minimum n_centers where FQI converges = {min_centers}\n")
else:
    print(f"\n  ⚠️  Search exhausted without finding minimum (all tried converge).\n")

# Best configs (converged=True) sorted by n_centers ascending, then ridge
import pandas as pd
df_search = pd.DataFrame(search_log)
print("Full search log:")
display(df_search)

converged_df = df_search[df_search["converged"]].sort_values(
    ["n_centers", "ridge", "sigma_mult"]
)
print("\nConverging configurations:")
display(converged_df)

# ── Save results to CSV ───────────────────────────────────────────────────────
SEARCH_LOG_PATH   = PROJECT_ROOT / "datasets" / "fqi_center_greedy_search.csv"
CONVERGED_LOG_PATH = PROJECT_ROOT / "datasets" / "fqi_center_greedy_search_converged.csv"
df_search.to_csv(str(SEARCH_LOG_PATH), index=False)
converged_df.to_csv(str(CONVERGED_LOG_PATH), index=False)
print(f"\n✅ Full log saved      → {SEARCH_LOG_PATH}")
print(f"✅ Converged log saved → {CONVERGED_LOG_PATH}")
