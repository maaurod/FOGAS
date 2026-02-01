
import itertools
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import shutil

# Adjust paths if necessary
import sys
# sys.path.append(...) 

from fogas_torch.algorithm import FQISolver, FQIEvaluator
from fogas_torch.dataset_collection import EnvDataCollector, LinearMDPEnv

def run_coverage_grid_search(mdp, base_save_dir="datasets/grid_search", device="cpu", seed=42, goal_state=99):
    """
    Grid search over:
      - Epsilon (quality of behavioral policy derived from pi*)
      - Proportion (mix of behavioral vs random policy)
      - N (dataset size)
    """
    
    # ---------------------------
    # 1. Grid Definitions
    # ---------------------------
    epsilons = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
    proportions_list = [(0.8, 0.2), (0.5, 0.5), (0.2, 0.8), (0.1, 0.9)] 
    # proportions: (prob_behavioral, prob_random)
    n_steps_list = [500, 1000, 2000, 4000, 8000, 16000]
    
    # Ensure directory exists
    base_path = Path(base_save_dir)
    if base_path.exists():
        shutil.rmtree(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    
    results = []

    # Initialize Collector
    # We need an env wrapper for LinearMDP
    env = LinearMDPEnv(mdp) 
    collector = EnvDataCollector(env, seed=seed)

    total_configs = len(epsilons) * len(proportions_list) * len(n_steps_list)
    idx = 0
    
    print(f"Starting Grid Search: {total_configs} configurations...")
    
    for n in n_steps_list:
        for eps in epsilons:
            for (p_beh, p_rand) in proportions_list:
                idx += 1
                
                behavior_policy_def = (mdp.pi_star, eps)
                
                dataset_name = f"data_n{n}_eps{eps}_p{p_beh}.csv"
                dataset_path = base_path / dataset_name
                
                collector.collect_mixed_dataset(
                    policies=[behavior_policy_def, "random"],
                    proportions=[p_beh, p_rand],
                    n_steps=n,
                    episode_based=True,
                    save_path=str(dataset_path),
                    verbose=False # Silence generation
                )
                
                solver = FQISolver(
                    mdp=mdp,
                    csv_path=str(dataset_path),
                    device=device,
                    seed=seed,
                    ridge=1e-2 # As requested
                )
                
                solver.run(K=1000, tau=0.1, verbose=False)
                
                evaluator = FQIEvaluator(solver)
                
                # Metric 1: Expected Reward (Theoretical)
                reward_metric = evaluator.final_reward()
                
                # Let's perform 20 simulations
                trajs = [evaluator.simulate_trajectory(max_steps=50) for _ in range(20)]
                
                # Check success: reached goal state
                # Each step is (s, a, r, s_next)
                success_count = 0
                for tr in trajs:
                    # Check if any next_state in the trajectory is the goal_state
                    reached = any(step[3] == goal_state for step in tr)
                    if reached:
                        success_count += 1
                
                success_rate = success_count / 20.0
                
                optimal_return = mdp.policy_return(mdp.pi_star)
                
                # Store results
                print(f"[{idx}/{total_configs}] N={n}, eps={eps}, prop={p_beh:.1f} | R={reward_metric:.4f}, Succ={success_rate:.2f}")
                
                results.append({
                    "n": n,
                    "epsilon": eps,
                    "mix_ratio": p_beh, # keeping just first number of tuple
                    "reward": reward_metric,
                    "success_rate": success_rate,
                    "optimal_return_ref": optimal_return
                })
                
                # Clean up dataset to save space?
                if os.path.exists(dataset_path):
                    os.remove(dataset_path)

    # ---------------------------
    # 5. Visualization / Summary
    # ---------------------------
    df = pd.DataFrame(results)
    
    # Identify Best Config
    best_row = df.loc[df['reward'].idxmax()]
    print("\n========== BEST CONFIGURATION ==========")
    print(best_row)
    print("========================================\n")
    
    # Pivot tables for Heatmaps
    # Since we have 3 dims (N, Eps, Mix), we can fix one and plot others, or average.
    
    # Plot 1: Reward vs (N, Epsilon) - Averaged over Mix
    plt.figure(figsize=(10, 6))
    pivot_n_eps = df.pivot_table(index="epsilon", columns="n", values="reward", aggfunc="mean")
    sns.heatmap(pivot_n_eps, annot=True, fmt=".2f", cmap="viridis")
    plt.title("Mean Reward: Epsilon vs Dataset Size (Averaged over Mix Ratios)")
    plt.show()
    
    # Plot 2: Reward vs (N, Mix Ratio) - Averaged over Epsilon
    plt.figure(figsize=(10, 6))
    pivot_n_mix = df.pivot_table(index="mix_ratio", columns="n", values="reward", aggfunc="mean")
    sns.heatmap(pivot_n_mix, annot=True, fmt=".2f", cmap="magma")
    plt.title("Mean Reward: Mix Ratio (Optimal%) vs Dataset Size")
    plt.ylabel("Proportion of Behavioral Policy")
    plt.show()
    
    return df

if __name__ == "__main__":
    pass
