import os
import numpy as np
import random
import torch
import pandas as pd
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root to sys.path
def find_root(current_path, marker="fogas_torch"):
    current_path = Path(current_path).resolve()
    for parent in [current_path] + list(current_path.parents):
        if (parent / marker).exists():
            return parent
    return current_path

PROJECT_ROOT = find_root(Path.cwd())
sys.path.append(str(PROJECT_ROOT))

from fogas_torch import PolicySolver, EnvDataCollector
from fogas_torch.algorithm import (
    FOGASSolverVectorized,
    FOGASEvaluator,
    FOGASDataset,
)
from fogas.dataset_collection.dataset_analyzer import DatasetAnalyzer

# --- Setup Parameters ---
seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# --- MDP Initialization (10x10 Gridworld) ---
states  = torch.arange(100, dtype=torch.int64)
actions = torch.arange(4, dtype=torch.int64)
N = len(states)
A = len(actions)
gamma = 0.9
x_0 = 0
goal = 99
pits = {18, 32, 57, 61, 75}
walls = {
    4, 11, 14, 17, 21, 22, 27, 34, 37,
    40, 42, 43, 44, 45, 46, 47, 49,
    54, 62, 64, 66, 72, 76, 82, 84, 86, 87, 94
}

def phi(x, a):
    vec = torch.zeros(N * A, dtype=torch.float64)
    vec[int(x) * A + int(a)] = 1.0
    return vec

step_cost = -0.1
goal_reward = 1.0
pit_reward  = -5.0
omega = torch.full((N * A,), step_cost, dtype=torch.float64)
omega[goal * A : goal * A + A] = goal_reward
for p in pits:
    omega[p * A : p * A + A] = pit_reward

def to_rc(s):  return divmod(s, 10)
def to_s(r, c): return r * 10 + c

def next_state(s, a):
    if s == goal or s in pits: return s
    r, c = to_rc(s)
    if a == 0: r2, c2 = max(0, r - 1), c
    elif a == 1: r2, c2 = min(9, r + 1), c
    elif a == 2: r2, c2 = r, max(0, c - 1)
    elif a == 3: r2, c2 = r, min(9, c + 1)
    else: raise ValueError("Invalid action")
    sp = to_s(r2, c2)
    if sp in walls: return s
    return sp

def psi(xp):
    v = torch.zeros(N * A, dtype=torch.float64)
    for x in states:
        for a in actions:
            if next_state(int(x), int(a)) == xp:
                v[int(x) * A + int(a)] = 1.0
    return v

mdp = PolicySolver(
    states=states, actions=actions, phi=phi, omega=omega,
    gamma=gamma, x0=x_0, psi=psi
)

# --- Grid Search Config ---
dataset_sizes = [8000, 12000, 16000, 20000, 24000]
epsilon_values = [0.3, 0.4, 0.5, 0.6, 0.7]

proportion_configs = {
    "100/0": ([1.0, 0.0], "100% Eps-Greedy"),
    "90/10": ([0.9, 0.1], "90% Eps-Greedy / 10% Random"),
    "80/20": ([0.8, 0.2], "80% Eps-Greedy / 20% Random"),
    "70/30": ([0.7, 0.3], "70% Eps-Greedy / 30% Random"),
}

reset_configs = {
    "Fixed": {'custom': 1.0},
    "Random": {'random': 0.8, 'x0': 0.2} 
}

FOGAS_PARAMS = {
    'alpha': 0.001,
    'eta': 0.0002,
    'rho': 0.05,
    'T': 12000
}

OPT_STATES_PARAMS = {
    'num_trajectories': 1000,
    'max_steps': 100,
    'seed': 42
}

beta_val = 1e-4
temp_dir = "temp_grid_search"
os.makedirs(temp_dir, exist_ok=True)

results = []
total_iters = len(dataset_sizes) * len(epsilon_values) * len(proportion_configs) * len(reset_configs)

print(f"🚀 Starting Extended Grid Search ({total_iters} scenarios)...")
print(f"   Computing 5 metrics per scenario:")
print(f"   1. Coverage Ratio")
print(f"   2. Task Success (J(π̂; ρ_start))")
print(f"   3. On-Data Quality (E_{{s~d_data}}[V*(s) - V^π̂(s)])")
print(f"   4. Optimal States Quality (E_{{s~d_π*}}[V*(s) - V^π̂(s)])")
print(f"   5. Final Reward (legacy metric)\n")

with tqdm(total=total_iters, desc="Grid Searching") as pbar:
    for reset_label, reset_probs in reset_configs.items():
        collector = EnvDataCollector(
            mdp=mdp,
            env_name="10grid_wall",
            restricted_states=list(walls),
            terminal_states=(list(pits) + [goal]),
            reset_probs=reset_probs,
            max_steps=50,
            seed=seed
        )
        
        for prop_label, (props, prop_name) in proportion_configs.items():
            for eps in epsilon_values:
                epsilon_policy = (mdp.pi_star, eps)
                
                for n_steps in dataset_sizes:
                    fname = f"gs_{reset_label}_{prop_label}_eps{eps}_n{n_steps}.csv"
                    save_path = os.path.join(temp_dir, fname)
                    
                    # A. Collect Dataset
                    try:
                        collector.collect_mixed_dataset(
                            policies=[epsilon_policy, "random"],
                            proportions=props,
                            n_steps=n_steps,
                            episode_based=True,
                            save_path=save_path,
                            verbose=False
                        )
                    except Exception as e:
                        print(f"\n⚠️  Dataset collection failed: {e}")
                        pbar.update(1)
                        continue
                    
                    # B. Analyze Feature Coverage
                    try:
                        analyzer = DatasetAnalyzer(save_path)
                        coverage_ratio = analyzer.feature_coverage_ratio(
                            mdp=mdp, beta=beta_val, use_optimal_policy=True, verbose=False
                        )
                    except:
                        coverage_ratio = np.nan
                        
                    # C. Train FOGAS and Compute Metrics
                    try:
                        temp_solver = FOGASSolverVectorized(
                            mdp=mdp, csv_path=save_path, device=device, 
                            beta=beta_val, seed=seed
                        )
                        temp_solver.run(
                            alpha=FOGAS_PARAMS['alpha'], 
                            eta=FOGAS_PARAMS['eta'], 
                            rho=FOGAS_PARAMS['rho'], 
                            T=FOGAS_PARAMS['T'], 
                            tqdm_print=False
                        )
                        
                        temp_eval = FOGASEvaluator(temp_solver)
                        temp_dataset = FOGASDataset(save_path, verbose=False)
                        
                        task_success = temp_eval.task_success()
                        on_data_quality = temp_eval.on_data_quality(temp_dataset)
                        optimal_states_quality = temp_eval.optimal_states_quality(
                            num_trajectories=OPT_STATES_PARAMS['num_trajectories'],
                            max_steps=OPT_STATES_PARAMS['max_steps'],
                            seed=OPT_STATES_PARAMS['seed']
                        )
                        final_reward = temp_eval.get_metric("reward")()
                        
                    except Exception as e:
                        print(f"\n⚠️  FOGAS training/evaluation failed: {e}")
                        task_success = np.nan
                        on_data_quality = np.nan
                        optimal_states_quality = np.nan
                        final_reward = np.nan
                    
                    results.append({
                        "Dataset Size": n_steps,
                        "Epsilon": eps,
                        "Proportions": prop_name,
                        "Init Mode": reset_label,
                        "Coverage Ratio": coverage_ratio,
                        "Log Coverage": np.log10(coverage_ratio) if coverage_ratio > 0 else np.nan,
                        "Task Success": task_success,
                        "On-Data Quality": on_data_quality,
                        "Optimal States Quality": optimal_states_quality,
                        "Final Reward": final_reward,
                    })
                    
                    if os.path.exists(save_path):
                        os.remove(save_path)
                    pbar.update(1)

# Cleanup
if os.path.exists(temp_dir):
    try:
        os.rmdir(temp_dir)
    except:
        pass

df_results = pd.DataFrame(results)
output_filename = 'grid_search_results_sbatch.csv'
df_results.to_csv(output_filename, index=False)

print("\n✅ Grid Search Complete!")
print(f"   Total scenarios: {len(df_results)}")
print(f"   Successful runs: {df_results['Task Success'].notna().sum()}")
print(f"   Failed runs: {df_results['Task Success'].isna().sum()}\n")
print(f"✅ Results saved to '{output_filename}'")
