"""
FOGASDataset
------------

Utility for loading and validating offline RL datasets for LinearMDPs.
Expected CSV columns:
    state, action, reward, next_state
"""

import numpy as np
import pandas as pd


class FOGASDataset:
    def __init__(self, csv_path, verbose=False):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)

        required = ["state", "action", "reward", "next_state"]
        for col in required:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")

        self.X = self.df["state"].to_numpy(np.int64)
        self.A = self.df["action"].to_numpy(np.int64)
        self.R = self.df["reward"].to_numpy(np.float32)
        self.X_next = self.df["next_state"].to_numpy(np.int64)

        self.n = len(self.df)

        if verbose:
            print(f"Loaded dataset {csv_path} with {self.n} transitions.")
            self.print_stats()

    def summary(self):
        return {
            "n": self.n,
            "unique_states": np.unique(self.X),
            "unique_actions": np.unique(self.A),
            "reward_mean": float(self.R.mean()),
        }

    def print_stats(self):
        """
        Pretty-print basic statistics of the offline RL dataset.
        """
        print("\n========== FOGAS DATASET SUMMARY ==========\n")

        print(f"Total transitions (n): {self.n}")
        print(f"Unique states:         {len(np.unique(self.X))}")
        print(f"Unique actions:        {len(np.unique(self.A))}")

        print("\nState distribution:")
        unique_states, counts_states = np.unique(self.X, return_counts=True)
        for s, c in zip(unique_states, counts_states):
            print(f"  State {s}: {c} samples")

        print("\nAction distribution:")
        unique_actions, counts_actions = np.unique(self.A, return_counts=True)
        for a, c in zip(unique_actions, counts_actions):
            print(f"  Action {a}: {c} samples")

        print(f"\nReward statistics:")
        print(f"  Mean reward:    {self.R.mean():.4f}")
        print(f"  Std deviation:  {self.R.std():.4f}")
        print(f"  Min reward:     {self.R.min():.4f}")
        print(f"  Max reward:     {self.R.max():.4f}")

        print("\nNext-state distribution:")
        unique_next, counts_next = np.unique(self.X_next, return_counts=True)
        for s, c in zip(unique_next, counts_next):
            print(f"  Next state {s}: {c} transitions")

        print("\n===========================================\n")
