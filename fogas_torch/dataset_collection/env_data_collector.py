"""
EnvDataCollector
----------------

Utility for building a Gymnasium environment and generating an offline RL
dataset in a unified way. It can construct an environment from:

    1) a LinearMDP / PolicySolver (via LinearMDPEnv),
    2) a Gym environment name (gym.make),
    3) or an already-instantiated Gym environment.

Given a behavior policy (object with .sample(state) or a string like "random"
or "uniform"), it runs the environment for a fixed number of steps and records
transitions (s, a, r, s') into a pandas DataFrame, optionally saving them to disk.
"""

import os
import random
import numpy as np
import torch
import pandas as pd
import gymnasium as gym

from .linear_mdp_env import LinearMDPEnv


class EnvDataCollector:
    """
    Helper class that:
      - builds an environment from:
          * a LinearMDP / PolicySolver (via LinearMDPEnv)
          * or a Gym env name (gym.make)
          * or a pre-instantiated env
      - collects datasets following a given policy
      - can instantiate policies by name ('random', 'uniform')
    """

    def __init__(self, mdp=None, env_name=None, env=None,
                 max_steps=1000, terminal_states=None, restricted_states=None, 
                 initial_states=None, reset_probs=None, seed=42):
        """
        seed: random seed for reproducibility in environment and policy sampling
        """
        self.seed = seed
        
        # Set random seed for reproducibility
        if seed is not None:
            random.seed(seed)

        if env is not None:
            self.env = env

        elif mdp is not None:
            self.env = LinearMDPEnv(
                mdp,
                max_steps=max_steps,
                terminal_states=terminal_states,
                restricted_states=restricted_states,
                initial_states=initial_states,
                reset_probs=reset_probs,
            )

        elif env_name is not None:
            self.env = gym.make(env_name)

        else:
            raise ValueError("You must provide either mdp, env_name, or env.")

    class RandomPolicy:
        """Chooses actions uniformly at random from the env's action space."""
        def __init__(self, action_space):
            self.action_space = action_space

        def sample(self, state):
            return self.action_space.sample()

    class UniformPolicy:
        """Uniform distribution over actions via random.choice."""
        def __init__(self, action_space):
            self.action_space = action_space

        def sample(self, state):
            return random.randrange(self.action_space.n)

    class MatrixPolicy:
        """Policy wrapper for torch policy matrices (e.g., mdp.pi_star)."""
        def __init__(self, policy_matrix):
            """
            Parameters
            ----------
            policy_matrix : tensor or ndarray of shape (N, A)
                Policy matrix where policy_matrix[s, a] = probability of action a in state s
            """
            if isinstance(policy_matrix, torch.Tensor):
                self.policy_matrix = policy_matrix.cpu().numpy()
            else:
                self.policy_matrix = policy_matrix

        def sample(self, state):
            """Sample action from policy distribution at given state."""
            action_probs = self.policy_matrix[state]
            return random.choices(range(len(action_probs)), weights=action_probs)[0]

    class EpsilonGreedyPolicy:
        """
        Wraps a base policy with epsilon-greedy exploration.
        1 - epsilon probability: follow base_policy.sample(state)
        epsilon probability: take a random action from action_space
        """
        def __init__(self, base_policy, epsilon, action_space):
            self.base_policy = base_policy
            self.epsilon = epsilon
            self.action_space = action_space

        def sample(self, state):
            if random.random() < self.epsilon:
                return self.action_space.sample()
            else:
                return self.base_policy.sample(state)

    # --------------------------------------------
    # Policy factory
    # --------------------------------------------
    def _make_policy(self, policy):
        """
        Accepts:
            - a policy object with .sample(state)
            - a torch tensor or numpy array (policy matrix of shape (N, A))
            - a string: "random", "uniform"
            - a tuple: (base_policy, epsilon) -> creates an EpsilonGreedyPolicy

        Returns:
            A policy object with method sample(state).
        """

        # Epsilon-greedy check: tuple (base_policy, epsilon)
        if isinstance(policy, tuple) and len(policy) == 2:
            base_pol_input, epsilon = policy
            if not isinstance(epsilon, (float, int)):
                raise ValueError(f"Epsilon must be a number, got {type(epsilon)}")
            
            base_policy = self._make_policy(base_pol_input)
            return self.EpsilonGreedyPolicy(base_policy, float(epsilon), self.env.action_space)

        # Already a policy object → return as is
        if hasattr(policy, "sample"):
            return policy

        # Tensor or numpy array (policy matrix) → wrap it
        if isinstance(policy, (torch.Tensor, np.ndarray)):
            return self.MatrixPolicy(policy)

        # String name → build the corresponding policy
        if isinstance(policy, str):
            policy = policy.lower()

            if policy == "random":
                return self.RandomPolicy(self.env.action_space)

            if policy == "uniform":
                return self.UniformPolicy(self.env.action_space)

            raise ValueError(
                f"Unknown policy name '{policy}'. Supported: 'random', 'uniform'."
            )

        raise ValueError(
            "Policy must be a string, tensor/array (policy matrix), or an object with a .sample() method."
        )

    # --------------------------------------------
    # Dataset collection
    # --------------------------------------------
    def collect_dataset(self, policy="random", n_steps=1000, save_path=None, verbose=True):
        """
        Collect transitions following behavior policy π_b and return an ordered DataFrame.

        Parameters
        ----------
        policy : str or object
            Behavior policy ("random", "uniform", or object with .sample(state)).
        n_steps : int
            Number of environment steps to collect.
        save_path : str or None
            If provided, CSV path where (s, a, r, s') transitions are saved.
        verbose : bool
            If True, prints basic info when saving.
        """
        # Reset seed for reproducible data collection
        if self.seed is not None:
            random.seed(self.seed)
        
        env = self.env
        policy = self._make_policy(policy)

        data = {
            "episode": [],
            "step": [],
            "state": [],
            "action": [],
            "reward": [],
            "next_state": [],
        }

        episode, step = 0, 0
        obs, _ = env.reset()

        for _ in range(n_steps):
            action = policy.sample(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            data["episode"].append(episode)
            data["step"].append(step)
            data["state"].append(obs)
            data["action"].append(action)
            data["reward"].append(reward)
            data["next_state"].append(next_obs)

            step += 1

            if terminated or truncated:
                episode += 1
                step = 0
                obs, _ = env.reset()
            else:
                obs = next_obs

        df = pd.DataFrame(data)

        # Optional saving
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df[["state", "action", "reward", "next_state"]].to_csv(save_path, index=False)

            if verbose:
                print(f"✅ Dataset saved to: {save_path}")
                print(f"   Total transitions: {len(df)}")

        return df

    def collect_mixed_dataset(self, policies, proportions=None, n_steps=1000, 
                             save_path=None, verbose=True, episode_based=True):
        """
        Collect transitions using multiple policies with specified proportions.
        
        Parameters
        ----------
        policies : list
            List of policies (each can be a string like "random" or object with .sample(state)).
            Example: ["random", policy_obj1, policy_obj2]
        proportions : list of float, optional
            Proportion of data to collect with each policy. Must sum to 1.0.
            If None, policies are used equally (e.g., [0.5, 0.5] for 2 policies).
            Example: [0.3, 0.5, 0.2] means 30% policy1, 50% policy2, 20% policy3.
        n_steps : int
            Total number of environment steps to collect.
        save_path : str or None
            If provided, CSV path where (s, a, r, s', policy_id) transitions are saved.
        verbose : bool
            If True, prints collection statistics.
        episode_based : bool
            If True, switches policy at episode boundaries (entire episodes use one policy).
            If False, switches policy according to proportions at each step.
            
        Returns
        -------
        df : pd.DataFrame
            DataFrame with columns: episode, step, state, action, reward, next_state, policy_id
        """
        # Reset seed for reproducible data collection
        if self.seed is not None:
            random.seed(self.seed)
        
        # Validate inputs
        if not isinstance(policies, list):
            raise ValueError("policies must be a list")
        
        num_policies = len(policies)
        if num_policies < 2:
            raise ValueError("Must provide at least 2 policies. Use collect_dataset() for single policy.")
        
        # Convert all policies to policy objects
        policies = [self._make_policy(p) for p in policies]
        
        # Set proportions
        if proportions is None:
            proportions = [1.0 / num_policies] * num_policies
        else:
            if len(proportions) != num_policies:
                raise ValueError(f"proportions must have {num_policies} elements to match policies")
            if abs(sum(proportions) - 1.0) > 1e-9:
                raise ValueError(f"proportions must sum to 1.0, got {sum(proportions)}")
        
        env = self.env
        
        data = {
            "episode": [],
            "step": [],
            "state": [],
            "action": [],
            "reward": [],
            "next_state": [],
            "policy_id": [],  # Track which policy was used
        }
        
        # Track statistics
        policy_step_counts = [0] * num_policies
        policy_episode_counts = [0] * num_policies
        
        episode, step = 0, 0
        obs, _ = env.reset()
        
        # Choose initial policy
        if episode_based:
            current_policy_idx = random.choices(range(num_policies), weights=proportions)[0]
            current_policy = policies[current_policy_idx]
            policy_episode_counts[current_policy_idx] += 1
        
        for global_step in range(n_steps):
            # Select policy for this step
            if episode_based:
                # Use the same policy for entire episode
                policy_idx = current_policy_idx
                policy = current_policy
            else:
                # Sample policy at each step according to proportions
                policy_idx = random.choices(range(num_policies), weights=proportions)[0]
                policy = policies[policy_idx]
            
            # Execute action
            action = policy.sample(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            
            # Record transition
            data["episode"].append(episode)
            data["step"].append(step)
            data["state"].append(obs)
            data["action"].append(action)
            data["reward"].append(reward)
            data["next_state"].append(next_obs)
            data["policy_id"].append(policy_idx)
            
            # Update statistics
            policy_step_counts[policy_idx] += 1
            step += 1
            
            # Handle episode termination
            if terminated or truncated:
                episode += 1
                step = 0
                obs, _ = env.reset()
                
                # Choose new policy for next episode (if episode-based)
                if episode_based:
                    current_policy_idx = random.choices(range(num_policies), weights=proportions)[0]
                    current_policy = policies[current_policy_idx]
                    policy_episode_counts[current_policy_idx] += 1
            else:
                obs = next_obs
        
        df = pd.DataFrame(data)
        
        # Print statistics
        if verbose:
            print(f"\n{'='*60}")
            print(f"  MIXED DATASET COLLECTION SUMMARY (TORCH)")
            print(f"{'='*60}")
            print(f"Total transitions: {len(df)}")
            print(f"Total episodes: {episode}")
            print(f"Mode: {'Episode-based' if episode_based else 'Step-based'}")
            print(f"\nPolicy Distribution:")
            for i, (count, prop) in enumerate(zip(policy_step_counts, proportions)):
                actual_prop = count / len(df) if len(df) > 0 else 0
                print(f"  Policy {i}: {count:5d} steps ({actual_prop:5.1%}) | "
                      f"Target: {prop:5.1%} | ", end="")
                if episode_based:
                    print(f"Episodes: {policy_episode_counts[i]}")
                else:
                    print()
            print(f"{'='*60}\n")
        
        # Optional saving
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df[["state", "action", "reward", "next_state", "policy_id"]].to_csv(
                save_path, index=False
            )
            
            if verbose:
                print(f"✅ Mixed dataset saved to: {save_path}")
        
        return df
