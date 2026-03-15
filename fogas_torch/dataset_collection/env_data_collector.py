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
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            if torch.backends.mps.is_available():
                torch.mps.manual_seed(seed)

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

        # Seed environment and spaces
        if seed is not None:
            self.env.action_space.seed(seed)
            self.env.observation_space.seed(seed)
            self.env.reset(seed=seed)

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

    @staticmethod
    def _fine_to_coarse_state(x_fine, fine_size=20, coarse_size=10, factor=2):
        """
        Map a fine-grid state index to its coarse-grid cell.
        """
        r_f, c_f = divmod(int(x_fine), fine_size)
        r_c, c_c = r_f // factor, c_f // factor
        return int(r_c * coarse_size + c_c)

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
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
            if torch.backends.mps.is_available():
                torch.mps.manual_seed(self.seed)
        
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

    def collect_dataset_terminal_aware(self, policy="random", n_steps=1000, save_path=None, verbose=True, extra_steps=5):
        """
        Collect transitions following behavior policy π_b and return an ordered DataFrame.
        Similar to collect_dataset, but when an absorbing state is reached, the trajectory
        stays for 'extra_steps' in that state before terminating and restarting.

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
        extra_steps : int
            Number of additional transitions to record in the absorbing state. Default is 5.
        """
        # Reset seed for reproducible data collection
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
            if torch.backends.mps.is_available():
                torch.mps.manual_seed(self.seed)
        
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
        
        # Track if we are currently staying in an absorbing state
        extra_steps_remaining = -1

        def _reward_from_env_state_action(state, action, fallback):
            return self._get_reward_from_env(state, action, fallback)

        for _ in range(n_steps):
            action = policy.sample(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            # During terminal self-loops, LinearMDPEnv.step returns 0.0 by design.
            # Replace it with the model-consistent reward for (terminal_state, action).
            if extra_steps_remaining != -1:
                reward = _reward_from_env_state_action(obs, action, reward)

            data["episode"].append(episode)
            data["step"].append(step)
            data["state"].append(obs)
            data["action"].append(action)
            data["reward"].append(reward)
            data["next_state"].append(next_obs)

            step += 1

            if truncated:
                # Truncation always causes an immediate reset
                episode += 1
                step = 0
                obs, _ = env.reset()
                extra_steps_remaining = -1
            elif terminated:
                # If we just reached a terminal (absorbing) state
                if extra_steps_remaining == -1:
                    # FIRST HIT: start extra-stay counter
                    extra_steps_remaining = extra_steps
                
                if extra_steps_remaining > 0:
                    # Stay in the absorbing state for extra steps
                    extra_steps_remaining -= 1
                    obs = next_obs # In LinearMDPEnv, next_obs == obs for terminal states
                else:
                    # After the extra steps, finally reset
                    episode += 1
                    step = 0
                    obs, _ = env.reset()
                    extra_steps_remaining = -1
            else:
                obs = next_obs

        df = pd.DataFrame(data)

        # Optional saving
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df[["state", "action", "reward", "next_state"]].to_csv(save_path, index=False)

            if verbose:
                print(f"✅ Terminal-aware dataset saved to: {save_path}")
                print(f"   Total transitions: {len(df)}")

        return df

    def collect_macro_dataset_repeated_actions(
        self,
        policy="random",
        n_macro_steps=1000,
        gamma=0.99,
        fine_size=20,
        coarse_size=10,
        factor=2,
        save_path=None,
        verbose=True,
    ):
        """
        Collect a macro dataset by repeating the same fine action twice.

        Each macro transition stores:
            (coarse(s_t), a_t, r_t + gamma * r_{t+1}, coarse(s_{t+2}))

        The second fine step is always executed. This assumes terminal states are
        absorbing, so stepping again after reaching one is valid.
        """
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
            if torch.backends.mps.is_available():
                torch.mps.manual_seed(self.seed)

        env = self.env
        policy = self._make_policy(policy)

        data = {
            "episode": [],
            "macro_step": [],
            "fine_state": [],
            "state": [],
            "action": [],
            "reward": [],
            "next_fine_state": [],
            "next_state": [],
            "macro_complete": [],
        }

        episode, macro_step = 0, 0
        obs, _ = env.reset()
        collected = 0

        while collected < n_macro_steps:
            fine_state = int(obs)
            coarse_state = self._fine_to_coarse_state(
                fine_state,
                fine_size=fine_size,
                coarse_size=coarse_size,
                factor=factor,
            )

            action = int(policy.sample(obs))

            obs_1, reward_1, terminated_1, truncated_1, _ = env.step(action)
            obs_2, reward_2, terminated_2, truncated_2, _ = env.step(action)

            next_fine_state = int(obs_2)
            next_coarse_state = self._fine_to_coarse_state(
                next_fine_state,
                fine_size=fine_size,
                coarse_size=coarse_size,
                factor=factor,
            )
            macro_reward = float(reward_1) + float(gamma) * float(reward_2)

            data["episode"].append(episode)
            data["macro_step"].append(macro_step)
            data["fine_state"].append(fine_state)
            data["state"].append(coarse_state)
            data["action"].append(action)
            data["reward"].append(macro_reward)
            data["next_fine_state"].append(next_fine_state)
            data["next_state"].append(next_coarse_state)
            data["macro_complete"].append(True)

            collected += 1
            macro_step += 1

            if terminated_1 or truncated_1 or terminated_2 or truncated_2:
                episode += 1
                macro_step = 0
                obs, _ = env.reset()
            else:
                obs = obs_2

        df = pd.DataFrame(data)

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df[["state", "action", "reward", "next_state"]].to_csv(save_path, index=False)

            if verbose:
                print(f"✅ Macro dataset saved to: {save_path}")
                print(f"   Total macro transitions: {len(df)}")

        elif verbose:
            print(f"Collected {len(df)} macro transitions.")

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
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
            if torch.backends.mps.is_available():
                torch.mps.manual_seed(self.seed)
        
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

    def collect_mixed_dataset_terminal_aware(self, policies, proportions=None, n_steps=1000, 
                                             save_path=None, verbose=True, episode_based=True, extra_steps=5,
                                             manual_states=None, manual_samples_per_pair=10):
        """
        Collect transitions using multiple policies with specified proportions,
        while handling absorbing states by staying for 'extra_steps'.

        Parameters
        ----------
        policies : list
            List of policies.
        proportions : list of float, optional
            Proportions for each policy.
        n_steps : int
            Total number of steps.
        save_path : str or None
            CSV save path.
        verbose : bool
            Print status info.
        episode_based : bool
            Switch policy at episode boundaries.
        extra_steps : int
            Number of additional transitions to record in the absorbing state.
        manual_states : list of int, optional
            List of states to manually sample every action for.
        manual_samples_per_pair : int
            Number of manual samples per (s, a) pair.

        Returns
        -------
        df : pd.DataFrame
        """
        # Reset seed for reproducible data collection
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
            if torch.backends.mps.is_available():
                torch.mps.manual_seed(self.seed)
        
        # Validate inputs
        if not isinstance(policies, list):
            raise ValueError("policies must be a list")
        
        num_policies = len(policies)
        if num_policies < 2:
            raise ValueError("Must provide at least 2 policies.")
        
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
            "policy_id": [],
        }
        
        # Track statistics
        policy_step_counts = [0] * num_policies
        policy_episode_counts = [0] * num_policies
        
        episode, step = 0, 0
        obs, _ = env.reset()
        extra_steps_remaining = -1
        
        if episode_based:
            current_policy_idx = random.choices(range(num_policies), weights=proportions)[0]
            current_policy = policies[current_policy_idx]
            policy_episode_counts[current_policy_idx] += 1
        
        for global_step in range(n_steps):
            if episode_based:
                policy_idx = current_policy_idx
                policy = current_policy
            else:
                policy_idx = random.choices(range(num_policies), weights=proportions)[0]
                policy = policies[policy_idx]
            
            action = policy.sample(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            
            # Use model reward for extra terminal steps
            if extra_steps_remaining != -1:
                reward = self._get_reward_from_env(obs, action, reward)
            
            # Record transition
            data["episode"].append(episode)
            data["step"].append(step)
            data["state"].append(obs)
            data["action"].append(action)
            data["reward"].append(reward)
            data["next_state"].append(next_obs)
            data["policy_id"].append(policy_idx)
            
            policy_step_counts[policy_idx] += 1
            step += 1
            
            if truncated:
                episode += 1
                step = 0
                obs, _ = env.reset()
                extra_steps_remaining = -1
                if episode_based:
                    current_policy_idx = random.choices(range(num_policies), weights=proportions)[0]
                    current_policy = policies[current_policy_idx]
                    policy_episode_counts[current_policy_idx] += 1
            elif terminated:
                if extra_steps_remaining == -1:
                    extra_steps_remaining = extra_steps
                
                if extra_steps_remaining > 0:
                    extra_steps_remaining -= 1
                    obs = next_obs
                else:
                    episode += 1
                    step = 0
                    obs, _ = env.reset()
                    extra_steps_remaining = -1
                    if episode_based:
                        current_policy_idx = random.choices(range(num_policies), weights=proportions)[0]
                        current_policy = policies[current_policy_idx]
                        policy_episode_counts[current_policy_idx] += 1
            else:
                obs = next_obs
        
        df = pd.DataFrame(data)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"  MIXED TERMINAL-AWARE DATASET COLLECTION SUMMARY (TORCH)")
            print(f"{'='*60}")
            print(f"Total transitions: {len(df)}")
            print(f"Total episodes: {episode}")
            print(f"Extra steps: {extra_steps}")
            print(f"\nPolicy Distribution:")
            for i, (count, prop) in enumerate(zip(policy_step_counts, proportions)):
                actual_prop = count / len(df) if len(df) > 0 else 0
                print(f"  Policy {i}: {count:5d} steps ({actual_prop:5.1%}) | Target: {prop:5.1%}", end="")
                if episode_based:
                    print(f" | Episodes: {policy_episode_counts[i]}")
                else:
                    print()
            print(f"{'='*60}\n")

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df[["state", "action", "reward", "next_state", "policy_id"]].to_csv(save_path, index=False)
            if verbose:
                print(f"✅ Mixed terminal-aware dataset saved to: {save_path}")
        
        # Integrated manual augmentation if requested
        if manual_states:
            if verbose:
                print(f"\nPerforming integrated manual augmentation for {len(manual_states)} states...")
            df_manual = self.add_manual_samples(
                states=manual_states, 
                samples_per_pair=manual_samples_per_pair, 
                save_path=save_path, 
                verbose=verbose
            )
            # Combine the DataFrames for the return value
            # We only keep the core overlapping columns
            cols = ["state", "action", "reward", "next_state"]
            df = pd.concat([df[cols], df_manual[cols]], ignore_index=True)

        return df

    def collect_mixed_dataset_terminal_aware_with_restricted(self, policies, proportions=None, n_steps=1000, 
                                             save_path=None, verbose=True, episode_based=True, extra_steps=5, restricted_max_steps=1):
        """
        Collect transitions using multiple policies with specified proportions,
        while handling absorbing states by staying for 'extra_steps', AND specifically
        handling transitions that start in restricted states by forcing random action 
        choice and a defined truncation length.

        Parameters
        ----------
        policies : list
            List of policies.
        proportions : list of float, optional
            Proportions for each policy.
        n_steps : int
            Total number of steps.
        save_path : str or None
            CSV save path.
        verbose : bool
            Print status info.
        episode_based : bool
            Switch policy at episode boundaries.
        extra_steps : int
            Number of additional transitions to record in the absorbing state.
        restricted_max_steps : int
            Number of max steps permitted when spawning randomly in a restricted state. Actions are completely random.

        Returns
        -------
        df : pd.DataFrame
        """
        # Reset seed for reproducible data collection
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
            if torch.backends.mps.is_available():
                torch.mps.manual_seed(self.seed)
        
        # Validate inputs
        if not isinstance(policies, list):
            raise ValueError("policies must be a list")
        
        num_policies = len(policies)
        if num_policies < 2:
            raise ValueError("Must provide at least 2 policies.")
        
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
            "policy_id": [],
        }
        
        # Track statistics
        policy_step_counts = [0] * num_policies
        policy_episode_counts = [0] * num_policies
        
        episode, step = 0, 0
        obs, info = env.reset()
        is_restricted_start = info.get("restricted_start", False)
        
        extra_steps_remaining = -1
        restricted_steps_count = 0
        
        if episode_based:
            current_policy_idx = random.choices(range(num_policies), weights=proportions)[0]
            current_policy = policies[current_policy_idx]
            policy_episode_counts[current_policy_idx] += 1
        
        for global_step in range(n_steps):
            if episode_based:
                policy_idx = current_policy_idx
                policy = current_policy
            else:
                policy_idx = random.choices(range(num_policies), weights=proportions)[0]
                policy = policies[policy_idx]
            
            if is_restricted_start:
                # Force Random Action to touch every action when randomly started on a wall.
                action = env.action_space.sample()
                restricted_steps_count += 1
            else:
                action = policy.sample(obs)
                
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            if is_restricted_start and restricted_steps_count >= restricted_max_steps:
               truncated = True
               
            # Use model reward for extra terminal steps
            if extra_steps_remaining != -1:
                reward = self._get_reward_from_env(obs, action, reward)
            
            # Record transition
            data["episode"].append(episode)
            data["step"].append(step)
            data["state"].append(obs)
            data["action"].append(action)
            data["reward"].append(reward)
            data["next_state"].append(next_obs)
            data["policy_id"].append(-1 if is_restricted_start else policy_idx)
            
            if not is_restricted_start:
               policy_step_counts[policy_idx] += 1
            step += 1
            
            if truncated:
                episode += 1
                step = 0
                obs, info = env.reset()
                is_restricted_start = info.get("restricted_start", False)
                extra_steps_remaining = -1
                restricted_steps_count = 0
                if episode_based:
                    current_policy_idx = random.choices(range(num_policies), weights=proportions)[0]
                    current_policy = policies[current_policy_idx]
                    policy_episode_counts[current_policy_idx] += 1
            elif terminated:
                if extra_steps_remaining == -1:
                    extra_steps_remaining = extra_steps
                
                if extra_steps_remaining > 0:
                    extra_steps_remaining -= 1
                    obs = next_obs
                else:
                    episode += 1
                    step = 0
                    obs, info = env.reset()
                    is_restricted_start = info.get("restricted_start", False)
                    extra_steps_remaining = -1
                    restricted_steps_count = 0
                    if episode_based:
                        current_policy_idx = random.choices(range(num_policies), weights=proportions)[0]
                        current_policy = policies[current_policy_idx]
                        policy_episode_counts[current_policy_idx] += 1
            else:
                obs = next_obs
        
        df = pd.DataFrame(data)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"  MIXED TERMINAL-AWARE (RESTRICTED) DATASET COLLECTION")
            print(f"{'='*60}")
            print(f"Total transitions: {len(df)}")
            print(f"Total episodes: {episode}")
            print(f"Extra steps: {extra_steps}")
            print(f"Max restricted start steps: {restricted_max_steps}")
            print(f"\nPolicy Distribution (excluding forced restricted random steps):")
            for i, (count, prop) in enumerate(zip(policy_step_counts, proportions)):
                actual_prop = count / len(df) if len(df) > 0 else 0
                print(f"  Policy {i}: {count:5d} steps ({actual_prop:5.1%}) | Target: {prop:5.1%}", end="")
                if episode_based:
                    print(f" | Episodes: {policy_episode_counts[i]}")
                else:
                    print()
            print(f"{'='*60}\n")

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df[["state", "action", "reward", "next_state", "policy_id"]].to_csv(save_path, index=False)
            if verbose:
                print(f"✅ Mixed restricted dataset saved to: {save_path}")
        
        return df

    def collect_uniform_dataset(self, samples_per_pair=1, save_path=None, verbose=True):
        """
        Collect exactly 'samples_per_pair' for every possible (state, action) pair
        in the environment. This performs a complete sweep of the MDP state-action
        space, ignoring trajectory-related constraints like 'max_steps', 'restricted_states',
        or 'reset_probs'.

        Parameters
        ----------
        samples_per_pair : int
            Number of transitions to record for each (s, a) pair.
        save_path : str or None
            Path to save the CSV.
        verbose : bool
            Whether to print progress.

        Returns
        -------
        df : pd.DataFrame
        """
        # Reset seed for reproducibility
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)

        env = self.env
        
        # Determine valid states and actions for a uniform sweep
        if hasattr(env, "N") and hasattr(env, "A"):
            # LinearMDPEnv path: Include ALL states
            states_to_sample = range(env.N)
            actions_to_sample = range(env.A)
        elif hasattr(env, "observation_space") and isinstance(env.observation_space, gym.spaces.Discrete) \
             and hasattr(env, "action_space") and isinstance(env.action_space, gym.spaces.Discrete):
            # General Discrete Env path
            states_to_sample = range(env.observation_space.n)
            actions_to_sample = range(env.action_space.n)
        else:
            raise ValueError("collect_uniform_dataset requires a discrete environment (like LinearMDPEnv).")


        data = {
            "episode": [],
            "step": [],
            "state": [],
            "action": [],
            "reward": [],
            "next_state": [],
        }

        if verbose:
            total = len(states_to_sample) * len(actions_to_sample) * samples_per_pair
            print(f"Collecting UNIFORM dataset: {len(states_to_sample)} states, {len(actions_to_sample)} actions, {samples_per_pair} samples/pair.")
            print(f"Total transitions: {total}")

        idx = 0
        for s in states_to_sample:
            for a in actions_to_sample:
                for _ in range(samples_per_pair):
                    # Manual state injection (requires env to support .state assignment)
                    try:
                        env.state = s
                    except AttributeError:
                        raise AttributeError("Environment does not support manual state assignment via .state.")
                    
                    if hasattr(env, "step_count"):
                        env.step_count = 0
                    
                    next_obs, reward, terminated, truncated, _ = env.step(a)
                    
                    # Consistent rewards for terminal self-loops
                    if terminated and s in getattr(env, "terminal_states", []):
                        reward = self._get_reward_from_env(s, a, reward)

                    data["episode"].append(idx)
                    data["step"].append(0)
                    data["state"].append(s)
                    data["action"].append(a)
                    data["reward"].append(reward)
                    data["next_state"].append(next_obs)
                    idx += 1

        df = pd.DataFrame(data)

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df[["state", "action", "reward", "next_state"]].to_csv(save_path, index=False)
            if verbose:
                print(f"✅ Uniform dataset saved to: {save_path}")

        return df

    def add_manual_samples(self, states, samples_per_pair=10, save_path=None, verbose=True):
        """
        Manually add 'samples_per_pair' for every possible action in each provided state.
        This is useful for ensuring specific states (e.g. near goals or difficult areas) 
        are sufficiently represented in the dataset.
        
        Parameters
        ----------
        states : list of int
            List of state indices to sample.
        samples_per_pair : int
            Number of transitions to record for each (state, action) pair.
        save_path : str or None
            If provided, the CSV path to update or save to.
        verbose : bool
            Whether to print progress.
            
        Returns
        -------
        df : pd.DataFrame
            The newly collected manual transitions.
        """
        # Reset seed for reproducibility
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)

        env = self.env
        
        # Determine valid actions
        if hasattr(env, "A"):
            actions_to_sample = range(env.A)
        elif hasattr(env, "action_space") and isinstance(env.action_space, gym.spaces.Discrete):
            actions_to_sample = range(env.action_space.n)
        else:
            raise ValueError("add_manual_samples requires a discrete environment.")

        data = {
            "episode": [],
            "step": [],
            "state": [],
            "action": [],
            "reward": [],
            "next_state": [],
        }

        if verbose:
            total = len(states) * len(actions_to_sample) * samples_per_pair
            print(f"Adding MANUAL samples: {len(states)} states, {len(actions_to_sample)} actions, {samples_per_pair} samples/pair.")
            print(f"Total transitions to add: {total}")

        idx = 0
        for s in states:
            # Ensure s is int
            s_val = int(s)
            for a in actions_to_sample:
                for _ in range(samples_per_pair):
                    # Manual state injection (requires env to support .state assignment)
                    try:
                        env.state = s_val
                    except AttributeError:
                        raise AttributeError("Environment does not support manual state assignment via .state.")
                    
                    if hasattr(env, "step_count"):
                        env.step_count = 0
                    
                    next_obs, reward, terminated, truncated, _ = env.step(a)
                    
                    # Consistent rewards for terminal self-loops
                    if terminated and s_val in getattr(env, "terminal_states", []):
                        reward = self._get_reward_from_env(s_val, a, reward)

                    data["episode"].append(-1) # Mark as manual/augmented
                    data["step"].append(0)
                    data["state"].append(s_val)
                    data["action"].append(a)
                    data["reward"].append(reward)
                    data["next_state"].append(next_obs)
                    idx += 1

        df_manual = pd.DataFrame(data)

        if save_path is not None:
            # If file exists, append; otherwise create
            if os.path.exists(save_path):
                # We only need the columns that matched the original dataset
                cols = ["state", "action", "reward", "next_state"]
                df_existing = pd.read_csv(save_path)
                df_combined = pd.concat([df_existing, df_manual[cols]], ignore_index=True)
                df_combined.to_csv(save_path, index=False)
                if verbose:
                    print(f"✅ Manual samples appended to: {save_path}")
                    print(f"   Total transitions now: {len(df_combined)}")
            else:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                df_manual.to_csv(save_path, index=False)
                if verbose:
                    print(f"✅ Manual samples saved to: {save_path}")

        return df_manual

    def _get_reward_from_env(self, state, action, fallback):
        """
        Internal helper to recover original reward from env model for terminal self-loops.
        """
        env = self.env
        if not (hasattr(env, "states") and hasattr(env, "A") and hasattr(env, "r")):
            return fallback

        try:
            if isinstance(env.states, torch.Tensor):
                state_idx_tensor = (env.states == int(state)).nonzero(as_tuple=True)[0]
                if len(state_idx_tensor) == 0:
                    return fallback
                state_idx = int(state_idx_tensor[0].item())
            else:
                state_idx = int(state)

            row_idx = state_idx * int(env.A) + int(action)
            r_val = env.r[row_idx]
            if isinstance(r_val, torch.Tensor):
                return float(r_val.item())
            return float(r_val)
        except Exception:
            return fallback
