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
                 max_steps=1000, terminal_states=None):

        if env is not None:
            self.env = env

        elif mdp is not None:
            self.env = LinearMDPEnv(
                mdp,
                max_steps=max_steps,
                terminal_states=terminal_states,
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

    # --------------------------------------------
    # Policy factory
    # --------------------------------------------
    def _make_policy(self, policy):
        """
        Accepts:
            - a policy object with .sample(state)
            - or a string: "random", "uniform"

        Returns:
            A policy object with method sample(state).
        """

        # Already a policy object → return as is
        if hasattr(policy, "sample"):
            return policy

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

        raise ValueError("Policy must be a string or an object with a .sample() method.")

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
