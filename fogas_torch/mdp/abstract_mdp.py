"""
Utilities for turning continuous-control style problems into finite abstract
MDPs that remain compatible with LinearMDP / FOGAS.

The abstraction is intentionally split into small pieces:

    - state discretizers map continuous observations to discrete state ids
    - action discretizers map discrete action ids to environment actions
    - feature maps define phi(state_id, action_id)
    - DiscretizedLinearMDP builds rewards and transitions automatically

This keeps FOGAS itself unchanged: the solver still only sees a standard
finite linear MDP.
"""

from __future__ import annotations

import itertools
from typing import Callable, Optional, Sequence

import numpy as np
import torch

from .linear_mdp import LinearMDP


class BoxStateDiscretizer:
    """
    Uniform axis-aligned discretization for a Box-like continuous state space.

    Parameters
    ----------
    low, high : array-like
        Lower and upper bounds for each state dimension.
    bins : array-like of int
        Number of bins per dimension.
    terminal_obs_predicate : callable or None
        Optional predicate terminal_obs_predicate(obs) -> bool. If provided,
        an absorbing state is appended after the core grid states.
    """

    def __init__(
        self,
        low: Sequence[float],
        high: Sequence[float],
        bins: Sequence[int],
        terminal_obs_predicate: Optional[Callable[[np.ndarray], bool]] = None,
    ):
        self.low = np.asarray(low, dtype=np.float64)
        self.high = np.asarray(high, dtype=np.float64)
        self.bins = np.asarray(bins, dtype=np.int64)

        if self.low.shape != self.high.shape or self.low.shape != self.bins.shape:
            raise ValueError("low, high, and bins must have the same shape")
        if np.any(self.high <= self.low):
            raise ValueError("Each component of high must be strictly greater than low")
        if np.any(self.bins <= 0):
            raise ValueError("All entries in bins must be positive")

        self.dim = int(self.low.size)
        self.terminal_obs_predicate = terminal_obs_predicate

        self.bin_edges = [
            np.linspace(lo, hi, int(n_bins) + 1, dtype=np.float64)
            for lo, hi, n_bins in zip(self.low, self.high, self.bins)
        ]
        self.bin_centers = [
            0.5 * (edges[:-1] + edges[1:]) for edges in self.bin_edges
        ]
        self.bin_widths = (self.high - self.low) / self.bins

        self.core_state_count = int(np.prod(self.bins))
        self.absorbing_state_id = (
            self.core_state_count if self.terminal_obs_predicate is not None else None
        )
        self.n_states = self.core_state_count + int(self.absorbing_state_id is not None)

    def clip(self, obs: Sequence[float]) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64)
        if obs.shape != self.low.shape:
            raise ValueError(f"Expected obs shape {self.low.shape}, got {obs.shape}")
        return np.clip(obs, self.low, self.high)

    def is_terminal_obs(self, obs: Sequence[float]) -> bool:
        if self.terminal_obs_predicate is None:
            return False
        return bool(self.terminal_obs_predicate(np.asarray(obs, dtype=np.float64)))

    def obs_to_multi_bin(self, obs: Sequence[float]) -> tuple[int, ...]:
        obs = self.clip(obs)
        scaled = np.floor((obs - self.low) / self.bin_widths).astype(np.int64)
        clipped = np.clip(scaled, 0, self.bins - 1)
        return tuple(int(v) for v in clipped)

    def multi_bin_to_state_id(self, multi_bin: Sequence[int]) -> int:
        return int(np.ravel_multi_index(tuple(multi_bin), self.bins))

    def state_id_to_multi_bin(self, state_id: int) -> tuple[int, ...]:
        self._validate_core_state_id(state_id)
        return tuple(int(v) for v in np.unravel_index(int(state_id), self.bins))

    def obs_to_state_id(self, obs: Sequence[float]) -> int:
        if self.is_terminal_obs(obs):
            if self.absorbing_state_id is None:
                raise ValueError("Terminal observation encountered without absorbing state")
            return int(self.absorbing_state_id)
        return self.multi_bin_to_state_id(self.obs_to_multi_bin(obs))

    def state_id_to_center_obs(self, state_id: int) -> Optional[np.ndarray]:
        state_id = int(state_id)
        if self.absorbing_state_id is not None and state_id == self.absorbing_state_id:
            return None
        multi_bin = self.state_id_to_multi_bin(state_id)
        return np.array(
            [self.bin_centers[d][idx] for d, idx in enumerate(multi_bin)],
            dtype=np.float64,
        )

    def _validate_core_state_id(self, state_id: int) -> None:
        state_id = int(state_id)
        if state_id < 0 or state_id >= self.core_state_count:
            raise ValueError(
                f"Core state_id must be in [0, {self.core_state_count - 1}], got {state_id}"
            )


class DiscreteActionDiscretizer:
    """
    Wraps a finite action set. The exported action ids are always 0..A-1.

    action_values can contain ints, floats, or vectors; these are the actual
    values passed to the transition function.
    """

    def __init__(
        self,
        action_values: Sequence,
        action_labels: Optional[dict[int, str]] = None,
    ):
        if len(action_values) == 0:
            raise ValueError("action_values must be non-empty")

        self._action_values = [self._normalize_action_value(v) for v in action_values]
        self.action_ids = np.arange(len(self._action_values), dtype=np.int64)
        self.n_actions = int(len(self._action_values))
        self.action_labels = (
            {int(k): str(v) for k, v in action_labels.items()}
            if action_labels is not None
            else None
        )

    def action_id_to_env_action(self, action_id: int):
        action_id = int(action_id)
        if action_id < 0 or action_id >= self.n_actions:
            raise ValueError(f"action_id must be in [0, {self.n_actions - 1}], got {action_id}")
        value = self._action_values[action_id]
        if isinstance(value, np.ndarray) and value.ndim == 0:
            return value.item()
        return value.copy() if isinstance(value, np.ndarray) else value

    def action_id_to_label(self, action_id: int) -> str:
        action_id = int(action_id)
        if self.action_labels is not None and action_id in self.action_labels:
            return self.action_labels[action_id]
        return str(self.action_id_to_env_action(action_id))

    @staticmethod
    def _normalize_action_value(value):
        arr = np.asarray(value)
        if arr.ndim == 0:
            return arr.item()
        return arr.astype(np.float64)


class BoxActionDiscretizer(DiscreteActionDiscretizer):
    """
    Uniform discretization for Box-like continuous action spaces.

    For a 1D action space, action_id_to_env_action returns a scalar float.
    For multi-dimensional spaces, it returns a numpy array.
    """

    def __init__(
        self,
        low: Sequence[float],
        high: Sequence[float],
        bins: Sequence[int],
    ):
        low = np.asarray(low, dtype=np.float64)
        high = np.asarray(high, dtype=np.float64)
        bins = np.asarray(bins, dtype=np.int64)

        if low.shape != high.shape or low.shape != bins.shape:
            raise ValueError("low, high, and bins must have the same shape")
        if np.any(high <= low):
            raise ValueError("Each component of high must be strictly greater than low")
        if np.any(bins <= 0):
            raise ValueError("All entries in bins must be positive")

        edges = [
            np.linspace(lo, hi, int(n_bins) + 1, dtype=np.float64)
            for lo, hi, n_bins in zip(low, high, bins)
        ]
        centers = [0.5 * (edge[:-1] + edge[1:]) for edge in edges]
        grid = np.array(list(itertools.product(*centers)), dtype=np.float64)
        action_values = [row[0] if row.shape == (1,) else row for row in grid]

        super().__init__(action_values=action_values)

        self.low = low
        self.high = high
        self.bins = bins
        self.centers = centers
        self.grid = grid


class TabularFeatureMap:
    """One-hot features over abstract state-action pairs."""

    def __init__(self, n_states: int, n_actions: int, dtype=torch.float64):
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.d = self.n_states * self.n_actions
        self.dtype = dtype

    def __call__(self, state_id, action_id):
        feat = torch.zeros(self.d, dtype=self.dtype)
        idx = int(state_id) * self.n_actions + int(action_id)
        feat[idx] = 1.0
        return feat


class DiscretizedLinearMDP(LinearMDP):
    """
    Builds a finite abstract MDP from continuous states/actions and a
    user-supplied transition function over representative observations/actions.

    Parameters
    ----------
    state_discretizer : BoxStateDiscretizer
    action_discretizer : DiscreteActionDiscretizer or BoxActionDiscretizer
    transition_fn : callable
        Called as transition_fn(obs, env_action). Supported return signatures:
            (next_obs, reward, done)
            (next_obs, reward, terminated, truncated)
            (next_obs, reward, terminated, truncated, info)
    gamma : float
    initial_obs : array-like
        Initial continuous observation used to derive x0.
    """

    def __init__(
        self,
        state_discretizer: BoxStateDiscretizer,
        action_discretizer: DiscreteActionDiscretizer,
        transition_fn: Callable,
        gamma: float,
        initial_obs: Sequence[float],
        dtype=torch.float64,
    ):
        self.state_discretizer = state_discretizer
        self.action_discretizer = action_discretizer
        self.transition_fn = transition_fn

        states = torch.arange(self.state_discretizer.n_states, dtype=torch.int64)
        actions = torch.arange(self.action_discretizer.n_actions, dtype=torch.int64)

        self.absorbing_state_id = self.state_discretizer.absorbing_state_id
        terminal_states = (
            {int(self.absorbing_state_id)}
            if self.absorbing_state_id is not None
            else set()
        )

        self.feature_map = TabularFeatureMap(
            n_states=self.state_discretizer.n_states,
            n_actions=self.action_discretizer.n_actions,
            dtype=dtype,
        )
        P_table, reward_table = self._build_model_tables()

        self.transition_table = torch.from_numpy(P_table).to(dtype=torch.float64)
        self.transition_matrix = self.transition_table.reshape(len(states) * len(actions), len(states))
        self.reward_table = torch.from_numpy(reward_table).to(dtype=torch.float64)
        x0 = self.state_discretizer.obs_to_state_id(initial_obs)

        super().__init__(
            states=states,
            actions=actions,
            phi=self.feature_map,
            reward_fn=self.reward_fn,
            gamma=gamma,
            x0=x0,
            P=self.transition_matrix,
            terminal_states=terminal_states,
        )

    def reward_fn(self, state_id, action_id):
        return float(self.reward_table[int(state_id), int(action_id)].item())

    def obs_to_state_id(self, obs: Sequence[float]) -> int:
        return self.state_discretizer.obs_to_state_id(obs)

    def state_id_to_center_obs(self, state_id: int) -> Optional[np.ndarray]:
        return self.state_discretizer.state_id_to_center_obs(state_id)

    def action_id_to_env_action(self, action_id: int):
        return self.action_discretizer.action_id_to_env_action(action_id)

    def action_id_to_label(self, action_id: int) -> str:
        return self.action_discretizer.action_id_to_label(action_id)

    def _build_model_tables(self):
        n_states = self.state_discretizer.n_states
        n_actions = self.action_discretizer.n_actions

        P = np.zeros((n_states, n_actions, n_states), dtype=np.float64)
        r = np.zeros((n_states, n_actions), dtype=np.float64)

        for state_id in range(n_states):
            is_absorbing = (
                self.absorbing_state_id is not None
                and state_id == self.absorbing_state_id
            )

            for action_id in range(n_actions):
                if is_absorbing:
                    P[state_id, action_id, state_id] = 1.0
                    r[state_id, action_id] = 0.0
                    continue

                obs = self.state_discretizer.state_id_to_center_obs(state_id)
                env_action = self.action_discretizer.action_id_to_env_action(action_id)
                next_obs, reward, done = self._parse_transition_output(
                    self.transition_fn(obs.copy(), env_action)
                )

                if done or self.state_discretizer.is_terminal_obs(next_obs):
                    if self.absorbing_state_id is None:
                        raise ValueError(
                            "Transition reached a terminal observation but no absorbing state is defined"
                        )
                    next_state_id = self.absorbing_state_id
                else:
                    next_state_id = self.state_discretizer.obs_to_state_id(next_obs)

                P[state_id, action_id, int(next_state_id)] = 1.0
                r[state_id, action_id] = float(reward)

        return P, r

    @staticmethod
    def _parse_transition_output(result):
        if not isinstance(result, tuple):
            raise TypeError("transition_fn must return a tuple")

        if len(result) == 3:
            next_obs, reward, done = result
            return np.asarray(next_obs, dtype=np.float64), float(reward), bool(done)

        if len(result) == 4:
            next_obs, reward, terminated, truncated = result
            done = bool(terminated) or bool(truncated)
            return np.asarray(next_obs, dtype=np.float64), float(reward), done

        if len(result) == 5:
            next_obs, reward, terminated, truncated, _ = result
            done = bool(terminated) or bool(truncated)
            return np.asarray(next_obs, dtype=np.float64), float(reward), done

        raise ValueError(
            "transition_fn must return (next_obs, reward, done), "
            "(next_obs, reward, terminated, truncated), or "
            "(next_obs, reward, terminated, truncated, info)"
        )
