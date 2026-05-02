from __future__ import annotations

import torch


class TabularStateFeatures:
    """One-hot state features phi(s)."""

    def __init__(self, n_states: int, dtype: torch.dtype = torch.float64):
        self.n_states = int(n_states)
        if self.n_states <= 0:
            raise ValueError("n_states must be positive")
        self.d = self.n_states
        self.dtype = dtype

    def __call__(self, state: int) -> torch.Tensor:
        state = int(state)
        if state < 0 or state >= self.n_states:
            raise ValueError(f"state must be in [0, {self.n_states}), got {state}")
        feat = torch.zeros(self.d, dtype=self.dtype)
        feat[state] = 1.0
        return feat


class TabularStateActionFeatures:
    """One-hot state-action features rho_features(s, a)."""

    def __init__(self, n_states: int, n_actions: int, dtype: torch.dtype = torch.float64):
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        if self.n_states <= 0:
            raise ValueError("n_states must be positive")
        if self.n_actions <= 0:
            raise ValueError("n_actions must be positive")
        self.d = self.n_states * self.n_actions
        self.dtype = dtype

    def __call__(self, state: int, action: int) -> torch.Tensor:
        state = int(state)
        action = int(action)
        if state < 0 or state >= self.n_states:
            raise ValueError(f"state must be in [0, {self.n_states}), got {state}")
        if action < 0 or action >= self.n_actions:
            raise ValueError(f"action must be in [0, {self.n_actions}), got {action}")
        feat = torch.zeros(self.d, dtype=self.dtype)
        feat[state * self.n_actions + action] = 1.0
        return feat
