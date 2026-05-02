from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import torch


@dataclass
class SBEEDDataset:
    """
    Transition buffer D for discrete SBEED experiments.

    In the paper, SBEED is written as an online algorithm with an experience
    replay buffer D. This class is the code representation of that buffer.
    The buffer is populated incrementally by the solver's online collection
    loop.

    Expected transition fields are state, action, reward, and next_state. There
    is no dependency on FOGAS or LinearMDP internals.
    """

    X: torch.Tensor
    A: torch.Tensor
    R: torch.Tensor
    X_next: torch.Tensor

    def __post_init__(self) -> None:
        self.X = torch.as_tensor(self.X, dtype=torch.int64).reshape(-1)
        self.A = torch.as_tensor(self.A, dtype=torch.int64).reshape(-1)
        self.R = torch.as_tensor(self.R, dtype=torch.float64).reshape(-1)
        self.X_next = torch.as_tensor(self.X_next, dtype=torch.int64).reshape(-1)

        lengths = {self.X.numel(), self.A.numel(), self.R.numel(), self.X_next.numel()}
        if len(lengths) != 1:
            raise ValueError("state, action, reward, and next_state must have the same length")
        self.n = int(self.X.numel())

    @classmethod
    def empty(cls, device: Optional[Union[torch.device, str]] = None) -> "SBEEDDataset":
        device = torch.device("cpu" if device is None else device)
        return cls(
            X=torch.empty(0, dtype=torch.int64, device=device),
            A=torch.empty(0, dtype=torch.int64, device=device),
            R=torch.empty(0, dtype=torch.float64, device=device),
            X_next=torch.empty(0, dtype=torch.int64, device=device),
        )

    def append(self, state: int, action: int, reward: float, next_state: int) -> None:
        device = self.X.device
        self.X = torch.cat([self.X, torch.tensor([state], dtype=torch.int64, device=device)])
        self.A = torch.cat([self.A, torch.tensor([action], dtype=torch.int64, device=device)])
        self.R = torch.cat([self.R, torch.tensor([reward], dtype=torch.float64, device=device)])
        self.X_next = torch.cat([self.X_next, torch.tensor([next_state], dtype=torch.int64, device=device)])
        self.n = int(self.X.numel())

    def append_fifo(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        capacity: int,
    ) -> None:
        capacity = int(capacity)
        if capacity <= 0:
            raise ValueError("capacity must be positive")

        self.append(state, action, reward, next_state)
        if self.n > capacity:
            self.X = self.X[-capacity:]
            self.A = self.A[-capacity:]
            self.R = self.R[-capacity:]
            self.X_next = self.X_next[-capacity:]
            self.n = int(self.X.numel())

    def extend(self, other: "SBEEDDataset") -> None:
        other = other.to(self.X.device)
        self.X = torch.cat([self.X, other.X])
        self.A = torch.cat([self.A, other.A])
        self.R = torch.cat([self.R, other.R])
        self.X_next = torch.cat([self.X_next, other.X_next])
        self.n = int(self.X.numel())

    def to(self, device: Union[torch.device, str]) -> "SBEEDDataset":
        device = torch.device(device)
        return SBEEDDataset(
            X=self.X.to(device),
            A=self.A.to(device),
            R=self.R.to(device),
            X_next=self.X_next.to(device),
        )

    def validate(self, n_states: int, n_actions: int) -> None:
        if torch.any((self.X < 0) | (self.X >= n_states)):
            raise ValueError("dataset states must be in [0, n_states)")
        if torch.any((self.X_next < 0) | (self.X_next >= n_states)):
            raise ValueError("dataset next_states must be in [0, n_states)")
        if torch.any((self.A < 0) | (self.A >= n_actions)):
            raise ValueError("dataset actions must be in [0, n_actions)")

    def summary(self) -> Dict[str, Any]:
        if self.n == 0:
            return {
                "n": 0,
                "unique_states": torch.empty(0, dtype=torch.int64, device=self.X.device),
                "unique_actions": torch.empty(0, dtype=torch.int64, device=self.A.device),
                "reward_mean": None,
            }
        return {
            "n": self.n,
            "unique_states": torch.unique(self.X),
            "unique_actions": torch.unique(self.A),
            "reward_mean": float(self.R.mean().item()),
        }
