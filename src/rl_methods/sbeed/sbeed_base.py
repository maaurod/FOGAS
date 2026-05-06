from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Protocol, runtime_checkable

import torch

from .sbeed_spec import DiscreteMDPSpec


@runtime_checkable
class SBEEDSolverProtocol(Protocol):
    """Shared public contract for SBEED-style solvers."""

    spec: DiscreteMDPSpec
    n_states: int
    n_actions: int
    gamma: float
    lambda_entropy: float
    n: int
    pi: Optional[torch.Tensor]
    theta: torch.Tensor
    PHI_S: torch.Tensor

    def run(
        self,
        transition_fn: Callable[[int, int], Any],
        reward_fn: Optional[Callable[[int, int, int], float]] = None,
        episodes: int = 100,
        collect_per_episode: int = 10,
        updates_per_episode: int = 10,
        initial_collect_steps: int = 0,
        start_state: Optional[int] = None,
        behavior: str = "policy",
        epsilon: float = 0.1,
        terminal_states: Optional[set] = None,
        reset_state_fn: Optional[Callable[[], int]] = None,
        verbose: bool = False,
        log_every: int = 10,
        tqdm_print: bool = True,
        store_history: bool = True,
    ) -> torch.Tensor:
        ...

    def get_policy_matrix(self, W: Optional[torch.Tensor] = None) -> torch.Tensor:
        ...

    def objective(self) -> Dict[str, float]:
        ...
