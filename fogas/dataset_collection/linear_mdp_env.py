"""
LinearMDPEnv
------------

Lightweight Gymnasium wrapper around a linear MDP model (typically a
PolicySolver or any object exposing .states, .actions, .N, .A, .P, .r).
It provides a standard discrete Gym interface for simulation and
policy testing, with optional automatic detection of absorbing states.

Episodes:
    - start from a uniformly random initial state,
    - terminate on reaching an absorbing state,
    - may truncate after a maximum number of steps.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class LinearMDPEnv(gym.Env):
    """
    Gym wrapper for LinearMDP / PolicySolver, with:
      - automatic detection of absorbing states from P (if terminal_states=None)
      - random initial state at reset (uniform over states)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, mdp, max_steps=100, terminal_states=None, tol=1e-8):
        super().__init__()

        self.mdp = mdp
        self.states = mdp.states
        self.actions = mdp.actions
        self.N = mdp.N
        self.A = mdp.A

        self.observation_space = spaces.Discrete(self.N)
        self.action_space = spaces.Discrete(self.A)

        self.max_steps = max_steps
        self.step_count = 0
        self.state = None

        # Expect mdp to expose precomputed P and r (shape (N*A, N) and (N*A,))
        self.P = mdp.P
        self.r = mdp.r

        # If no terminal_states are provided, detect absorbing states from P
        if terminal_states is None:
            self.terminal_states = self._detect_absorbing_states(self.P, tol=tol)
        else:
            self.terminal_states = set(terminal_states)

    # ------------------------------------------------
    # Helper: detect absorbing states from P
    # ------------------------------------------------
    def _detect_absorbing_states(self, P, tol=1e-8):
        """
        Detect absorbing states from P of shape (N*A, N).

        A state s is marked absorbing if for ALL actions a:
          - P(s'|s,a) is (approximately) a Dirac at s:
                sum_s' P(s'|s,a) ≈ 1
                P(s|s,a) ≈ 1
                P(s'!=s | s,a) ≈ 0
        """
        absorbing = []
        for s_idx, s in enumerate(self.states):
            is_absorbing = True
            for a_idx in range(self.A):
                row_idx = s_idx * self.A + a_idx
                probs = P[row_idx]  # size N

                # Conditions for absorbing under this action
                if not np.isclose(probs.sum(), 1.0, atol=tol):
                    is_absorbing = False
                    break
                if not np.isclose(probs[s_idx], 1.0, atol=tol):
                    is_absorbing = False
                    break

                mask_others = np.ones(self.N, dtype=bool)
                mask_others[s_idx] = False
                if np.any(probs[mask_others] > tol):
                    is_absorbing = False
                    break

            if is_absorbing:
                absorbing.append(s)

        return set(absorbing)

    # ------------------------------------------------
    # Reset: random initial state
    # ------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        # Random initial state (uniform over states)
        self.state = self.mdp.x0
        return int(self.state), {}

    # ------------------------------------------------
    # Step
    # ------------------------------------------------
    def step(self, action):
        assert self.action_space.contains(action)

        # If already in a terminal state: stay there, zero reward, terminated
        if self.state in self.terminal_states:
            return int(self.state), 0.0, True, False, {"terminal_state": True}

        x_idx = int(np.where(self.states == self.state)[0][0])
        a_idx = int(action)

        row_idx = x_idx * self.A + a_idx

        probs = self.P[row_idx]
        next_state = int(np.random.choice(self.states, p=probs))
        reward = float(self.r[row_idx])

        self.state = next_state
        self.step_count += 1

        terminated = next_state in self.terminal_states
        truncated = self.step_count >= self.max_steps

        info = {}
        if terminated:
            info["terminal_state"] = True

        return int(next_state), reward, terminated, truncated, info

    def render(self):
        print(f"Current state: {self.state}")

    def close(self):
        pass
