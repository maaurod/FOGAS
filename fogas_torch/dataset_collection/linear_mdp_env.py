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

import torch
import random
import gymnasium as gym
from gymnasium import spaces


class LinearMDPEnv(gym.Env):
    """
    Gym wrapper for LinearMDP / PolicySolver, with:
      - automatic detection of absorbing states from P (if terminal_states=None)
      - random initial state at reset (uniform over states)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, mdp, max_steps=100, terminal_states=None, 
                 restricted_states=None, initial_states=None, 
                 reset_probs=None, tol=1e-8):
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

        # Handle restricted states for random restarts (walls, pits, etc.)
        if restricted_states is None:
            self.forbidden_resets = self.terminal_states.copy()
        else:
            self.forbidden_resets = set(restricted_states) | self.terminal_states
        
        # List of states that are valid for a random restart
        self.valid_start_states = [int(s) for s in range(self.N) if int(s) not in self.forbidden_resets]
        
        # Custom initial states if provided by the user
        self.custom_start_states = initial_states if initial_states is not None else []
        
        # Reset probabilities: e.g., {'x0': 0.2, 'random': 0.8, 'custom': 0.0}
        self.reset_probs = reset_probs

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
                one = torch.tensor(1.0, dtype=P.dtype, device=P.device)
                if not torch.isclose(probs.sum(), one, atol=tol):
                    is_absorbing = False
                    break
                if not torch.isclose(probs[s_idx], one, atol=tol):
                    is_absorbing = False
                    break

                mask_others = torch.ones(self.N, dtype=torch.bool, device=P.device)
                mask_others[s_idx] = False
                if torch.any(probs[mask_others] > tol):
                    is_absorbing = False
                    break

            if is_absorbing:
                absorbing.append(s.item())

        return set(absorbing)

    # ------------------------------------------------
    # Reset: random initial state
    # ------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        # Determine starting state based on reset_probs
        if self.reset_probs is not None:
            # We expect reset_probs to be a dict e.g. {'x0': 0.2, 'random': 0.8}
            modes = list(self.reset_probs.keys())
            weights = list(self.reset_probs.values())
            
            mode = random.choices(modes, weights=weights)[0]
            
            if mode == 'random' and self.valid_start_states:
                self.state = random.choice(self.valid_start_states)
            elif mode == 'custom' and self.custom_start_states:
                self.state = random.choice(self.custom_start_states)
            else:
                self.state = self.mdp.x0
        else:
            # Default behavior: always start at x0
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

        x_idx = int((self.states == self.state).nonzero(as_tuple=True)[0][0])
        a_idx = int(action)

        row_idx = x_idx * self.A + a_idx

        probs = self.P[row_idx]
        # Sample next state using torch.multinomial
        next_state_idx = torch.multinomial(probs, num_samples=1).item()
        next_state = int(self.states[next_state_idx].item())
        reward = float(self.r[row_idx].item())

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
