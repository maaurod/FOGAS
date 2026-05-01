# mdp/linear_mdp.py

"""
Linear MDP model where rewards and transitions are linear in a shared
state–action feature map phi(x, a).

Rewards are defined as:
    r(x, a) = <phi(x, a), omega>

Transitions can be specified either by:
    (1) transition weight vectors psi[x'], giving
            P(x' | x, a) = <phi(x, a), psi[x']>
    or
    (2) an explicit transition matrix P of shape (N*A, N).

The class computes basic structural properties (feature dimension, feature norm
bound R), stores gamma and the initial state x0, and provides utilities for:

    - building the reward vector,
    - constructing or validating the transition matrix,
    - printing a policy in a readable form.
"""

# mdp/linear_mdp.py

import torch

class LinearMDP:
    def __init__(
        self,
        states,
        actions,
        phi,
        omega=None,
        gamma=None,
        x0=None,
        psi=None,   # callable psi(xp)->(d,) OR dict {xp:(d,)} OR None
        P=None,
        terminal_states=None,
        reward_fn=None,  # Optional: direct reward function r(x, a) -> float
    ):
        """
        Linear MDP with feature-based rewards and transitions.

        Rewards:
            Either provide omega: r(x,a) = <phi(x,a), omega>
            Or provide reward_fn: r(x,a) = reward_fn(x, a)

        Transitions:
            Either provide psi as a callable:
                P(x'|x,a) = <phi(x,a), psi(x')>
            Or provide an explicit transition matrix P with shape (N*A, N).
        """

        self.states = states.clone()
        self.actions = actions.clone()
        self.N = len(self.states)
        self.A = len(self.actions)

        self.terminal_states = set(terminal_states) if terminal_states is not None else set()

        self.phi = phi
        self.d = len(phi(self.states[0], self.actions[0]))

        # Upper bound on feature norms
        norms = []
        for s in self.states:
            for a in self.actions:
                phi_val = phi(s.item(), a.item())
                norms.append(torch.linalg.norm(phi_val))
        self.R = torch.max(torch.stack(norms))

        # Handle reward specification: either omega or reward_fn
        self.reward_fn = reward_fn
        if omega is not None and reward_fn is not None:
            raise ValueError("Cannot specify both omega and reward_fn")
        
        if omega is not None:
            self.omega = omega.clone().to(dtype=torch.float64).reshape(-1)
            if self.omega.shape != (self.d,):
                raise ValueError(f"omega must have shape ({self.d},), got {self.omega.shape}")
        elif reward_fn is not None:
            # omega will be None, will be computed when needed
            self.omega = None
        else:
            raise ValueError("Must provide either omega or reward_fn")

        # Store psi or P
        self.P = None
        self.given_psi = psi is not None

        if self.given_psi:
            # Allow dict form for backward compatibility, but convert to callable.
            if isinstance(psi, dict):
                psi_dict = {int(k): v.clone().to(dtype=torch.float64).reshape(-1) for k, v in psi.items()}
                for xp, v in psi_dict.items():
                    if v.shape != (self.d,):
                        raise ValueError(f"psi[{xp}] must have shape ({self.d},), got {v.shape}")

                def psi_fn(xp):
                    xp = int(xp)
                    if xp not in psi_dict:
                        raise KeyError(f"psi is missing key for next state {xp}")
                    return psi_dict[xp]

                self.psi = psi_fn
            else:
                if not callable(psi):
                    raise TypeError("psi must be a callable psi(xp)->(d,), a dict {xp: (d,)}, or None")
                # Light shape check on one call
                test = psi(self.states[0].item()).to(dtype=torch.float64).reshape(-1)
                if test.shape != (self.d,):
                    raise ValueError(f"psi(xp) must return shape ({self.d},), got {test.shape}")
                self.psi = psi
        else:
            if P is None:
                raise ValueError("If psi is None, you must provide explicit transition matrix P.")
            self.P = P.clone().to(dtype=torch.float64)

        self.gamma = float(gamma)
        self.x0 = int(x0)

        self.nu0 = torch.zeros(self.N, dtype=torch.float64)
        self.nu0[self.x0] = 1.0

        # Precompute Phi (N*A, d)
        Phi = []
        for x in range(self.N):
            s = self.states[x].item()
            for a in range(self.A):
                act = self.actions[a].item()
                Phi.append(self.phi(s, act).to(dtype=torch.float64))
        self.Phi = torch.vstack(Phi)

        self.Psi = None  # cache

    def to(self, device):
        """Move all internal tensors to the specified device."""
        self.states = self.states.to(device)
        self.actions = self.actions.to(device)
        if self.omega is not None:
            self.omega = self.omega.to(device)
        self.nu0 = self.nu0.to(device)
        self.Phi = self.Phi.to(device)
        if self.P is not None:
            self.P = self.P.to(device)
        if self.Psi is not None:
            self.Psi = self.Psi.to(device)
        return self

    # ------------------------------------------------------------
    # Reward computation: r = Φω
    # ------------------------------------------------------------
    def get_reward(self, verbose=False):
        if self.omega is not None:
            # r = Phi @ omega
            # Shapes: (N*A, d) @ (d,) -> (N*A,)
            r = self.Phi @ self.omega
        else:
            # Use reward_fn directly
            r_list = []
            for x in range(self.N):
                for a in range(self.A):
                    r_list.append(self.reward_fn(self.states[x].item(), self.actions[a].item()))
            r = torch.tensor(r_list, dtype=torch.float64)

        if verbose:
            print("\n=== Reward Vector r ===")
            print("Shape:", r.shape)
            idx = 0
            for x in range(self.N):
                for a in range(self.A):
                    print(f"r(s={self.states[x].item()}, a={self.actions[a].item()}) = {r[idx].item():.4f}")
                    idx += 1
            print("")
        return r

    # ------------------------------------------------------------
    # Transition matrix P of shape (N*A, N)
    # ------------------------------------------------------------
    def get_transition_matrix(self, verbose=False):
        if self.given_psi:
            # P = Phi @ Psi
            # Shapes: (N*A, d) @ (d, N) -> (N*A, N)
            Psi = self.get_Psi()
            P = self.Phi @ Psi
        else:
            P = self.P.clone()

        # Validation
        eps = 1e-12
        negative_entries = torch.any(P < -eps)
        row_sums = P.sum(dim=1)
        invalid_rows = torch.where(torch.abs(row_sums - 1) > 1e-6)[0]

        if verbose:
            print("\n=== Transition Matrix P ===")
            print("Shape:", P.shape)
            print("First few rows:\n", P[:min(5, len(P)), :], "\n")

            print("✔ No negative probabilities detected." if not negative_entries
                  else "✘ WARNING: Negative probabilities detected!")

            if len(invalid_rows) == 0:
                print("✔ All rows sum to 1 (valid probability matrix).")
            else:
                print("✘ WARNING: Some rows do not sum to 1:")
                for i in invalid_rows[:10]:
                    print(f"  Row {i}: sum = {row_sums[i]:.6f}")

        return P

    # ------------------------------------------------------------
    # Pretty-print policy
    # ------------------------------------------------------------
    def print_policy(self, pi):
        for i, s in enumerate(self.states):
            best_a_idx = torch.argmax(pi[i]).item()
            best_action = self.actions[best_a_idx]

            print(f"  State {s}: ", end="")
            for j, a in enumerate(self.actions):
                print(f"π(a={a}|s={s}) = {pi[i, j].item():.2f}  ", end="")
            print(f"--> best action: {best_action}")
        print()

    # ------------------------------------------------------------
    # Get Psi matrix (d, N)
    # ------------------------------------------------------------
    def get_Psi(self):
        """
        Returns Psi with shape (d, N), where column xp is psi(xp).

        - If psi callable was provided: builds Psi by calling psi(xp).
        - If P was provided explicitly: computes Psi = pinv(Phi) @ P.
        """
        if self.Psi is not None:
            return self.Psi

        if self.given_psi:
            device = self.Phi.device
            Psi = torch.zeros((self.d, self.N), dtype=torch.float64, device=device)
            for xp in range(self.N):
                s_next = self.states[xp].item()
                v = self.psi(s_next).to(dtype=torch.float64, device=device).reshape(-1)
                if v.shape != (self.d,):
                    raise ValueError(f"psi({s_next}) must return shape ({self.d},), got {v.shape}")
                Psi[:, xp] = v
            self.Psi = Psi
            return self.Psi

        # Recover Psi from explicit P and Phi
        Phi = self.Phi
        P = self.P

        NA = self.N * self.A
        if Phi.shape[0] != NA:
            raise ValueError(f"Phi must have {NA} rows (N*A). Got {Phi.shape[0]}.")
        if P.shape != (NA, self.N):
            raise ValueError(f"P must have shape ({NA}, {self.N}). Got {P.shape}.")

        self.Psi = torch.linalg.pinv(Phi) @ P  # (d, N)
        return self.Psi