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

import numpy as np

class LinearMDP:
    def __init__(
        self,
        states,
        actions,
        phi,
        omega,
        gamma,
        x0,
        psi=None,   # callable psi(xp)->(d,) OR dict {xp:(d,)} OR None
        P=None,
    ):
        """
        Linear MDP with feature-based rewards and transitions.

        Rewards:
            r(x,a) = <phi(x,a), omega>

        Transitions:
            Either provide psi as a callable:
                P(x'|x,a) = <phi(x,a), psi(x')>
            Or provide an explicit transition matrix P with shape (N*A, N).
        """

        self.states = np.array(states)
        self.actions = np.array(actions)
        self.N = len(self.states)
        self.A = len(self.actions)

        self.phi = phi
        self.d = len(phi(self.states[0], self.actions[0]))

        # Upper bound on feature norms
        self.R = np.max([
            np.linalg.norm(phi(s, a))
            for s in self.states
            for a in self.actions
        ])

        self.omega = np.array(omega).reshape(-1)
        if self.omega.shape != (self.d,):
            raise ValueError(f"omega must have shape ({self.d},), got {self.omega.shape}")

        # Store psi or P
        self.P = None
        self.given_psi = psi is not None

        if self.given_psi:
            # Allow dict form for backward compatibility, but convert to callable.
            if isinstance(psi, dict):
                psi_dict = {int(k): np.asarray(v).reshape(-1) for k, v in psi.items()}
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
                test = np.asarray(psi(self.states[0])).reshape(-1)
                if test.shape != (self.d,):
                    raise ValueError(f"psi(xp) must return shape ({self.d},), got {test.shape}")
                self.psi = psi
        else:
            if P is None:
                raise ValueError("If psi is None, you must provide explicit transition matrix P.")
            self.P = np.asarray(P)

        self.gamma = float(gamma)
        self.x0 = int(x0)

        self.nu0 = np.zeros(self.N)
        self.nu0[self.x0] = 1.0

        # Precompute Phi (N*A, d)
        Phi = []
        for x in range(self.N):
            for a in range(self.A):
                Phi.append(self.phi(x, a))
        self.Phi = np.vstack(Phi)

        self.Psi = None  # cache

    # ------------------------------------------------------------
    # Reward computation: r = Φω
    # ------------------------------------------------------------
    def get_reward(self, verbose=False):
        r = np.zeros(self.N * self.A)
        idx = 0
        for x in range(self.N):
            for a in range(self.A):
                r[idx] = self.phi(x, a) @ self.omega
                idx += 1

        if verbose:
            print("\n=== Reward Vector r ===")
            print("Shape:", r.shape)
            for x in range(self.N):
                for a in range(self.A):
                    print(f"r(s={x}, a={a}) = {self.phi(x,a) @ self.omega:.4f}")
            print("")
        return r

    # ------------------------------------------------------------
    # Transition matrix P of shape (N*A, N)
    # ------------------------------------------------------------
    def get_transition_matrix(self, verbose=False):
        if self.given_psi:
            P = np.zeros((self.N * self.A, self.N))
            idx = 0
            for x in range(self.N):
                for a in range(self.A):
                    ph = self.phi(x, a)
                    for xp in range(self.N):
                        P[idx, xp] = ph @ self.psi(xp)
                    idx += 1
        else:
            P = self.P.copy()

        # Validation
        eps = 1e-12
        negative_entries = np.any(P < -eps)
        row_sums = P.sum(axis=1)
        invalid_rows = np.where(np.abs(row_sums - 1) > 1e-6)[0]

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
            best_a_idx = np.argmax(pi[i])
            best_action = self.actions[best_a_idx]

            print(f"  State {s}: ", end="")
            for j, a in enumerate(self.actions):
                print(f"π(a={a}|s={s}) = {pi[i, j]:.2f}  ", end="")
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
            Psi = np.zeros((self.d, self.N))
            for xp in range(self.N):
                v = np.asarray(self.psi(xp)).reshape(-1)
                if v.shape != (self.d,):
                    raise ValueError(f"psi({xp}) must return shape ({self.d},), got {v.shape}")
                Psi[:, xp] = v
            self.Psi = Psi
            return self.Psi

        # Recover Psi from explicit P and Phi
        Phi = np.asarray(self.Phi)
        P = np.asarray(self.P)

        NA = self.N * self.A
        if Phi.shape[0] != NA:
            raise ValueError(f"Phi must have {NA} rows (N*A). Got {Phi.shape[0]}.")
        if P.shape != (NA, self.N):
            raise ValueError(f"P must have shape ({NA}, {self.N}). Got {P.shape}.")

        self.Psi = np.linalg.pinv(Phi) @ P  # (d, N)
        return self.Psi