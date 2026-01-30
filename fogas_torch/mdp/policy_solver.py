"""
PolicySolver extends LinearMDP with dynamic-programming tools for planning and
analysis. It constructs reward and transition structures from the linear MDP,
computes an optimal deterministic (or softmax) policy via policy iteration, and
derives the corresponding value function, Q-function, and discounted occupancy
measure.

Main utilities:
    - evaluate any policy (value v and action-value q),
    - run deterministic or softmax policy iteration,
    - compute occupancy measures μ^π,
    - print value tables, Q-values, policies, and occupancy summaries.

Designed for linear MDP experimentation and validation of algorithms such as FOGAS.
"""

import torch
from .linear_mdp import LinearMDP


class PolicySolver(LinearMDP):
    """
    Extends LinearMDP with evaluation, iteration, occupancy measure,
    and pretty-printing utilities.
    """

    def __init__(self, mode="deterministic", *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Precompute reward & transition matrix
        self.r = self.get_reward(verbose=False)
        self.P = self.get_transition_matrix(verbose=False)

        # Compute optimal policy π*
        self.pi_star, self.v_star, self.q_star = self.policy_iteration(
            mode=mode,
            verbose=False
        )

        # Compute discounted occupancy μ*
        self.mu_star = self.occupancy_measure(pi=self.pi_star)

    def to(self, device):
        """Move all internal tensors to the specified device."""
        LinearMDP.to(self, device)
        self.r = self.r.to(device)
        self.P = self.P.to(device)
        self.pi_star = self.pi_star.to(device)
        self.v_star = self.v_star.to(device)
        self.q_star = self.q_star.to(device)
        self.mu_star = self.mu_star.to(device)
        return self

    # ------------------------------------------------------------
    # Policy Evaluation
    # ------------------------------------------------------------
    def evaluate_policy(self, pi, verbose=False):
        device = self.r.device
        pi = pi.to(device)
        r = self.r
        P = self.P

        # r_pi(s) = sum_a pi(s,a) r(s,a)
        r_pi = (pi * r.reshape(self.N, self.A)).sum(dim=1)

        # P_pi(s, s') = sum_a pi(s,a) P(s' | s,a)
        P_pi = torch.zeros((self.N, self.N), dtype=torch.float64, device=device)
        for a in range(self.A):
            P_pi += torch.diag(pi[:, a]) @ P[a::self.A, :]

        # v = (I - γ P_pi)^{-1} r_pi
        v = torch.linalg.solve(torch.eye(self.N, dtype=torch.float64, device=device) - self.gamma * P_pi, r_pi)

        # q(s,a) = r(s,a) + γ ∑_{s'} P(s'|s,a) v(s')
        q = (r + self.gamma * P @ v).reshape(self.N, self.A)

        if verbose:
            self.print_results(pi=pi, v=v, q=q)

        return v, q

    # ------------------------------------------------------------
    # Policy Iteration
    # ------------------------------------------------------------
    def policy_iteration(self, mode="deterministic", temperature=1.0,
                         max_iter=1000, eps=1e-8, verbose=False):

        device = self.r.device
        pi = torch.ones((self.N, self.A), dtype=torch.float64, device=device) / self.A  # uniform start

        for it in range(max_iter):
            v, q = self.evaluate_policy(pi)

            # Improve policy
            if mode == "deterministic":
                new_pi = torch.zeros_like(pi)
                best_a = torch.argmax(q, dim=1)
                new_pi[torch.arange(self.N), best_a] = 1.0

            elif mode == "softmax":
                logits = q / temperature
                logits -= logits.max(dim=1, keepdim=True)[0]
                new_pi = torch.exp(logits)
                new_pi /= new_pi.sum(dim=1, keepdim=True)

            else:
                raise ValueError("mode must be 'deterministic' or 'softmax'")

            # Check for convergence
            if torch.allclose(new_pi, pi, atol=eps):
                if verbose:
                    print(f"Converged at iteration {it+1}")
                break

            pi = new_pi

        return pi, v, q

    # ------------------------------------------------------------
    # Pretty-print results (π, v, q)
    # ------------------------------------------------------------
    def print_results(self, pi=None, v=None, q=None):
        print("\n========== POLICY - VALUE RESULTS ==========\n")

        # Print values
        if v is not None:
            for idx, s in enumerate(self.states):
                print(f"State {s}: V = {v[idx].item():.4f}")
            print()

        # Print Q-values
        if q is not None:
            print("Action-Value Function (Q):")
            for i, s in enumerate(self.states):
                print(f"  State {s}:")
                for j, a in enumerate(self.actions):
                    print(f"    Action {a}: Q(s={s}, a={a}) = {q[i, j].item():.4f}")
                print()

        # Print policy
        if pi is not None:
            self.print_policy(pi)

        print("=============================================\n")

    def print_optimals(self, occupancy=False):
        self.print_results(self.pi_star, self.v_star, self.q_star)
        if occupancy:
            self.print_occupancy(self.mu_star)

    # ------------------------------------------------------------
    # Compute Occupancy Measure μ^π  
    # ------------------------------------------------------------
    def occupancy_measure(self, pi):
        # Build (N*A × N) matrix that expands pi into action probabilities
        device = self.r.device
        pi = pi.to(device)
        Comp_pi = torch.zeros((self.N * self.A, self.N), dtype=torch.float64, device=device)
        for x in range(self.N):
            for a in range(self.A):
                Comp_pi[x * self.A + a, x] = pi[x, a]

        I = torch.eye(self.N * self.A, dtype=torch.float64, device=device)
        rhs = (1 - self.gamma) * (Comp_pi @ self.nu0)

        # Solve: μ = (I - γ * Comp_pi * P^T)^(-1) * rhs
        mu = torch.linalg.solve(I - self.gamma * Comp_pi @ self.P.T, rhs)

        return mu

    # ------------------------------------------------------------
    # Pretty-print occupancy measure 
    # ------------------------------------------------------------
    def print_occupancy(self, mu):
        N, A = self.N, self.A
        mu_matrix = mu.reshape(N, A)

        print("\nDiscounted Occupancy Measure μ^π")
        header = "State | " + " | ".join(f"{a:>8}" for a in self.actions) + " |   Sum"
        print(header)

        for i, s in enumerate(self.states):
            row_vals = " | ".join(f"{mu_matrix[i, j].item():8.5f}" for j in range(A))
            row_sum = torch.sum(mu_matrix[i]).item()
            print(f"{str(s):>5} | {row_vals} | {row_sum:6.5f}")

        total_sum = torch.sum(mu_matrix).item()
        print(f"Total sum = {total_sum:.6f} (should be ≈ 1.0)\n")

    # ------------------------------------------------------------
    # Discounted Final Reward (Value-based)
    # ------------------------------------------------------------
    def policy_return(self, pi):
        """
        Compute discounted final return rho(pi) using the value function.
        """
        v, _ = self.evaluate_policy(pi)
        return float((1.0 - self.gamma) * self.nu0 @ v)

    def optimal_policy_return(self):
        """
        Compute discounted final return (pi*).
        """
        return float((1.0 - self.gamma) * self.nu0 @ self.v_star)

    def optimal_q_feature_weights(self, ridge=1e-10):
        """
        Compute theta* such that Q*(x,a) ≈ <phi(x,a), theta*>.
        """
        N, A, d = self.N, self.A, self.d
        y = []

        for x in range(N):
            for a in range(A):
                y.append(self.q_star[x, a])

        y = torch.stack(y)  # (N*A,)

        device = self.r.device
        A_mat = self.Phi.T @ self.Phi + ridge * torch.eye(d, dtype=torch.float64, device=device)
        b = self.Phi.T @ y
        theta_star = torch.linalg.solve(A_mat, b)

        return theta_star

    def optimal_feature_occupancy(self):
        """
        Feature occupancy lamda* = sum lambda*(x,a) phi(x,a).
        """
        mu_star = self.mu_star
        return self.Phi.T @ mu_star

