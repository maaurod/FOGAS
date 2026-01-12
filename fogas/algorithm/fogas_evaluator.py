import numpy as np
import matplotlib.pyplot as plt

class FOGASEvaluator:
    """
    Collection of evaluation metrics for FOGAS.
    Metrics are exposed as CALLABLES returning a scalar.
    """

    def __init__(self, solver):
        self.solver = solver
        self.mdp = solver.mdp

    # =====================================================
    # --- Concrete metric implementations ---
    # =====================================================

    def expected_gap(self, num_J_samples=50):
        if self.solver.theta_bar_history is None:
            raise ValueError("Run solver.run() before evaluating.")

        theta_list = self.solver.theta_bar_history
        T = len(theta_list)

        mu_opt = self.mdp.mu_star
        r = self.mdp.r
        alpha = self.solver.mod_alpha

        gaps = []
        for _ in range(num_J_samples):
            J = np.random.randint(1, T + 1)
            theta_bar = theta_list[J - 1]

            pi_J = self.solver.softmax_policy(
                theta_bar, alpha, return_matrix=True
            )
            mu_J = self.mdp.occupancy_measure(pi_J)

            gaps.append(np.dot(mu_opt - mu_J, r))

        return float(np.mean(gaps))


    def final_reward(self):
        if self.solver.pi is None:
            raise ValueError("Run solver.run() first.")
        return -1 * self.mdp.policy_return(self.solver.pi)

    # =====================================================
    # --- Metric factory ---
    # =====================================================

    def get_metric(self, name, **kwargs):
        """
        Returns a ZERO-ARG callable metric.
        The optimizer will ONLY see the callable.
        """

        if name == "expected_gap":
            num_J_samples = kwargs.get("num_J_samples", 50)
            return lambda: self.expected_gap(num_J_samples)

        if name == "baseline_gap":
            return lambda: self.baseline_gap()

        if name == "reward":
            return lambda: self.final_reward()

        raise ValueError(f"Unknown metric '{name}'")

    def plot_metric_vs_T_single_run(
        self,
        evaluator,
        metric_name,
        growth_rate,
        num_T_values,
        metric_kwargs=None,
        solver_run_kwargs=None,
        log_x=True,
        log_y=True,
        title=None,
    ):
        metric_kwargs = metric_kwargs or {}
        solver_run_kwargs = solver_run_kwargs or {}

        Ts = [int(self.solver.T + (growth_rate *k)) for k in range(num_T_values)]
        values = []

        for T in Ts:
            evaluator.solver.run(T=T, **solver_run_kwargs)
            metric = evaluator.get_metric(metric_name, **metric_kwargs)
            values.append(metric())

        plt.figure(figsize=(6, 4))
        plt.plot(Ts, values, marker="o", label=metric_name)

        if log_x:
            plt.xscale("log")
        if log_y:
            plt.yscale("log")

        plt.xlabel("Horizon T")
        plt.ylabel(metric_name)
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.legend()
        if title is not None:
            plt.title(title)
        plt.tight_layout()
        plt.show()

        return Ts, values
    
    def compare_value_functions(self, print_each=True):
        """
        Print and compare optimal vs learned V and Q functions.
        """

        if self.solver.pi is None:
            raise ValueError("Run solver.run() first.")

        mdp = self.mdp

        # Optimal
        v_star = mdp.v_star
        q_star = mdp.q_star

        # Learned
        v_pi, q_pi = mdp.evaluate_policy(self.solver.pi)

        print("\n========== VALUE FUNCTION COMPARISON ==========\n")

        if print_each:
            print("State-wise V comparison:")
            for x in range(mdp.N):
                diff = v_pi[x] - v_star[x]
                print(
                    f"State {x}: "
                    f"V*(x) = {v_star[x]: .6f} | "
                    f"V^π(x) = {v_pi[x]: .6f} | "
                    f"Δ = {diff: .6e}"
                )

            print("\nAction-value Q comparison:")
            for x in range(mdp.N):
                for a in range(mdp.A):
                    diff = q_pi[x, a] - q_star[x, a]
                    print(
                        f"(x={x}, a={a}): "
                        f"Q*(x,a) = {q_star[x,a]: .6f} | "
                        f"Q^π(x,a) = {q_pi[x,a]: .6f} | "
                        f"Δ = {diff: .6e}"
                    )

        # Norm diagnostics
        v_err = np.linalg.norm(v_pi - v_star)
        q_err = np.linalg.norm(q_pi - q_star)

        print("\nNorm diagnostics:")
        print(f"||V^π - V*||_2 = {v_err:.6e}")
        print(f"||Q^π - Q*||_2 = {q_err:.6e}")

        print("\n===============================================\n")

    def compare_primal_dual(
        self,
        top_k=None,
        rel_eps=1e-12
    ):
        """
        Compare learned theta_bar and lambda against optimal feature-space references
        in a feature-wise (coordinate-wise) manner.

        Parameters
        ----------
        top_k : int or None
            If not None, only print the top_k features with largest absolute error.
        rel_eps : float
            Small constant to avoid division by zero in relative error.
        """

        if self.solver.theta_bar_history is None:
            raise ValueError("Run solver.run() first.")

        mdp = self.mdp

        # Learned quantities
        theta_bar_T = self.solver.theta_bar_history[-1]
        lambda_pi = self.solver.lambda_T

        # Optimal references
        theta_star = mdp.optimal_q_feature_weights()
        lambda_star = mdp.optimal_feature_occupancy()

        print("\n========== PRIMAL–DUAL FEATURE-WISE COMPARISON ==========\n")

        def report_vector(name, x, x_star):
            diff = x - x_star
            abs_err = np.abs(diff)
            rel_err = abs_err / (np.abs(x_star) + rel_eps)

            d = len(x)
            idxs = np.arange(d)

            # Sort by absolute error
            order = np.argsort(-abs_err)
            if top_k is not None:
                order = order[:top_k]

            print(f"\n--- {name} (feature-wise) ---")
            print(
                " idx |     value     |   optimal    |    diff      |  abs err   |  rel err"
            )
            print("-" * 78)

            for i in order:
                print(
                    f"{i:4d} | "
                    f"{x[i]: .6e} | "
                    f"{x_star[i]: .6e} | "
                    f"{diff[i]: .6e} | "
                    f"{abs_err[i]: .6e} | "
                    f"{rel_err[i]: .6e}"
                )

            # Summary diagnostics
            print("\nSummary:")
            print(f"  max |diff| = {abs_err.max():.6e}")
            print(f"  mean |diff| = {abs_err.mean():.6e}")
            print(
                f"  cosine similarity = "
                f"{np.dot(x, x_star)/(np.linalg.norm(x)*np.linalg.norm(x_star)+1e-12):.6f}"
            )

        report_vector("theta_bar", theta_bar_T, theta_star)
        report_vector("lambda", lambda_pi, lambda_star)

    # =====================================================
    # --- Reward comparison ---
    # =====================================================

    def compare_final_rewards(self):
        """
        Compare final policy reward against optimal policy reward.
        """

        if self.solver.pi is None:
            raise ValueError("Run solver.run() first.")

        mdp = self.mdp

        # Optimal policy reward
        J_star = mdp.policy_return(mdp.pi_star)

        # Learned policy reward
        J_pi = mdp.policy_return(self.solver.pi)

        gap = J_star - J_pi

        print("\n========== FINAL REWARD COMPARISON ==========\n")
        print(f"J*(π*)   = {J_star:.6f}")
        print(f"J(π_FOGAS) = {J_pi:.6f}")
        print(f"Gap (J* − J) = {gap:.6e}")
        print("\n============================================\n")

    def print_policy(self):
        self.mdp.print_policy(self.solver.pi)

