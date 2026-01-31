import torch
import random
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
            J = torch.randint(1, T + 1, (1,)).item()
            theta_bar = theta_list[J - 1]

            pi_J = self.solver.softmax_policy(
                theta_bar, alpha, return_matrix=True
            )
            mu_J = self.mdp.occupancy_measure(pi_J)

            gaps.append(((mu_opt - mu_J) @ r).item())

        return float(torch.tensor(gaps).mean().item())


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
                    f"V*(x) = {v_star[x].item(): .6f} | "
                    f"V^π(x) = {v_pi[x].item(): .6f} | "
                    f"Δ = {diff.item(): .6e}"
                )

            print("\nAction-value Q comparison:")
            for x in range(mdp.N):
                for a in range(mdp.A):
                    diff = q_pi[x, a] - q_star[x, a]
                    print(
                        f"(x={x}, a={a}): "
                        f"Q*(x,a) = {q_star[x,a].item(): .6f} | "
                        f"Q^π(x,a) = {q_pi[x,a].item(): .6f} | "
                        f"Δ = {diff.item(): .6e}"
                    )

        # Norm diagnostics
        v_err = torch.linalg.norm(v_pi - v_star)
        q_err = torch.linalg.norm(q_pi - q_star)

        print("\nNorm diagnostics:")
        print(f"||V^π - V*||_2 = {v_err.item():.6e}")
        print(f"||Q^π - Q*||_2 = {q_err.item():.6e}")

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
            abs_err = torch.abs(diff)
            rel_err = abs_err / (torch.abs(x_star) + rel_eps)

            d = len(x)
            idxs = torch.arange(d)

            # Sort by absolute error
            order = torch.argsort(-abs_err)
            if top_k is not None:
                order = order[:top_k]

            print(f"\n--- {name} (feature-wise) ---")
            print(
                " idx |     value     |   optimal    |    diff      |  abs err   |  rel err"
            )
            print("-" * 78)

            for i in order:
                print(
                    f"{i.item():4d} | "
                    f"{x[i].item(): .6e} | "
                    f"{x_star[i].item(): .6e} | "
                    f"{diff[i].item(): .6e} | "
                    f"{abs_err[i].item(): .6e} | "
                    f"{rel_err[i].item(): .6e}"
                )

            # Summary diagnostics
            print("\nSummary:")
            print(f"  max |diff| = {abs_err.max().item():.6e}")
            print(f"  mean |diff| = {abs_err.mean().item():.6e}")
            cos_sim = (x @ x_star) / (torch.linalg.norm(x) * torch.linalg.norm(x_star) + 1e-12)
            print(
                f"  cosine similarity = "
                f"{cos_sim.item():.6f}"
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

    # =====================================================
    # --- Optimal Path Visualization ---
    # =====================================================

    def simulate_trajectory(self, pi=None, max_steps=100, seed=None, goal_state=None):
        """
        Simulate a single trajectory following a given policy.
        
        Parameters
        ----------
        pi : tensor, optional
            Policy to follow. If None, uses the learned policy from solver.
        max_steps : int
            Maximum number of steps to simulate.
        seed : int, optional
            Random seed for reproducibility.
        goal_state : int, optional
            If provided, simulation stops when this state is reached.
            
        Returns
        -------
        trajectory : list of dict
            Each element contains: {'state': s, 'action': a, 'reward': r, 'next_state': sp}
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            if torch.backends.mps.is_available():
                torch.mps.manual_seed(seed)
            
        mdp = self.mdp
        
        if pi is None:
            if self.solver.pi is None:
                raise ValueError("Run solver.run() first or provide a policy.")
            pi = self.solver.pi
        
        # Ensure pi is a tensor
        if not isinstance(pi, torch.Tensor):
            pi = torch.tensor(pi, dtype=torch.float64)
            
        trajectory = []
        state = int(mdp.x0)
        
        for step in range(max_steps):
            action_probs = pi[state]
            action = torch.argmax(action_probs).item()
            
            reward = mdp.r[state * mdp.A + action]
            if isinstance(reward, torch.Tensor):
                reward = reward.item()
                
            transition_probs = mdp.P[state * mdp.A + action]
            if isinstance(transition_probs, torch.Tensor):
                next_state = torch.multinomial(transition_probs, num_samples=1).item()
            else:
                next_state = torch.multinomial(torch.tensor(transition_probs), num_samples=1).item()
            
            trajectory.append({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'step': step,
                'was_self_loop': (next_state == state),
                'reached_goal': (next_state == goal_state) if goal_state is not None else False
            })
            
            if goal_state is not None and next_state == goal_state:
                break
                
            state = next_state
            
        return trajectory

    def print_optimal_path(self, use_optimal=False, num_trajectories=1, 
                          max_steps=50, seed=42, show_probabilities=False,
                          show_probabilities_first_n=5, goal_state=None):
        """
        Display optimal path(s) from initial state in a pretty format.
        """
        mdp = self.mdp
        
        if use_optimal:
            pi = mdp.pi_star
            policy_name = "Optimal Policy (π*)"
        else:
            if self.solver.pi is None:
                raise ValueError("Run solver.run() first.")
            pi = self.solver.pi
            policy_name = "Learned Policy (π_FOGAS)"
            
        print(f"\n{'='*70}")
        print(f"  OPTIMAL PATH VISUALIZATION - {policy_name}")
        print(f"{'='*70}\n")
        print(f"Initial State: {mdp.states[mdp.x0]}")
        if goal_state is not None:
            print(f"Goal State: {mdp.states[goal_state]}")
        print(f"Discount Factor (γ): {mdp.gamma}")
        print(f"\n{'-'*70}\n")
        
        for traj_idx in range(num_trajectories):
            current_seed = seed + traj_idx if seed is not None else None
            trajectory = self.simulate_trajectory(
                pi=pi, 
                max_steps=max_steps, 
                seed=current_seed,
                goal_state=goal_state
            )
            
            if num_trajectories > 1:
                print(f"\n  --- Trajectory {traj_idx + 1} ---\n")
                
            discounted_return = 0.0
            
            for i, step in enumerate(trajectory):
                s, a, r, sp = step['state'], step['action'], step['reward'], step['next_state']
                
                state_name = mdp.states[s]
                action_name = mdp.actions[a]
                next_state_name = mdp.states[sp]
                
                discount_factor = (mdp.gamma ** i)
                discounted_return += discount_factor * r
                
                step_str = f"Step {i:3d}"
                state_str = f"State: {state_name}"
                action_str = f"Action: {action_name}"
                reward_str = f"Reward: {r:7.3f}"
                next_str = f"→ {next_state_name}"
                
                indicators = ""
                if step.get('was_self_loop', False):
                    indicators += " ⚠️ SELF-LOOP"
                if step.get('reached_goal', False):
                    indicators += " 🎯 GOAL REACHED!"
                
                print(f"  {step_str} │ {state_str:15s} │ {action_str:15s} │ {reward_str} │ {next_str}{indicators}")
                
                if show_probabilities and i < show_probabilities_first_n:
                    print(f"           │ Policy at state {s}:")
                    for act_idx in range(mdp.A):
                        prob = pi[s, act_idx]
                        if isinstance(prob, torch.Tensor):
                            prob = prob.item()
                        bar_len = int(prob * 20)
                        bar = "█" * bar_len + "░" * (20 - bar_len)
                        print(f"           │   π(a={act_idx}|s={s}) = {prob:.3f} {bar}")
                    print(f"           │")
                    
            final_state = trajectory[-1]['next_state']
            
            print(f"\n  {'─'*66}")
            print(f"  Trajectory Length: {len(trajectory)} steps")
            print(f"  Discounted Return: {discounted_return:.6f}")
            print(f"  Final State: {mdp.states[final_state]}")
            
        print(f"\n{'-'*70}")
        
        v_pi, _ = mdp.evaluate_policy(pi)
        v_x0 = v_pi[mdp.x0]
        if isinstance(v_x0, torch.Tensor):
            v_x0 = v_x0.item()
        print(f"  Expected Return (from V): {v_x0:.6f}")
        
        v_star = mdp.v_star[mdp.x0]
        if isinstance(v_star, torch.Tensor):
            v_star = v_star.item()
        print(f"  Optimal Return (π*): {v_star:.6f}")
        
        print(f"\n{'='*70}\n")
