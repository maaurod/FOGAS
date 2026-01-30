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

    # =====================================================
    # --- Optimal Path Visualization ---
    # =====================================================

    def simulate_trajectory(self, pi=None, max_steps=100, seed=None, goal_state=None):
        """
        Simulate a single trajectory following a given policy.
        
        Parameters
        ----------
        pi : ndarray, optional
            Policy to follow. If None, uses the learned policy from solver.
        max_steps : int
            Maximum number of steps to simulate.
        seed : int, optional
            Random seed for reproducibility.
        goal_state : int, optional
            If provided, simulation stops when this state is reached.
            If None, runs for max_steps (no early termination).
            
        Returns
        -------
        trajectory : list of dict
            Each element contains: {'state': s, 'action': a, 'reward': r, 'next_state': sp}
        """
        if pi is None:
            if self.solver.pi is None:
                raise ValueError("Run solver.run() first or provide a policy.")
            pi = self.solver.pi
            
        if seed is not None:
            np.random.seed(seed)
            
        mdp = self.mdp
        state = mdp.x0
        trajectory = []
        
        for step in range(max_steps):
            # Sample action from policy
            action_probs = pi[state]
            action = np.argmax(action_probs)
            
            # Get reward
            state_action_idx = state * mdp.A + action
            reward = mdp.r[state_action_idx]
            
            # Sample next state
            transition_probs = mdp.P[state_action_idx]
            next_state = np.random.choice(mdp.N, p=transition_probs)
            
            # Record the step
            trajectory.append({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'step': step,
                'was_self_loop': (next_state == state),
                'reached_goal': (next_state == goal_state) if goal_state is not None else False
            })
            
            # Check for goal state (if specified)
            if goal_state is not None and next_state == goal_state:
                break
                
            state = next_state
            
        return trajectory

    def print_optimal_path(self, use_optimal=False, num_trajectories=1, 
                          max_steps=50, seed=42, show_probabilities=False,
                          show_probabilities_first_n=5, goal_state=None):
        """
        Display optimal path(s) from initial state in a pretty format.
        
        Parameters
        ----------
        use_optimal : bool
            If True, use the optimal policy π*. If False, use learned policy.
        num_trajectories : int
            Number of trajectory samples to display.
        max_steps : int
            Maximum steps per trajectory.
        seed : int
            Random seed for reproducibility.
        show_probabilities : bool
            If True, show action probabilities at each state.
        show_probabilities_first_n : int
            Number of initial steps to show policy probabilities for.
            Set to 0 to disable, or a large number to show for all steps.
        goal_state : int, optional
            If provided, simulation stops when this state is reached.
            If None, runs for max_steps without early termination.
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
        
        # Simulate multiple trajectories
        total_return = 0.0
        
        for traj_idx in range(num_trajectories):
            current_seed = seed + traj_idx if seed is not None else None
            trajectory = self.simulate_trajectory(
                pi=pi, 
                max_steps=max_steps, 
                seed=current_seed,
                goal_state=goal_state
            )
            
            if num_trajectories > 1:
                print(f"\n{'─'*70}")
                print(f"  Trajectory {traj_idx + 1} / {num_trajectories}")
                print(f"{'─'*70}\n")
            
            # Calculate discounted return for this trajectory
            discounted_return = 0.0
            
            # Print trajectory steps
            for i, step in enumerate(trajectory):
                s = step['state']
                a = step['action']
                r = step['reward']
                sp = step['next_state']
                
                state_name = mdp.states[s]
                action_name = mdp.actions[a]
                next_state_name = mdp.states[sp]
                
                # Accumulate discounted return
                discount_factor = (mdp.gamma ** i)
                discounted_return += discount_factor * r
                
                # Format step info
                step_str = f"Step {i:3d}"
                state_str = f"State: {state_name}"
                action_str = f"Action: {action_name}"
                reward_str = f"Reward: {r:7.3f}"
                next_str = f"→ {next_state_name}"
                
                # Add visual indicators
                indicators = ""
                if step.get('was_self_loop', False):
                    indicators += " ⚠️ SELF-LOOP"
                if step.get('reached_goal', False):
                    indicators += " 🎯 GOAL REACHED!"
                
                print(f"  {step_str} │ {state_str:15s} │ {action_str:15s} │ {reward_str} │ {next_str}{indicators}")
                
                # Show policy probabilities if requested (for first N steps)
                if show_probabilities and i < show_probabilities_first_n:
                    print(f"           │ Policy at state {state_name}:")
                    for action_idx, action_name in enumerate(mdp.actions):
                        prob = pi[s, action_idx]
                        bar_length = int(prob * 20)
                        bar = '█' * bar_length + '░' * (20 - bar_length)
                        print(f"           │   π(a={action_name}|s={state_name}) = {prob:.3f} {bar}")
                    print(f"           │")
                
            # Print trajectory summary
            total_return += discounted_return
            print(f"\n  {'─'*66}")
            print(f"  Trajectory Length: {len(trajectory)} steps")
            print(f"  Discounted Return: {discounted_return:.6f}")
            print(f"  Final State: {mdp.states[trajectory[-1]['next_state']]}")
            
        # Print overall summary
        if num_trajectories > 1:
            avg_return = total_return / num_trajectories
            print(f"\n{'-'*70}")
            print(f"  Average Discounted Return: {avg_return:.6f}")
            
        # Show expected return from value function
        v_pi, _ = mdp.evaluate_policy(pi)
        expected_return = (1.0 - mdp.gamma) * v_pi[mdp.x0]
        
        print(f"\n{'-'*70}")
        print(f"  Expected Return (from V): {expected_return:.6f}")
        
        if use_optimal:
            optimal_return = mdp.optimal_policy_return()
            print(f"  Optimal Return (π*): {optimal_return:.6f}")
        else:
            optimal_return = mdp.optimal_policy_return()
            learned_return = mdp.policy_return(pi)
            gap = optimal_return - learned_return
            print(f"  Optimal Return (π*): {optimal_return:.6f}")
            print(f"  Gap (J* - J): {gap:.6e}")
        
        print(f"\n{'='*70}\n")

    def debug_state_policy(self, state, use_optimal=False):
        """
        Debug what the policy is doing at a specific state.
        
        Parameters
        ----------
        state : int
            The state to inspect
        use_optimal : bool
            If True, inspect optimal policy. If False, inspect learned policy.
        """
        mdp = self.mdp
        
        if use_optimal:
            pi = mdp.pi_star
            v, q = mdp.v_star, mdp.q_star
            policy_name = "Optimal Policy (π*)"
        else:
            if self.solver.pi is None:
                raise ValueError("Run solver.run() first.")
            pi = self.solver.pi
            v, q = mdp.evaluate_policy(pi)
            policy_name = "Learned Policy (π_FOGAS)"
        
        print(f"\n{'='*70}")
        print(f"  STATE POLICY DIAGNOSTICS - {policy_name}")
        print(f"{'='*70}\n")
        print(f"Inspecting State: {state}")
        print(f"Value V(s={state}): {v[state]:.6f}\n")
        
        # Policy at this state
        print("Policy Probabilities:")
        for a in range(mdp.A):
            prob = pi[state, a]
            bar_length = int(prob * 30)
            bar = '█' * bar_length + '░' * (30 - bar_length)
            print(f"  π(a={a}|s={state}) = {prob:.4f} {bar}")
        
        # Q-values
        print(f"\nAction-Values (Q):")
        best_action = np.argmax(q[state])
        for a in range(mdp.A):
            marker = "← BEST" if a == best_action else ""
            print(f"  Q(s={state}, a={a}) = {q[state, a]:.6f} {marker}")
        
        # Transition probabilities for each action
        print(f"\nTransition Probabilities from State {state}:")
        for a in range(mdp.A):
            state_action_idx = state * mdp.A + a
            trans_probs = mdp.P[state_action_idx]
            reward = mdp.r[state_action_idx]
            
            print(f"\n  Action {a} (reward: {reward:.3f}):")
            
            # Show top 5 most likely next states
            top_indices = np.argsort(-trans_probs)[:5]
            for next_s in top_indices:
                if trans_probs[next_s] > 0.01:  # Only show if prob > 1%
                    self_loop = " ← SELF-LOOP!" if next_s == state else ""
                    print(f"    → State {next_s}: {trans_probs[next_s]:.4f}{self_loop}")
        
        # Check for self-loop trap - IMPROVED VERSION
        print(f"\n{'─'*70}")
        print("SELF-LOOP ANALYSIS:")
        
        # Find all actions that cause self-loops
        self_loop_actions = []
        for a in range(mdp.A):
            state_action_idx = state * mdp.A + a
            trans_probs = mdp.P[state_action_idx]
            if trans_probs[state] > 0.99:
                self_loop_actions.append(a)
        
        if self_loop_actions:
            print(f"  Actions that cause self-loops: {self_loop_actions}")
            
            # Calculate total probability of getting stuck
            total_self_loop_prob = sum(pi[state, a] for a in self_loop_actions)
            print(f"  Combined probability of self-loop: {total_self_loop_prob:.1%}")
            
            for a in self_loop_actions:
                print(f"    - Action {a}: π={pi[state, a]:.1%}, Q={q[state, a]:.6f}")
            
            if total_self_loop_prob > 0.5:
                print(f"\n  ⚠️  WARNING: State {state} has HIGH risk of self-loop!")
                print(f"  The policy assigns {total_self_loop_prob:.1%} probability to actions")
                print(f"  that keep the agent stuck in state {state}.")
                print(f"  Trajectory simulation will likely terminate here.")
            elif total_self_loop_prob > 0.0:
                print(f"\n  ⚠️  CAUTION: {total_self_loop_prob:.1%} chance of getting stuck in state {state}")
        else:
            print(f"  ✓ No deterministic self-loop actions.")
        
        # Check for policy-Q mismatch (best Q-value but low policy probability)
        print(f"\n{'─'*70}")
        print("POLICY-Q MISMATCH DETECTION:")
        
        best_q_action = np.argmax(q[state])
        best_q_value = q[state, best_q_action]
        best_q_policy_prob = pi[state, best_q_action]
        
        if best_q_policy_prob < 0.01:  # Less than 1% probability
            print(f"  ⚠️  CRITICAL: Best action (a={best_q_action}) has near-zero policy probability!")
            print(f"  Q(s={state}, a={best_q_action}) = {best_q_value:.6f} ← BEST")
            print(f"  π(a={best_q_action}|s={state}) = {best_q_policy_prob:.4f} ← NEARLY ZERO!")
            print(f"\n  This indicates the learned policy is NOT aligned with learned Q-values.")
            print(f"  Possible causes:")
            print(f"    - FOGAS didn't converge (need more iterations)")
            print(f"    - Learning rate (eta) issues")
            print(f"    - Feature representation problems")
        elif best_q_policy_prob < 0.3:
            print(f"  ⚠️  WARNING: Best action (a={best_q_action}) has low probability ({best_q_policy_prob:.1%})")
        else:
            print(f"  ✓ Policy generally aligned with Q-values.")
        
        print(f"\n{'='*70}\n")

