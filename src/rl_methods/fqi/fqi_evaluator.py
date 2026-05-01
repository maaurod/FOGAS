
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

class FQIEvaluator:
    """
    Evaluator specific for FQI Solver.
    """

    def __init__(self, solver):
        self.solver = solver
        self.mdp = solver.mdp

    def final_reward(self):
        """Calculates the expected discounted return of the learned policy."""
        if self.solver.pi is None:
            raise ValueError("Run solver.run() first.")
        # policy_return returns (1-gamma) * value. 
        # But often we just want the value. FOGASEvaluator returns -1 * return for minimization.
        # Here we just return the raw expected return (positive)
        return self.mdp.policy_return(self.solver.pi)
    
    def converged(self, tol=1e-8):
        """
        Check if FQI has converged by comparing the last two theta values.
        
        Parameters
        ----------
        tol : float
            Convergence tolerance for change in theta.
            
        Returns
        -------
        bool
            True if converged, False otherwise.
        """
        if not hasattr(self.solver, 'theta_history') or len(self.solver.theta_history) < 2:
            return False
        
        theta_last = self.solver.theta_history[-1]
        theta_prev = self.solver.theta_history[-2]
        diff = torch.linalg.norm(theta_last - theta_prev).item()
        
        return diff < tol

    def compare_final_rewards(self):
        """
        Compare final policy reward against optimal policy reward.
        """
        if self.solver.pi is None:
            raise ValueError("Run solver.run() first.")

        mdp = self.mdp
        # Optimal return (pi*)
        J_star = mdp.policy_return(mdp.pi_star)
        # Learned return (pi_fqi)
        J_pi = mdp.policy_return(self.solver.pi)
        gap = J_star - J_pi

        print("\n========== FINAL REWARD COMPARISON ==========\n")
        print(f"J*(π*)   = {J_star:.6f}")
        print(f"J(π_FQI) = {J_pi:.6f}")
        print(f"Gap (J* − J) = {gap:.6e}")
        print("\n============================================\n")

    def print_policy(self):
        """Print the learned policy."""
        print("\n========== LEARNED POLICY (FQI) ==========")
        self.mdp.print_policy(self.solver.pi)
        print("==========================================\n")

    def compare_value_functions(self):
        """
        Compare learned Q values (from theta) vs Optimal Q*.
        Note: FQI learns Q(s,a) = theta^T phi(s,a).
        """
        if self.solver.final_theta is None:
            raise ValueError("Run solver.run() first.")
        
        mdp = self.mdp
        theta = self.solver.final_theta
        
        # Reconstruct Q_fqi table
        Q_fqi = torch.zeros((mdp.N, mdp.A), dtype=torch.float64, device=self.solver.device)
        for s in range(mdp.N):
            for a in range(mdp.A):
                phi_sa = mdp.phi(s, a).to(dtype=torch.float64, device=self.solver.device)
                Q_fqi[s, a] = torch.dot(theta, phi_sa)
        
        Q_star = mdp.q_star
        
        print("\n========== VALUE FUNCTION COMPARISON ==========\n")
        print("state | action | Q*(s,a)   | Q_fqi(s,a)|  diff")
        print("--------------------------------------------------")
        
        for s in range(mdp.N):
            for a in range(mdp.A):
                diff = (Q_star[s, a] - Q_fqi[s, a]).item()
                print(f"{s:5d} | {a:6d} | {Q_star[s,a].item():9.4f} | {Q_fqi[s,a].item():9.4f} | {diff:9.4f}")
                
        err = torch.linalg.norm(Q_star - Q_fqi).item()
        print(f"\nL2 Error ||Q* - Q_fqi|| = {err:.6f}")
        print("\n===============================================\n")

    def simulate_trajectory(self, max_steps=20, seed=None):
        """
        Simulate a trajectory using the learned policy.
        """
        if self.solver.pi is None:
            raise ValueError("Run solver.run() first.")
        
        # Use simple simulation logic (assuming deterministic policy for FQI usually)
        # But we can reuse the generic logic if we want.
        # Writing simple one here.
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        mdp = self.mdp
        s = mdp.x0
        traj = []
        
        for step in range(max_steps):
            # Pick action from pi
            # pi is (N, A)
            dist = self.solver.pi[s]
            a = torch.argmax(dist).item()
            
            # Transition
            # mdp.P is (N*A, N)
            # Row index = s*A + a
            row_idx = s * mdp.A + a
            probs = mdp.P[row_idx]
            
            # Sample next state
            s_next = torch.multinomial(probs, 1).item()
            r = mdp.r[row_idx].item()
            
            traj.append((s, a, r, s_next))
            
            s = s_next
        
        return traj

    def print_optimal_path(self, max_steps=10):
        """
        Print the path taken by the learned policy from x0.
        """
        print("\n========== OPTIMAL PATH (FQI Policy) ==========")
        traj = self.simulate_trajectory(max_steps=max_steps)
        
        total_r = 0.0
        
        # Get optimal path for comparison logic if needed, but let's just print FQI path
        for i, (s, a, r, sn) in enumerate(traj):
            print(f"Step {i}: State {s} -> Action {a} -> Reward {r:.4f} -> Next {sn}")
            total_r += (self.mdp.gamma ** i) * r
            
        print(f"Discounted Return (Simulated {max_steps} steps): {total_r:.4f}")
        print("===============================================")
