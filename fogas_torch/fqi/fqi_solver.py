import torch
import random
import numpy as np
from tqdm import trange

from ..algorithm.fogas_dataset import FOGASDataset


class FQISolver:
    """
    Fitted Q Iteration implementation.
    
    Update rule:
      1. target_i = r_i + gamma * max_a' Q(x'_i, a'; theta_k)
      2. theta_{k+1}^+ = argmin_theta sum_i (target_i - Q(x_i, a_i; theta))^2
                       = (Phi^T Phi)^(-1) Phi^T Y
      3. theta_{k+1} = tau * theta_k + (1 - tau) * theta_{k+1}^+
      
    Where Q(s, a; theta) = phi(s, a)^T theta.
    """

    def __init__(
        self,
        mdp,
        csv_path,
        gamma=None,
        ridge=1e-6,
        dataset_verbose=False,
        seed=42,
        device=None,
    ):
        self.mdp = mdp
        # Allow overriding gamma, otherwise take from MDP
        self.gamma = gamma if gamma is not None else mdp.gamma
        self.ridge = ridge
        self.seed = seed

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Set random seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        # Move MDP to device
        # Note: We assume mdp has a .to() method (LinearMDP does)
        if hasattr(self.mdp, 'to'):
            self.mdp.to(self.device)

        # Global MDP info
        self.A = mdp.A
        self.d = mdp.d
        self.phi = mdp.phi  # Function phi(s, a)

        # ------------------------------
        # Dataset
        # ------------------------------
        self.dataset = FOGASDataset(csv_path=csv_path, verbose=dataset_verbose)
        self.n = self.dataset.n

        # Move dataset tensors to device
        self.Xs = self.dataset.X.to(self.device)         # (n,)
        self.As = self.dataset.A.to(self.device)         # (n,)
        self.Rs = self.dataset.R.to(self.device)         # (n,)
        self.X_nexts = self.dataset.X_next.to(self.device) # (n,)

        # ------------------------------
        # Done mask (terminal next-state)
        # ------------------------------
        if hasattr(self.mdp, "terminal_states"):
            terminal_list = list(self.mdp.terminal_states)
        elif hasattr(self.mdp, "T"):
            terminal_list = list(self.mdp.T)
        elif hasattr(self.mdp, "terminal_set"):
            terminal_list = list(self.mdp.terminal_set)
        else:
            raise AttributeError("MDP must expose terminal states (e.g., mdp.T or mdp.terminal_states).")

        terminal_states = torch.tensor(terminal_list, device=self.device)
        self.done = torch.isin(self.X_nexts, terminal_states).to(torch.float64)  # (n,)

        # ------------------------------
        # Filter terminal-current samples
        # ------------------------------
        self.is_terminal_s = torch.isin(self.Xs, terminal_states)
        keep = ~self.is_terminal_s

        self.Xs = self.Xs[keep]
        self.As = self.As[keep]
        self.Rs = self.Rs[keep]
        self.X_nexts = self.X_nexts[keep]
        self.done = self.done[keep]
        self.n = self.Xs.shape[0]

        # ------------------------------
        # Precompute Features
        # ------------------------------
        self._precompute_dataset_features()
        self._build_regression_solver()
        
        # Results
        self.theta_history = []
        self.final_theta = None
        self.pi = None

    def _precompute_dataset_features(self):
        """
        1. Phi: Features for (x_i, a_i). Shape (n, d).
        2. Phi_next_all: Features for (x'_i, a') for all a'. Shape (n, A, d).
           Used for efficiently computing max_a' Q(x'_i, a').
        """
        n, d, A = self.n, self.d, self.A
        phi = self.phi
        device = self.device

        # 1. Phi for current state-actions (x_i, a_i)
        Phi_list = [
            phi(int(x.item()), int(a.item())).to(dtype=torch.float64, device=device)
            for x, a in zip(self.Xs, self.As)
        ]
        self.Phi = torch.vstack(Phi_list)  # (n, d)

        # 2. Phi for next states (x'_i, for all actions)
        # We need this to compute max_a' Q(x'_i, a') = max_a' (theta^T phi(x'_i, a'))
        # Storing as (n, A, d) tensor.
        Phi_next_list = []
        for x_next in self.X_nexts:
            x_next_val = int(x_next.item())
            # For each action, compute feature
            feats = [
                phi(x_next_val, a).to(dtype=torch.float64, device=device)
                for a in range(A)
            ]
            Phi_next_list.append(torch.stack(feats)) # (A, d)
        
        self.Phi_next_all = torch.stack(Phi_next_list) # (n, A, d)

    def _build_regression_solver(self):
        """
        Precompute the matrix M = (Phi^T Phi + ridge*I)^(-1) Phi^T
        Then theta_new = M @ targets
        """
        d = self.d
        n = self.n
        ridge = self.ridge
        
        Phi = self.Phi # (n, d)
        
        # Regularized Gram matrix: Phi^T Phi + lambda*I
        Gram = Phi.T @ Phi + ridge * torch.eye(d, dtype=torch.float64, device=self.device)
        
        # Inverse
        Gram_inv = torch.linalg.inv(Gram)
        
        # Projection matrix M: (d, d) @ (d, n) -> (d, n)
        self.M = Gram_inv @ Phi.T

    def run(self, K=100, tau=0.1, theta_init=None, verbose=False):
        """
        Run Algorithm 9.
        
        Args:
            K (int): Number of iterations
            tau (float): Step size / Soft update parameter (tau * theta_old + (1-tau) * theta_new)
                         Note: The prompt algorithm says: theta_{k+1} = tau * theta_k + (1 - tau) * theta_{k+1}^+
            theta_init (torch.Tensor): Initial theta (d,). If None, starts at 0.
        """
        d = self.d
        device = self.device
        
        if theta_init is None:
            theta = torch.zeros(d, dtype=torch.float64, device=device)
        else:
            theta = theta_init.clone().to(dtype=torch.float64, device=device)
            
        params_history = []
        
        iterator = trange(K, desc="FQI", disable=not verbose)
        
        for k in iterator:
            # 1. Compute Regression Targets
            # Q(x'_i, a') for all a': (n, A, d) @ theta -> (n, A)
            Q_next_all = torch.einsum('nad,d->na', self.Phi_next_all, theta)
            
            # max_a' Q(x'_i, a') -> (n,)
            max_Q_next, _ = torch.max(Q_next_all, dim=1)
            
            # Target y_i = r_i + gamma * (1 - done_i) * max_Q_next
            targets = self.Rs + self.gamma * (1.0 - self.done) * max_Q_next
            
            # 2. Least Squares Solution (theta_{k+1}^+)
            # theta_plus = M @ targets
            theta_plus = self.M @ targets
            
            # 3. Soft Update
            # theta_{k+1} = tau * theta_k + (1 - tau) * theta_plus
            theta_next = tau * theta + (1 - tau) * theta_plus
            
            # Update
            theta = theta_next
            
            # Store
            params_history.append(theta.clone())
            
            if verbose and (k % 10 == 0):
                norm = torch.linalg.norm(theta).item()
                iterator.set_postfix(theta_norm=f"{norm:.4f}")
        
        self.final_theta = theta
        self.theta_history = params_history
        self.pi = self.get_policy_matrix(theta)
        
        # Return a policy function wrapper
        return self.get_greedy_policy(theta)

    def get_greedy_policy(self, theta=None):
        """
        Returns a function pi(x) -> probabilities (one-hot greedy)
        or can return a full (N, A) matrix if needed.
        """
        if theta is None:
            theta = self.final_theta
            
        def policy_fn(state_idx):
            state_idx = int(state_idx)
            # Compute Q(s, a) for all a
            q_values = []
            for a in range(self.A):
                feat = self.phi(state_idx, a).to(dtype=torch.float64, device=self.device)
                q = torch.dot(theta, feat)
                q_values.append(q)
            q_values = torch.stack(q_values)
            
            # Greedy
            best_a = torch.argmax(q_values)
            probs = torch.zeros(self.A, dtype=torch.float64, device=self.device)
            probs[best_a] = 1.0
            return probs
            
        return policy_fn
    
    def get_policy_matrix(self, theta=None):
        """
        Returns the (N, A) policy matrix for the greedy policy w.r.t theta.
        """
        if theta is None:
            theta = self.final_theta
            
        N = self.mdp.N
        A = self.A
        pi_mat = torch.zeros((N, A), dtype=torch.float64, device=self.device)
        
        for x in range(N):
            # Compute Q(x, :)
            q_vals = torch.zeros(A, dtype=torch.float64, device=self.device)
            for a in range(A):
                feat = self.phi(x, a).to(dtype=torch.float64, device=self.device)
                q_vals[a] = torch.dot(theta, feat)
            
            best_a = torch.argmax(q_vals)
            pi_mat[x, best_a] = 1.0
            
        return pi_mat

    @property
    def theta_bar_history(self):
        """Alias for compatibility with Evaluators."""
        return self.theta_history
    
    @property
    def mod_alpha(self):
        """FQI doesn't use alpha, but Evaluator expects it for some metrics."""
        return 1.0
