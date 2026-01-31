import torch
import random
import numpy as np

from .fogas_parameters import FOGASParameters
from tqdm import trange


class FOGASOracleSolver:
    """Oracle FOGAS solver with CUDA support."""
    
    def __init__(
        self,
        mdp,
        delta=0.05,
        n=None,
        T=None,
        alpha=None,
        eta=None,
        rho=None,
        D_theta=None,
        print_params=False,
        cov_matrix="identity",
        seed=42,
        device=None,
    ):
        self.mdp = mdp
        self.delta = delta
        self.seed = seed

        # Set device (CUDA if available)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Set random seed for reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            if torch.backends.mps.is_available():
                torch.mps.manual_seed(seed)

        self.n = 10e6 if n is None else n

        # ------------------------------
        # MDP info
        # ------------------------------
        self.N = mdp.N
        self.A = mdp.A
        self.d = mdp.d
        self.gamma = mdp.gamma
        self.R = mdp.R
        self.phi = mdp.phi
        self.omega = mdp.omega.to(self.device) if isinstance(mdp.omega, torch.Tensor) else torch.tensor(mdp.omega, dtype=torch.float64, device=self.device)
        self.x0 = mdp.x0
        self.Phi = mdp.Phi.to(self.device) if isinstance(mdp.Phi, torch.Tensor) else torch.tensor(mdp.Phi, dtype=torch.float64, device=self.device)

        # ------------------------------
        # Theoretical parameters
        # ------------------------------
        self.params = FOGASParameters(
            mdp=mdp,
            n=self.n,
            delta=delta,
            T=T,
            alpha=alpha,
            eta=eta,
            rho=rho,
            D_theta=D_theta,
            print_params=print_params,
        )

        self.T = self.params.T
        self.alpha = self.params.alpha
        self.eta = self.params.eta
        self.rho = self.params.rho if n is not None else 10e-3
        self.D_theta = self.params.D_theta
        self.beta = self.params.beta
        self.D_pi = self.params.D_pi

        # Results to be filled by run()
        self.theta_bar_history = None
        self.pi = None
        self.mod_alpha = self.alpha
        self.lambda_T = None

        self.cov_matrix = cov_matrix

    # ------------------------------------------------------------------
    # Softmax policy
    # ------------------------------------------------------------------
    def softmax_policy(self, theta_bar, alpha, return_matrix=False):
        phi = self.phi
        A = self.A
        N = self.N

        def compute_probs(x):
            logits = []
            for a in range(A):
                phi_val = phi(x, a).to(dtype=torch.float64)
                logits.append(alpha * torch.dot(phi_val, theta_bar))
            logits = torch.stack(logits)
            exp_logits = torch.exp(logits - torch.max(logits))
            return exp_logits / exp_logits.sum()

        if not return_matrix:
            def pi(x):
                return compute_probs(x)
            return pi
        else:
            out = torch.zeros((N, A), dtype=torch.float64)
            for x in range(N):
                out[x] = compute_probs(x)
            return out

    # ------------------------------------------------------------------
    # RUN FOGAS
    # ------------------------------------------------------------------
    def run(
        self,
        T=None,
        alpha=None,
        eta=None,
        rho=None,
        n=None,
        D_theta=None,
        lambda_init=None,
        theta_bar_init=None,
        print_policies=False,
        verbose=False,
        tqdm_print=False, 
        cov_matrix=None,
    ):
        # -------------------------
        # Override parameters
        # -------------------------
        T = self.params.T if T is None else T
        alpha = self.params.alpha if alpha is None else alpha
        eta = self.params.eta if eta is None else eta
        rho = self.params.rho if rho is None else rho
        D_theta = self.params.D_theta if D_theta is None else D_theta
        cov_matrix = self.cov_matrix if cov_matrix is None else cov_matrix

        self.mod_alpha = alpha  # store alpha used

        Phi = self.Phi
        gamma = self.gamma
        omega = self.omega
        d = self.d
        A = self.A
        N = self.N

        # -------------------------
        # Initialization
        # -------------------------
        lambda_t = torch.zeros(d, dtype=torch.float64) if lambda_init is None else lambda_init.clone()
        theta_bar_t = torch.zeros(d, dtype=torch.float64) if theta_bar_init is None else theta_bar_init.clone()
        theta_bar_history = []
        pi_t = lambda x: torch.ones(A, dtype=torch.float64) / A  # start uniform

        # -------------------------
        # Main loop
        # -------------------------
        use_tqdm = not verbose and not print_policies and tqdm_print
        iterator = trange(T, desc="FOGAS", disable=not use_tqdm)

        for t in iterator:

            # ---------------------------
            # μ̂ term
            # ---------------------------
            pi_matrix = self.softmax_policy(theta_bar=theta_bar_t, alpha=alpha, return_matrix=True)
            occ_measure = self.mdp.occupancy_measure(pi_matrix)

            true_feature_occupancy = Phi.T @ occ_measure

            # ---------------------------
            # θ_t update
            # ---------------------------
            c_t = true_feature_occupancy - lambda_t
            norm_c = torch.linalg.norm(c_t)
            theta_t = torch.zeros_like(c_t) if norm_c < 1e-12 else -D_theta * c_t / norm_c

            # ---------------------------
            # Ψ̂ v term
            # ---------------------------
            # q_theta(s,a) = <theta, phi(s,a)>
            q = (Phi @ theta_t).reshape(N, A)      # (N,A)
            v_theta_pi = (pi_matrix * q).sum(dim=1)                  # (N,)

            Psi_v = self.mdp.get_Psi() @ v_theta_pi                   # (d,)

            # ---------------------------
            # λ update
            # ---------------------------
            g = omega + gamma * Psi_v - theta_t
            
            if cov_matrix == "identity":
                Lambda = torch.eye(d, dtype=torch.float64)
            else:
                if cov_matrix == "cov_uniform":
                    mu = torch.ones(N * A, dtype=torch.float64) / (N * A)  # Fixed: added parentheses
                elif cov_matrix == "cov_opt":
                    mu = self.mdp.mu_star
                elif cov_matrix == "cov_dynamic":
                    mu = occ_measure
                Lambda = self.beta * torch.eye(d, dtype=torch.float64) + Phi.T @ (mu[:, None] * Phi)

            lambda_t = (1 / (1 + rho * eta)) * (lambda_t + eta * Lambda @ g)

            # ---------------------------
            # Policy update
            # ---------------------------
            theta_bar_t += theta_t
            theta_bar_history.append(theta_bar_t.clone())
            pi_t = self.softmax_policy(theta_bar_t, alpha)

            if print_policies and (t % max(1, T // 10) == 0):
                print(f"\nIteration {t+1}")
                pi_matrix = self.softmax_policy(theta_bar_t, alpha, return_matrix=True)
                self.mdp.print_policy(pi_matrix)

            if verbose and (t % max(1, T // 10) == 0):
                print(f"\n[FOGAS] Iter {t+1}/{T}")
                print(f"  θ_t     = {theta_t}")
                print(f"  ||θ_t|| = {torch.linalg.norm(theta_t).item():.3e}")
                print(f"  λ_t     = {lambda_t}")
                print(f"  ||λ_t|| = {torch.linalg.norm(lambda_t).item():.3e}")
    
        self.theta_bar_history = theta_bar_history
        self.pi = self.softmax_policy(theta_bar_t, alpha, return_matrix=True)
        self.lambda_T = lambda_t

        return pi_t

