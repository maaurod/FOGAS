import torch
import random
import numpy as np

from ..algorithm.fogas_dataset import FOGASDataset
from ..algorithm.fogas_parameters import FOGASParameters
from tqdm import trange


class FOGASSolverPolicy:
    """
    FOGAS implementation: runs the optimization algorithm, stores θ̄-history and final π.
    Evaluation utilities are moved to FOGASEvaluator.
    Supports CUDA acceleration.
    """

    def __init__(
        self,
        mdp,
        csv_path,
        csv_path_omega=None,
        delta=0.05,
        T=None,
        alpha=None,
        eta=None,
        rho=None,
        D_theta=None,
        print_params=False,
        dataset_verbose=False,
        seed=42,
        device=None,
    ):
        self.mdp = mdp
        self.csv_path = csv_path
        self.csv_path_omega = csv_path_omega if csv_path_omega is not None else csv_path
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

        # Move MDP to device
        self.mdp.to(self.device)

        # ------------------------------
        # Dataset
        # ------------------------------
        self.dataset = FOGASDataset(csv_path=csv_path, verbose=dataset_verbose)
        self.Xs = self.dataset.X.to(self.device)
        self.As = self.dataset.A.to(self.device)
        self.Rs = self.dataset.R.to(self.device)
        self.X_nexts = self.dataset.X_next.to(self.device)
        self.n = self.dataset.n

        # ------------------------------
        # MDP info
        # ------------------------------
        self.N = mdp.N
        self.A = mdp.A
        self.d = mdp.d
        self.gamma = mdp.gamma
        self.R = mdp.R
        self.phi = mdp.phi
        
        # Handle omega: use from MDP if available, otherwise estimate from dataset
        if mdp.omega is not None:
            self.omega = mdp.omega.to(self.device) if isinstance(mdp.omega, torch.Tensor) else torch.tensor(mdp.omega, dtype=torch.float64, device=self.device)
        else:
            if print_params:
                print("MDP omega not provided. Estimating from dataset...")
            self.omega = None
        self.x0 = mdp.x0

        # ------------------------------
        # Theoretical parameters
        # ------------------------------
        if print_params:
            print(f"\nDevice: {self.device}")
            print(f"Dataset: {csv_path} (n={self.n})")

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
        self.rho = self.params.rho
        self.D_theta = self.params.D_theta
        self.beta = self.params.beta
        self.D_pi = self.params.D_pi

        # Build covariance matrices
        self._build_covariances()

        # Estimate omega if needed
        if self.omega is None:
            self._estimate_omega()
            if print_params:
                print(f"Estimated omega (first 5 components): {self.omega[:5]}")

        # Results to be filled by run()
        self.theta_bar_history = None
        self.pi = None
        self.mod_alpha = self.alpha
        self.beta_T = None
        self.lambda_T = None
        self.policy_variant = None
        self.policy_objective_before_history = None
        self.policy_objective_after_history = None
        self.policy_grad_norm_history = None
        self.policy_cosine_similarity_history = None

    # ------------------------------------------------------------------
    # Covariance
    # ------------------------------------------------------------------
    def _build_covariances(self):
        n = self.n
        d = self.d
        beta = self.beta
        phi = self.phi
        Xs = self.Xs
        As = self.As

        Phi_list = [phi(int(x.item()), int(a.item())).to(dtype=torch.float64, device=self.device) 
                    for x, a in zip(Xs, As)]
        Phi = torch.vstack(Phi_list)
        Cov_emp = beta * torch.eye(d, dtype=torch.float64, device=self.device) + (Phi.T @ Phi) / n
        Cov_emp_inv = torch.linalg.inv(Cov_emp)

        self.Phi = Phi
        self.Cov_emp = Cov_emp
        self.Cov_emp_inv = Cov_emp_inv

    # ------------------------------------------------------------------
    # Estimate omega from dataset
    # ------------------------------------------------------------------
    def _estimate_omega(self):
        """
        Estimate omega from the omega dataset using regularized least squares.
        """
        # If the paths are the same, use precomputed tensors to save time
        if self.csv_path_omega == self.csv_path:
            n = self.n
            Phi = self.Phi
            R = self.Rs
            Cov_inv = self.Cov_emp_inv
        else:
            # Load the second dataset
            ds_omega = FOGASDataset(csv_path=self.csv_path_omega, verbose=False)
            Xs_o = ds_omega.X.to(self.device)
            As_o = ds_omega.A.to(self.device)
            R = ds_omega.R.to(self.device)
            n = ds_omega.n
            
            # Compute features for this dataset
            Phi_list = [self.phi(int(x.item()), int(a.item())).to(dtype=torch.float64, device=self.device) 
                        for x, a in zip(Xs_o, As_o)]
            Phi = torch.vstack(Phi_list)
            
            # Compute local covariance for estimation
            Cov = self.beta * torch.eye(self.d, dtype=torch.float64, device=self.device) + (Phi.T @ Phi) / n
            Cov_inv = torch.linalg.inv(Cov)
        
        # omega_hat = Cov_inv @ (Phi^T @ R / n)
        sum_phi_r = (Phi.T @ R) / n
        self.omega = Cov_inv @ sum_phi_r

    # ------------------------------------------------------------------
    # Softmax policy
    # ------------------------------------------------------------------
    def softmax_policy(self, w, alpha, return_matrix=False):
        """
        Return the softmax-linear policy induced by w.
        """
        pi_matrix = self.softmax_policy_matrix_from_w(w, alpha)

        if return_matrix:
            return pi_matrix

        def pi(x):
            return pi_matrix[int(x)]

        return pi

    def softmax_policy_matrix_from_w(self, w, alpha):
        """
        Return policy matrix pi_w of shape (N, A), where
        pi_w[x, a] = softmax_a(alpha * <phi(x,a), w>).
        Differentiable w.r.t. w.
        """
        N, A = self.N, self.A
        device = self.device

        pi = torch.zeros((N, A), dtype=torch.float64, device=device)
        for x in range(N):
            logits = []
            for a in range(A):
                phi_xa = self.phi(x, a).to(dtype=torch.float64, device=device)
                logits.append(alpha * torch.dot(phi_xa, w))
            logits = torch.stack(logits)
            pi[x] = torch.softmax(logits, dim=0)
        return pi
    
    def policy_objective(self, w, theta_t, beta_t, alpha):
        """
        J_t(w) = (gamma/n) sum_i u_beta(X_i,A_i) * sum_a pi_w(a|X'_i) Q_theta(X'_i,a)
        """
        device = self.device
        gamma = self.gamma
        n = self.n
        A = self.A
        Phi = self.Phi
        X_nexts = self.X_nexts
        phi = self.phi

        pi_matrix = self.softmax_policy_matrix_from_w(w, alpha)  # (N, A)

        obj = torch.tensor(0.0, dtype=torch.float64, device=device)

        for i in range(n):
            x_next = int(X_nexts[i].item())

            # u_beta(X_i, A_i)
            u_i = torch.dot(Phi[i], beta_t)

            # V_theta^pi(X'_i) = sum_a pi(a|x'_i) Q_theta(x'_i,a)
            v_i = torch.tensor(0.0, dtype=torch.float64, device=device)
            for a in range(A):
                phi_xa = phi(x_next, a).to(dtype=torch.float64, device=device)
                q_xa = torch.dot(theta_t, phi_xa)
                v_i = v_i + pi_matrix[x_next, a] * q_xa

            obj = obj + u_i * v_i

        return (gamma / n) * obj
    
    def policy_gradient_step(self, w_t, theta_t, beta_t, alpha, eta_pi, K_pi=1):
        """
        Perform K_pi gradient ascent steps on the policy objective J_t(w).
        """
        w = w_t.clone().detach().to(self.device)
        w.requires_grad_(True)

        for _ in range(K_pi):
            J = self.policy_objective(w, theta_t, beta_t, alpha)
            grad_w = torch.autograd.grad(J, w)[0] # automatic differentiation to get gradient([0] to take element 0)

            with torch.no_grad():
                w += eta_pi * grad_w

            w.requires_grad_(True)

        return w.detach()

    def policy_gradient_info(self, w, theta_t, beta_t, alpha):
        """
        Return J_t(w), grad_w J_t(w), ||grad||, and cosine(theta_t, grad_w).
        """
        w_local = w.clone().detach().to(self.device)
        w_local.requires_grad_(True)

        J = self.policy_objective(w_local, theta_t, beta_t, alpha)
        grad_w = torch.autograd.grad(J, w_local)[0].detach()
        grad_norm = torch.linalg.norm(grad_w)
        theta_norm = torch.linalg.norm(theta_t)

        if grad_norm < 1e-12 or theta_norm < 1e-12:
            cosine = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        else:
            cosine = torch.dot(theta_t, grad_w) / (theta_norm * grad_norm)

        return J.detach(), grad_w, grad_norm.detach(), cosine.detach()

    # ------------------------------------------------------------------
    # RUN FOGAS
    # ------------------------------------------------------------------
    def run(
        self,
        T=None,
        alpha=None,
        eta=None,
        rho=None,
        D_theta=None,
        beta_init=None,
        theta_bar_init=None,
        w_init=None,
        eta_pi=1e-2,
        K_pi=1,
        policy_variant="gradient",
        tau_pi=0.5,
        check_cosine_similarity=True,
        print_policies=False,
        verbose=False,
        tqdm_print=False
    ):
        # -------------------------
        # Override parameters
        # -------------------------
        T = self.params.T if T is None else T
        alpha = self.params.alpha if alpha is None else alpha
        eta = self.params.eta if eta is None else eta
        rho = self.params.rho if rho is None else rho
        D_theta = self.params.D_theta if D_theta is None else D_theta

        self.mod_alpha = alpha  # store alpha used

        Cov_emp, Cov_emp_inv = self.Cov_emp, self.Cov_emp_inv
        Phi = self.Phi
        phi = self.phi
        gamma = self.gamma
        omega = self.omega
        x0 = self.x0
        n = self.n
        d = self.d
        A = self.A
        X_nexts = self.X_nexts

        # -------------------------
        # Initialization
        # -------------------------
        device = self.device
        if theta_bar_init is not None and w_init is not None:
            raise ValueError("Provide at most one of theta_bar_init and w_init.")

        initial_w = w_init if w_init is not None else theta_bar_init
        beta_t = torch.zeros(d, dtype=torch.float64, device=device) if beta_init is None else beta_init.clone().to(device)
        w_t = torch.zeros(d, dtype=torch.float64, device=device) if initial_w is None else initial_w.clone().to(device)
        w_history = []
        policy_objective_before_history = []
        policy_objective_after_history = []
        policy_grad_norm_history = []
        policy_cosine_similarity_history = []
        pi_t = self.softmax_policy(w_t, alpha)

        if policy_variant in (0, "mirror", "theta"):
            policy_variant_name = "mirror"
        elif policy_variant in (1, "gradient", "grad"):
            policy_variant_name = "gradient"
        elif policy_variant in (2, "hybrid", "blend"):
            policy_variant_name = "hybrid"
        else:
            raise ValueError(
                "policy_variant must be one of {0, 1, 2, 'mirror', 'gradient', 'hybrid'}."
            )

        # -------------------------
        # Main loop
        # -------------------------
        use_tqdm = not verbose and not print_policies and tqdm_print
        iterator = trange(T, desc="FOGAS", disable=not use_tqdm)

        for t in iterator:

            # ---------------------------
            # μ̂ term
            # ---------------------------
            lambda_emp_sum1 = (1 - gamma) * sum(
                pi_t(x0)[a] * phi(x0, a).to(dtype=torch.float64, device=device) for a in range(A)
            )

            lambda_emp_sum2 = torch.zeros(d, dtype=torch.float64, device=device)

            for i in range(n):
                coeff = Phi[i] @ beta_t
                inner = sum(pi_t(int(X_nexts[i].item()))[a] * phi(int(X_nexts[i].item()), a).to(dtype=torch.float64, device=device)
                            for a in range(A))
                lambda_emp_sum2 += coeff * inner
            lambda_emp_sum2 *= gamma / n

            emp_feature_occupancy = lambda_emp_sum1 + lambda_emp_sum2

            # ---------------------------
            # θ_t update
            # ---------------------------
            c_t = emp_feature_occupancy - (Cov_emp @ beta_t)
            norm_c = torch.linalg.norm(c_t)
            theta_t = torch.zeros_like(c_t) if norm_c < 1e-12 else -D_theta * c_t / norm_c

            # ---------------------------
            # Ψ̂ v term
            # ---------------------------
            sum_term = torch.zeros(d, dtype=torch.float64, device=device)
            for i in range(n):
                probs = pi_t(int(X_nexts[i].item()))
                v = sum(
                    probs[a] * torch.dot(theta_t, phi(int(X_nexts[i].item()), a).to(dtype=torch.float64, device=device))
                    for a in range(A)
                )
                sum_term += Phi[i] * v

            Psi_hat_v = (1 / n) * (Cov_emp_inv @ sum_term)

            # ---------------------------
            # λ update
            # ---------------------------
            g = omega + gamma * Psi_hat_v - theta_t
            beta_t = (1 / (1 + rho * eta)) * (beta_t + eta * g)

            # ---------------------------
            # Policy update
            # ---------------------------
            J_before, grad_w, grad_norm, cosine_similarity = self.policy_gradient_info(
                w=w_t,
                theta_t=theta_t,
                beta_t=beta_t,
                alpha=alpha,
            )

            if policy_variant_name == "mirror":
                w_t = w_t + theta_t
            elif policy_variant_name == "gradient":
                w_t = self.policy_gradient_step(
                    w_t=w_t,
                    theta_t=theta_t,
                    beta_t=beta_t,
                    alpha=alpha,
                    eta_pi=eta_pi,
                    K_pi=K_pi,
                )
            else:
                w_candidate = w_t.clone()
                for _ in range(K_pi):
                    _, grad_hybrid, _, _ = self.policy_gradient_info(
                        w=w_candidate,
                        theta_t=theta_t,
                        beta_t=beta_t,
                        alpha=alpha,
                    )
                    w_candidate = (
                        w_candidate
                        + tau_pi * theta_t
                        + (1.0 - tau_pi) * eta_pi * grad_hybrid
                    )
                w_t = w_candidate

            J_after = self.policy_objective(w_t, theta_t, beta_t, alpha).detach()

            w_history.append(w_t.clone())
            policy_objective_before_history.append(J_before)
            policy_objective_after_history.append(J_after)
            policy_grad_norm_history.append(grad_norm)
            if check_cosine_similarity:
                policy_cosine_similarity_history.append(cosine_similarity)
            pi_t = self.softmax_policy(w_t, alpha)

            if print_policies and (t % max(1, T // 10) == 0):
                print(f"\nIteration {t+1}")
                pi_matrix = self.softmax_policy(w_t, alpha, return_matrix=True)
                self.mdp.print_policy(pi_matrix)

            if verbose and (t % max(1, T // 10) == 0):
                print(f"\n[FOGAS] Iter {t+1}/{T}")
                print(f"  θ_t     = {theta_t}")
                print(f"  ||θ_t|| = {torch.linalg.norm(theta_t).item():.3e}")
                print(f"  β_t     = {beta_t}")
                print(f"  ||β_t|| = {torch.linalg.norm(beta_t).item():.3e}")
                print(f"  J_before = {J_before.item():.3e}")
                print(f"  J_after  = {J_after.item():.3e}")
                print(f"  ||∇J||   = {grad_norm.item():.3e}")
                if check_cosine_similarity:
                    print(f"  cos(θ,∇J) = {cosine_similarity.item():.3e}")

        self.theta_bar_history = w_history
        self.pi = self.softmax_policy(w_t, alpha, return_matrix=True)
        self.beta_T = beta_t
        self.lambda_T = Cov_emp @ beta_t
        self.policy_variant = policy_variant_name
        self.policy_objective_before_history = policy_objective_before_history
        self.policy_objective_after_history = policy_objective_after_history
        self.policy_grad_norm_history = policy_grad_norm_history
        self.policy_cosine_similarity_history = (
            policy_cosine_similarity_history if check_cosine_similarity else None
        )

        return self.pi
