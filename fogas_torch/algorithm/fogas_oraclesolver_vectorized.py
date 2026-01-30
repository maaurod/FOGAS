import torch

from .fogas_parameters import FOGASParameters
from tqdm import trange


class FOGASOracleSolverVectorized:
    """
    Vectorized FOGAS Oracle implementation: precomputes PHI tensors and uses
    batch operations for policy computation and occupancy measures.
    Supports CUDA acceleration.
    """

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
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

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

        # Precompute feature tensors
        self._build_feature_tensors()

        # Results to be filled by run()
        self.theta_bar_history = None
        self.pi = None
        self.mod_alpha = self.alpha
        self.lambda_T = None

        self.cov_matrix = cov_matrix

    # ------------------------------------------------------------------
    # Feature tensors
    # ------------------------------------------------------------------
    def _build_feature_tensors(self):
        # PHI_XA[x, a] = phi(x, a)
        PHI_XA = torch.stack(
            [torch.stack([self.phi(x, a) for a in range(self.A)]) for x in range(self.N)],
            dim=0,
        ).to(dtype=torch.float64, device=self.device)
        self.PHI_XA = PHI_XA

    # ------------------------------------------------------------------
    # Softmax policy (matrix)
    # ------------------------------------------------------------------
    @staticmethod
    def _row_softmax(logits):
        z = logits - logits.max(dim=1, keepdim=True).values
        ez = torch.exp(z)
        return ez / ez.sum(dim=1, keepdim=True)

    def softmax_policy(self, theta_bar, alpha, return_matrix=True):
        """
        Compute softmax policy in vectorized form.
        Returns: policy matrix of shape (N, A)
        """
        logits = alpha * torch.tensordot(self.PHI_XA, theta_bar, dims=([2], [0]))
        pi_mat = self._row_softmax(logits)
        return pi_mat

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
        device = self.device

        # -------------------------
        # Initialization
        # -------------------------
        lambda_t = torch.zeros(d, dtype=torch.float64, device=device) if lambda_init is None else lambda_init.clone().to(device)
        theta_bar_t = torch.zeros(d, dtype=torch.float64, device=device) if theta_bar_init is None else theta_bar_init.clone().to(device)
        theta_bar_history = []

        # -------------------------
        # Main loop
        # -------------------------
        use_tqdm = not verbose and not print_policies and tqdm_print
        iterator = trange(T, desc="FOGAS Oracle", disable=not use_tqdm)

        for t in iterator:

            # ---------------------------
            # π_t computation (vectorized)
            # ---------------------------
            pi_matrix = self.softmax_policy(theta_bar=theta_bar_t, alpha=alpha, return_matrix=True)

            # ---------------------------
            # μ̂ term (true occupancy measure)
            # ---------------------------
            occ_measure = self.mdp.occupancy_measure(pi_matrix.cpu() if device.type == 'cuda' else pi_matrix)
            if isinstance(occ_measure, torch.Tensor):
                occ_measure = occ_measure.to(device)
            else:
                occ_measure = torch.tensor(occ_measure, dtype=torch.float64, device=device)
            true_feature_occupancy = Phi.T @ occ_measure

            # ---------------------------
            # θ_t update
            # ---------------------------
            c_t = true_feature_occupancy - lambda_t
            norm_c = torch.linalg.norm(c_t)
            theta_t = torch.zeros_like(c_t) if norm_c < 1e-12 else -D_theta * c_t / norm_c

            # ---------------------------
            # Ψ̂ v term (vectorized)
            # ---------------------------
            # q_theta(s,a) = <theta, phi(s,a)>
            q = (Phi @ theta_t).reshape(N, A)  # (N,A)
            v_theta_pi = (pi_matrix * q).sum(dim=1)  # (N,)

            Psi = self.mdp.get_Psi()
            if isinstance(Psi, torch.Tensor):
                Psi = Psi.to(device)
            else:
                Psi = torch.tensor(Psi, dtype=torch.float64, device=device)
            Psi_v = Psi @ v_theta_pi  # (d,)

            # ---------------------------
            # λ update
            # ---------------------------
            g = omega + gamma * Psi_v - theta_t

            if cov_matrix == "identity":
                Lambda = torch.eye(d, dtype=torch.float64, device=device)
            else:
                if cov_matrix == "cov_uniform":
                    mu = torch.ones(N * A, dtype=torch.float64, device=device) / (N * A)
                elif cov_matrix == "cov_opt":
                    mu_star = self.mdp.mu_star
                    if isinstance(mu_star, torch.Tensor):
                        mu = mu_star.to(device)
                    else:
                        mu = torch.tensor(mu_star, dtype=torch.float64, device=device)
                elif cov_matrix == "cov_dynamic":
                    mu = occ_measure
                Lambda = self.beta * torch.eye(d, dtype=torch.float64, device=device) + Phi.T @ (mu[:, None] * Phi)

            lambda_t = (1 / (1 + rho * eta)) * (lambda_t + eta * Lambda @ g)

            # ---------------------------
            # Policy update
            # ---------------------------
            theta_bar_t = theta_bar_t + theta_t
            theta_bar_history.append(theta_bar_t.clone())

            if print_policies and (t % max(1, T // 10) == 0):
                print(f"\nIteration {t+1}")
                self.mdp.print_policy(pi_matrix.cpu())

            if verbose and (t % max(1, T // 10) == 0):
                print(f"\n[FOGAS Oracle] Iter {t+1}/{T}")
                print(f"  θ_t     = {theta_t}")
                print(f"  ||θ_t|| = {torch.linalg.norm(theta_t).item():.3e}")
                print(f"  λ_t     = {lambda_t}")
                print(f"  ||λ_t|| = {torch.linalg.norm(lambda_t).item():.3e}")

        self.theta_bar_history = theta_bar_history
        self.pi = pi_matrix
        self.lambda_T = lambda_t

        return pi_matrix
