import numpy as np

from .fogas_dataset import FOGASDataset
from .fogas_parameters import FOGASParameters
from tqdm import trange


class FOGASSolverVectorized:
    """
    Vectorized FOGAS implementation: precomputes PHI tensors and runs the
    core loop with batch operations.
    """

    def __init__(
        self,
        mdp,
        csv_path,
        delta=0.05,
        T=None,
        alpha=None,
        eta=None,
        rho=None,
        D_theta=None,
        print_params=False,
        dataset_verbose=False,
        seed=42,
    ):
        self.mdp = mdp
        self.delta = delta
        self.seed = seed
        
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # ------------------------------
        # Dataset
        # ------------------------------
        self.dataset = FOGASDataset(csv_path=csv_path, verbose=dataset_verbose)
        self.Xs = self.dataset.X
        self.As = self.dataset.A
        self.Rs = self.dataset.R
        self.X_nexts = self.dataset.X_next
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
        self.omega = mdp.omega
        self.x0 = mdp.x0

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
        self.rho = self.params.rho
        self.D_theta = self.params.D_theta
        self.beta = self.params.beta
        self.D_pi = self.params.D_pi

        # Precompute feature tensors and covariances
        self._build_feature_tensors()
        self._build_covariances()

        # Results to be filled by run()
        self.theta_bar_history = None
        self.pi = None
        self.mod_alpha = self.alpha
        self.lambda_T = None

    # ------------------------------------------------------------------
    # Feature tensors
    # ------------------------------------------------------------------
    def _build_feature_tensors(self):
        # PHI_XA[x, a] = phi(x, a)
        PHI_XA = np.stack(
            [[self.phi(x, a) for a in range(self.A)] for x in range(self.N)],
            axis=0,
        )

        # Dataset features via indexing: Phi_i = phi(X_i, A_i)
        Phi = PHI_XA[self.Xs.astype(int), self.As.astype(int)]

        self.PHI_XA = PHI_XA
        self.Phi = Phi

    # ------------------------------------------------------------------
    # Covariance
    # ------------------------------------------------------------------
    def _build_covariances(self):
        n = self.n
        d = self.d
        beta = self.beta
        Phi = self.Phi

        Cov_emp = beta * np.eye(d) + (Phi.T @ Phi) / n
        Cov_emp_inv = np.linalg.inv(Cov_emp)

        self.Cov_emp = Cov_emp
        self.Cov_emp_inv = Cov_emp_inv

    # ------------------------------------------------------------------
    # Softmax policy (matrix)
    # ------------------------------------------------------------------
    @staticmethod
    def _row_softmax(logits):
        z = logits - logits.max(axis=1, keepdims=True)
        ez = np.exp(z)
        return ez / ez.sum(axis=1, keepdims=True)

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
        lambda_init=None,
        theta_bar_init=None,
        print_policies=False,
        verbose=False,
        tqdm_print=False,
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

        N, A, d = self.N, self.A, self.d
        n = self.n
        gamma = self.gamma
        PHI_XA = self.PHI_XA
        Phi = self.Phi
        Xn = self.X_nexts.astype(int)
        x0 = int(self.x0)

        Cov = self.Cov_emp
        Cov_inv = self.Cov_emp_inv
        omega = self.omega

        # -------------------------
        # Initialization
        # -------------------------
        lambda_t = np.zeros(d) if lambda_init is None else lambda_init.copy()
        theta_bar_t = np.zeros(d) if theta_bar_init is None else theta_bar_init.copy()
        theta_bar_history = []

        # -------------------------
        # Main loop
        # -------------------------
        use_tqdm = not verbose and not print_policies and tqdm_print
        iterator = trange(T, desc="FOGAS", disable=not use_tqdm)

        for t in iterator:
            # ---------- Policy matrix π_t (N,A) ----------
            logits = alpha * np.tensordot(PHI_XA, theta_bar_t, axes=([2], [0]))
            pi_mat = self._row_softmax(logits)

            # E_phi_pi[x] = sum_a pi(x,a) * phi(x,a)
            E_phi_pi = (pi_mat[..., None] * PHI_XA).sum(axis=1)

            # ---------- μ̂ term ----------
            lambda_emp_sum1 = (1.0 - gamma) * E_phi_pi[x0]

            Lambda_term = Cov_inv @ lambda_t
            coeff = Phi @ Lambda_term
            inner = E_phi_pi[Xn]
            lambda_emp_sum2 = (gamma / n) * (coeff[:, None] * inner).sum(axis=0)
            emp_feature_occupancy = lambda_emp_sum1 + lambda_emp_sum2

            # ---------- θ_t update ----------
            c_t = emp_feature_occupancy - lambda_t
            norm_c = np.linalg.norm(c_t)
            theta_t = np.zeros_like(c_t) if norm_c < 1e-12 else -D_theta * c_t / norm_c

            # ---------- Ψ̂ v term ----------
            q = np.tensordot(PHI_XA[Xn], theta_t, axes=([2], [0]))
            v = (pi_mat[Xn] * q).sum(axis=1)
            sum_term = (Phi * v[:, None]).sum(axis=0)
            Psi_hat_v = (Cov_inv @ sum_term) / n

            # ---------- λ update ----------
            g = omega + gamma * Psi_hat_v - theta_t
            lambda_t = (1.0 / (1.0 + rho * eta)) * (lambda_t + eta * (Cov @ g))

            # ---------- θ̄ update ----------
            theta_bar_t += theta_t
            theta_bar_history.append(theta_bar_t.copy())

            if print_policies and (t % max(1, T // 10) == 0):
                print(f"\nIteration {t+1}")
                self.mdp.print_policy(pi_mat)

            if verbose and (t % max(1, T // 10) == 0):
                print(f"\n[FOGAS] Iter {t+1}/{T}")
                print(f"  θ_t     = {theta_t}")
                print(f"  ||θ_t|| = {np.linalg.norm(theta_t):.3e}")
                print(f"  λ_t     = {lambda_t}")
                print(f"  ||λ_t|| = {np.linalg.norm(lambda_t):.3e}")

        self.theta_bar_history = theta_bar_history
        self.pi = pi_mat
        self.lambda_T = lambda_t

        return pi_mat
