import numpy as np

from .fogas_parameters import FOGASParameters
from tqdm import trange


class FOGASOracleSolverVectorized:
    """
    Vectorized FOGAS Oracle implementation: precomputes PHI tensors and uses
    batch operations for policy computation and occupancy measures.
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
    ):
        self.mdp = mdp
        self.delta = delta
        self.seed = seed
        
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)

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
        self.omega = mdp.omega
        self.x0 = mdp.x0
        self.Phi = mdp.Phi

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
        PHI_XA = np.stack(
            [[self.phi(x, a) for a in range(self.A)] for x in range(self.N)],
            axis=0,
        )
        self.PHI_XA = PHI_XA

    # ------------------------------------------------------------------
    # Softmax policy (matrix)
    # ------------------------------------------------------------------
    @staticmethod
    def _row_softmax(logits):
        z = logits - logits.max(axis=1, keepdims=True)
        ez = np.exp(z)
        return ez / ez.sum(axis=1, keepdims=True)

    def softmax_policy(self, theta_bar, alpha, return_matrix=True):
        """
        Compute softmax policy in vectorized form.
        Returns: policy matrix of shape (N, A)
        """
        logits = alpha * np.tensordot(self.PHI_XA, theta_bar, axes=([2], [0]))
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
        iterator = trange(T, desc="FOGAS Oracle", disable=not use_tqdm)

        for t in iterator:

            # ---------------------------
            # π_t computation (vectorized)
            # ---------------------------
            pi_matrix = self.softmax_policy(theta_bar=theta_bar_t, alpha=alpha, return_matrix=True)

            # ---------------------------
            # μ̂ term (true occupancy measure)
            # ---------------------------
            occ_measure = self.mdp.occupancy_measure(pi_matrix)
            true_feature_occupancy = Phi.T @ occ_measure

            # ---------------------------
            # θ_t update
            # ---------------------------
            c_t = true_feature_occupancy - lambda_t
            norm_c = np.linalg.norm(c_t)
            theta_t = np.zeros_like(c_t) if norm_c < 1e-12 else -D_theta * c_t / norm_c

            # ---------------------------
            # Ψ̂ v term (vectorized)
            # ---------------------------
            # q_theta(s,a) = <theta, phi(s,a)>
            q = (Phi @ theta_t).reshape(N, A)  # (N,A)
            v_theta_pi = (pi_matrix * q).sum(axis=1)  # (N,)

            Psi_v = self.mdp.get_Psi() @ v_theta_pi  # (d,)

            # ---------------------------
            # λ update
            # ---------------------------
            g = omega + gamma * Psi_v - theta_t

            if cov_matrix == "identity":
                Lambda = np.eye(d)
            else:
                if cov_matrix == "cov_uniform":
                    mu = np.ones(N * A) / (N * A)
                elif cov_matrix == "cov_opt":
                    mu = self.mdp.mu_star
                elif cov_matrix == "cov_dynamic":
                    mu = occ_measure
                Lambda = self.beta * np.eye(d) + Phi.T @ (mu[:, None] * Phi)

            lambda_t = (1 / (1 + rho * eta)) * (lambda_t + eta * Lambda @ g)

            # ---------------------------
            # Policy update
            # ---------------------------
            theta_bar_t += theta_t
            theta_bar_history.append(theta_bar_t.copy())

            if print_policies and (t % max(1, T // 10) == 0):
                print(f"\nIteration {t+1}")
                self.mdp.print_policy(pi_matrix)

            if verbose and (t % max(1, T // 10) == 0):
                print(f"\n[FOGAS Oracle] Iter {t+1}/{T}")
                print(f"  θ_t     = {theta_t}")
                print(f"  ||θ_t|| = {np.linalg.norm(theta_t):.3e}")
                print(f"  λ_t     = {lambda_t}")
                print(f"  ||λ_t|| = {np.linalg.norm(lambda_t):.3e}")

        self.theta_bar_history = theta_bar_history
        self.pi = pi_matrix
        self.lambda_T = lambda_t

        return pi_matrix
