import numpy as np

from .fogas_dataset import FOGASDataset
from .fogas_parameters import FOGASParameters
from tqdm import trange


class FOGASSolver:
    """
    FOGAS implementation: runs the optimization algorithm, stores θ̄-history and final π.
    Evaluation utilities are moved to FOGASEvaluator.
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

        # Build covariance matrices
        self._build_covariances()

        # Results to be filled by run()
        self.theta_bar_history = None
        self.pi = None
        self.mod_alpha = self.alpha
        self.lambda_T = None

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

        Phi = np.stack([phi(int(x), int(a)) for x, a in zip(Xs, As)], axis=0)
        Cov_emp = beta * np.eye(d) + (Phi.T @ Phi) / n
        Cov_emp_inv = np.linalg.inv(Cov_emp)

        self.Phi = Phi
        self.Cov_emp = Cov_emp
        self.Cov_emp_inv = Cov_emp_inv

    # ------------------------------------------------------------------
    # Softmax policy
    # ------------------------------------------------------------------
    def softmax_policy(self, theta_bar, alpha, return_matrix=False):
        phi = self.phi
        A = self.A
        N = self.N

        def compute_probs(x):
            logits = np.array([alpha * np.dot(phi(x, a), theta_bar) for a in range(A)])
            exp_logits = np.exp(logits - np.max(logits))
            return exp_logits / exp_logits.sum()

        if not return_matrix:
            def pi(x):
                return compute_probs(x)
            return pi
        else:
            out = np.zeros((N, A))
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
        D_theta=None,
        lambda_init=None,
        theta_bar_init=None,
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
        lambda_t = np.zeros(d) if lambda_init is None else lambda_init.copy()
        theta_bar_t = np.zeros(d) if theta_bar_init is None else theta_bar_init.copy()
        theta_bar_history = []
        pi_t = lambda x: np.ones(A) / A  # start uniform

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
                pi_t(x0)[a] * phi(x0, a) for a in range(A)
            )

            Lambda_term = Cov_emp_inv @ lambda_t
            lambda_emp_sum2 = np.zeros(d)

            for i in range(n):
                coeff = np.dot(Phi[i], Lambda_term)
                inner = sum(pi_t(int(X_nexts[i]))[a] * phi(int(X_nexts[i]), a)
                            for a in range(A))
                lambda_emp_sum2 += coeff * inner
            lambda_emp_sum2 *= gamma / n

            emp_feature_occupancy = lambda_emp_sum1 + lambda_emp_sum2

            # ---------------------------
            # θ_t update
            # ---------------------------
            c_t = emp_feature_occupancy - lambda_t
            norm_c = np.linalg.norm(c_t)
            theta_t = np.zeros_like(c_t) if norm_c < 1e-12 else -D_theta * c_t / norm_c

            # ---------------------------
            # Ψ̂ v term
            # ---------------------------
            sum_term = np.zeros(d)
            for i in range(n):
                probs = pi_t(int(X_nexts[i]))
                v = sum(
                    probs[a] * np.dot(theta_t, phi(int(X_nexts[i]), a))
                    for a in range(A)
                )
                sum_term += Phi[i] * v

            Psi_hat_v = (1 / n) * (Cov_emp_inv @ sum_term)

            # ---------------------------
            # λ update
            # ---------------------------
            g = omega + gamma * Psi_hat_v - theta_t
            lambda_t = (1 / (1 + rho * eta)) * (lambda_t + eta * Cov_emp @ g)

            # ---------------------------
            # Policy update
            # ---------------------------
            theta_bar_t += theta_t
            theta_bar_history.append(theta_bar_t.copy())
            pi_t = self.softmax_policy(theta_bar_t, alpha)

            if print_policies and (t % max(1, T // 10) == 0):
                print(f"\nIteration {t+1}")
                pi_matrix = self.softmax_policy(theta_bar_t, alpha, return_matrix=True)
                self.mdp.print_policy(pi_matrix)

            if verbose and (t % max(1, T // 10) == 0):
                print(f"\n[FOGAS] Iter {t+1}/{T}")
                print(f"  θ_t     = {theta_t}")
                print(f"  ||θ_t|| = {np.linalg.norm(theta_t):.3e}")
                print(f"  λ_t     = {lambda_t}")
                print(f"  ||λ_t|| = {np.linalg.norm(lambda_t):.3e}")

    
        self.theta_bar_history = theta_bar_history
        self.pi = self.softmax_policy(theta_bar_t, alpha, return_matrix=True)
        self.lambda_T = lambda_t

        return pi_t
