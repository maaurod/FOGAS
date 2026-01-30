import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

class FOGASHyperOptimizer:
    """
    Hyperparameter optimizer for FOGAS.
    Completely agnostic to the metric being optimized.
    """

    # ---------------------------
    # Default bounds (constant)
    # ---------------------------
    # Each entry:
    #   - "bounds": either a (low, high) tuple OR a callable(theory, current)->(low, high)
    #   - "log_scale": default log_scale for that parameter
    DEFAULT_BOUNDS = {
        "alpha": {"bounds": lambda theory, cur: (theory["alpha"], 5.0), "log_scale": True},
        "rho":   {"bounds": lambda theory, cur: (1e-2, 5.0),  "log_scale": True},
        "eta":   {"bounds": lambda theory, cur: (theory["eta"], 3.0),   "log_scale": True},
    }

    def __init__(self, solver, metric_callable, seed=42):
        """
        metric_callable: zero-argument function returning a scalar
        seed: random seed for reproducibility (default: 42)
        """
        self.solver = solver
        self.metric = metric_callable
        self.seed = seed

        # Set random seeds for reproducibility
        if seed is not None:
            import torch
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

    # =====================================================
    # Objective (metric-agnostic)
    # =====================================================
    def _evaluate(self, num_runs=3, **params):
        import torch  # Import here to avoid requiring torch if not using solver with tensors
        vals = []
        for _ in range(num_runs):
            self.solver.run(**params)
            metric_val = self.metric()
            # Handle torch tensors if solver returns them
            if isinstance(metric_val, torch.Tensor):
                vals.append(metric_val.item())
            else:
                vals.append(metric_val)
        return float(np.mean(vals))


    # =====================================================
    # Optimal search 
    # =====================================================
    def optimal_search(
        self,
        param_name,
        bounds,
        fixed_params=None,
        coarse_points=10,
        bo_iters=15,
        log_scale=False,
        search_method="random",
        random_candidates=15,
        print_each=False,
        return_history=False,
        num_runs=3,
    ):
        assert search_method in {"bo", "random"}
        fixed_params = fixed_params or {}
        a, b = bounds

        # ---------------- coarse sweep ----------------
        if log_scale:
            grid = np.logspace(np.log10(a), np.log10(b), coarse_points)
        else:
            grid = np.linspace(a, b, coarse_points)

        X, y = [], []
        for v in grid:
            params = dict(fixed_params)
            params[param_name] = v

            val = self._evaluate(num_runs=num_runs, **params)
            X.append(v)
            y.append(val)

            if print_each:
                print(f"{param_name}={v}, metric={val}")

        X = np.array(X)
        y = np.array(y)

        # ---------------- refinement ----------------
        idx = int(np.argmin(y))
        left = X[max(idx - 1, 0)]
        right = X[min(idx + 1, len(X) - 1)]
        if print_each:
            print(f"Refinement: from {left} to {right}")

        # ---------------- optimization ----------------
        if search_method == "bo":
            X, y = self._optimize_bo(
                param_name,
                left,
                right,
                X.reshape(-1, 1),
                y,
                fixed_params,
                bo_iters,
                print_each,
                num_runs
            )
        else:
            X, y = self._optimize_random(
                param_name,
                left,
                right,
                X.reshape(-1, 1),
                y,
                fixed_params,
                random_candidates,
                log_scale,
                print_each,
                num_runs
            )

        best_idx = int(np.argmin(y))
        best_param = float(X[best_idx][0])

        if return_history:
            return best_param, X.flatten(), y
        return best_param

    def _optimize_bo(
        self,
        param_name,
        left,
        right,
        X,
        y,
        fixed_params,
        bo_iters,
        print_each,
        num_runs,
    ):
        def expected_improvement(x, gp, y_best, xi=0.01):
            mu, sigma = gp.predict(x.reshape(-1, 1), return_std=True)
            sigma = np.maximum(sigma, 1e-9)
            Z = (y_best - mu - xi) / sigma
            return (
                (y_best - mu - xi) * norm.cdf(Z)
                + sigma * norm.pdf(Z)
            )

        kernel = (
            ConstantKernel(1.0)
            * Matern(length_scale=0.5, nu=2.5)
            + WhiteKernel(noise_level=0.05)
        )

        gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            optimizer=None,
        )

        for t in range(bo_iters):
            gp.fit(X, y)

            candidates = np.linspace(left, right, 300)
            ei = expected_improvement(
                candidates, gp, np.min(y)
            )

            x_next = float(candidates[np.argmax(ei)])

            params = dict(fixed_params)
            params[param_name] = x_next

            y_next = self._evaluate(num_runs=num_runs, **params)

            X = np.vstack([X, [[x_next]]])
            y = np.append(y, y_next)

            if print_each:
                print(
                    f"BO {t+1}: {param_name}={x_next:.4g}, metric={y_next:.4f}"
                )

        return X, y

    def _optimize_random(
        self,
        param_name,
        left,
        right,
        X,
        y,
        fixed_params,
        random_candidates,
        log_scale,
        print_each,
        num_runs,
    ):
        if log_scale:
            candidates = np.exp(
                np.random.uniform(
                    np.log(left), np.log(right), random_candidates
                )
            )
        else:
            candidates = np.random.uniform(
                left, right, random_candidates
            )

        for i, x in enumerate(candidates):
            params = dict(fixed_params)
            params[param_name] = x

            y_val = self._evaluate(num_runs=num_runs, **params)

            X = np.vstack([X, [[x]]])
            y = np.append(y, y_val)

            if print_each:
                print(
                    f"Random {i+1}: {param_name}={x:.4g}, metric={y_val:.4f}"
                )

        return X, y

    def optimize_fogas_hyperparameters(
        self,
        search_method="bo",
        coarse_points=10,
        bo_iters=15,
        random_candidates=30,
        print_main=True,
        print_search=False,
        plot=True,
        num_runs=3,
        order=("alpha", "rho", "eta"),
        bounds_overrides=None,
    ):
        """
        Sequential tuning in arbitrary order.

        - order: tuple/list, e.g. ("rho","alpha","eta")
        - bounds_overrides: optional dict, e.g.
              {"alpha": (1e-6, 20.0), "rho": {"bounds": (1e-4, 1e-1), "log_scale": True}}
          If you pass a tuple, it's interpreted as (low, high).
          If you pass a dict, it can override "bounds" and/or "log_scale".
        """
        # --- validate order ---
        valid = set(self.DEFAULT_BOUNDS.keys())
        for p in order:
            if p not in valid:
                raise ValueError(f"Unknown parameter '{p}'. Valid: {sorted(valid)}")

        bounds_overrides = bounds_overrides or {}

        # --- theory (starting point) ---
        theory = {
            "alpha": float(self.solver.alpha),
            "rho":   float(self.solver.rho),
            "eta":   float(self.solver.eta),
        }

        # current best-known params start at theory values
        current = dict(theory)

        metrics = {}
        metrics["theory"] = self._evaluate(num_runs=num_runs, **current)

        if print_main:
            print("\n=== FOGAS Hyperparameter Optimization ===")
            print(f"[Theory] metric = {metrics['theory']:.4f}")
            print(f"Order: {order}")

        stage_labels = ["theory"]
        stage_values = [metrics["theory"]]

        # --- sequential loop ---
        for p in order:
            if print_main:
                print(f"\nOptimizing {p}")

            # build config = DEFAULT + override
            cfg = dict(self.DEFAULT_BOUNDS[p])  # shallow copy
            override = bounds_overrides.get(p, None)

            if override is not None:
                if isinstance(override, tuple) or isinstance(override, list):
                    cfg["bounds"] = tuple(override)
                elif isinstance(override, dict):
                    cfg.update(override)
                else:
                    raise TypeError(
                        f"bounds_overrides['{p}'] must be tuple/list or dict, got {type(override)}"
                    )

            # resolve bounds (callable or tuple)
            bspec = cfg["bounds"]
            bounds = bspec(theory, current) if callable(bspec) else bspec
            log_scale = bool(cfg.get("log_scale", False))

            # fixed params = all except p
            fixed_params = {k: v for k, v in current.items() if k != p}

            p_star = self.optimal_search(
                p,
                bounds=bounds,
                fixed_params=fixed_params,
                log_scale=log_scale,
                search_method=search_method,
                coarse_points=coarse_points,
                bo_iters=bo_iters,
                random_candidates=random_candidates,
                print_each=print_search,
                num_runs=num_runs,
            )

            # update and evaluate exactly at the new triplet
            current[p] = float(p_star)
            self.solver.run(**current)
            key = " + ".join(order[: order.index(p) + 1])
            metrics[key] = float(self.metric())

            if print_main:
                pretty = ", ".join([f"{k}={current[k]:.4e}" for k in ("alpha","rho","eta")])
                print(f"[After {p}*] {pretty} | metric = {metrics[key]:.4f}")

            stage_labels.append(f"{p}*")
            stage_values.append(metrics[key])

        # --- plot (optional) ---
        if plot:
            plt.figure(figsize=(8, 5))
            plt.plot(stage_values, marker="o", linewidth=2)
            plt.xticks(range(len(stage_values)), stage_labels, rotation=20)
            plt.ylabel("Metric value")
            plt.title("Sequential FOGAS hyperparameter optimization")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        return {
            "alpha": current["alpha"],
            "rho": current["rho"],
            "eta": current["eta"],
            "metrics": metrics,
            "order": tuple(order),
        }
