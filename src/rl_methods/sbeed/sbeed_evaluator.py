from __future__ import annotations

import random
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch


ArrayLike = Union[np.ndarray, torch.Tensor]


class SBEEDEvaluator:
    """
    Evaluation utilities for a trained or partially trained discrete SBEED solver.

    What you can call with only a solver:
        learned_value()
            Returns the current SBEED value vector V_theta.
        learned_policy()
            Returns the current policy matrix pi(a | s).
        print_policy()
            Prints the learned policy as a generic discrete-MDP table.
        get_metric("objective")
            Returns a zero-argument callable for the current SBEED objective.

    What additionally needs an exact tabular model:
        value_iteration_hard()
            Solves the standard optimal Bellman equation for V*.
        value_iteration_soft()
            Solves the smoothed SBEED target equation for V_lambda*.
        bellman_residual_hard(), bellman_residual_soft()
            Measures fixed-point residuals.
        compare_to_optimal_values()
            Prints and returns SBEED vs V_lambda* and V* diagnostics.
        evaluate_policy(), policy_return(), simulate_trajectory(), print_optimal_path()
            Evaluates or simulates policies under the model.

    The model can be supplied in three equivalent ways:
        1. Explicit arrays:
               P with shape [S, A, S] or [S*A, S]
               R with shape [S, A] or [S*A]
        2. A project MDP object with `.P` and `.r`.
        3. The deterministic functions used by SBEED:
               next_state_fn(s, a)
               reward_fn(s, a, next_state)

    For SBEED with lambda_entropy > 0, compare primarily against
    V_lambda*, not V*, because the algorithm targets the smoothed Bellman
    consistency equation from the paper.

    Expensive dynamic-programming references are computed only when requested.
    """

    def __init__(
        self,
        solver,
        P: Optional[ArrayLike] = None,
        R: Optional[ArrayLike] = None,
        mdp: Optional[Any] = None,
        next_state_fn: Optional[Callable[[int, int], Any]] = None,
        transition_fn: Optional[Callable[[int, int], Any]] = None,
        reward_fn: Optional[Callable[..., float]] = None,
        terminal_states: Optional[set] = None,
        state_names: Optional[Sequence[str]] = None,
        action_names: Optional[Sequence[str]] = None,
    ):
        """
        Build an evaluator around an existing SBEED solver.

        Args:
            solver:
                The SBEEDSolver instance. It supplies n_states, n_actions,
                gamma, lambda_entropy, learned theta, and learned policy.
            P, R:
                Optional exact tabular model arrays. Use these when the MDP is
                stochastic or already materialized.
            mdp:
                Optional project MDP object. If present, `.P`, `.r`, `.states`,
                `.actions`, and `.x0` are read when available.
            next_state_fn:
                Deterministic transition function with the same shape as the
                function passed to solver.run(...). It may return next_state,
                (next_state, reward), or gym-like tuples accepted by SBEEDSolver.
            transition_fn:
                Alias for next_state_fn.
            reward_fn:
                Reward function used when next_state_fn does not return reward.
                Preferred signature is reward_fn(s, a, next_state); reward_fn(s, a)
                is also accepted for older notebooks.
            terminal_states:
                Optional episodic terminal states. If supplied, evaluator model
                rows for those states are absorbing with zero outgoing reward.
            state_names, action_names:
                Optional labels used by pretty printers.
        """
        self.solver = solver
        self.spec = solver.spec
        self.n_states = int(solver.n_states)
        self.n_actions = int(solver.n_actions)
        self.gamma = float(solver.gamma)
        self.x0 = self.spec.x0
        self.terminal_states = set() if terminal_states is None else {int(s) for s in terminal_states}

        self.P = None
        self.R = None
        if mdp is not None:
            P = getattr(mdp, "P", P)
            R = getattr(mdp, "r", R)
            if state_names is None:
                state_names = [str(x.item()) for x in getattr(mdp, "states", [])]
            if action_names is None:
                action_names = [str(a.item()) for a in getattr(mdp, "actions", [])]
            if hasattr(mdp, "x0") and self.x0 is None:
                self.x0 = int(mdp.x0)
        if P is not None or R is not None:
            if P is None or R is None:
                raise ValueError("P and R must be provided together")
            self.P = self._as_transition_tensor(P)
            self.R = self._as_reward_matrix(R)
        else:
            model_fn = next_state_fn if next_state_fn is not None else transition_fn
            if model_fn is not None:
                self.P, self.R = self._model_from_deterministic_functions(
                    next_state_fn=model_fn,
                    reward_fn=reward_fn,
                )
        if self.P is not None and self.terminal_states:
            self._apply_terminal_states()

        self.state_names = list(state_names) if state_names is not None and len(state_names) else None
        self.action_names = list(action_names) if action_names is not None and len(action_names) else None

    def available_methods(self) -> Dict[str, Sequence[str]]:
        """
        Return a small capability map for notebooks.

        This is only documentation as data; it does not run evaluations.
        """
        solver_only = [
            "learned_value",
            "learned_policy",
            "print_policy",
            "get_metric('objective')",
        ]
        model_based = [
            "value_iteration_hard",
            "value_iteration_soft",
            "bellman_residual_hard",
            "bellman_residual_soft",
            "compare_to_optimal_values",
            "evaluate_policy",
            "policy_return",
            "simulate_trajectory",
            "print_optimal_path",
            "get_metric('soft_value_error')",
            "get_metric('soft_value_error_l2')",
            "get_metric('soft_residual')",
            "get_metric('hard_value_error')",
            "get_metric('hard_value_error_l2')",
            "get_metric('hard_residual')",
            "get_metric('policy_return')",
        ]
        return {"solver_only": solver_only, "requires_model": model_based}

    @staticmethod
    def _to_numpy(x: ArrayLike) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _as_transition_tensor(self, P: ArrayLike) -> np.ndarray:
        P_np = self._to_numpy(P).astype(np.float64, copy=False)
        if P_np.shape == (self.n_states * self.n_actions, self.n_states):
            P_np = P_np.reshape(self.n_states, self.n_actions, self.n_states)
        if P_np.shape != (self.n_states, self.n_actions, self.n_states):
            raise ValueError(
                "P must have shape "
                f"({self.n_states}, {self.n_actions}, {self.n_states}) "
                f"or ({self.n_states * self.n_actions}, {self.n_states}); got {P_np.shape}"
            )
        row_sums = P_np.sum(axis=2)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            raise ValueError("Each P[s, a, :] row must sum to 1")
        if np.any(P_np < -1e-12):
            raise ValueError("P contains negative transition probabilities")
        return P_np

    def _as_reward_matrix(self, R: ArrayLike) -> np.ndarray:
        R_np = self._to_numpy(R).astype(np.float64, copy=False)
        if R_np.shape == (self.n_states * self.n_actions,):
            R_np = R_np.reshape(self.n_states, self.n_actions)
        if R_np.shape != (self.n_states, self.n_actions):
            raise ValueError(
                f"R must have shape ({self.n_states}, {self.n_actions}) "
                f"or ({self.n_states * self.n_actions},); got {R_np.shape}"
            )
        return R_np

    def _apply_terminal_states(self) -> None:
        for state in self.terminal_states:
            if state < 0 or state >= self.n_states:
                raise ValueError(f"terminal state {state} is outside [0, n_states)")
            self.P[state, :, :] = 0.0
            self.P[state, :, state] = 1.0
            self.R[state, :] = 0.0

    def _model_from_deterministic_functions(
        self,
        next_state_fn: Callable[[int, int], Any],
        reward_fn: Optional[Callable[..., float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Materialize P and R by enumerating every state-action pair.

        This matches common finite-discrete notebooks where next_state_fn(s, a)
        is deterministic. For each (s, a), this creates one transition with
        probability 1.0. If next_state_fn returns a reward, that reward is used;
        otherwise reward_fn is required.
        """
        P = np.zeros((self.n_states, self.n_actions, self.n_states), dtype=np.float64)
        R = np.zeros((self.n_states, self.n_actions), dtype=np.float64)

        for s in range(self.n_states):
            for a in range(self.n_actions):
                next_state, reward, _ = self._parse_transition_result(next_state_fn(s, a))
                if next_state < 0 or next_state >= self.n_states:
                    raise ValueError(
                        f"next_state_fn({s}, {a}) returned invalid next_state {next_state}"
                    )
                if reward is None:
                    if reward_fn is None:
                        raise ValueError(
                            "reward_fn is required when next_state_fn does not return reward"
                        )
                    reward = self._call_reward_fn(reward_fn, s, a, next_state)
                P[s, a, next_state] = 1.0
                R[s, a] = float(reward)

        return P, R

    @staticmethod
    def _parse_transition_result(result: Any) -> Tuple[int, Optional[float], bool]:
        """
        Parse the same deterministic transition signatures accepted by SBEEDSolver.

        Supported returns:
            next_state
            (next_state, reward)
            (next_state, reward, done)
            (next_state, reward, terminated, truncated)
            (next_state, reward, terminated, truncated, info)
        """
        if isinstance(result, tuple):
            if len(result) == 2:
                next_state, reward = result
                return int(next_state), float(reward), False
            if len(result) == 3:
                next_state, reward, done = result
                return int(next_state), float(reward), bool(done)
            if len(result) == 4:
                next_state, reward, terminated, truncated = result
                return int(next_state), float(reward), bool(terminated) or bool(truncated)
            if len(result) == 5:
                next_state, reward, terminated, truncated, _ = result
                return int(next_state), float(reward), bool(terminated) or bool(truncated)
            raise ValueError("Unsupported transition tuple length")
        return int(result), None, False

    @staticmethod
    def _call_reward_fn(
        reward_fn: Callable[..., float],
        state: int,
        action: int,
        next_state: int,
    ) -> float:
        """
        Call reward_fn with the preferred 3-argument signature, falling back to
        the older reward_fn(s, a) style used in some experiments.
        """
        try:
            return float(reward_fn(state, action, next_state))
        except TypeError:
            return float(reward_fn(state, action))

    def _require_model(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the tabular model or raise a clear error for model-based methods.
        """
        if self.P is None or self.R is None:
            raise ValueError(
                "This evaluation requires a model. Pass P and R, mdp=..., "
                "or next_state_fn=... with reward_fn=... to SBEEDEvaluator."
            )
        return self.P, self.R

    def _require_policy(self) -> torch.Tensor:
        """
        Return the learned policy if solver.run() set it, otherwise compute it
        from the current policy weights W.
        """
        if self.solver.pi is None:
            return self.solver.get_policy_matrix()
        return self.solver.pi.detach().clone()

    @staticmethod
    def stable_logsumexp(x: np.ndarray, axis=None) -> np.ndarray:
        """
        Numerically stable log-sum-exp used by the smoothed Bellman operator.
        """
        x_max = np.max(x, axis=axis, keepdims=True)
        y = x_max + np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))
        if axis is not None:
            y = np.squeeze(y, axis=axis)
        return y

    def learned_value(self) -> np.ndarray:
        """
        Return the current learned value vector V_theta(s).

        This does not require an exact model and can be called before or after
        solver.run(); before training it returns the value induced by the
        current theta, usually zeros.
        """
        with torch.no_grad():
            value = self.solver.PHI_S @ self.solver.theta
        return value.detach().cpu().numpy()

    def learned_policy(self) -> np.ndarray:
        """
        Return the current learned policy matrix pi(a | s).

        If solver.run() has not assigned solver.pi yet, this computes the policy
        from the current policy parameter W.
        """
        return self._require_policy().detach().cpu().numpy()

    def value_iteration_hard(
        self,
        tol: float = 1e-10,
        max_iter: int = 100_000,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Solve the standard optimal Bellman fixed point V* by value iteration.

        Equation:
            V*(s) = max_a [R(s,a) + gamma * E[V*(s') | s,a]]

        Returns:
            V:
                Optimal hard-max value vector with shape [S].
            pi:
                Greedy deterministic action index for each state, shape [S].
            info:
                Convergence diagnostics: iterations, final_diff, converged.

        Requires:
            A model from P/R, mdp, or next_state_fn + reward_fn.
        """
        P, R = self._require_model()
        V = np.zeros(self.n_states, dtype=np.float64)
        diff = np.inf
        it = -1

        for it in range(int(max_iter)):
            Q = R + self.gamma * np.einsum("sat,t->sa", P, V)
            V_new = np.max(Q, axis=1)
            diff = float(np.max(np.abs(V_new - V)))
            V = V_new
            if diff < tol:
                break

        Q = R + self.gamma * np.einsum("sat,t->sa", P, V)
        pi = np.argmax(Q, axis=1)
        info = {"iterations": it + 1, "final_diff": diff, "converged": diff < tol}
        return V, pi, info

    def value_iteration_soft(
        self,
        lam: Optional[float] = None,
        tol: float = 1e-10,
        max_iter: int = 100_000,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Solve the entropy-smoothed SBEED Bellman fixed point V_lambda*.

        Equation:
            V_lambda*(s) =
                lam * log sum_a exp(
                    [R(s,a) + gamma * E[V_lambda*(s') | s,a]] / lam
                )

        Args:
            lam:
                Entropy smoothing coefficient. Defaults to
                solver.lambda_entropy.
            tol, max_iter:
                Value-iteration stopping controls.

        Returns:
            V:
                Smoothed optimal value vector with shape [S].
            pi:
                Soft optimal policy matrix with shape [S, A].
            info:
                Convergence diagnostics: iterations, final_diff, converged.

        Requires:
            A model from P/R, mdp, or next_state_fn + reward_fn.
        """
        P, R = self._require_model()
        lam = self.solver.lambda_entropy if lam is None else float(lam)
        if lam <= 0.0:
            raise ValueError("lam must be positive for soft value iteration")

        V = np.zeros(self.n_states, dtype=np.float64)
        diff = np.inf
        it = -1

        for it in range(int(max_iter)):
            Q = R + self.gamma * np.einsum("sat,t->sa", P, V)
            V_new = lam * self.stable_logsumexp(Q / lam, axis=1)
            diff = float(np.max(np.abs(V_new - V)))
            V = V_new
            if diff < tol:
                break

        Q = R + self.gamma * np.einsum("sat,t->sa", P, V)
        logits = Q / lam
        logits = logits - np.max(logits, axis=1, keepdims=True)
        pi = np.exp(logits)
        pi = pi / np.sum(pi, axis=1, keepdims=True)
        info = {"iterations": it + 1, "final_diff": diff, "converged": diff < tol}
        return V, pi, info

    def bellman_residual_hard(self, V: Optional[ArrayLike] = None) -> float:
        """
        Compute ||V - T V||_infinity for the hard Bellman operator.

        Args:
            V:
                Value vector to test. If omitted, uses learned_value().

        Use this to see whether a value function is close to the ordinary
        optimal Bellman equation. For SBEED with lambda_entropy > 0, the soft
        residual is usually the primary diagnostic.
        """
        P, R = self._require_model()
        V_np = self.learned_value() if V is None else self._to_numpy(V).astype(np.float64, copy=False)
        Q = R + self.gamma * np.einsum("sat,t->sa", P, V_np)
        TV = np.max(Q, axis=1)
        return float(np.max(np.abs(V_np - TV)))

    def bellman_residual_soft(
        self,
        V: Optional[ArrayLike] = None,
        lam: Optional[float] = None,
    ) -> float:
        """
        Compute ||V - T_lambda V||_infinity for the smoothed Bellman operator.

        Args:
            V:
                Value vector to test. If omitted, uses learned_value().
            lam:
                Smoothing coefficient. Defaults to solver.lambda_entropy.

        This is the main fixed-point residual to inspect for SBEED when
        lambda_entropy > 0.
        """
        P, R = self._require_model()
        lam = self.solver.lambda_entropy if lam is None else float(lam)
        if lam <= 0.0:
            raise ValueError("lam must be positive for soft Bellman residual")
        V_np = self.learned_value() if V is None else self._to_numpy(V).astype(np.float64, copy=False)
        Q = R + self.gamma * np.einsum("sat,t->sa", P, V_np)
        T_lam_V = lam * self.stable_logsumexp(Q / lam, axis=1)
        return float(np.max(np.abs(V_np - T_lam_V)))

    def evaluate_policy(self, pi: Optional[ArrayLike] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate a policy exactly under the tabular model.

        Args:
            pi:
                Policy matrix with shape [S, A]. If omitted, uses the learned
                SBEED policy.

        Returns:
            V_pi:
                State-value vector with shape [S].
            Q_pi:
                Action-value matrix with shape [S, A].

        Requires:
            A model from P/R, mdp, or next_state_fn + reward_fn.
        """
        P, R = self._require_model()
        pi_np = self.learned_policy() if pi is None else self._to_numpy(pi).astype(np.float64, copy=False)
        if pi_np.shape != (self.n_states, self.n_actions):
            raise ValueError(f"pi must have shape ({self.n_states}, {self.n_actions})")

        r_pi = np.sum(pi_np * R, axis=1)
        P_pi = np.einsum("sa,sat->st", pi_np, P)
        V = np.linalg.solve(np.eye(self.n_states) - self.gamma * P_pi, r_pi)
        Q = R + self.gamma * np.einsum("sat,t->sa", P, V)
        return V, Q

    def policy_return(
        self,
        pi: Optional[ArrayLike] = None,
        start_distribution: Optional[ArrayLike] = None,
        normalized: bool = False,
    ) -> float:
        """
        Compute the expected discounted return of a policy.

        Args:
            pi:
                Policy matrix. If omitted, uses the learned SBEED policy.
            start_distribution:
                Optional initial state distribution with shape [S]. If omitted,
                uses solver.spec.x0, mdp.x0, or state 0.
            normalized:
                If False, returns E[V(s0)]. If True, returns
                (1 - gamma) * E[V(s0)], matching discounted occupancy scaling.
        """
        V, _ = self.evaluate_policy(pi)
        if start_distribution is None:
            rho0 = np.zeros(self.n_states, dtype=np.float64)
            rho0[self.x0 if self.x0 is not None else 0] = 1.0
        else:
            rho0 = self._to_numpy(start_distribution).astype(np.float64, copy=False).reshape(-1)
            if rho0.shape != (self.n_states,):
                raise ValueError(f"start_distribution must have shape ({self.n_states},)")
        value = float(rho0 @ V)
        return float((1.0 - self.gamma) * value) if normalized else value

    def compare_to_optimal_values(
        self,
        lam: Optional[float] = None,
        tol: float = 1e-10,
        max_iter: int = 100_000,
        print_each: bool = True,
    ) -> Dict[str, Any]:
        """
        Compare learned SBEED values to both V_lambda* and V*.

        This method computes:
            V_sbeed:
                Current learned V_theta.
            V_lambda_star:
                Soft fixed point from value_iteration_soft().
            V_star:
                Hard fixed point from value_iteration_hard().
            linf_to_soft:
                max_s |V_sbeed(s) - V_lambda*(s)|.
            soft_residual:
                ||V_sbeed - T_lambda V_sbeed||_infinity.

        For SBEED experiments, `linf_to_soft` and `soft_residual` are the most
        relevant numbers when lambda_entropy > 0.
        """
        V_sbeed = self.learned_value()
        V_star, pi_star, hard_info = self.value_iteration_hard(tol=tol, max_iter=max_iter)
        V_lam, pi_lam, soft_info = self.value_iteration_soft(lam=lam, tol=tol, max_iter=max_iter)

        if print_each:
            print("\n========== SBEED VALUE COMPARISON ==========\n")
            for s in range(self.n_states):
                name = self._state_name(s)
                print(
                    f"State {name}: "
                    f"V_sbeed={V_sbeed[s]: .6f} | "
                    f"V_lambda*={V_lam[s]: .6f} | "
                    f"V*={V_star[s]: .6f} | "
                    f"|SBEED-lambda|={abs(V_sbeed[s] - V_lam[s]): .6e}"
                )
            print("\nNorm diagnostics:")

        stats = {
            "hard_info": hard_info,
            "soft_info": soft_info,
            "V_sbeed": V_sbeed,
            "V_star": V_star,
            "V_lambda_star": V_lam,
            "pi_star": pi_star,
            "pi_lambda_star": pi_lam,
            "linf_to_soft": float(np.max(np.abs(V_sbeed - V_lam))),
            "l2_to_soft": float(np.linalg.norm(V_sbeed - V_lam)),
            "linf_to_hard": float(np.max(np.abs(V_sbeed - V_star))),
            "l2_to_hard": float(np.linalg.norm(V_sbeed - V_star)),
            "soft_residual": self.bellman_residual_soft(V_sbeed, lam=lam),
            "hard_residual": self.bellman_residual_hard(V_sbeed),
        }
        if print_each:
            print(f"||V_sbeed - V_lambda*||_inf = {stats['linf_to_soft']:.6e}")
            print(f"||V_sbeed - V_lambda*||_2   = {stats['l2_to_soft']:.6e}")
            print(f"||V_sbeed - V*||_inf        = {stats['linf_to_hard']:.6e}")
            print(f"||V_sbeed - V*||_2          = {stats['l2_to_hard']:.6e}")
            print(f"soft Bellman residual       = {stats['soft_residual']:.6e}")
            print(f"hard Bellman residual       = {stats['hard_residual']:.6e}")
            print("\n============================================\n")
        return stats

    def get_metric(self, name: str, **kwargs):
        """
        Return a zero-argument callable metric, matching the FOGAS evaluator style.

        Supported names:
            "soft_value_error" / "value_error_soft":
                ||V_sbeed - V_lambda*||_infinity.
            "soft_value_error_l2" / "value_error_soft_l2":
                ||V_sbeed - V_lambda*||_2.
            "soft_residual" / "bellman_residual_soft":
                ||V_sbeed - T_lambda V_sbeed||_infinity.
            "hard_value_error" / "value_error_hard":
                ||V_sbeed - V*||_infinity.
            "hard_value_error_l2" / "value_error_hard_l2":
                ||V_sbeed - V*||_2.
            "hard_residual" / "bellman_residual_hard":
                ||V_sbeed - T V_sbeed||_infinity.
            "return" / "policy_return" / "reward":
                Exact expected return of the learned policy.
            "objective" / "sbeed_objective":
                Current empirical SBEED objective from solver.objective().

        Example:
            metric = evaluator.get_metric("soft_residual")
            value = metric()
        """
        if name in {"soft_value_error", "value_error_soft"}:
            lam = kwargs.get("lam", None)
            tol = kwargs.get("tol", 1e-10)
            max_iter = kwargs.get("max_iter", 100_000)
            return lambda: self.compare_to_optimal_values(
                lam=lam, tol=tol, max_iter=max_iter, print_each=False
            )["linf_to_soft"]

        if name in {"soft_value_error_l2", "value_error_soft_l2"}:
            lam = kwargs.get("lam", None)
            tol = kwargs.get("tol", 1e-10)
            max_iter = kwargs.get("max_iter", 100_000)
            return lambda: self.compare_to_optimal_values(
                lam=lam, tol=tol, max_iter=max_iter, print_each=False
            )["l2_to_soft"]

        if name in {"soft_residual", "bellman_residual_soft"}:
            lam = kwargs.get("lam", None)
            return lambda: self.bellman_residual_soft(lam=lam)

        if name in {"hard_value_error", "value_error_hard"}:
            tol = kwargs.get("tol", 1e-10)
            max_iter = kwargs.get("max_iter", 100_000)
            return lambda: self.compare_to_optimal_values(
                tol=tol, max_iter=max_iter, print_each=False
            )["linf_to_hard"]

        if name in {"hard_value_error_l2", "value_error_hard_l2"}:
            tol = kwargs.get("tol", 1e-10)
            max_iter = kwargs.get("max_iter", 100_000)
            return lambda: self.compare_to_optimal_values(
                tol=tol, max_iter=max_iter, print_each=False
            )["l2_to_hard"]

        if name in {"hard_residual", "bellman_residual_hard"}:
            return lambda: self.bellman_residual_hard()

        if name in {"return", "policy_return", "reward"}:
            start_distribution = kwargs.get("start_distribution", None)
            normalized = kwargs.get("normalized", False)
            return lambda: self.policy_return(
                start_distribution=start_distribution,
                normalized=normalized,
            )

        if name in {"objective", "sbeed_objective"}:
            return lambda: self.solver.objective()["objective"]

        raise ValueError(f"Unknown metric '{name}'")

    def print_policy(
        self,
        pi: Optional[ArrayLike] = None,
    ) -> None:
        """
        Pretty-print a policy as a generic discrete-MDP table.

        Args:
            pi:
                Policy matrix. If omitted, uses the learned SBEED policy.

        This method does not require P/R.
        """
        pi_np = self.learned_policy() if pi is None else self._to_numpy(pi).astype(np.float64, copy=False)
        if pi_np.shape != (self.n_states, self.n_actions):
            raise ValueError(f"pi must have shape ({self.n_states}, {self.n_actions})")

        print("\n========== SBEED POLICY ==========\n")
        for s in range(self.n_states):
            probs = "  ".join(
                f"pi({self._action_name(a)}|{self._state_name(s)})={pi_np[s, a]:.3f}"
                for a in range(self.n_actions)
            )
            best = int(np.argmax(pi_np[s]))
            print(f"State {self._state_name(s)}: {probs}  --> best action: {self._action_name(best)}")
        print("\n==================================\n")

    def simulate_trajectory(
        self,
        pi: Optional[ArrayLike] = None,
        max_steps: int = 100,
        seed: Optional[int] = None,
        start_state: Optional[int] = None,
        goal_state: Optional[int] = None,
        greedy: bool = True,
    ) -> list[Dict[str, Any]]:
        """
        Simulate one trajectory under the tabular model.

        Args:
            pi:
                Policy matrix. If omitted, uses the learned SBEED policy.
            max_steps:
                Maximum number of transitions to generate.
            seed:
                Optional numpy/random seed.
            start_state:
                Optional initial state. Defaults to x0 or 0.
            goal_state:
                If supplied, stop when this state is reached.
            greedy:
                If True, take argmax_a pi(a|s). If False, sample actions from
                pi(a|s).

        Returns:
            A list of dictionaries with state, action, reward, next_state, step,
            and reached_goal fields.
        """
        P, R = self._require_model()
        pi_np = self.learned_policy() if pi is None else self._to_numpy(pi).astype(np.float64, copy=False)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        state = int(start_state if start_state is not None else (self.x0 if self.x0 is not None else 0))

        trajectory = []
        for step in range(int(max_steps)):
            if greedy:
                action = int(np.argmax(pi_np[state]))
            else:
                action = int(np.random.choice(self.n_actions, p=pi_np[state]))
            next_state = int(np.random.choice(self.n_states, p=P[state, action]))
            reward = float(R[state, action])
            trajectory.append(
                {
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                    "step": step,
                    "reached_goal": next_state == goal_state if goal_state is not None else False,
                }
            )
            if goal_state is not None and next_state == goal_state:
                break
            state = next_state
        return trajectory

    def print_optimal_path(
        self,
        max_steps: int = 50,
        seed: Optional[int] = 42,
        goal_state: Optional[int] = None,
        use_soft_optimal: bool = False,
        use_hard_optimal: bool = False,
    ) -> None:
        """
        Print a readable trajectory for learned, soft-optimal, or hard-optimal policy.

        Args:
            max_steps:
                Maximum simulation length.
            seed:
                Optional random seed for next-state sampling.
            goal_state:
                If supplied, stop printing when reached.
            use_soft_optimal:
                If True, simulate pi_lambda* from value_iteration_soft().
            use_hard_optimal:
                If True, simulate greedy pi* from value_iteration_hard().

        By default this prints the learned SBEED policy trajectory.
        """
        if use_soft_optimal and use_hard_optimal:
            raise ValueError("Choose only one of use_soft_optimal or use_hard_optimal")
        if use_soft_optimal:
            _, pi, _ = self.value_iteration_soft()
            policy_name = "Soft Optimal Policy (pi_lambda*)"
        elif use_hard_optimal:
            _, hard_pi, _ = self.value_iteration_hard()
            pi = np.zeros((self.n_states, self.n_actions), dtype=np.float64)
            pi[np.arange(self.n_states), hard_pi] = 1.0
            policy_name = "Hard Optimal Policy (pi*)"
        else:
            pi = self.learned_policy()
            policy_name = "Learned Policy (pi_SBEED)"

        trajectory = self.simulate_trajectory(
            pi=pi,
            max_steps=max_steps,
            seed=seed,
            goal_state=goal_state,
        )
        print(f"\n========== SBEED PATH - {policy_name} ==========\n")
        discounted_return = 0.0
        for i, item in enumerate(trajectory):
            discounted_return += (self.gamma ** i) * item["reward"]
            print(
                f"Step {i:3d}: "
                f"{self._state_name(item['state'])} --{self._action_name(item['action'])}/"
                f"{item['reward']:.4f}--> {self._state_name(item['next_state'])}"
            )
            if item["reached_goal"]:
                print("Goal reached.")
                break
        print(f"\nDiscounted return: {discounted_return:.6f}")
        print("\n===============================================\n")

    def _state_name(self, state: int) -> str:
        if self.state_names is not None and int(state) < len(self.state_names):
            return str(self.state_names[int(state)])
        return str(int(state))

    def _action_name(self, action: int) -> str:
        if self.action_names is not None and int(action) < len(self.action_names):
            return str(self.action_names[int(action)])
        return str(int(action))
