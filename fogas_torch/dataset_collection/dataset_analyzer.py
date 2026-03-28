"""
DatasetAnalyzer
---------------

Utility class for analyzing (state, action) pair frequencies in datasets
created by EnvDataCollector. Supports both specific pair queries and
global dataset statistics.
"""

import pandas as pd
import numpy as np
from typing import Union, List, Tuple, Optional


class DatasetAnalyzer:
    """
    Analyzer for (state, action) pair frequency in collected datasets.
    
    Can load data from CSV files created by EnvDataCollector and provide:
    - Counts for specific (state, action) pairs
    - Global statistics about pair coverage and distribution
    
    Parameters
    ----------
    csv_path : str
        Path to CSV file with columns: state, action, reward, next_state
    """
    
    def __init__(self, csv_path: str):
        """Load dataset from CSV file."""
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        
        # Pre-compute counts for efficiency
        self._state_counts = self.df['state'].value_counts()
        self._pair_counts = self.df.groupby(['state', 'action']).size()
        self._pair_counts_df = self._pair_counts.reset_index(name='count')

    @staticmethod
    def _to_numpy(value) -> np.ndarray:
        """
        Convert numpy arrays, lists, or torch tensors to numpy arrays.
        """
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        return np.asarray(value)
    
    # -------------------------------------------------------------------------
    # Specific Pair Queries
    # -------------------------------------------------------------------------
    
    def count_pair(self, state: int, action: Union[int, str], action_map: Optional[dict] = None) -> int:
        """
        Get the count of a specific (state, action) pair.
        
        Parameters
        ----------
        state : int
            State index
        action : int or str
            Action index or action name (requires action_map)
        action_map : dict, optional
            Mapping from action names to indices (e.g., {'Down': 1})
            
        Returns
        -------
        int
            Number of times this (state, action) pair appears in the dataset
        """
        if isinstance(action, str):
            if action_map is None:
                # Default mapping for standard GridWorld
                action_map = {"Up": 0, "Down": 1, "Left": 2, "Right": 3}
            
            action_id = action_map.get(action)
            if action_id is None:
                return 0
            action = action_id

        try:
            return self._pair_counts[(state, action)]
        except KeyError:
            return 0

    def count_state(self, state: int) -> int:
        """
        Get the count of a specific state.
        
        Parameters
        ----------
        state : int
            State index
            
        Returns
        -------
        int
            Number of times this state appears in the dataset
        """
        return int(self._state_counts.get(state, 0))

    def count_states(self, states: List[int]) -> dict:
        """
        Get counts for multiple states.
        
        Parameters
        ----------
        states : list of int
            List of state indices to query
            
        Returns
        -------
        dict
            Dictionary mapping state -> count
        """
        return {s: self.count_state(s) for s in states}
    
    def count_pairs(self, pairs: List[Tuple[int, int]]) -> dict:
        """
        Get counts for multiple (state, action) pairs.
        
        Parameters
        ----------
        pairs : list of tuples
            List of (state, action) pairs to query
            
        Returns
        -------
        dict
            Dictionary mapping (state, action) -> count
        """
        return {(s, a): self.count_pair(s, a) for s, a in pairs}
    
    def get_state_actions(self, state: int) -> pd.DataFrame:
        """
        Get all actions taken from a specific state and their counts.
        
        Parameters
        ----------
        state : int
            State index
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: action, count
        """
        state_data = self._pair_counts_df[self._pair_counts_df['state'] == state]
        return state_data[['action', 'count']].sort_values('count', ascending=False)
    
    def get_action_states(self, action: int) -> pd.DataFrame:
        """
        Get all states where a specific action was taken and their counts.
        
        Parameters
        ----------
        action : int
            Action index
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: state, count
        """
        action_data = self._pair_counts_df[self._pair_counts_df['action'] == action]
        return action_data[['state', 'count']].sort_values('count', ascending=False)

    def analyze_custom_stats(self, states: List[int] = None, 
                             pairs: List[Tuple[int, Union[int, str]]] = None, 
                             action_map: Optional[dict] = None) -> None:
        """
        Print analysis for specific states and (state, action) pairs.
        
        Parameters
        ----------
        states : list of int, optional
            States to count
        pairs : list of (state, action) tuples, optional
            Pairs to count
        action_map : dict, optional
            Mapping for action names
        """
        print("\n" + "=" * 40)
        print("     CUSTOM DATASET ANALYSIS")
        print("=" * 40)
        
        if states:
            print("\n[State Visit Counts]")
            for s in states:
                print(f"  State {s:3d}: {self.count_state(s):6d}")
                
        if pairs:
            print("\n[State-Action Pair Counts]")
            for s, a in pairs:
                count = self.count_pair(s, a, action_map)
                print(f"  Pair ({s}, {a}): {count:6d}")
        
        print("\n" + "=" * 40 + "\n")
    
    # -------------------------------------------------------------------------
    # Global Statistics
    # -------------------------------------------------------------------------
    
    def get_global_stats(self, n_states: Optional[int] = None, 
                         n_actions: Optional[int] = None) -> dict:
        """
        Get global statistics about (state, action) pair distribution.
        
        Parameters
        ----------
        n_states : int, optional
            Total number of states in MDP (for coverage calculation)
        n_actions : int, optional
            Total number of actions in MDP (for coverage calculation)
            
        Returns
        -------
        dict
            Dictionary with global statistics
        """
        counts = self._pair_counts.values
        
        stats = {
            'total_transitions': len(self.df),
            'unique_pairs': len(self._pair_counts),
            'unique_states': self.df['state'].nunique(),
            'unique_actions': self.df['action'].nunique(),
            'min_count': int(counts.min()),
            'max_count': int(counts.max()),
            'mean_count': float(counts.mean()),
            'std_count': float(counts.std()),
            'median_count': float(np.median(counts)),
        }
        
        # Coverage statistics if MDP dimensions provided
        if n_states is not None and n_actions is not None:
            total_possible = n_states * n_actions
            stats['total_possible_pairs'] = total_possible
            stats['coverage_percent'] = 100.0 * stats['unique_pairs'] / total_possible
            stats['missing_pairs'] = total_possible - stats['unique_pairs']
        
        return stats

    # -------------------------------------------------------------------------
    # Feature Coverage (FOGAS)
    # -------------------------------------------------------------------------
    def feature_coverage_ratio(
        self,
        mdp,
        beta: float = 0.0,
        policy: Optional[np.ndarray] = None,
        use_optimal_policy: bool = True,
        return_details: bool = False,
        verbose: bool = False,
    ) -> Union[float, dict]:
        """
        Compute feature coverage ratio ||lambda_pi||_{Lambda_n^{-1}}^2.

        Lambda_n = beta I + (1/n) sum_i phi_i phi_i^T
        lambda_pi = Phi^T mu_pi

        Parameters
        ----------
        mdp : LinearMDP or PolicySolver
            Must expose states, actions, N, A, d, Phi, gamma, nu0 and
            get_transition_matrix(). If policy is None and use_optimal_policy
            is True, mdp must expose pi_star.
        beta : float
            Ridge term for empirical covariance.
        policy : ndarray, optional
            Policy matrix with shape (N, A). If None, uses mdp.pi_star.
        use_optimal_policy : bool
            Whether to default to mdp.pi_star when policy is None.
        return_details : bool
            If True, returns a dict with intermediate values.
        verbose : bool
            If True, prints a short summary.

        Returns
        -------
        float or dict
            Feature coverage ratio, or a dict with details.
        """
        if len(self.df) == 0:
            raise ValueError("Dataset is empty; cannot compute coverage ratio.")

        if policy is None:
            if not use_optimal_policy:
                raise ValueError("Provide policy or set use_optimal_policy=True.")
            if not hasattr(mdp, "pi_star"):
                raise ValueError("mdp does not expose pi_star; provide policy explicitly.")
            policy = mdp.pi_star

        # Map dataset states/actions to MDP indices (robust to non-0..N-1 labels)
        states_arr = self._to_numpy(mdp.states).astype(int)
        actions_arr = self._to_numpy(mdp.actions).astype(int)
        state_index = {int(s): i for i, s in enumerate(states_arr)}
        action_index = {int(a): i for i, a in enumerate(actions_arr)}

        X_raw = self.df["state"].to_numpy(dtype=np.int64)
        A_raw = self.df["action"].to_numpy(dtype=np.int64)
        try:
            X_idx = np.array([state_index[int(s)] for s in X_raw], dtype=np.int64)
            A_idx = np.array([action_index[int(a)] for a in A_raw], dtype=np.int64)
        except KeyError as exc:
            raise ValueError("Dataset contains states/actions not present in the MDP.") from exc

        N, A, d = int(mdp.N), int(mdp.A), int(mdp.d)
        policy_np = self._to_numpy(policy).astype(float)
        if policy_np.shape != (N, A):
            raise ValueError(f"Policy must have shape ({N}, {A}); got {policy_np.shape}.")

        # Build Phi tensors from MDP (N*A, d) -> (N, A, d)
        Phi_full = self._to_numpy(mdp.Phi).reshape(N, A, d)
        Phi_data = Phi_full[X_idx, A_idx]  # (n, d)

        n = len(self.df)
        Cov = beta * np.eye(d) + (Phi_data.T @ Phi_data) / n

        # Compute occupancy measure mu_pi using numpy
        if hasattr(mdp, "P") and mdp.P is not None:
            P = mdp.P
        else:
            P = mdp.get_transition_matrix()
        P_np = self._to_numpy(P).astype(float)
        nu0_np = self._to_numpy(mdp.nu0).reshape(-1).astype(float)
        gamma = float(mdp.gamma)

        Comp_pi = np.zeros((N * A, N), dtype=float)
        for x in range(N):
            Comp_pi[x * A:(x + 1) * A, x] = policy_np[x]

        rhs = (1.0 - gamma) * (Comp_pi @ nu0_np)
        mu_pi = np.linalg.solve(np.eye(N * A) - gamma * Comp_pi @ P_np.T, rhs)

        Phi_flat = Phi_full.reshape(N * A, d)
        lambda_pi = Phi_flat.T @ mu_pi

        ratio = float(lambda_pi.T @ np.linalg.solve(Cov, lambda_pi))

        if verbose:
            policy_src = "mdp.pi_star" if (policy is None and use_optimal_policy) else "provided"
            mu_sum = float(mu_pi.sum())
            mu_min = float(mu_pi.min())
            mu_max = float(mu_pi.max())
            lambda_l2 = float(np.linalg.norm(lambda_pi))
            lambda_l1 = float(np.linalg.norm(lambda_pi, 1))
            lambda_max = float(np.max(np.abs(lambda_pi)))

            diag = np.diag(Cov)
            diag_min = float(diag.min())
            diag_max = float(diag.max())
            diag_mean = float(diag.mean())

            try:
                eigs = np.linalg.eigvalsh(Cov)
                eig_min = float(eigs.min())
                eig_max = float(eigs.max())
                cond = float(eig_max / eig_min) if eig_min > 0 else float("inf")
            except np.linalg.LinAlgError:
                eig_min = float("nan")
                eig_max = float("nan")
                cond = float("nan")

            print("\nFeature Coverage Ratio Details")
            print("------------------------------")
            print(f"  Dataset size (n):         {n}")
            print(f"  MDP dims (N, A, d):        ({N}, {A}, {d})")
            print(f"  gamma:                    {gamma:.6g}")
            print(f"  beta (ridge):             {beta:.6g}")
            print(f"  policy source:            {policy_src}")
            print("")
            print("  Occupancy μ_pi summary:")
            print(f"    sum:                    {mu_sum:.6g}")
            print(f"    min / max:              {mu_min:.6g} / {mu_max:.6g}")
            print("")
            print("  Feature occupancy λ_pi summary:")
            print(f"    ||λ||_2:                {lambda_l2:.6g}")
            print(f"    ||λ||_1:                {lambda_l1:.6g}")
            print(f"    max |λ_i|:              {lambda_max:.6g}")
            print("")
            print("  Empirical covariance Λ_n:")
            print(f"    diag min / mean / max:  {diag_min:.6g} / {diag_mean:.6g} / {diag_max:.6g}")
            print(f"    eig min / max:          {eig_min:.6g} / {eig_max:.6g}")
            print(f"    condition number:       {cond:.6g}")
            print("")
            print(f"  Coverage ratio:           {ratio:.6g}")

        if return_details:
            details = {
                "coverage_ratio": ratio,
                "lambda_pi": lambda_pi,
                "mu_pi": mu_pi,
                "covariance": Cov,
                "policy": policy_np,
                "beta": beta,
                "n": n,
            }
            return details

        return ratio
    
    def get_all_pair_counts(self, sort_by: str = 'count', 
                            ascending: bool = False) -> pd.DataFrame:
        """
        Get all (state, action) pairs with their counts.
        
        Parameters
        ----------
        sort_by : str
            Column to sort by: 'count', 'state', or 'action'
        ascending : bool
            Sort order
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: state, action, count
        """
        return self._pair_counts_df.sort_values(sort_by, ascending=ascending)
    
    def get_top_pairs(self, n: int = 10) -> pd.DataFrame:
        """
        Get the n most frequent (state, action) pairs.
        
        Parameters
        ----------
        n : int
            Number of pairs to return
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: state, action, count
        """
        return self.get_all_pair_counts().head(n)
    
    def get_rare_pairs(self, n: int = 10) -> pd.DataFrame:
        """
        Get the n least frequent (state, action) pairs.
        
        Parameters
        ----------
        n : int
            Number of pairs to return
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: state, action, count
        """
        return self.get_all_pair_counts(ascending=True).head(n)
    
    def get_missing_pairs(self, n_states: int, n_actions: int) -> List[Tuple[int, int]]:
        """
        Get all (state, action) pairs that never appear in the dataset.
        
        Parameters
        ----------
        n_states : int
            Total number of states in MDP
        n_actions : int
            Total number of actions in MDP
            
        Returns
        -------
        list of tuples
            List of (state, action) pairs with zero occurrences
        """
        existing = set(zip(self._pair_counts_df['state'], 
                          self._pair_counts_df['action']))
        all_pairs = set((s, a) for s in range(n_states) for a in range(n_actions))
        return sorted(all_pairs - existing)
    
    # -------------------------------------------------------------------------
    # Comparative Analysis Methods
    # -------------------------------------------------------------------------
    
    def analyze_pair(self, state: int, action: int, verbose: bool = True) -> dict:
        """
        Analyze a specific (state, action) pair compared to global statistics.
        
        Parameters
        ----------
        state : int
            State index
        action : int
            Action index
        verbose : bool
            If True, prints the analysis
            
        Returns
        -------
        dict
            Analysis results including count, comparison to mean, percentile, etc.
        """
        count = self.count_pair(state, action)
        global_stats = self.get_global_stats()
        
        mean = global_stats['mean_count']
        std = global_stats['std_count']
        
        # Z-score: how many standard deviations from mean
        z_score = (count - mean) / std if std > 0 else 0.0
        
        # Percentile rank among all pairs
        all_counts = self._pair_counts.values
        percentile = 100.0 * np.sum(all_counts <= count) / len(all_counts)
        
        # Ratio to mean
        ratio_to_mean = count / mean if mean > 0 else float('inf')
        
        analysis = {
            'state': state,
            'action': action,
            'count': count,
            'global_mean': mean,
            'global_std': std,
            'diff_from_mean': count - mean,
            'ratio_to_mean': ratio_to_mean,
            'z_score': z_score,
            'percentile': percentile,
        }
        
        if verbose:
            print(f"Analysis for pair (state={state}, action={action}):")
            print(f"  Count: {count}")
            print(f"  Global mean: {mean:.2f}")
            print(f"  Difference from mean: {count - mean:+.2f} ({ratio_to_mean:.2f}x)")
            print(f"  Z-score: {z_score:+.2f} std")
            print(f"  Percentile: {percentile:.1f}%")
            if z_score > 2:
                print("  ⬆️  Significantly ABOVE average")
            elif z_score < -2:
                print("  ⬇️  Significantly BELOW average")
            else:
                print("  ➡️  Within normal range")
        
        return analysis
    
    def analyze_pairs(self, pairs: List[Tuple[int, int]], verbose: bool = True) -> pd.DataFrame:
        """
        Analyze multiple (state, action) pairs compared to global statistics.
        
        Parameters
        ----------
        pairs : list of tuples
            List of (state, action) pairs to analyze
        verbose : bool
            If True, prints the analysis table
            
        Returns
        -------
        pd.DataFrame
            DataFrame with analysis for each pair
        """
        analyses = [self.analyze_pair(s, a, verbose=False) for s, a in pairs]
        df = pd.DataFrame(analyses)
        
        if verbose and len(df) > 0:
            print("Pair Analysis (compared to global mean):")
            print("-" * 70)
            display_df = df[['state', 'action', 'count', 'diff_from_mean', 
                            'ratio_to_mean', 'z_score', 'percentile']].copy()
            display_df['diff_from_mean'] = display_df['diff_from_mean'].apply(lambda x: f"{x:+.1f}")
            display_df['ratio_to_mean'] = display_df['ratio_to_mean'].apply(lambda x: f"{x:.2f}x")
            display_df['z_score'] = display_df['z_score'].apply(lambda x: f"{x:+.2f}")
            display_df['percentile'] = display_df['percentile'].apply(lambda x: f"{x:.1f}%")
            print(display_df.to_string(index=False))
        
        return df
    
    def analyze_state(self, state: int, n_actions: Optional[int] = None, 
                      verbose: bool = True) -> dict:
        """
        Analyze all (state, action) pairs for a specific state.
        
        Parameters
        ----------
        state : int
            State index to analyze
        n_actions : int, optional
            Total number of actions (to include missing actions with count=0)
        verbose : bool
            If True, prints the analysis
            
        Returns
        -------
        dict
            Dictionary with state-level and per-action analysis
        """
        global_stats = self.get_global_stats()
        global_mean = global_stats['mean_count']
        global_std = global_stats['std_count']
        
        # Get existing pairs for this state
        state_data = self._pair_counts_df[self._pair_counts_df['state'] == state].copy()
        
        # If n_actions provided, include missing actions with count=0
        if n_actions is not None:
            existing_actions = set(state_data['action'])
            for a in range(n_actions):
                if a not in existing_actions:
                    state_data = pd.concat([state_data, pd.DataFrame({
                        'state': [state], 'action': [a], 'count': [0]
                    })], ignore_index=True)
        
        state_data = state_data.sort_values('action').reset_index(drop=True)
        
        # Compute comparison metrics
        state_data['diff_from_mean'] = state_data['count'] - global_mean
        state_data['ratio_to_mean'] = state_data['count'] / global_mean if global_mean > 0 else 0
        state_data['z_score'] = (state_data['count'] - global_mean) / global_std if global_std > 0 else 0
        
        # State-level statistics
        counts = state_data['count'].values
        state_stats = {
            'state': state,
            'total_visits': int(counts.sum()),
            'n_actions_seen': int((counts > 0).sum()),
            'n_actions_missing': int((counts == 0).sum()),
            'state_mean': float(counts.mean()),
            'state_std': float(counts.std()),
            'min_action_count': int(counts.min()),
            'max_action_count': int(counts.max()),
            'global_mean': global_mean,
            'avg_diff_from_global': float(state_data['diff_from_mean'].mean()),
        }
        
        result = {
            'state_stats': state_stats,
            'action_details': state_data,
        }
        
        if verbose:
            print("=" * 60)
            print(f"Analysis for State {state}")
            print("=" * 60)
            print(f"Total visits to this state: {state_stats['total_visits']:,}")
            print(f"Actions seen: {state_stats['n_actions_seen']}, "
                  f"Missing: {state_stats['n_actions_missing']}")
            print(f"State mean count: {state_stats['state_mean']:.2f} "
                  f"(Global mean: {global_mean:.2f})")
            print(f"Avg diff from global mean: {state_stats['avg_diff_from_global']:+.2f}")
            print("-" * 60)
            print("Per-Action Breakdown:")
            print("-" * 60)
            
            for _, row in state_data.iterrows():
                indicator = ""
                if row['z_score'] > 2:
                    indicator = "⬆️"
                elif row['z_score'] < -2:
                    indicator = "⬇️"
                elif row['count'] == 0:
                    indicator = "❌"
                    
                print(f"  Action {int(row['action']):2d}: count={int(row['count']):4d}, "
                      f"diff={row['diff_from_mean']:+7.1f}, "
                      f"z={row['z_score']:+5.2f} {indicator}")
            
            print("=" * 60)
        
        return result
    
    # -------------------------------------------------------------------------
    # Summary Methods
    # -------------------------------------------------------------------------
    
    def summary(self, n_states: Optional[int] = None, 
                n_actions: Optional[int] = None) -> None:
        """
        Print a summary of the dataset analysis.
        
        Parameters
        ----------
        n_states : int, optional
            Total number of states in MDP (for coverage calculation)
        n_actions : int, optional
            Total number of actions in MDP (for coverage calculation)
        """
        stats = self.get_global_stats(n_states, n_actions)
        
        print("=" * 50)
        print("Dataset Analysis Summary")
        print("=" * 50)
        print(f"Source: {self.csv_path}")
        print(f"Total transitions: {stats['total_transitions']:,}")
        print(f"Unique (state, action) pairs: {stats['unique_pairs']:,}")
        print(f"Unique states visited: {stats['unique_states']:,}")
        print(f"Unique actions taken: {stats['unique_actions']:,}")
        print("-" * 50)
        print("Pair Frequency Statistics:")
        print(f"  Min count:    {stats['min_count']:,}")
        print(f"  Max count:    {stats['max_count']:,}")
        print(f"  Mean count:   {stats['mean_count']:.2f}")
        print(f"  Std count:    {stats['std_count']:.2f}")
        print(f"  Median count: {stats['median_count']:.1f}")
        
        if n_states is not None and n_actions is not None:
            print("-" * 50)
            print("Coverage Statistics:")
            print(f"  Total possible pairs: {stats['total_possible_pairs']:,}")
            print(f"  Coverage: {stats['coverage_percent']:.2f}%")
            print(f"  Missing pairs: {stats['missing_pairs']:,}")
        
        print("=" * 50)
    
    def __repr__(self) -> str:
        return (f"DatasetAnalyzer(csv_path='{self.csv_path}', "
                f"transitions={len(self.df)}, unique_pairs={len(self._pair_counts)})")
