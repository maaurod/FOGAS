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
        
        # Pre-compute pair counts for efficiency
        self._pair_counts = self.df.groupby(['state', 'action']).size()
        self._pair_counts_df = self._pair_counts.reset_index(name='count')
    
    # -------------------------------------------------------------------------
    # Specific Pair Queries
    # -------------------------------------------------------------------------
    
    def count_pair(self, state: int, action: int) -> int:
        """
        Get the count of a specific (state, action) pair.
        
        Parameters
        ----------
        state : int
            State index
        action : int
            Action index
            
        Returns
        -------
        int
            Number of times this (state, action) pair appears in the dataset
        """
        try:
            return self._pair_counts[(state, action)]
        except KeyError:
            return 0
    
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
