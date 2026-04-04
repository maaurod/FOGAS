# fogas_torch/__init__.py

# --- MDP core classes ---
from .mdp.linear_mdp import LinearMDP
from .mdp.policy_solver import PolicySolver
from .mdp.abstract_mdp import (
    BoxActionDiscretizer,
    BoxStateDiscretizer,
    DiscreteActionDiscretizer,
    DiscretizedLinearMDP,
    TabularFeatureMap,
)
from .q_learning.q_learning_solver import QLearningResult, QLearningSolver, run_q_learning

# --- FOGAS algorithm ---
from .algorithm.fogas_solver import FOGASSolver
from .algorithm.fogas_evaluator import FOGASEvaluator
from .algorithm.fogas_dataset import FOGASDataset
from .algorithm.fogas_parameters import FOGASParameters
from .algorithm.fogas_hyperoptimizer import FOGASHyperOptimizer
from .algorithm.fogas_oraclesolver import FOGASOracleSolver

# --- Dataset collection utilities ---
from .dataset_collection.linear_mdp_env import LinearMDPEnv
from .dataset_collection.env_data_collector import EnvDataCollector
from .dataset_collection.continuous_env_data_collector import ContinuousEnvDataCollector

__all__ = [
    "LinearMDP",
    "PolicySolver",
    "BoxStateDiscretizer",
    "DiscreteActionDiscretizer",
    "BoxActionDiscretizer",
    "TabularFeatureMap",
    "DiscretizedLinearMDP",
    "QLearningResult",
    "QLearningSolver",
    "run_q_learning",
    "FOGASSolver",
    "FOGASDataset",
    "FOGASParameters",
    "LinearMDPEnv",
    "EnvDataCollector",
    "ContinuousEnvDataCollector",
    "FOGASEvaluator",
    "FOGASHyperOptimizer",
    "FOGASOracleSolver",
]
