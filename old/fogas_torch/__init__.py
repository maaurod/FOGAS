# fogas_torch/__init__.py

# --- MDP core classes ---
from .mdp.linear_mdp import LinearMDP
from .mdp.policy_solver import PolicySolver
from .mdp.abstract_mdp import (
    BoxActionDiscretizer,
    BoxStateDiscretizer,
    DiscreteActionDiscretizer,
    DiscretizedLinearMDP,
    FeatureOnlyAbstractMDP,
    TabularFeatureMap,
)
from .q_learning.q_learning_solver import QLearningResult, QLearningSolver, run_q_learning

# --- FOGAS algorithm ---
from .algorithm.fogas_solver import FOGASSolver
from .algorithm_gen.fogas_solver_gen import FOGASSolverBeta
from .algorithm_gen.fogas_solver_gen_vectorized import FOGASSolverBetaVectorized
from .algorithm_gen.solver_policy import FOGASSolverPolicy
from .algorithm.fogas_evaluator import FOGASEvaluator
from .algorithm.fogas_dataset import FOGASDataset
from .algorithm.fogas_parameters import FOGASParameters
from .algorithm.fogas_hyperoptimizer import FOGASHyperOptimizer
from .algorithm.fogas_oraclesolver import FOGASOracleSolver

# --- Dataset collection utilities ---
from .dataset_collection.linear_mdp_env import LinearMDPEnv
from .dataset_collection.env_data_collector import EnvDataCollector
from .dataset_collection.continuous_env_data_collector import ContinuousEnvDataCollector
from .dataset_collection.abstract_env_data_collector import (
    build_uniform_reset_distribution_from_policy_trajectory,
    collect_change_of_state_dataset_from_env_policy,
)

__all__ = [
    "LinearMDP",
    "PolicySolver",
    "BoxStateDiscretizer",
    "DiscreteActionDiscretizer",
    "BoxActionDiscretizer",
    "TabularFeatureMap",
    "DiscretizedLinearMDP",
    "FeatureOnlyAbstractMDP",
    "QLearningResult",
    "QLearningSolver",
    "run_q_learning",
    "FOGASSolver",
    "FOGASSolverBeta",
    "FOGASSolverBetaVectorized",
    "FOGASSolverPolicy",
    "FOGASDataset",
    "FOGASParameters",
    "LinearMDPEnv",
    "EnvDataCollector",
    "ContinuousEnvDataCollector",
    "build_uniform_reset_distribution_from_policy_trajectory",
    "collect_change_of_state_dataset_from_env_policy",
    "FOGASEvaluator",
    "FOGASHyperOptimizer",
    "FOGASOracleSolver",
]
