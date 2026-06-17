"""Generalized FOGAS exports."""

from .solvers import (
    BetaSolver,
    FinalLinearSolver,
    FinalParametrizedSolver,
    LinearBetaPiSolver,
    LinearSolver,
    LossThetaBetaPiSolver,
    LinearPolicyFOGAS,
    RegularizedLossThetaBetaPiSolver,
    VBetaLogitSolver,
    VBetaObjectivePolicySolver,
    VBetaSolver,
)
from .fogas_parameters import GeneralizedFOGASParameters, StandaloneFOGASParameters
from .features import (
    FeatureFunction,
    LinearFunction,
    LinearQParam,
    LinearQFunction,
    LinearUParam,
    LinearUFunction,
    NeuralPolicyParam,
    NeuralQParam,
    NeuralUParam,
    PolicyParam,
    RBFStateActionFeatures,
    RBFStateFeatures,
    QParam,
    SoftmaxLinearPolicyParam,
    StateActionMLPModule,
    StateMLPPolicyModule,
    TabularFeatures,
    TabularPolicyFeatures,
    UParam,
    build_feature_table,
    build_policy_feature_table,
    build_q_feature_table,
    build_u_feature_table,
)
from .u_functions import UFunction

try:
    from .solver_policy import FOGASSolverPolicy
except ModuleNotFoundError as exc:
    if exc.name != f"{__name__}.solver_policy":
        raise
    FOGASSolverPolicy = None

__all__ = [
    "BetaSolver",
    "VBetaSolver",
    "VBetaObjectivePolicySolver",
    "VBetaLogitSolver",
    "LinearPolicyFOGAS",
    "LinearSolver",
    "FinalLinearSolver",
    "FinalParametrizedSolver",
    "LinearBetaPiSolver",
    "LossThetaBetaPiSolver",
    "RegularizedLossThetaBetaPiSolver",
    "GeneralizedFOGASParameters",
    "StandaloneFOGASParameters",
    "TabularPolicyFeatures",
    "TabularFeatures",
    "RBFStateFeatures",
    "RBFStateActionFeatures",
    "UFunction",
    "FeatureFunction",
    "UParam",
    "QParam",
    "PolicyParam",
    "LinearFunction",
    "LinearUFunction",
    "LinearQFunction",
    "LinearUParam",
    "LinearQParam",
    "SoftmaxLinearPolicyParam",
    "NeuralUParam",
    "NeuralQParam",
    "NeuralPolicyParam",
    "StateActionMLPModule",
    "StateMLPPolicyModule",
    "build_feature_table",
    "build_u_feature_table",
    "build_q_feature_table",
    "build_policy_feature_table",
]

if FOGASSolverPolicy is not None:
    __all__.append("FOGASSolverPolicy")
