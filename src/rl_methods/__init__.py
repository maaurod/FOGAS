"""Top-level package for the RL method implementations."""

from importlib import import_module

_SUBPACKAGES = (
    "data_collection",
    "fogas",
    "fogas_generalization",
    "fqi",
    "mdp",
    "q_learning",
    "sbeed",
)

_EXPORTS = {
    "mdp": (
        "DiscreteMDP",
        "FeaturesMDP",
        "TabularFeatureMap",
        "Planner",
        "StateDiscretizer",
        "ActionDiscretizer",
    ),
    "fogas": (
        "FOGASSolver",
        "FOGASEvaluator",
        "FOGASDataset",
        "ContinuousFOGASDataset",
        "FOGASParameters",
        "FOGASHyperOptimizer",
        "FOGASOracleSolver",
    ),
    "fogas_generalization": (
        "BetaSolver",
        "ContinuousFinalParametrizedSolver",
        "FinalLinearSolver",
        "FinalParametrizedSolver",
        "LinearBetaPiSolver",
        "LinearSolver",
        "LossThetaBetaPiSolver",
        "LinearPolicyFOGAS",
        "PrimalAlgaeDICESolver",
        "RegularizedLossThetaBetaPiSolver",
        "VBetaLogitSolver",
        "VBetaObjectivePolicySolver",
        "VBetaSolver",
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
        "ContinuousStateActionMLPModule",
        "ContinuousStateMLPPolicyModule",
        "ContinuousGaussianPolicyModule",
        "ContinuousRBFStateActionFeatures",
        "ContinuousLinearRBFUParam",
        "ContinuousLinearRBFQParam",
        "ContinuousSoftmaxLinearRBFPolicyParam",
        "ContinuousNeuralUParam",
        "ContinuousNeuralQParam",
        "ContinuousDiscretePolicyParam",
        "ContinuousGaussianPolicyParam",
        "StateActionMLPModule",
        "StateMLPPolicyModule",
        "build_feature_table",
        "build_u_feature_table",
        "build_q_feature_table",
        "build_policy_feature_table",
    ),
    "fqi": ("FQISolver",),
    "data_collection": ("DiscreteDataBuffer", "DatasetAnalyzer", "GymDataBuffer"),
    "q_learning": ("QLearningResult", "QLearningSolver", "run_q_learning"),
    "sbeed": (
        "SBEED",
        "DiscreteSBEED",
        "DiscreteSBEEDDataset",
        "ContinuousSBEED",
        "ContinuousSBEEDDataset",
        "SBEEDSolver",
        "SBEEDSolverSGDRho",
        "SBEEDOptimizers",
        "MultiLinearSBEED",
        "MultiParametrizedSBEED",
        "SBEEDSolverProtocol",
        "SBEEDEvaluator",
        "SBEEDDataset",
        "DiscreteMDPSpec",
        "ValueParam",
        "RhoParam",
        "ContinuousValueParam",
        "ContinuousRhoParam",
        "LinearValueParam",
        "LinearRhoParam",
        "RFFGaussianPolicyParam",
        "IdentityHead",
        "StateMLPValueModule",
        "StateActionMLPModule",
        "StateMLPPolicyModule",
        "ContinuousStateMLPValueModule",
        "ContinuousStateActionMLPModule",
        "NeuralValueParam",
        "NeuralRhoParam",
        "NeuralPolicyParam",
        "ContinuousNeuralValueParam",
        "ContinuousNeuralRhoParam",
    ),
}

_EXPORT_TO_SUBPACKAGE = {
    name: subpackage
    for subpackage, names in _EXPORTS.items()
    for name in names
}

__all__ = [*_SUBPACKAGES, *_EXPORT_TO_SUBPACKAGE]


def __getattr__(name):
    if name in _SUBPACKAGES:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module

    subpackage = _EXPORT_TO_SUBPACKAGE.get(name)
    if subpackage is not None:
        module = import_module(f".{subpackage}", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
