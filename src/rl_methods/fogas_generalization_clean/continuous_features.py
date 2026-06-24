"""Continuous-observation parametrizations for generalized FOGAS."""

from __future__ import annotations

import math
from typing import Callable, Sequence, Union

import torch
from torch import nn


def _activation_factory(activation: Union[type[nn.Module], Callable[[], nn.Module], nn.Module]):
    if isinstance(activation, nn.Module):
        return lambda: activation.__class__()
    return activation


def _build_mlp(
    input_dim,
    output_dim,
    hidden_sizes: Sequence[int] = (64, 64),
    activation: Union[type[nn.Module], Callable[[], nn.Module], nn.Module] = nn.Tanh,
    dtype=torch.float64,
):
    input_dim = int(input_dim)
    output_dim = int(output_dim)
    if input_dim <= 0 or output_dim <= 0:
        raise ValueError("input_dim and output_dim must be positive")
    make_activation = _activation_factory(activation)
    layers = []
    prev_dim = input_dim
    for hidden_size in hidden_sizes:
        hidden_size = int(hidden_size)
        if hidden_size <= 0:
            raise ValueError("hidden_sizes entries must be positive")
        layers.append(nn.Linear(prev_dim, hidden_size, dtype=dtype))
        layers.append(make_activation())
        prev_dim = hidden_size
    layers.append(nn.Linear(prev_dim, output_dim, dtype=dtype))
    return nn.Sequential(*layers)


class ContinuousStateActionMLPModule(nn.Module):
    """MLP over concatenated continuous observations and action descriptors."""

    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_sizes: Sequence[int] = (64, 64),
        output_dim=1,
        activation=nn.Tanh,
        dtype=torch.float64,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.output_dim = int(output_dim)
        if self.obs_dim <= 0 or self.action_dim <= 0 or self.output_dim <= 0:
            raise ValueError("obs_dim, action_dim, and output_dim must be positive")
        self.net = _build_mlp(
            input_dim=self.obs_dim + self.action_dim,
            output_dim=self.output_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            dtype=dtype,
        )

    def forward(self, observations, actions):
        observations = torch.as_tensor(observations, dtype=self._dtype(), device=self._device())
        actions = torch.as_tensor(actions, dtype=observations.dtype, device=observations.device)
        if observations.ndim == 1:
            observations = observations.reshape(1, -1)
        if actions.ndim == 1:
            actions = actions.reshape(-1, self.action_dim)
        if observations.shape[-1] != self.obs_dim:
            raise ValueError(f"observations last dimension must be {self.obs_dim}")
        if actions.shape[-1] != self.action_dim:
            raise ValueError(f"actions last dimension must be {self.action_dim}")
        if observations.shape[0] == 1 and actions.shape[0] > 1:
            observations = observations.expand(actions.shape[0], self.obs_dim)
        elif actions.shape[0] == 1 and observations.shape[0] > 1:
            actions = actions.expand(observations.shape[0], self.action_dim)
        elif observations.shape[0] != actions.shape[0]:
            raise ValueError(
                "observations and actions must have the same batch size, "
                "unless one batch size is 1"
            )
        return self.net(torch.cat([observations, actions], dim=-1))

    def _dtype(self):
        return next(self.parameters()).dtype

    def _device(self):
        return next(self.parameters()).device


class ContinuousStateMLPPolicyModule(nn.Module):
    """Categorical policy-logit MLP over continuous observations."""

    def __init__(
        self,
        obs_dim,
        n_actions,
        hidden_sizes: Sequence[int] = (64, 64),
        activation=nn.Tanh,
        dtype=torch.float64,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        if self.obs_dim <= 0 or self.n_actions <= 0:
            raise ValueError("obs_dim and n_actions must be positive")
        self.net = _build_mlp(
            input_dim=self.obs_dim,
            output_dim=self.n_actions,
            hidden_sizes=hidden_sizes,
            activation=activation,
            dtype=dtype,
        )

    def forward(self, observations):
        observations = torch.as_tensor(observations, dtype=self._dtype(), device=self._device())
        if observations.ndim == 1:
            observations = observations.reshape(1, -1)
        if observations.shape[-1] != self.obs_dim:
            raise ValueError(f"observations last dimension must be {self.obs_dim}")
        return self.net(observations)

    def _dtype(self):
        return next(self.parameters()).dtype

    def _device(self):
        return next(self.parameters()).device


class ContinuousGaussianPolicyModule(nn.Module):
    """Gaussian policy with an MLP mean and diagonal log standard deviation."""

    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_sizes: Sequence[int] = (64, 64),
        init_log_std=-0.5,
        activation=nn.Tanh,
        dtype=torch.float64,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        if self.obs_dim <= 0 or self.action_dim <= 0:
            raise ValueError("obs_dim and action_dim must be positive")
        self.mean_net = _build_mlp(
            input_dim=self.obs_dim,
            output_dim=self.action_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            dtype=dtype,
        )
        self.log_std = nn.Parameter(torch.full((self.action_dim,), float(init_log_std), dtype=dtype))

    def mean(self, observations):
        observations = torch.as_tensor(observations, dtype=self._dtype(), device=self._device())
        if observations.ndim == 1:
            observations = observations.reshape(1, -1)
        if observations.shape[-1] != self.obs_dim:
            raise ValueError(f"observations last dimension must be {self.obs_dim}")
        return self.mean_net(observations)

    def std(self):
        return torch.exp(self.log_std).clamp_min(1e-8)

    def sample(self, observations, deterministic=False):
        mean = self.mean(observations)
        if deterministic:
            return mean
        return mean + torch.randn_like(mean) * self.std().to(dtype=mean.dtype, device=mean.device)

    def log_prob_actions(self, observations, actions):
        mean = self.mean(observations)
        actions = torch.as_tensor(actions, dtype=mean.dtype, device=mean.device)
        if actions.ndim == 1:
            actions = actions.reshape(-1, self.action_dim)
        std = self.std().to(dtype=mean.dtype, device=mean.device)
        var = std.square()
        log_probs = -0.5 * (
            ((actions - mean).square() / var)
            + 2.0 * torch.log(std)
            + math.log(2.0 * math.pi)
        )
        return log_probs.sum(dim=-1)

    def forward(self, observations):
        return self.mean(observations)

    def _dtype(self):
        return next(self.parameters()).dtype

    def _device(self):
        return next(self.parameters()).device


class ContinuousNeuralUParam(nn.Module):
    """Neural FOGAS u wrapper for continuous observations."""

    is_linear_fast_path = False

    def __init__(self, module):
        super().__init__()
        self.module = module

    def u(self, observations, actions):
        return self.module(observations, actions).squeeze(-1)

    def forward(self, observations, actions):
        return self.u(observations, actions)


class ContinuousNeuralQParam(nn.Module):
    """Neural FOGAS Q wrapper for continuous observations."""

    is_linear_fast_path = False

    def __init__(self, module):
        super().__init__()
        self.module = module

    def q(self, observations, actions):
        return self.module(observations, actions).squeeze(-1)

    def forward(self, observations, actions):
        return self.q(observations, actions)


class ContinuousDiscretePolicyParam(nn.Module):
    """Categorical policy over finite actions from continuous observations."""

    is_linear_fast_path = False

    def __init__(self, module):
        super().__init__()
        self.module = module

    def logits(self, observations):
        return self.module(observations)

    def probs(self, observations):
        return torch.softmax(self.logits(observations), dim=-1)

    def log_probs(self, observations):
        return torch.log_softmax(self.logits(observations), dim=-1)

    def log_prob_actions(self, observations, actions):
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.logits(observations).device)
        return self.log_probs(observations).gather(-1, actions.reshape(-1, 1)).squeeze(-1)

    def sample(self, observations, deterministic=False):
        probs = self.probs(observations)
        if deterministic:
            return torch.argmax(probs, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def forward(self, observations):
        return self.probs(observations)


class ContinuousGaussianPolicyParam(nn.Module):
    """Gaussian policy wrapper for continuous-action FOGAS."""

    is_linear_fast_path = False

    def __init__(self, module):
        super().__init__()
        self.module = module

    def mean(self, observations):
        return self.module.mean(observations)

    def sample(self, observations, deterministic=False):
        return self.module.sample(observations, deterministic=deterministic)

    def log_prob_actions(self, observations, actions):
        return self.module.log_prob_actions(observations, actions)

    def forward(self, observations):
        return self.mean(observations)


__all__ = [
    "ContinuousStateActionMLPModule",
    "ContinuousStateMLPPolicyModule",
    "ContinuousGaussianPolicyModule",
    "ContinuousNeuralUParam",
    "ContinuousNeuralQParam",
    "ContinuousDiscretePolicyParam",
    "ContinuousGaussianPolicyParam",
]
