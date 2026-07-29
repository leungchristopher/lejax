"""Projection head."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from flax import linen as nn


@dataclass(frozen=True)
class ProjectorConfig:
    input_dim: int
    hidden_dims: tuple[int, ...] = (2048, 2048)
    output_dim: int = 16

    def __post_init__(self) -> None:
        if self.output_dim <= 0:
            raise ValueError("output_dim must be non-zero")
        if any(width <= 0 for width in self.hidden_dims):
            raise ValueError(f"hidden widths must be non-zero, got {self.hidden_dims}")

    def init(self) -> "Projector":
        return Projector(config=self)


class Projector(nn.Module):
    """(Linear -> BatchNorm -> ReLU) x N, then a bare Linear."""

    config: ProjectorConfig

    @nn.compact
    def __call__(self, embeddings: jnp.ndarray, use_running_average: bool = True) -> jnp.ndarray:
        x = embeddings
        for hidden in self.config.hidden_dims:
            x = nn.Dense(hidden)(x)
            x = nn.BatchNorm(use_running_average=use_running_average)(x)
            x = nn.relu(x)
        return nn.Dense(self.config.output_dim)(x)
