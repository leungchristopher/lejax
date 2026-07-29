"""Linear probe over a frozen encoder."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import linen as nn

from ljx.models.backbone import ViT

NUM_CLASSES = 200


@dataclass(frozen=True)
class LinearClassifierConfig:
    embed_dim: int
    num_classes: int = NUM_CLASSES

    def __post_init__(self) -> None:
        if self.num_classes <= 0:
            raise ValueError("num_classes must be non-zero")


class LinearClassifier(nn.Module):
    backbone: ViT
    config: LinearClassifierConfig

    @nn.compact
    def __call__(self, images: jnp.ndarray) -> jnp.ndarray:
        """images: [batch, height, width, 3] -> [batch, num_classes]."""
        embeddings = jax.lax.stop_gradient(self.backbone(images, deterministic=True))
        return nn.Dense(self.config.num_classes, name="head")(embeddings)

    def encode(self, images: jnp.ndarray) -> jnp.ndarray:
        return jax.lax.stop_gradient(self.backbone(images, deterministic=True))
