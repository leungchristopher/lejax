"""LeJEPA objective: encoder + projector + SIGReg.

total = (1 - lambda) * prediction + lambda * sigreg
prediction = mean over views of || projection_v - mean_v(projection) ||^2
sigreg     = mean over views of SIGReg(projection_v)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import jax.numpy as jnp
from flax import linen as nn

from ljx.losses.sigreg import SigRegConfig, sigreg_loss_at_step
from ljx.models.backbone import ViT, ViTConfig
from ljx.models.projector import Projector, ProjectorConfig

DEFAULT_LAMBDA = 0.02
DEFAULT_NUM_PROJECTIONS = 256


@dataclass(frozen=True)
class LeJEPAConfig:
    backbone: ViTConfig = field(default_factory=ViTConfig)
    projector_hidden_dims: tuple[int, ...] = (2048, 2048)
    projector_output_dim: int = 16
    lejepa_lambda: float = DEFAULT_LAMBDA
    sigreg: SigRegConfig = field(
        default_factory=lambda: SigRegConfig(num_projections=DEFAULT_NUM_PROJECTIONS)
    )

    def __post_init__(self) -> None:
        if not (0.0 <= self.lejepa_lambda <= 1.0):
            raise ValueError(
                f"lambda is a convex weight and must lie in [0, 1], got {self.lejepa_lambda}"
            )

    def projector_config(self) -> ProjectorConfig:
        return ProjectorConfig(
            input_dim=self.backbone.embed_dim,
            hidden_dims=self.projector_hidden_dims,
            output_dim=self.projector_output_dim,
        )

    def init(self) -> "LeJEPA":
        return LeJEPA(backbone_config=self.backbone, projector_config=self.projector_config())


class LeJEPAEmbedding(NamedTuple):
    embedding: jnp.ndarray  # [batch, embed_dim]
    projection: jnp.ndarray  # [batch, projector_output_dim]


class LeJEPALoss(NamedTuple):
    total: jnp.ndarray
    prediction: jnp.ndarray
    sigreg: jnp.ndarray


class LeJEPA(nn.Module):
    backbone_config: ViTConfig
    projector_config: ProjectorConfig

    def setup(self) -> None:
        self.backbone = ViT(config=self.backbone_config)
        self.projector = Projector(config=self.projector_config)

    def encode(self, images: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        return self.backbone(images, deterministic=deterministic)

    def encode_and_project(
        self,
        images: jnp.ndarray,
        deterministic: bool = True,
        use_running_average: bool = True,
    ) -> LeJEPAEmbedding:
        embedding = self.backbone(images, deterministic=deterministic)
        projection = self.projector(embedding, use_running_average=use_running_average)
        return LeJEPAEmbedding(embedding=embedding, projection=projection)

    def __call__(
        self,
        global_views: list[jnp.ndarray],
        local_views: list[jnp.ndarray],
        deterministic: bool = True,
        use_running_average: bool = True,
    ) -> list[jnp.ndarray]:
        views = list(global_views) + list(local_views)
        if not views:
            raise ValueError("LeJEPA needs at least one view to train on")
        return [
            self.encode_and_project(
                view, deterministic=deterministic, use_running_average=use_running_average
            ).projection
            for view in views
        ]


def invariance_loss(projections: list[jnp.ndarray]) -> jnp.ndarray:
    stacked = jnp.stack(projections, axis=0)
    centre = jnp.mean(stacked, axis=0)
    residual = centre[None, :, :] - stacked
    return jnp.mean(jnp.mean(residual * residual, axis=(1, 2)))


def lejepa_loss(
    projections: list[jnp.ndarray], config: LeJEPAConfig, step: int
) -> LeJEPALoss:
    if not projections:
        raise ValueError("LeJEPA needs at least one view to train on")

    prediction = invariance_loss(projections)
    sigreg = jnp.mean(
        jnp.stack(
            [sigreg_loss_at_step(config.sigreg, p, step) for p in projections]
        )
    )
    total = (1.0 - config.lejepa_lambda) * prediction + config.lejepa_lambda * sigreg
    return LeJEPALoss(total=total, prediction=prediction, sigreg=sigreg)
