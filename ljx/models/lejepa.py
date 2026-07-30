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

from ljx.data.jax_augment import ViewConfig
from ljx.losses.sigreg import SigRegConfig, sigreg_loss_multi_view
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
    global_view: ViewConfig = field(default_factory=lambda: ViewConfig(size=64, scale=(0.30, 1.0)))
    local_view: ViewConfig = field(default_factory=lambda: ViewConfig(size=32, scale=(0.05, 0.30)))
    num_global_views: int = 2
    # 6 is the DINO/iBOT-style convention the reference repo inherited, tuned
    # for larger-scale pretraining. 4 trades some of that multi-crop signal
    # for ~15% fewer tokens through the MLP/QKV projections per step at this
    # ViT-Tiny/64px scale, where the original ratio is unverified.
    num_local_views: int = 4

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
        global_views = list(global_views)
        local_views = list(local_views)
        if not global_views and not local_views:
            raise ValueError("LeJEPA needs at least one view to train on")

        # Views within a group share a resolution, so they can be concatenated
        # into one batched forward pass instead of one call per view — same
        # per-image result (nothing in the ViT mixes across the batch axis),
        # far fewer kernel launches.
        projections = []
        for views in (global_views, local_views):
            if not views:
                continue
            stacked = jnp.concatenate(views, axis=0)
            projection = self.encode_and_project(
                stacked, deterministic=deterministic, use_running_average=use_running_average
            ).projection
            projections.extend(jnp.split(projection, len(views), axis=0))
        return projections


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
    sigreg = jnp.mean(sigreg_loss_multi_view(config.sigreg, jnp.stack(projections, axis=0), step))
    total = (1.0 - config.lejepa_lambda) * prediction + config.lejepa_lambda * sigreg
    return LeJEPALoss(total=total, prediction=prediction, sigreg=sigreg)
