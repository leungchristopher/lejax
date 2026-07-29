"""Named TrainingConfig presets for a param sweep"""

from __future__ import annotations

from dataclasses import replace

from ljx.losses.sigreg import SigRegConfig
from ljx.models.backbone import ViTConfig
from ljx.models.lejepa import LeJEPAConfig
from ljx.training.train_loop import TrainingConfig

_BACKBONE = ViTConfig()  # fixed across the sweep: ViT-Tiny, 64px, patch 8


def _config(
    *,
    lejepa_lambda: float,
    num_projections: int,
    projector_hidden_dims: tuple[int, ...],
    projector_output_dim: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    t_max: float = 3.0,
    n_points: int = 33,
) -> TrainingConfig:
    model = LeJEPAConfig(
        backbone=_BACKBONE,
        projector_hidden_dims=projector_hidden_dims,
        projector_output_dim=projector_output_dim,
        lejepa_lambda=lejepa_lambda,
        sigreg=SigRegConfig(num_projections=num_projections, t_max=t_max, n_points=n_points),
    )
    return TrainingConfig(
        model=model,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )


# lambda=0.02, proj_dim=16, 256 slices, lr=2e-3, bs=256 — MINIMAL.md's
# recommended recipe for ViT-S/8 + Imagenette on a single GPU.
MINIMAL_RECIPE = _config(
    lejepa_lambda=0.02,
    num_projections=256,
    projector_hidden_dims=(2048, 2048),
    projector_output_dim=16,
    learning_rate=2e-3,
    weight_decay=5e-2,
    batch_size=256,
)

# This project's existing default (LeJEPAConfig()/TrainingConfig() as-is),
# included so the sweep has a control to compare the others against.
PROJECT_DEFAULT = _config(
    lejepa_lambda=0.02,
    num_projections=256,
    projector_hidden_dims=(2048, 2048),
    projector_output_dim=16,
    learning_rate=1e-4,
    weight_decay=0.05,
    batch_size=64,
)

# lambda=0.05, 1024 slices, lr=5e-4 — launch_proj_ablation.md's setting, at
# their smaller projector_dim=128 rather than 512/1024 (which target a
# 512-2048-wide encoder, not this project's 192-wide ViT-Tiny).
WIDE_PROJECTOR = _config(
    lejepa_lambda=0.05,
    num_projections=1024,
    projector_hidden_dims=(512, 512),
    projector_output_dim=128,
    learning_rate=5e-4,
    weight_decay=5e-2,
    batch_size=128,
)

# Cheapest projector in the sweep, larger batch — tests whether a lighter
# head changes T4 throughput meaningfully now that DALI has moved
# augmentation off the CPU.
LIGHT_PROJECTOR_FAST = _config(
    lejepa_lambda=0.02,
    num_projections=512,
    projector_hidden_dims=(512, 512),
    projector_output_dim=64,
    learning_rate=2e-3,
    weight_decay=5e-2,
    batch_size=256,
)

# lambda=0.1 — top of launch_inet10.py's swept range (0.01, 0.02, 0.05, 0.1),
# with launch_proj_ablation.md's 1000-ish slice count. Tests whether a much
# larger SIGReg weight destabilizes training at this small scale.
HIGH_LAMBDA = _config(
    lejepa_lambda=0.1,
    num_projections=1000,
    projector_hidden_dims=(2048, 2048),
    projector_output_dim=16,
    learning_rate=1e-4,
    weight_decay=0.05,
    batch_size=64,
)

# lr=3e-3, weight_decay=3e-2 — the upper end of launch_inet10.py's swept
# lr/weight_decay pairs (3e-3,1e-4 / 3e-2,1e-5).
HIGH_LR = _config(
    lejepa_lambda=0.02,
    num_projections=256,
    projector_hidden_dims=(2048, 2048),
    projector_output_dim=16,
    learning_rate=3e-3,
    weight_decay=3e-2,
    batch_size=256,
)

# n_points=41 (vs this project's default 33, reference's default 17),
# t_max=5 — from launch_epps_ablation.md's swept range
# (num_slices in {512,1024,4096}, t_max in {1,3,5}, n_points in {5,17,41}).
# Tests whether a finer quadrature grid changes anything at this scale, since
# sigreg.py already deviates from the reference default for aliasing reasons.
FINE_QUADRATURE = _config(
    lejepa_lambda=0.02,
    num_projections=512,
    projector_hidden_dims=(2048, 2048),
    projector_output_dim=16,
    learning_rate=1e-4,
    weight_decay=0.05,
    batch_size=64,
    t_max=5.0,
    n_points=41,
)

SWEEP: dict[str, TrainingConfig] = {
    "minimal_recipe": MINIMAL_RECIPE,
    "project_default": PROJECT_DEFAULT,
    "wide_projector": WIDE_PROJECTOR,
    "light_projector_fast": LIGHT_PROJECTOR_FAST,
    "high_lambda": HIGH_LAMBDA,
    "high_lr": HIGH_LR,
    "fine_quadrature": FINE_QUADRATURE,
}
