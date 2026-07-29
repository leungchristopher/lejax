"""
SIGReg: sketched isotropic Gaussian regularization.
Reference implementation: https://github.com/galilai-group/lejepa
Uses n_points=33 rather than the reference's 17: the 17-point grid can be
aliased by a non-Gaussian sample set that happens to match the reference
characteristic function at exactly those 17 points: 33 reduces the probability.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

NORM_EPS = 1e-12


@dataclass(frozen=True)
class SigRegConfig:
    num_projections: int = 1000
    t_max: float = 3.0
    n_points: int = 33
    clip_value: float | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if self.num_projections <= 0:
            raise ValueError("SigReg requires at least one projection")
        if self.n_points % 2 == 0:
            raise ValueError(f"n_points must be odd, got {self.n_points}")
        if self.t_max <= 0:
            raise ValueError(f"t_max must be positive, got {self.t_max}")


def _directions(config: SigRegConfig, dim: int, step: int) -> jnp.ndarray:
    """[dim, num_projections], unit-norm columns, seeded by (config.seed, step)."""
    key = jax.random.fold_in(jax.random.PRNGKey(config.seed), step)
    values = jax.random.normal(key, (dim, config.num_projections))
    norms = jnp.sqrt(jnp.sum(values * values, axis=0, keepdims=True))
    return values / jnp.maximum(norms, NORM_EPS)


def _epps_pulley_quadrature(samples: jnp.ndarray, t_max: float, n_points: int) -> jnp.ndarray:
    """samples: [num_projections, batch] -> [num_projections]."""
    num_projections, batch = samples.shape
    n = batch

    dt = t_max / (n_points - 1)
    k = jnp.arange(n_points)
    t = dt * k
    phi = jnp.exp(-0.5 * t * t)
    trapezoid = jnp.where((k == 0) | (k == n_points - 1), dt, 2.0 * dt)
    weights = trapezoid * phi

    x_t = samples[:, :, None] * t[None, None, :]
    cos_mean = jnp.mean(jnp.cos(x_t), axis=1)
    sin_mean = jnp.mean(jnp.sin(x_t), axis=1)

    real_error = cos_mean - phi[None, :]
    error = real_error * real_error + sin_mean * sin_mean

    return (error @ weights) * n


def _epps_pulley_exact(samples: jnp.ndarray) -> jnp.ndarray:
    """samples: [num_projections, batch] -> [num_projections]."""
    num_projections, batch = samples.shape
    n = batch

    xi = samples[:, :, None]
    xj = samples[:, None, :]
    diff = xi - xj
    pairwise = jnp.sum(jnp.exp(-0.5 * diff * diff), axis=(1, 2)) / (n * n)

    cross = jnp.sum(jnp.exp(-0.25 * samples * samples), axis=1) * (-2.0 / (n * jnp.sqrt(2.0)))

    return (pairwise + cross + 1.0 / jnp.sqrt(3.0)) * (n * jnp.sqrt(2.0 * jnp.pi))


def sigreg_loss_at_step(
    config: SigRegConfig, embeddings: jnp.ndarray, step: int, exact: bool = False
) -> jnp.ndarray:
    """embeddings: [batch, dim] -> scalar."""
    batch, dim = embeddings.shape
    if batch == 0:
        raise ValueError("SigReg requires a non-empty batch")

    directions = _directions(config, dim, step)
    samples = (embeddings @ directions).T

    statistics = (
        _epps_pulley_exact(samples)
        if exact
        else _epps_pulley_quadrature(samples, config.t_max, config.n_points)
    )

    if config.clip_value is not None:
        statistics = jnp.where(statistics < config.clip_value, 0.0, statistics)

    return jnp.mean(statistics)


def sigreg_loss(embeddings: jnp.ndarray, num_projections: int, step: int = 0) -> jnp.ndarray:
    return sigreg_loss_at_step(SigRegConfig(num_projections=num_projections), embeddings, step)
