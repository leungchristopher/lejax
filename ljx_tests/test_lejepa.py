import dataclasses

import jax
import jax.numpy as jnp
import pytest

from ljx.losses.sigreg import SigRegConfig
from ljx.models.backbone import ViTConfig
from ljx.models.lejepa import LeJEPAConfig, invariance_loss, lejepa_loss


def tiny_config():
    # float32: CPU dot_general doesn't support fp16 (the production default)
    return LeJEPAConfig(
        backbone=ViTConfig(depth=2, embed_dim=24, num_heads=2, compute_dtype="float32"),
        projector_hidden_dims=(32, 32),
        projector_output_dim=16,
        sigreg=SigRegConfig(num_projections=32, seed=7),
    )


def views(key, batch):
    keys = jax.random.split(key, 8)
    globals_ = [jax.random.normal(k, (batch, 64, 64, 3)) for k in keys[:2]]
    locals_ = [jax.random.normal(k, (batch, 32, 32, 3)) for k in keys[2:]]
    return globals_, locals_


def test_forward_returns_finite_scalar():
    config = tiny_config()
    model = config.init()
    key = jax.random.PRNGKey(0)
    globals_, locals_ = views(key, 4)
    variables = model.init(key, globals_, locals_)

    projections = model.apply(variables, globals_, locals_)
    loss = lejepa_loss(projections, config, 0)
    assert jnp.isfinite(loss.total)


def test_total_is_convex_combination():
    config = dataclasses.replace(tiny_config(), lejepa_lambda=0.25)
    model = config.init()
    key = jax.random.PRNGKey(0)
    globals_, locals_ = views(key, 4)
    variables = model.init(key, globals_, locals_)

    projections = model.apply(variables, globals_, locals_)
    loss = lejepa_loss(projections, config, 3)

    convex = 0.75 * loss.prediction + 0.25 * loss.sigreg
    assert jnp.allclose(loss.total, convex, atol=1e-5)


def test_lambda_bounds_select_a_single_term():
    key = jax.random.PRNGKey(0)
    globals_, locals_ = views(key, 4)
    base = tiny_config()

    for lam, term in [(0.0, "prediction"), (1.0, "sigreg")]:
        config = dataclasses.replace(base, lejepa_lambda=lam)
        model = config.init()
        variables = model.init(key, globals_, locals_)
        projections = model.apply(variables, globals_, locals_)
        loss = lejepa_loss(projections, config, 0)
        assert jnp.allclose(loss.total, getattr(loss, term), atol=1e-6)


def test_invariance_is_symmetric_over_every_view():
    make = lambda v: jnp.ones((2, 4)) * v
    loss = invariance_loss([make(-1.0), make(0.0), make(1.0)])
    assert jnp.allclose(loss, 2.0 / 3.0, atol=1e-6)


def test_invariance_vanishes_when_views_agree():
    value = jnp.ones((4, 8)) * 2.5
    loss = invariance_loss([value, value, value])
    assert loss < 1e-10


def test_rejects_lambda_outside_unit_interval():
    with pytest.raises(ValueError):
        LeJEPAConfig(lejepa_lambda=1.5)
    with pytest.raises(ValueError):
        LeJEPAConfig(lejepa_lambda=-0.1)


def test_rejects_empty_view_set():
    config = tiny_config()
    model = config.init()
    key = jax.random.PRNGKey(0)
    globals_, locals_ = views(key, 4)
    variables = model.init(key, globals_, locals_)
    with pytest.raises(ValueError):
        model.apply(variables, [], [])
