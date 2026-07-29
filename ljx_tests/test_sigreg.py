import jax
import jax.numpy as jnp
import pytest

from ljx.losses.sigreg import SigRegConfig, sigreg_loss, sigreg_loss_at_step


def normal(key, batch, dim):
    return jax.random.normal(key, (batch, dim))


def test_forward_is_finite_and_has_gradient():
    key = jax.random.PRNGKey(0)
    embeddings = normal(key, 32, 128)

    loss, grad = jax.value_and_grad(lambda e: sigreg_loss(e, 64))(embeddings)
    assert jnp.isfinite(loss)
    assert jnp.all(jnp.isfinite(grad))
    assert jnp.any(grad != 0)


def test_collapsed_embeddings_score_higher_than_healthy():
    key = jax.random.PRNGKey(1)
    healthy = normal(key, 64, 128)
    collapsed = jnp.ones((64, 128))

    config = SigRegConfig(num_projections=128, seed=3)
    healthy_loss = sigreg_loss_at_step(config, healthy, 0)
    collapsed_loss = sigreg_loss_at_step(config, collapsed, 0)
    assert collapsed_loss > 10.0 * healthy_loss


def test_clip_value_zeroes_slices_below_the_floor():
    key = jax.random.PRNGKey(3)
    embeddings = normal(key, 64, 128)

    unclipped = sigreg_loss_at_step(SigRegConfig(num_projections=64, seed=3), embeddings, 0)
    assert unclipped > 0.0

    clipped = sigreg_loss_at_step(
        SigRegConfig(num_projections=64, seed=3, clip_value=1.0e6), embeddings, 0
    )
    assert clipped == 0.0


def test_same_seed_and_step_reproduce_directions():
    embeddings = normal(jax.random.PRNGKey(5), 32, 64)
    config_a = SigRegConfig(num_projections=16, seed=1234)
    config_b = SigRegConfig(num_projections=16, seed=1234)

    assert sigreg_loss_at_step(config_a, embeddings, 5) == sigreg_loss_at_step(config_b, embeddings, 5)


def test_different_steps_give_different_losses():
    embeddings = normal(jax.random.PRNGKey(7), 32, 64)
    config = SigRegConfig(num_projections=16, seed=7)

    assert sigreg_loss_at_step(config, embeddings, 0) != sigreg_loss_at_step(config, embeddings, 1)


def test_quadrature_matches_exact_on_a_fine_grid():
    embeddings = normal(jax.random.PRNGKey(9), 64, 16)
    fine = sigreg_loss_at_step(SigRegConfig(num_projections=64, t_max=8.0, n_points=257), embeddings, 0)
    exact = sigreg_loss_at_step(SigRegConfig(num_projections=64), embeddings, 0, exact=True)
    assert abs(float(fine) - float(exact)) < 0.2


def test_rejects_even_n_points():
    with pytest.raises(ValueError):
        SigRegConfig(n_points=16)
