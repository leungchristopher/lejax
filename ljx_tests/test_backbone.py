import jax
import jax.numpy as jnp
import pytest

from ljx.models.backbone import ViT, ViTConfig, resample_matrix


def tiny_config():
    # float32: CPU dot_general doesn't support fp16 (the production default)
    return ViTConfig(depth=2, embed_dim=24, num_heads=2, compute_dtype="float32")


def images(key, batch, size):
    return jax.random.normal(key, (batch, size, size, 3))


def test_config_defaults():
    config = ViTConfig()
    assert (config.image_size, config.patch_size, config.embed_dim, config.depth, config.num_heads) == (
        64,
        8,
        192,
        12,
        3,
    )
    assert config.grid_size == 8
    assert config.num_patches == 64


def test_forward_shape_at_native_and_other_resolutions():
    model = ViT(config=tiny_config())
    key = jax.random.PRNGKey(0)
    variables = model.init(key, images(key, 2, 64))

    for size in (64, 32, 96):
        out = model.apply(variables, images(key, 2, size))
        assert out.shape == (2, 24)


def test_forward_rejects_non_multiple_of_patch_size():
    model = ViT(config=tiny_config())
    key = jax.random.PRNGKey(0)
    variables = model.init(key, images(key, 1, 64))
    with pytest.raises(ValueError):
        model.apply(variables, images(key, 1, 60))


def test_gradients_reach_cls_token_and_pos_embed_at_multiple_sizes():
    model = ViT(config=tiny_config())
    key = jax.random.PRNGKey(0)
    variables = model.init(key, images(key, 2, 64))

    for size in (64, 32):
        def loss_fn(params, size=size):
            out = model.apply({"params": params}, images(key, 2, size))
            return jnp.sum(out * out)

        grads = jax.grad(loss_fn)(variables["params"])
        for name in ("cls_token", "pos_embed"):
            grad = grads[name]
            assert jnp.all(jnp.isfinite(grad))
            assert jnp.any(grad != 0)


def test_resample_matrix_rows_sum_to_one():
    for source, target in [(8, 4), (8, 16), (8, 8), (8, 3), (3, 8)]:
        matrix = resample_matrix(source, target)
        assert matrix.shape == (target, source)
        assert jnp.allclose(jnp.sum(matrix, axis=1), 1.0, atol=1e-6)


def test_resample_matrix_is_identity_when_sizes_match():
    matrix = resample_matrix(5, 5)
    assert jnp.allclose(matrix, jnp.eye(5))


def test_config_rejects_bad_shapes():
    with pytest.raises(ValueError):
        ViTConfig(image_size=60, patch_size=8)
    with pytest.raises(ValueError):
        ViTConfig(embed_dim=50, num_heads=3)
