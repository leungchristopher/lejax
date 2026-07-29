import jax
import jax.numpy as jnp

from ljx.data.jax_augment import ViewConfig, _crop_resize_one, _sample_crop_boxes, generate_views


def dummy_images(key, batch, size=64):
    return jax.random.uniform(key, (batch, size, size, 3))


def test_crop_boxes_stay_within_bounds():
    boxes = _sample_crop_boxes(jax.random.PRNGKey(0), 256, (0.05, 1.0))
    y0, x0, y1, x1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    assert jnp.all(y0 >= 0.0) and jnp.all(x0 >= 0.0)
    assert jnp.all(y1 <= 1.0 + 1e-6) and jnp.all(x1 <= 1.0 + 1e-6)
    assert jnp.all(y1 > y0) and jnp.all(x1 > x0)


def test_crop_resize_one_matches_source_on_a_full_frame_crop():
    # A (0,0,1,1) box at the source's own resolution is the identity.
    image = jax.random.uniform(jax.random.PRNGKey(1), (32, 32, 3))
    box = jnp.array([0.0, 0.0, 1.0, 1.0])

    out = _crop_resize_one(image, box, output_size=32)
    assert out.shape == (32, 32, 3)
    assert jnp.allclose(out, image, atol=1e-4)


def test_generate_views_shape_and_dtype():
    images = dummy_images(jax.random.PRNGKey(2), 4)
    config = ViewConfig(size=64, scale=(0.3, 1.0))

    views = generate_views(jax.random.PRNGKey(3), images, config, num_views=2)
    assert views.shape == (2, 4, 64, 64, 3)
    assert views.dtype == jnp.float32 or views.dtype == jnp.float64


def test_local_views_have_the_configured_size():
    images = dummy_images(jax.random.PRNGKey(4), 4)
    config = ViewConfig(size=32, scale=(0.05, 0.30))

    views = generate_views(jax.random.PRNGKey(5), images, config, num_views=6)
    assert views.shape == (6, 4, 32, 32, 3)


def test_views_differ_from_one_another():
    images = dummy_images(jax.random.PRNGKey(6), 2)
    config = ViewConfig(size=64, scale=(0.3, 1.0))

    views = generate_views(jax.random.PRNGKey(7), images, config, num_views=2)
    assert not jnp.allclose(views[0], views[1])


def test_same_seed_reproduces_the_same_views():
    images = dummy_images(jax.random.PRNGKey(8), 2)
    config = ViewConfig(size=64, scale=(0.3, 1.0))

    a = generate_views(jax.random.PRNGKey(42), images, config, num_views=2)
    b = generate_views(jax.random.PRNGKey(42), images, config, num_views=2)
    c = generate_views(jax.random.PRNGKey(43), images, config, num_views=2)

    assert jnp.array_equal(a, b)
    assert not jnp.array_equal(a, c)


def test_generate_views_is_differentiable_through_the_source_image():
    images = dummy_images(jax.random.PRNGKey(9), 2)
    config = ViewConfig(size=32, scale=(0.3, 1.0))

    def loss(images):
        views = generate_views(jax.random.PRNGKey(1), images, config, num_views=2)
        return jnp.sum(views * views)

    grad = jax.grad(loss)(images)
    assert jnp.all(jnp.isfinite(grad))
    assert jnp.any(grad != 0.0)
