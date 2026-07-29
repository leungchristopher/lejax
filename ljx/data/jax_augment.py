"""Multi-crop augmentation as pure JAX ops. No DALI; CPU only decodes JPEGs
(raw_loader.py), crop/resize/flip/jitter/normalize run here on-device.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.scipy.ndimage

ASPECT_RATIO_RANGE = (3.0 / 4.0, 4.0 / 3.0)

TINY_IMAGENET_MEAN = jnp.array([0.4802, 0.4481, 0.3975])
TINY_IMAGENET_STD = jnp.array([0.2770, 0.2691, 0.2821])


@dataclass(frozen=True)
class ViewConfig:
    size: int
    scale: tuple[float, float]
    flip_probability: float = 0.5
    brightness: float = 0.2
    contrast: float = 0.2


def _sample_crop_boxes(rng: jax.Array, n: int, scale: tuple[float, float]) -> jnp.ndarray:
    """[n, 4] rows of (y0, x0, y1, x1), normalized to the source's [0, 1] extent."""
    area_key, aspect_key, y_key, x_key = jax.random.split(rng, 4)

    area = jax.random.uniform(area_key, (n,), minval=scale[0], maxval=scale[1])
    log_ratio = jax.random.uniform(
        aspect_key, (n,), minval=jnp.log(ASPECT_RATIO_RANGE[0]), maxval=jnp.log(ASPECT_RATIO_RANGE[1])
    )
    ratio = jnp.exp(log_ratio)

    w = jnp.clip(jnp.sqrt(area * ratio), 0.0, 1.0)
    h = jnp.clip(jnp.sqrt(area / ratio), 0.0, 1.0)

    x0 = jax.random.uniform(x_key, (n,)) * (1.0 - w)
    y0 = jax.random.uniform(y_key, (n,)) * (1.0 - h)

    return jnp.stack([y0, x0, y0 + h, x0 + w], axis=1)


def _crop_resize_one(image: jnp.ndarray, box: jnp.ndarray, output_size: int) -> jnp.ndarray:
    """image: [H, W, 3] float32 in [0, 1]. box: (y0, x0, y1, x1) normalized."""
    height, width, _ = image.shape
    y0, x0, y1, x1 = box

    i = jnp.arange(output_size)
    j = jnp.arange(output_size)
    src_y = y0 * height + (i + 0.5) / output_size * (y1 - y0) * height - 0.5
    src_x = x0 * width + (j + 0.5) / output_size * (x1 - x0) * width - 0.5
    grid_y, grid_x = jnp.meshgrid(src_y, src_x, indexing="ij")
    coords = jnp.stack([grid_y, grid_x], axis=0)

    def sample_channel(channel):
        return jax.scipy.ndimage.map_coordinates(channel, coords, order=1, mode="nearest")

    return jax.vmap(sample_channel, in_axes=2, out_axes=2)(image)


_crop_resize_batch = jax.vmap(_crop_resize_one, in_axes=(0, 0, None))


def _maybe_flip(image: jnp.ndarray, flip: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(flip, image[:, ::-1, :], image)


def _jitter(image: jnp.ndarray, brightness: jnp.ndarray, contrast: jnp.ndarray) -> jnp.ndarray:
    image = image * brightness
    mean = jnp.mean(image)
    image = mean + (image - mean) * contrast
    return jnp.clip(image, 0.0, 1.0)


def generate_views(
    rng: jax.Array, images: jnp.ndarray, config: ViewConfig, num_views: int
) -> jnp.ndarray:
    """images: [batch, H, W, 3] float32 in [0, 1] -> [num_views, batch, size, size, 3]."""
    batch = images.shape[0]
    total = num_views * batch
    box_key, flip_key, bright_key, contrast_key = jax.random.split(rng, 4)

    boxes = _sample_crop_boxes(box_key, total, config.scale)
    flips = jax.random.bernoulli(flip_key, config.flip_probability, (total,))
    brightness = jax.random.uniform(
        bright_key, (total,), minval=1.0 - config.brightness, maxval=1.0 + config.brightness
    )
    contrast = jax.random.uniform(
        contrast_key, (total,), minval=1.0 - config.contrast, maxval=1.0 + config.contrast
    )

    tiled = jnp.tile(images[None], (num_views, 1, 1, 1, 1)).reshape(total, *images.shape[1:])

    cropped = _crop_resize_batch(tiled, boxes, config.size)
    flipped = jax.vmap(_maybe_flip)(cropped, flips)
    jittered = jax.vmap(_jitter)(flipped, brightness, contrast)
    normalized = (jittered - TINY_IMAGENET_MEAN) / TINY_IMAGENET_STD

    return normalized.reshape(num_views, batch, config.size, config.size, 3)
