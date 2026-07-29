"""ViT-Tiny backbone, NHWC layout."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

IN_CHANNELS = 3
EMBED_INIT_STD = 0.02


@dataclass(frozen=True)
class ViTConfig:
    image_size: int = 64
    patch_size: int = 8
    embed_dim: int = 192
    depth: int = 12
    num_heads: int = 3
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    compute_dtype: str = "float16"

    def __post_init__(self) -> None:
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size {self.image_size} must be a multiple of "
                f"patch_size {self.patch_size}"
            )
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim {self.embed_dim} must be a multiple of "
                f"num_heads {self.num_heads}"
            )

    @property
    def grid_size(self) -> int:
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.grid_size * self.grid_size

    def init(self) -> "ViT":
        return ViT(config=self)


class _EncoderBlock(nn.Module):
    """Pre-norm: LN -> fused MHSA -> residual, LN -> MLP -> residual."""

    embed_dim: int
    num_heads: int
    mlp_dim: int
    dropout: float
    deterministic: bool
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, carry: jnp.ndarray, _):
        x = carry
        batch, seq, _ = x.shape
        head_dim = self.embed_dim // self.num_heads

        y = nn.LayerNorm(dtype=self.dtype)(x)
        qkv = nn.Dense(3 * self.embed_dim, dtype=self.dtype, name="qkv")(y)
        qkv = qkv.reshape(batch, seq, 3, self.num_heads, head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        attn = jax.nn.dot_product_attention(q, k, v)
        attn = attn.reshape(batch, seq, self.embed_dim)
        y = nn.Dense(self.embed_dim, dtype=self.dtype, name="attn_out")(attn)
        y = nn.Dropout(self.dropout)(y, deterministic=self.deterministic)
        x = x + y

        y = nn.LayerNorm(dtype=self.dtype)(x)
        y = nn.Dense(self.mlp_dim, dtype=self.dtype)(y)
        y = nn.gelu(y)
        y = nn.Dropout(self.dropout)(y, deterministic=self.deterministic)
        y = nn.Dense(self.embed_dim, dtype=self.dtype)(y)
        y = nn.Dropout(self.dropout)(y, deterministic=self.deterministic)
        return x + y, None


def resample_matrix(source: int, target: int) -> jnp.ndarray:
    """Bilinear resample matrix, half-pixel centers. [target, source]."""
    weights = np.zeros((target, source), dtype=np.float32)
    scale = source / target
    for i in range(target):
        center = min(max((i + 0.5) * scale - 0.5, 0.0), source - 1)
        low = int(np.floor(center))
        high = min(low + 1, source - 1)
        frac = center - low
        weights[i, low] += 1.0 - frac
        weights[i, high] += frac
    return jnp.asarray(weights)


class ViT(nn.Module):
    config: ViTConfig

    @nn.compact
    def __call__(self, images: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """images: [batch, height, width, 3] -> [batch, embed_dim]."""
        cfg = self.config
        compute_dtype = jnp.dtype(cfg.compute_dtype)
        batch, height, width, channels = images.shape
        if channels != IN_CHANNELS:
            raise ValueError(f"ViT expects {IN_CHANNELS}-channel input, got {channels}")
        if height % cfg.patch_size != 0 or width % cfg.patch_size != 0:
            raise ValueError(
                f"input {height}x{width} must divide evenly into "
                f"{cfg.patch_size}x{cfg.patch_size} patches"
            )
        grid_h, grid_w = height // cfg.patch_size, width // cfg.patch_size

        patch_embed = nn.Conv(
            features=cfg.embed_dim,
            kernel_size=(cfg.patch_size, cfg.patch_size),
            strides=(cfg.patch_size, cfg.patch_size),
            padding="VALID",
            name="patch_embed",
            dtype=compute_dtype,
        )
        tokens = patch_embed(images.astype(compute_dtype)).reshape(batch, grid_h * grid_w, cfg.embed_dim)

        normal = nn.initializers.normal(stddev=EMBED_INIT_STD)
        cls_token = self.param("cls_token", normal, (1, 1, cfg.embed_dim))
        pos_embed = self.param(
            "pos_embed", normal, (1, cfg.num_patches + 1, cfg.embed_dim)
        )

        cls = jnp.broadcast_to(cls_token, (batch, 1, cfg.embed_dim)).astype(compute_dtype)
        tokens = jnp.concatenate([cls, tokens], axis=1)
        pos = self._positional_embedding(pos_embed, grid_h, grid_w).astype(compute_dtype)
        tokens = tokens + pos

        mlp_dim = int(cfg.embed_dim * cfg.mlp_ratio)
        scanned_encoder = nn.scan(
            _EncoderBlock,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            length=cfg.depth,
        )
        tokens, _ = scanned_encoder(
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            mlp_dim=mlp_dim,
            dropout=cfg.dropout,
            deterministic=deterministic,
            dtype=compute_dtype,
        )(tokens, None)

        tokens = nn.LayerNorm(dtype=compute_dtype)(tokens)
        return tokens[:, 0, :].astype(jnp.float32)

    def _positional_embedding(
        self, pos_embed: jnp.ndarray, grid_h: int, grid_w: int
    ) -> jnp.ndarray:
        cfg = self.config
        if grid_h == cfg.grid_size and grid_w == cfg.grid_size:
            return pos_embed

        source = cfg.grid_size
        dim = cfg.embed_dim

        cls = pos_embed[:, :1, :]
        grid = pos_embed[:, 1:, :].reshape(1, source, source, dim)
        grid = jnp.transpose(grid, (0, 3, 1, 2))

        rows = resample_matrix(source, grid_h).reshape(1, 1, grid_h, source)
        cols = resample_matrix(source, grid_w).T.reshape(1, 1, source, grid_w)

        grid = rows @ grid @ cols
        grid = jnp.transpose(grid, (0, 2, 3, 1)).reshape(1, grid_h * grid_w, dim)

        return jnp.concatenate([cls, grid], axis=1)
