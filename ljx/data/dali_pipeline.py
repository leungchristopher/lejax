"""
DALI augments multi-crop: decode/crop/jitter/normalize all on GPU.

Note: contrast pivots around a fixed 127.5 centre, not each view's own mean.
"""

from __future__ import annotations

from dataclasses import dataclass

import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali import pipeline_def
from nvidia.dali.plugin.jax import DALIGenericIterator

NUM_GLOBAL_VIEWS = 2
NUM_LOCAL_VIEWS = 6

TINY_IMAGENET_MEAN = (0.4802, 0.4481, 0.3975)
TINY_IMAGENET_STD = (0.2770, 0.2691, 0.2821)


@dataclass(frozen=True)
class MultiCropConfig:
    global_size: int = 64
    global_scale: tuple[float, float] = (0.30, 1.0)
    local_size: int = 32
    local_scale: tuple[float, float] = (0.05, 0.30)
    flip_probability: float = 0.5
    brightness: float = 0.2
    contrast: float = 0.2
    mean: tuple[float, float, float] = TINY_IMAGENET_MEAN
    std: tuple[float, float, float] = TINY_IMAGENET_STD


def _view(images, scale: tuple[float, float], size: int, config: MultiCropConfig):
    cropped = fn.random_resized_crop(
        images,
        size=[size, size],
        random_area=list(scale),
        random_aspect_ratio=[3.0 / 4.0, 4.0 / 3.0],
        device="gpu",
    )

    mirror = fn.random.coin_flip(probability=config.flip_probability)
    brightness = fn.random.uniform(range=(1.0 - config.brightness, 1.0 + config.brightness))
    contrast = fn.random.uniform(range=(1.0 - config.contrast, 1.0 + config.contrast))
    jittered = fn.brightness_contrast(
        cropped, brightness=brightness, contrast=contrast, contrast_center=127.5
    )

    mean = [m * 255.0 for m in config.mean]
    std = [s * 255.0 for s in config.std]
    return fn.crop_mirror_normalize(
        jittered,
        mirror=mirror,
        mean=mean,
        std=std,
        output_layout="HWC",
        dtype=types.FLOAT,
    )


@pipeline_def
def multicrop_pipeline(
    file_root: str,
    config: MultiCropConfig,
    shuffle: bool,
    seed: int,
    file_list: str | None = None,
):
    """2 global (64px) + 6 local (32px) views per image. `file_list` (see
    tiny_imagenet.split_pretrain_file_lists) selects an explicit split."""
    jpegs, labels = fn.readers.file(
        file_root=file_root,
        file_list=file_list,
        random_shuffle=shuffle,
        seed=seed,
        name="reader",
    )
    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)

    outputs = [_view(images, config.global_scale, config.global_size, config) for _ in range(NUM_GLOBAL_VIEWS)]
    outputs += [_view(images, config.local_scale, config.local_size, config) for _ in range(NUM_LOCAL_VIEWS)]
    return tuple(outputs) + (labels,)


def _output_names() -> list[str]:
    names = [f"global_{i}" for i in range(NUM_GLOBAL_VIEWS)]
    names += [f"local_{i}" for i in range(NUM_LOCAL_VIEWS)]
    names.append("labels")
    return names


@pipeline_def
def labeled_pipeline(
    file_root: str | None,
    file_list: str | None,
    size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    shuffle: bool,
    seed: int,
):
    """Resize + centre-crop + normalize, no flip/jitter. For the linear probe."""
    jpegs, labels = fn.readers.file(
        file_root=file_root,
        file_list=file_list,
        random_shuffle=shuffle,
        seed=seed,
        name="reader",
    )
    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
    images = fn.resize(images, size=size, mode="not_smaller")
    images = fn.crop(images, crop=(size, size))
    images = fn.crop_mirror_normalize(
        images,
        mean=[m * 255.0 for m in mean],
        std=[s * 255.0 for s in std],
        output_layout="HWC",
        dtype=types.FLOAT,
    )
    return images, labels


def build_labeled_iterator(
    *,
    file_root: str | None = None,
    file_list: str | None = None,
    batch_size: int,
    num_threads: int,
    device_id: int = 0,
    size: int = 64,
    mean: tuple[float, float, float] = TINY_IMAGENET_MEAN,
    std: tuple[float, float, float] = TINY_IMAGENET_STD,
    shuffle: bool = True,
    seed: int = 0,
    prefetch_queue_depth: int = 4,
) -> DALIGenericIterator:
    """Pass exactly one of file_root (class-per-dir tree) or file_list."""
    if (file_root is None) == (file_list is None):
        raise ValueError("pass exactly one of file_root or file_list")

    pipeline = labeled_pipeline(
        file_root=file_root,
        file_list=file_list,
        size=size,
        mean=mean,
        std=std,
        shuffle=shuffle,
        seed=seed,
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        prefetch_queue_depth=prefetch_queue_depth,
    )
    return DALIGenericIterator(
        pipeline,
        output_map=["images", "labels"],
        reader_name="reader",
        auto_reset=True,
    )


def build_multicrop_iterator(
    file_root: str,
    batch_size: int,
    num_threads: int,
    device_id: int = 0,
    config: MultiCropConfig | None = None,
    shuffle: bool = True,
    seed: int = 0,
    prefetch_queue_depth: int = 4,
    file_list: str | None = None,
) -> DALIGenericIterator:
    """Yields dicts: global_0, global_1, local_0..5 (each [batch,size,size,3]
    float32), labels."""
    config = config or MultiCropConfig()
    pipeline = multicrop_pipeline(
        file_root=file_root,
        file_list=file_list,
        config=config,
        shuffle=shuffle,
        seed=seed,
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        prefetch_queue_depth=prefetch_queue_depth,
    )
    return DALIGenericIterator(
        pipeline,
        output_map=_output_names(),
        reader_name="reader",
        auto_reset=True,
    )
