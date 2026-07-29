"""CPU-only JPEG decode + batching, no augmentation. Feeds jax_augment.py."""

from __future__ import annotations

import pathlib
import queue
import random
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image


def _load_image(path: pathlib.Path, size: int) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB")
        if img.size != (size, size):
            img = img.resize((size, size), Image.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0


class RawImageLoader:
    """Iterates batches of decoded images as float32 [0, 1] numpy arrays.

    `file_list` lines are `<relative path> <label>` (label unused here)."""

    def __init__(
        self,
        file_list: pathlib.Path,
        dataset_root: pathlib.Path,
        batch_size: int,
        size: int = 64,
        shuffle: bool = True,
        seed: int = 0,
        num_workers: int = 4,
        prefetch: int = 4,
    ):
        dataset_root = pathlib.Path(dataset_root)
        self.paths = [
            dataset_root / line.split(" ")[0]
            for line in pathlib.Path(file_list).read_text().strip().splitlines()
        ]
        self.batch_size = batch_size
        self.size = size
        self.shuffle = shuffle
        self.rng = random.Random(seed)
        self.num_workers = num_workers
        self.prefetch = prefetch

    def __len__(self) -> int:
        return len(self.paths) // self.batch_size

    def __iter__(self):
        order = list(range(len(self.paths)))
        if self.shuffle:
            self.rng.shuffle(order)

        batches = [
            order[i : i + self.batch_size]
            for i in range(0, len(order) - self.batch_size + 1, self.batch_size)
        ]

        q: queue.Queue = queue.Queue(maxsize=self.prefetch)
        sentinel = object()

        def worker():
            with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
                for indices in batches:
                    paths = [self.paths[i] for i in indices]
                    images = list(pool.map(lambda p: _load_image(p, self.size), paths))
                    q.put(np.stack(images, axis=0))
            q.put(sentinel)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        while True:
            item = q.get()
            if item is sentinel:
                break
            yield item


class CachedImageLoader:
    """Decodes the whole `file_list` into RAM once at construction, instead of
    per-batch like RawImageLoader. Tiny ImageNet fits easily (~1.2GB for
    100k 64x64 images), so this trades a one-time up-front decode for zero
    per-step CPU JPEG decode — the actual bottleneck on 2-vCPU Colab
    instances, where RawImageLoader's per-batch decode can't keep the GPU
    fed no matter how many worker threads you throw at it."""

    def __init__(
        self,
        file_list: pathlib.Path,
        dataset_root: pathlib.Path,
        batch_size: int,
        size: int = 64,
        shuffle: bool = True,
        seed: int = 0,
        num_workers: int = 4,
    ):
        dataset_root = pathlib.Path(dataset_root)
        paths = [
            dataset_root / line.split(" ")[0]
            for line in pathlib.Path(file_list).read_text().strip().splitlines()
        ]
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            images = list(pool.map(lambda p: _load_image(p, size), paths))
        self.images = np.stack(images, axis=0)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.images) // self.batch_size

    def __iter__(self):
        order = list(range(len(self.images)))
        if self.shuffle:
            self.rng.shuffle(order)

        for i in range(0, len(order) - self.batch_size + 1, self.batch_size):
            yield self.images[order[i : i + self.batch_size]]


class LabeledImageLoader(RawImageLoader):
    """Same batching/prefetch as RawImageLoader, but also yields the label
    column from `file_list` (`<relative path> <label>` — label used here,
    unlike RawImageLoader's pretraining split lists, whose label column is
    just a placeholder to match this shared file_list format)."""

    def __init__(self, file_list: pathlib.Path, dataset_root: pathlib.Path, batch_size: int, **kwargs):
        super().__init__(file_list, dataset_root, batch_size, **kwargs)
        self.labels = [
            int(line.split(" ")[1])
            for line in pathlib.Path(file_list).read_text().strip().splitlines()
        ]

    def __iter__(self):
        order = list(range(len(self.paths)))
        if self.shuffle:
            self.rng.shuffle(order)

        batches = [
            order[i : i + self.batch_size]
            for i in range(0, len(order) - self.batch_size + 1, self.batch_size)
        ]

        q: queue.Queue = queue.Queue(maxsize=self.prefetch)
        sentinel = object()

        def worker():
            with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
                for indices in batches:
                    paths = [self.paths[i] for i in indices]
                    images = list(pool.map(lambda p: _load_image(p, self.size), paths))
                    labels = np.array([self.labels[i] for i in indices], dtype=np.int32)
                    q.put((np.stack(images, axis=0), labels))
            q.put(sentinel)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        while True:
            item = q.get()
            if item is sentinel:
                break
            yield item
