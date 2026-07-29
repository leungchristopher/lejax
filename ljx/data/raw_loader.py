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
