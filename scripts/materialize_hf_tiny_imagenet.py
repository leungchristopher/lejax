#!/usr/bin/env python3
"""Materialize the HuggingFace Tiny ImageNet parquet files into the canonical
on-disk layout this project's loaders expect.

    tiny-imagenet-200/
      train/<wnid>/images/*.jpeg
      val/images/*.jpeg
      val/val_annotations.txt
      wnids.txt

Why materialize rather than read parquet directly: every loader in the crate
(`TinyImageNetDataset` for pretraining, `LabeledDataset` for the probe) already
consumes this directory layout, and it is the same layout the official
cs231n archive unpacks into. Converting once keeps a single code path in Rust
and avoids pulling an Arrow dependency into an already slow build.

The images are stored as encoded JPEG bytes, so they are written straight to
disk without a decode/re-encode round trip — no generation loss, and fast.

Usage:
    python3 scripts/materialize_hf_tiny_imagenet.py \
        --parquet-dir datasets/hf-tiny-imagenet/data \
        --dest datasets
"""

import argparse
import json
import pathlib
import sys
import urllib.request

import pyarrow.parquet as pq

DATASET = "zh-plus/tiny-imagenet"
INFOS_URL = f"https://huggingface.co/datasets/{DATASET}/resolve/main/dataset_infos.json"
DATASET_DIR = "tiny-imagenet-200"
COMPLETION_MARKER = ".extraction-complete"


def label_names():
    """WordNet ids in HuggingFace label order (index 0..199)."""
    with urllib.request.urlopen(INFOS_URL, timeout=60) as response:
        infos = json.load(response)
    config = next(iter(infos.values()))
    names = config["features"]["label"]["names"]
    if len(names) != 200:
        sys.exit(f"expected 200 classes, got {len(names)}")
    return names


def find_parquet(directory, prefix):
    matches = sorted(pathlib.Path(directory).glob(f"{prefix}-*.parquet"))
    if not matches:
        sys.exit(f"no {prefix} parquet found in {directory}")
    return matches


def write_train(paths, root, names):
    """train/<wnid>/images/<file> — one directory per class."""
    for wnid in names:
        (root / "train" / wnid / "images").mkdir(parents=True, exist_ok=True)

    written = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=2048, columns=["image", "label"]):
            for image, label in zip(
                batch.column("image").to_pylist(), batch.column("label").to_pylist()
            ):
                wnid = names[label]
                target = root / "train" / wnid / "images" / f"{wnid}_{written}.jpeg"
                target.write_bytes(image["bytes"])
                written += 1
    return written


def write_val(paths, root, names):
    """val/images/ plus val_annotations.txt, matching the official archive.

    The labels live only in the annotations file, exactly as they do upstream,
    so the Rust loader's flat-layout branch is what gets exercised.
    """
    images = root / "val" / "images"
    images.mkdir(parents=True, exist_ok=True)

    annotations = []
    written = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=2048, columns=["image", "label"]):
            for image, label in zip(
                batch.column("image").to_pylist(), batch.column("label").to_pylist()
            ):
                name = f"val_{written}.jpeg"
                (images / name).write_bytes(image["bytes"])
                # Bounding boxes are unused here; the loader reads fields 1-2.
                annotations.append(f"{name}\t{names[label]}\t0\t0\t63\t63")
                written += 1

    (root / "val" / "val_annotations.txt").write_text("\n".join(annotations) + "\n")
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = pathlib.Path(args.dest) / DATASET_DIR
    if (root / COMPLETION_MARKER).exists() and not args.force:
        print(f"{root} already complete; pass --force to rebuild")
        return

    names = label_names()
    print(f"{len(names)} classes, first: {names[0]}")

    train = write_train(find_parquet(args.parquet_dir, "train"), root, names)
    print(f"wrote {train} training images")

    val = write_val(find_parquet(args.parquet_dir, "valid"), root, names)
    print(f"wrote {val} validation images")

    (root / "wnids.txt").write_text("\n".join(names) + "\n")

    # Written last, so an interrupted run is detected as incomplete by
    # `ensure_tiny_imagenet` rather than silently used.
    (root / COMPLETION_MARKER).write_text(f"materialized from {DATASET}\n")
    print(f"dataset ready at {root}")


if __name__ == "__main__":
    main()
