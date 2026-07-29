"""
Tiny ImageNet path/label handling for the labeled (probe) pipeline.

train/ is one directory per class (DALI labels it automatically); val/ is
flat plus val_annotations.txt, so this builds an explicit file_list for it
using the same alphabetical class order DALI's file_root reader uses.
"""

from __future__ import annotations

import pathlib


def class_names(train_root: pathlib.Path) -> list[str]:
    train_root = pathlib.Path(train_root)
    names = sorted(p.name for p in train_root.iterdir() if p.is_dir())
    if not names:
        raise ValueError(f"found no class directories under {train_root}")
    return names


def build_train_file_list(dataset_root: pathlib.Path, class_to_index: dict[str, int]) -> pathlib.Path:
    """train/<wnid>/images/*.jpeg -> explicit file_list, same layout as
    build_val_file_list, so both feed the same labeled loader."""
    train_root = pathlib.Path(dataset_root) / "train"
    lines = []
    for wnid, index in class_to_index.items():
        for image_path in sorted((train_root / wnid / "images").glob("*.jpeg")):
            lines.append(f"{wnid}/images/{image_path.name} {index}")

    file_list = train_root / "file_list.txt"
    file_list.write_text("\n".join(lines) + "\n")
    return file_list


def build_val_file_list(dataset_root: pathlib.Path, class_to_index: dict[str, int]) -> pathlib.Path:
    """val_annotations.txt line: <filename>\\t<wnid>\\t<x0>\\t<y0>\\t<x1>\\t<y1>."""
    val_root = pathlib.Path(dataset_root) / "val"
    annotations = (val_root / "val_annotations.txt").read_text().strip().splitlines()

    lines = []
    for line in annotations:
        filename, wnid = line.split("\t")[:2]
        lines.append(f"images/{filename} {class_to_index[wnid]}")

    file_list = val_root / "file_list.txt"
    file_list.write_text("\n".join(lines) + "\n")
    return file_list


def num_classes(dataset_root: pathlib.Path) -> int:
    return len(class_names(pathlib.Path(dataset_root) / "train"))


IMAGE_EXTENSIONS = {"jpeg", "jpg", "png", "bmp", "tif", "tiff"}


def _scan_images(root: pathlib.Path) -> list[pathlib.Path]:
    paths = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lstrip(".").lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"found no images under {root}")
    return paths


def split_pretrain_file_lists(
    dataset_path: pathlib.Path, num_valid_images: int, artifact_directory: pathlib.Path
) -> tuple[pathlib.Path, pathlib.Path]:
    """Sorted recursive scan, tail split off as validation. Writes two
    file_list manifests (relative paths, dummy label — DALI's reader
    requires some label column) under artifact_directory."""
    dataset_path = pathlib.Path(dataset_path)
    artifact_directory = pathlib.Path(artifact_directory)

    paths = _scan_images(dataset_path)
    num_valid_images = min(num_valid_images, max(len(paths) - 1, 0))
    boundary = len(paths) - num_valid_images

    artifact_directory.mkdir(parents=True, exist_ok=True)
    train_list = artifact_directory / "train_files.txt"
    valid_list = artifact_directory / "valid_files.txt"

    train_list.write_text(
        "\n".join(f"{p.relative_to(dataset_path)} 0" for p in paths[:boundary]) + "\n"
    )
    valid_list.write_text(
        "\n".join(f"{p.relative_to(dataset_path)} 0" for p in paths[boundary:]) + "\n"
    )
    return train_list, valid_list
