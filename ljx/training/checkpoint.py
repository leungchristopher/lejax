"""Checkpoint save/load with msgpack"""

from __future__ import annotations

import pathlib

from flax import serialization


def save(directory: pathlib.Path, epoch: int, payload: dict) -> pathlib.Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"model-{epoch}.msgpack"
    path.write_bytes(serialization.to_bytes(payload))
    return path


def load(path: pathlib.Path, template: dict) -> dict:
    """`template` must have the same pytree shape as what was saved."""
    return serialization.from_bytes(template, path.read_bytes())
