"""Per-epoch metrics logging: CSV file plus a one-time config dump.

Print statements alone don't survive a Colab disconnect and don't carry the
prediction/sigreg breakdown; this persists both to artifact_directory.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import pathlib
import time


class MetricsLogger:
    def __init__(self, artifact_directory: pathlib.Path, filename: str = "metrics.csv"):
        self.path = pathlib.Path(artifact_directory) / filename
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._wrote_header = self.path.exists() and self.path.stat().st_size > 0

    def log(self, **fields) -> None:
        fields = {"timestamp": time.time(), **fields}
        write_header = not self._wrote_header
        with self.path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(fields.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(fields)
        self._wrote_header = True


def dump_config(artifact_directory: pathlib.Path, config, filename: str = "config.json") -> None:
    path = pathlib.Path(artifact_directory) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataclasses.asdict(config), indent=2, default=str))
