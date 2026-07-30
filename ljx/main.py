"""CLI for the JAX LeJEPA training/eval pipeline.

Dataset must already be materialized (see scripts/materialize_hf_tiny_imagenet.py):

    tiny-imagenet-200/
      train/<wnid>/images/*.jpeg
      val/images/*.jpeg
      val/val_annotations.txt
"""

from __future__ import annotations

import argparse
import pathlib

from ljx.training.eval import EvalConfig, EvalRun
from ljx.training.train_loop import TrainingConfig, TrainingRun


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ljx", description=__doc__)
    parser.add_argument("--mode", choices=["pretrain", "evaluate"], default="pretrain")
    parser.add_argument("--dataset-path", required=True, type=pathlib.Path)
    parser.add_argument("--checkpoint-path", type=pathlib.Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--resume-from-epoch", type=int, default=None)
    parser.add_argument("--artifact-directory", type=pathlib.Path, default=pathlib.Path("artifacts"))
    parser.add_argument("--seed", type=int, default=0)
    return parser


def training_config(args: argparse.Namespace) -> TrainingConfig:
    config = TrainingConfig(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        checkpoint_every=args.checkpoint_every,
        resume_from_epoch=args.resume_from_epoch,
        seed=args.seed,
    )
    return config.dry_run() if args.dry_run else config


def eval_config(args: argparse.Namespace) -> EvalConfig:
    default = EvalConfig()
    return EvalConfig(
        num_epochs=default.num_epochs if args.epochs == 100 else args.epochs,
        num_workers=args.num_workers,
        seed=args.seed,
    )


def pretrain(args: argparse.Namespace) -> None:
    config = training_config(args)
    run = TrainingRun(config, args.dataset_path, args.artifact_directory)
    run.launch()


def evaluate(args: argparse.Namespace) -> None:
    if args.checkpoint_path is None:
        raise SystemExit("--mode evaluate requires --checkpoint-path")

    run = EvalRun(
        eval_config(args),
        args.dataset_path,
        args.checkpoint_path,
        args.artifact_directory / "probe",
    )
    print(f"linear probe: {run.num_classes()} classes")
    run.launch()


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "pretrain":
        pretrain(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
