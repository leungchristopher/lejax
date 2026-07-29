"""Linear-probe evaluation loop using ljx.data.raw_loader instead of DALI."""

from __future__ import annotations

import pathlib
import time

import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze
from flax.training import dynamic_scale as dynamic_scale_lib

from ljx.data.jax_augment import TINY_IMAGENET_MEAN, TINY_IMAGENET_STD
from ljx.data.raw_loader import LabeledImageLoader
from ljx.data.tiny_imagenet import build_train_file_list, build_val_file_list, class_names
from ljx.models.backbone import ViT
from ljx.models.linear import LinearClassifier, LinearClassifierConfig
from ljx.training import checkpoint, metrics
from ljx.training.eval import EvalConfig, ProbeState, _optimizer, eval_step, load_backbone_params, train_step


def _normalize(images: jnp.ndarray) -> jnp.ndarray:
    return (images - TINY_IMAGENET_MEAN) / TINY_IMAGENET_STD


class NativeEvalRun:
    def __init__(
        self,
        config: EvalConfig,
        dataset_root: pathlib.Path,
        checkpoint_path: pathlib.Path,
        artifact_directory: pathlib.Path,
    ):
        self.config = config
        self.dataset_root = pathlib.Path(dataset_root)
        self.checkpoint_path = pathlib.Path(checkpoint_path)
        self.artifact_directory = pathlib.Path(artifact_directory)
        self.artifact_directory.mkdir(parents=True, exist_ok=True)
        self.classes = class_names(self.dataset_root / "train")

    def num_classes(self) -> int:
        return len(self.classes)

    def launch(self) -> None:
        config = self.config
        class_to_index = {name: i for i, name in enumerate(self.classes)}
        size = config.model.backbone.image_size

        train_file_list = build_train_file_list(self.dataset_root, class_to_index)
        val_file_list = build_val_file_list(self.dataset_root, class_to_index)

        train_loader = LabeledImageLoader(
            train_file_list, self.dataset_root / "train", config.batch_size,
            size=size, shuffle=True, seed=config.seed, num_workers=config.num_workers,
        )
        valid_loader = LabeledImageLoader(
            val_file_list, self.dataset_root / "val", config.batch_size,
            size=size, shuffle=False, seed=config.seed, num_workers=config.num_workers,
        )

        rng = jax.random.PRNGKey(config.seed)
        embed_dim = config.model.backbone.embed_dim
        probe_model = LinearClassifierConfig(embed_dim=embed_dim, num_classes=self.num_classes())

        backbone = ViT(config=config.model.backbone)
        classifier = LinearClassifier(backbone=backbone, config=probe_model)

        sample = jnp.zeros((1, size, size, 3))
        variables = classifier.init(rng, sample)
        pretrained_backbone = load_backbone_params(
            self.checkpoint_path, config.model, variables["params"]["backbone"]
        )
        params = unfreeze(variables["params"])
        params["backbone"] = pretrained_backbone
        params = freeze(params)

        state = ProbeState.create(apply_fn=classifier.apply, params=params, tx=_optimizer(config))

        metrics.dump_config(self.artifact_directory, config)
        logger = metrics.MetricsLogger(self.artifact_directory)

        def _mean(values):
            return float(jnp.mean(jnp.stack(values))) if values else float("nan")

        for epoch in range(1, config.num_epochs + 1):
            epoch_start = time.time()

            train_losses, train_acc = [], []
            for images, labels in train_loader:
                state, loss, acc = train_step(state, _normalize(jnp.asarray(images)), jnp.asarray(labels))
                train_losses.append(loss)
                train_acc.append(acc)

            valid_losses, valid_acc = [], []
            for images, labels in valid_loader:
                loss, acc = eval_step(state, _normalize(jnp.asarray(images)), jnp.asarray(labels))
                valid_losses.append(loss)
                valid_acc.append(acc)

            print(
                f"probe epoch {epoch}/{config.num_epochs}: "
                f"train_loss={_mean(train_losses):.4f} "
                f"train_acc={_mean(train_acc):.4f} "
                f"valid_acc={_mean(valid_acc):.4f}"
            )

            logger.log(
                epoch=epoch,
                epoch_seconds=time.time() - epoch_start,
                train_loss=_mean(train_losses),
                train_acc=_mean(train_acc),
                valid_acc=_mean(valid_acc),
            )
