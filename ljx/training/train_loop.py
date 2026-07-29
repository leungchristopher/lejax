"""Pretraining LeJEPA"""

from __future__ import annotations

import functools
import pathlib
import time
from dataclasses import dataclass, field, replace

import jax
import jax.numpy as jnp
import optax
from flax.training import dynamic_scale as dynamic_scale_lib
from flax.training import train_state

from ljx.data.tiny_imagenet import split_pretrain_file_lists
from ljx.models.lejepa import LeJEPA, LeJEPAConfig, LeJEPALoss, lejepa_loss
from ljx.training import checkpoint, metrics


@dataclass(frozen=True)
class TrainingConfig:
    model: LeJEPAConfig = field(default_factory=LeJEPAConfig)
    batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 0.05
    num_workers: int = 4  # DALI num_threads
    checkpoint_every: int | None = 10
    resume_from_epoch: int | None = None
    num_valid_images: int = 1024
    seed: int = 0
    max_train_images: int | None = None

    def dry_run(self) -> "TrainingConfig":
        return replace(
            self,
            num_epochs=1,
            batch_size=min(self.batch_size, 8),
            num_workers=1,
            checkpoint_every=None,
            resume_from_epoch=None,
            num_valid_images=8,
            max_train_images=32,
        )


class LeJEPATrainState(train_state.TrainState):
    batch_stats: dict
    sigreg_step: jnp.ndarray
    dynamic_scale: dynamic_scale_lib.DynamicScale


def _views_from_batch(batch: dict) -> tuple[list[jnp.ndarray], list[jnp.ndarray]]:
    from ljx.data.dali_pipeline import NUM_GLOBAL_VIEWS, NUM_LOCAL_VIEWS

    globals_ = [batch[f"global_{i}"] for i in range(NUM_GLOBAL_VIEWS)]
    locals_ = [batch[f"local_{i}"] for i in range(NUM_LOCAL_VIEWS)]
    return globals_, locals_


@functools.partial(jax.jit, static_argnames=("config",))
def train_step(
    state: LeJEPATrainState,
    global_views: list[jnp.ndarray],
    local_views: list[jnp.ndarray],
    config: LeJEPAConfig,
) -> tuple[LeJEPATrainState, LeJEPALoss]:
    def loss_fn(params):
        variables = {"params": params, "batch_stats": state.batch_stats}
        projections, mutated = state.apply_fn(
            variables,
            global_views,
            local_views,
            deterministic=False,
            use_running_average=False,
            mutable=["batch_stats"],
        )
        loss = lejepa_loss(projections, config, state.sigreg_step)
        return loss.total, (loss, mutated["batch_stats"])

    dynamic_scale, finite, (_, (loss, batch_stats)), grads = state.dynamic_scale.value_and_grad(
        loss_fn, has_aux=True
    )(state.params)

    new_state = state.apply_gradients(grads=grads)
    select = lambda new, old: jnp.where(finite, new, old)
    params = jax.tree_util.tree_map(select, new_state.params, state.params)
    opt_state = jax.tree_util.tree_map(select, new_state.opt_state, state.opt_state)

    state = new_state.replace(
        params=params,
        opt_state=opt_state,
        batch_stats=batch_stats,
        sigreg_step=state.sigreg_step + 1,
        dynamic_scale=dynamic_scale,
    )
    return state, loss


@functools.partial(jax.jit, static_argnames=("config",))
def eval_step(
    state: LeJEPATrainState,
    global_views: list[jnp.ndarray],
    local_views: list[jnp.ndarray],
    config: LeJEPAConfig,
) -> LeJEPALoss:
    variables = {"params": state.params, "batch_stats": state.batch_stats}
    projections = state.apply_fn(
        variables, global_views, local_views, deterministic=True, use_running_average=True
    )
    return lejepa_loss(projections, config, state.sigreg_step)


class TrainingRun:
    def __init__(self, config: TrainingConfig, dataset_path: pathlib.Path, artifact_directory: pathlib.Path):
        self.config = config
        self.dataset_path = pathlib.Path(dataset_path)
        self.artifact_directory = pathlib.Path(artifact_directory)
        self.artifact_directory.mkdir(parents=True, exist_ok=True)

    def launch(self) -> None:
        from ljx.data.dali_pipeline import MultiCropConfig, build_multicrop_iterator

        config = self.config
        train_list, valid_list = split_pretrain_file_lists(
            self.dataset_path, config.num_valid_images, self.artifact_directory
        )
        if config.max_train_images is not None:
            lines = train_list.read_text().strip().splitlines()[: config.max_train_images]
            train_list.write_text("\n".join(lines) + "\n")

        train_iter = build_multicrop_iterator(
            file_root=str(self.dataset_path),
            file_list=str(train_list),
            batch_size=config.batch_size,
            num_threads=config.num_workers,
            seed=config.seed,
            shuffle=True,
        )
        valid_iter = build_multicrop_iterator(
            file_root=str(self.dataset_path),
            file_list=str(valid_list),
            batch_size=config.batch_size,
            num_threads=config.num_workers,
            seed=config.seed,
            shuffle=False,
        )

        num_train_images = sum(1 for _ in train_list.read_text().strip().splitlines())
        steps_per_epoch = max(-(-num_train_images // config.batch_size), 1)
        total_steps = max(steps_per_epoch * config.num_epochs, 1)

        schedule = optax.cosine_decay_schedule(
            init_value=config.learning_rate,
            decay_steps=total_steps,
            alpha=config.min_learning_rate / config.learning_rate,
        )
        optimizer = optax.adamw(learning_rate=schedule, weight_decay=config.weight_decay)

        rng = jax.random.PRNGKey(config.seed)
        crop_config = MultiCropConfig()
        dummy_global = jnp.zeros((1, crop_config.global_size, crop_config.global_size, 3))
        dummy_local = jnp.zeros((1, crop_config.local_size, crop_config.local_size, 3))

        model = config.model.init()
        variables = model.init(
            rng, [dummy_global], [dummy_local], deterministic=True, use_running_average=True
        )
        state = LeJEPATrainState.create(
            apply_fn=model.apply,
            params=variables["params"],
            tx=optimizer,
            batch_stats=variables.get("batch_stats", {}),
            sigreg_step=jnp.array(0, dtype=jnp.uint32),
            dynamic_scale=dynamic_scale_lib.DynamicScale(),
        )

        metrics.dump_config(self.artifact_directory, config)
        logger = metrics.MetricsLogger(self.artifact_directory)

        start_epoch = 1
        checkpoint_dir = self.artifact_directory / "checkpoint"
        if config.resume_from_epoch is not None:
            path = checkpoint_dir / f"model-{config.resume_from_epoch}.msgpack"
            template = {
                "params": state.params,
                "batch_stats": state.batch_stats,
                "opt_state": state.opt_state,
                "sigreg_step": state.sigreg_step,
                "dynamic_scale": state.dynamic_scale,
            }
            restored = checkpoint.load(path, template)
            state = state.replace(**restored)
            start_epoch = config.resume_from_epoch + 1

        for epoch in range(start_epoch, config.num_epochs + 1):
            epoch_start = time.time()

            train_losses, train_predictions, train_sigregs = [], [], []
            for batch in train_iter:
                global_views, local_views = _views_from_batch(batch)
                state, loss = train_step(state, global_views, local_views, config.model)
                train_losses.append(loss.total)
                train_predictions.append(loss.prediction)
                train_sigregs.append(loss.sigreg)

            valid_losses, valid_predictions, valid_sigregs = [], [], []
            for batch in valid_iter:
                global_views, local_views = _views_from_batch(batch)
                loss = eval_step(state, global_views, local_views, config.model)
                valid_losses.append(loss.total)
                valid_predictions.append(loss.prediction)
                valid_sigregs.append(loss.sigreg)

            def _mean(values):
                return float(jnp.mean(jnp.stack(values))) if values else float("nan")

            train_mean = _mean(train_losses)
            valid_mean = _mean(valid_losses)
            print(
                f"epoch {epoch}/{config.num_epochs}: "
                f"train_loss={train_mean:.4f} (pred={_mean(train_predictions):.4f} sigreg={_mean(train_sigregs):.4f}) "
                f"valid_loss={valid_mean:.4f} (pred={_mean(valid_predictions):.4f} sigreg={_mean(valid_sigregs):.4f})"
            )

            logger.log(
                epoch=epoch,
                epoch_seconds=time.time() - epoch_start,
                train_loss=train_mean,
                train_prediction=_mean(train_predictions),
                train_sigreg=_mean(train_sigregs),
                valid_loss=valid_mean,
                valid_prediction=_mean(valid_predictions),
                valid_sigreg=_mean(valid_sigregs),
            )

            should_checkpoint = config.checkpoint_every is not None and (
                epoch % config.checkpoint_every == 0 or epoch == config.num_epochs
            )
            if should_checkpoint:
                checkpoint.save(
                    checkpoint_dir,
                    epoch,
                    {
                        "params": state.params,
                        "batch_stats": state.batch_stats,
                        "opt_state": state.opt_state,
                        "sigreg_step": state.sigreg_step,
                        "dynamic_scale": state.dynamic_scale,
                    },
                )
