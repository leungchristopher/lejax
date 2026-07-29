"""Data-parallel pretraining across multiple local devices (e.g. all 8 cores
of a TPU v2-8), via jax.pmap.

Config.batch_size in TrainingConfig is per-device here — the effective
global batch is batch_size * jax.local_device_count(). Gradients (and
batch_stats) are averaged across devices each step via lax.pmean, so this
is standard synchronous data parallelism, not model parallelism: the model
itself must already fit on one core (see the memory estimate in
docs/tpu_capacity_notes — this ViT-Tiny's params+optimizer state is tens of
MB, far under a TPU v2 core's 8GB HBM; the real per-core cost is
activations, which scale with the per-device batch size)."""

from __future__ import annotations

import functools
import pathlib
import time
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import jax_utils

from ljx.data.jax_augment import generate_views
from ljx.data.raw_loader import CachedImageLoader
from ljx.data.tiny_imagenet import split_pretrain_file_lists
from ljx.models.lejepa import LeJEPALoss, lejepa_loss
from ljx.training import checkpoint, metrics
from ljx.training.train_loop import (
    GLOBAL_VIEW,
    LOCAL_VIEW,
    NUM_GLOBAL_VIEWS,
    NUM_LOCAL_VIEWS,
    LeJEPATrainState,
    TrainingConfig,
)


@functools.partial(jax.pmap, axis_name="devices", static_broadcasted_argnums=(3, 4))
def pmap_train_step(state, images, rng, config, loss_scale=1.0):
    def loss_fn(params):
        variables = {"params": params, "batch_stats": state.batch_stats}
        rng_global, rng_local = jax.random.split(rng)
        global_views = list(generate_views(rng_global, images, GLOBAL_VIEW, NUM_GLOBAL_VIEWS))
        local_views = list(generate_views(rng_local, images, LOCAL_VIEW, NUM_LOCAL_VIEWS))
        projections, mutated = state.apply_fn(
            variables,
            global_views,
            local_views,
            deterministic=False,
            use_running_average=False,
            mutable=["batch_stats"],
        )
        loss = lejepa_loss(projections, config, state.sigreg_step)
        scaled_total = loss.total * loss_scale if loss_scale != 1.0 else loss.total
        return scaled_total, (loss, mutated["batch_stats"])

    (_, (loss, batch_stats)), scaled_grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    grads = scaled_grads if loss_scale == 1.0 else jax.tree_util.tree_map(lambda g: g / loss_scale, scaled_grads)

    grads = jax.lax.pmean(grads, axis_name="devices")
    batch_stats = jax.lax.pmean(batch_stats, axis_name="devices")
    loss = jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, axis_name="devices"), loss)

    state = state.apply_gradients(grads=grads).replace(
        batch_stats=batch_stats,
        sigreg_step=state.sigreg_step + 1,
    )
    return state, loss


@functools.partial(jax.pmap, axis_name="devices", static_broadcasted_argnums=(3,))
def pmap_eval_step(state, images, rng, config):
    variables = {"params": state.params, "batch_stats": state.batch_stats}
    rng_global, rng_local = jax.random.split(rng)
    global_views = list(generate_views(rng_global, images, GLOBAL_VIEW, NUM_GLOBAL_VIEWS))
    local_views = list(generate_views(rng_local, images, LOCAL_VIEW, NUM_LOCAL_VIEWS))
    projections = state.apply_fn(
        variables, global_views, local_views, deterministic=True, use_running_average=True
    )
    loss = lejepa_loss(projections, config, state.sigreg_step)
    return jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, axis_name="devices"), loss)


def _device_grouped_batches(loader, num_devices: int):
    """Groups exactly num_devices consecutive loader batches into one stack
    with a leading device axis — the shape pmap shards across devices.
    Drops a final partial group (pmap needs a fixed device-axis size)."""
    group = []
    for batch in loader:
        group.append(batch)
        if len(group) == num_devices:
            yield np.stack(group, axis=0)
            group = []


class PmapTrainingRun:
    def __init__(self, config: TrainingConfig, dataset_path: pathlib.Path, artifact_directory: pathlib.Path):
        self.config = config
        self.dataset_path = pathlib.Path(dataset_path)
        self.artifact_directory = pathlib.Path(artifact_directory)
        self.artifact_directory.mkdir(parents=True, exist_ok=True)
        self.num_devices = jax.local_device_count()

    def launch(self) -> None:
        config = self.config
        num_devices = self.num_devices
        train_list, valid_list = split_pretrain_file_lists(
            self.dataset_path, config.num_valid_images, self.artifact_directory
        )
        if config.max_train_images is not None:
            lines = train_list.read_text().strip().splitlines()[: config.max_train_images]
            train_list.write_text("\n".join(lines) + "\n")

        train_loader = CachedImageLoader(
            train_list, self.dataset_path, config.batch_size,
            shuffle=True, seed=config.seed, num_workers=config.num_workers,
        )
        valid_loader = CachedImageLoader(
            valid_list, self.dataset_path, config.batch_size,
            shuffle=False, seed=config.seed, num_workers=config.num_workers,
        )

        num_train_images = sum(1 for _ in train_list.read_text().strip().splitlines())
        steps_per_epoch = max(len(train_loader) // num_devices, 1)
        total_steps = max(steps_per_epoch * config.num_epochs, 1)

        schedule = optax.cosine_decay_schedule(
            init_value=config.learning_rate,
            decay_steps=total_steps,
            alpha=config.min_learning_rate / config.learning_rate,
        )
        optimizer = optax.adamw(learning_rate=schedule, weight_decay=config.weight_decay)

        rng = jax.random.PRNGKey(config.seed)
        rng, init_rng = jax.random.split(rng)
        dummy_images = jnp.zeros((1, 64, 64, 3))
        dummy_global = list(generate_views(init_rng, dummy_images, GLOBAL_VIEW, NUM_GLOBAL_VIEWS))
        dummy_local = list(generate_views(init_rng, dummy_images, LOCAL_VIEW, NUM_LOCAL_VIEWS))

        model = config.model.init()
        variables = model.init(
            init_rng, dummy_global, dummy_local, deterministic=True, use_running_average=True
        )
        state = LeJEPATrainState.create(
            apply_fn=model.apply,
            params=variables["params"],
            tx=optimizer,
            batch_stats=variables.get("batch_stats", {}),
            sigreg_step=jnp.array(0, dtype=jnp.uint32),
        )

        start_epoch = 1
        checkpoint_dir = self.artifact_directory / "checkpoint"
        if config.resume_from_epoch is not None:
            path = checkpoint_dir / f"model-{config.resume_from_epoch}.msgpack"
            template = {
                "params": state.params,
                "batch_stats": state.batch_stats,
                "opt_state": state.opt_state,
                "sigreg_step": state.sigreg_step,
            }
            restored = checkpoint.load(path, template)
            state = state.replace(**restored)
            start_epoch = config.resume_from_epoch + 1

        state = jax_utils.replicate(state)
        metrics.dump_config(self.artifact_directory, config)
        logger = metrics.MetricsLogger(self.artifact_directory)

        for epoch in range(start_epoch, config.num_epochs + 1):
            epoch_start = time.time()

            train_losses, train_predictions, train_sigregs = [], [], []
            for batch_group in _device_grouped_batches(train_loader, num_devices):
                rng, step_rng = jax.random.split(rng)
                device_rngs = jax.random.split(step_rng, num_devices)
                images = jnp.asarray(batch_group, dtype=jnp.float32) / 255.0
                state, loss = pmap_train_step(state, images, device_rngs, config.model, config.loss_scale)
                train_losses.append(loss.total[0])
                train_predictions.append(loss.prediction[0])
                train_sigregs.append(loss.sigreg[0])

            valid_losses, valid_predictions, valid_sigregs = [], [], []
            for batch_group in _device_grouped_batches(valid_loader, num_devices):
                rng, step_rng = jax.random.split(rng)
                device_rngs = jax.random.split(step_rng, num_devices)
                images = jnp.asarray(batch_group, dtype=jnp.float32) / 255.0
                loss = pmap_eval_step(state, images, device_rngs, config.model)
                valid_losses.append(loss.total[0])
                valid_predictions.append(loss.prediction[0])
                valid_sigregs.append(loss.sigreg[0])

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
                unreplicated = jax_utils.unreplicate(state)
                checkpoint.save(
                    checkpoint_dir,
                    epoch,
                    {
                        "params": unreplicated.params,
                        "batch_stats": unreplicated.batch_stats,
                        "opt_state": unreplicated.opt_state,
                        "sigreg_step": unreplicated.sigreg_step,
                    },
                )
