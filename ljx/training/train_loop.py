"""Pretraining LeJEPA"""

from __future__ import annotations

import functools
import pathlib
import time
from dataclasses import dataclass, field, replace

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from ljx.data.jax_augment import ViewConfig, generate_views
from ljx.data.raw_loader import CachedImageLoader
from ljx.data.tiny_imagenet import split_pretrain_file_lists
from ljx.models.lejepa import LeJEPAConfig, LeJEPALoss, lejepa_loss
from ljx.training import checkpoint, metrics

GLOBAL_VIEW = ViewConfig(size=64, scale=(0.30, 1.0))
LOCAL_VIEW = ViewConfig(size=32, scale=(0.05, 0.30))
NUM_GLOBAL_VIEWS = 2
# 6 is the DINO/iBOT-style convention the reference repo inherited, tuned for
# larger-scale pretraining. 4 trades some of that multi-crop signal for ~15%
# fewer tokens through the MLP/QKV projections per step at this ViT-Tiny/64px
# scale, where the original ratio is unverified.
NUM_LOCAL_VIEWS = 4


@dataclass(frozen=True)
class TrainingConfig:
    model: LeJEPAConfig = field(default_factory=LeJEPAConfig)
    batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 0.05
    num_workers: int = 4
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


@functools.partial(jax.jit, static_argnames=("config",), donate_argnums=(0,))
def train_step(
    state: LeJEPATrainState,
    images: jnp.ndarray,
    rng: jax.Array,
    config: LeJEPAConfig,
) -> tuple[LeJEPATrainState, LeJEPALoss]:
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
        return loss.total, (loss, mutated["batch_stats"])

    (_, (loss, batch_stats)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    state = state.apply_gradients(grads=grads).replace(
        batch_stats=batch_stats,
        sigreg_step=state.sigreg_step + 1,
    )
    return state, loss


@functools.partial(jax.jit, static_argnames=("config",))
def eval_step(
    state: LeJEPATrainState,
    images: jnp.ndarray,
    rng: jax.Array,
    config: LeJEPAConfig,
) -> LeJEPALoss:
    variables = {"params": state.params, "batch_stats": state.batch_stats}
    rng_global, rng_local = jax.random.split(rng)
    global_views = list(generate_views(rng_global, images, GLOBAL_VIEW, NUM_GLOBAL_VIEWS))
    local_views = list(generate_views(rng_local, images, LOCAL_VIEW, NUM_LOCAL_VIEWS))
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
        config = self.config
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
        steps_per_epoch = max(len(train_loader), 1)
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
            }
            restored = checkpoint.load(path, template)
            state = state.replace(**restored)
            start_epoch = config.resume_from_epoch + 1

        for epoch in range(start_epoch, config.num_epochs + 1):
            epoch_start = time.time()

            train_losses, train_predictions, train_sigregs = [], [], []
            for batch in train_loader:
                rng, step_rng = jax.random.split(rng)
                images = jnp.asarray(batch, dtype=jnp.float32) / 255.0
                state, loss = train_step(state, images, step_rng, config.model)
                train_losses.append(loss.total)
                train_predictions.append(loss.prediction)
                train_sigregs.append(loss.sigreg)

            valid_losses, valid_predictions, valid_sigregs = [], [], []
            for batch in valid_loader:
                rng, step_rng = jax.random.split(rng)
                images = jnp.asarray(batch, dtype=jnp.float32) / 255.0
                loss = eval_step(state, images, step_rng, config.model)
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
                    },
                )
