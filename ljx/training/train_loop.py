"""Pretraining LeJEPA"""

from __future__ import annotations

import functools
import pathlib
from dataclasses import dataclass, field, replace

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from ljx.data.dali_pipeline import MultiCropConfig, build_multicrop_iterator
from ljx.data.tiny_imagenet import split_pretrain_file_lists
from ljx.models.lejepa import LeJEPA, LeJEPAConfig, LeJEPALoss, lejepa_loss
from ljx.training import checkpoint


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

    def dry_run(self) -> "TrainingConfig":
        return replace(
            self,
            num_epochs=1,
            batch_size=min(self.batch_size, 8),
            num_workers=1,
            checkpoint_every=None,
            resume_from_epoch=None,
            num_valid_images=8,
        )


class LeJEPATrainState(train_state.TrainState):
    batch_stats: dict
    sigreg_step: jnp.ndarray


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

    grads, (loss, batch_stats) = jax.grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=batch_stats, sigreg_step=state.sigreg_step + 1)
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
        config = self.config
        train_list, valid_list = split_pretrain_file_lists(
            self.dataset_path, config.num_valid_images, self.artifact_directory
        )

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
        first_batch = next(iter(train_iter))
        global_views, local_views = _views_from_batch(first_batch)

        model = config.model.init()
        variables = model.init(
            rng, [global_views[0]], [local_views[0]], deterministic=True, use_running_average=True
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

        for epoch in range(start_epoch, config.num_epochs + 1):
            train_losses = []
            for batch in train_iter:
                global_views, local_views = _views_from_batch(batch)
                state, loss = train_step(state, global_views, local_views, config.model)
                train_losses.append(loss.total)

            valid_losses = []
            for batch in valid_iter:
                global_views, local_views = _views_from_batch(batch)
                loss = eval_step(state, global_views, local_views, config.model)
                valid_losses.append(loss.total)

            train_mean = float(jnp.mean(jnp.stack(train_losses)))
            valid_mean = float(jnp.mean(jnp.stack(valid_losses))) if valid_losses else float("nan")
            print(f"epoch {epoch}/{config.num_epochs}: train_loss={train_mean:.4f} valid_loss={valid_mean:.4f}")

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
