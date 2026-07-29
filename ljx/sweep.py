"""Short, fixed-step comparison runs across configs.py's presets."""

from __future__ import annotations

import pathlib
import time

import jax
import jax.numpy as jnp
import optax

from ljx.data.jax_augment import generate_views
from ljx.data.raw_loader import CachedImageLoader
from ljx.data.tiny_imagenet import split_pretrain_file_lists
from ljx.training.train_loop import (
    GLOBAL_VIEW,
    LOCAL_VIEW,
    NUM_GLOBAL_VIEWS,
    NUM_LOCAL_VIEWS,
    LeJEPATrainState,
    TrainingConfig,
    train_step,
)


def run_one(
    name: str,
    config: TrainingConfig,
    dataset_path: pathlib.Path,
    artifact_directory: pathlib.Path,
    num_steps: int = 50,
) -> dict:
    dataset_path = pathlib.Path(dataset_path)
    run_dir = pathlib.Path(artifact_directory) / name
    train_list, _ = split_pretrain_file_lists(dataset_path, config.num_valid_images, run_dir)

    train_loader = CachedImageLoader(
        train_list, dataset_path, config.batch_size,
        shuffle=True, seed=config.seed, num_workers=config.num_workers,
    )

    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng = jax.random.split(rng)
    dummy_images = jnp.zeros((1, 64, 64, 3))
    dummy_global = list(generate_views(init_rng, dummy_images, GLOBAL_VIEW, NUM_GLOBAL_VIEWS))
    dummy_local = list(generate_views(init_rng, dummy_images, LOCAL_VIEW, NUM_LOCAL_VIEWS))

    model = config.model.init()
    variables = model.init(
        init_rng, dummy_global, dummy_local, deterministic=True, use_running_average=True
    )
    optimizer = optax.adamw(config.learning_rate, weight_decay=config.weight_decay)
    state = LeJEPATrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=optimizer,
        batch_stats=variables.get("batch_stats", {}),
        sigreg_step=jnp.array(0, dtype=jnp.uint32),
    )

    losses, predictions, sigregs = [], [], []
    start = time.time()
    steps_run = 0
    for batch in train_loader:
        if steps_run >= num_steps:
            break
        rng, step_rng = jax.random.split(rng)
        images = jnp.asarray(batch, dtype=jnp.float32) / 255.0
        state, loss = train_step(state, images, step_rng, config.model)
        losses.append(loss.total)
        predictions.append(loss.prediction)
        sigregs.append(loss.sigreg)
        steps_run += 1
    elapsed = time.time() - start

    return {
        "name": name,
        "steps": steps_run,
        "seconds": elapsed,
        "seconds_per_step": elapsed / max(steps_run, 1),
        "loss": float(jnp.mean(jnp.stack(losses))),
        "prediction": float(jnp.mean(jnp.stack(predictions))),
        "sigreg": float(jnp.mean(jnp.stack(sigregs))),
    }


def run_sweep(
    configs: dict[str, TrainingConfig],
    dataset_path: pathlib.Path,
    artifact_directory: pathlib.Path,
    num_steps: int = 50,
) -> list[dict]:
    results = []
    for name, config in configs.items():
        print(f"--- {name} ---")
        result = run_one(name, config, dataset_path, artifact_directory, num_steps)
        print(result)
        results.append(result)
    return results
