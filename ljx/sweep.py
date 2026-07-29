"""Short, fixed-step comparison runs across configs.py's presets."""

from __future__ import annotations

import pathlib
import time

import jax
import jax.numpy as jnp
import optax

from ljx.data.dali_pipeline import build_multicrop_iterator
from ljx.data.tiny_imagenet import split_pretrain_file_lists
from ljx.training.train_loop import LeJEPATrainState, TrainingConfig, _views_from_batch, train_step


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

    train_iter = build_multicrop_iterator(
        file_root=str(dataset_path),
        file_list=str(train_list),
        batch_size=config.batch_size,
        num_threads=config.num_workers,
        seed=config.seed,
        shuffle=True,
    )

    rng = jax.random.PRNGKey(config.seed)
    first_batch = next(iter(train_iter))
    global_views, local_views = _views_from_batch(first_batch)

    model = config.model.init()
    variables = model.init(
        rng, [global_views[0]], [local_views[0]], deterministic=True, use_running_average=True
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
    for batch in train_iter:
        if steps_run >= num_steps:
            break
        global_views, local_views = _views_from_batch(batch)
        state, loss = train_step(state, global_views, local_views, config.model)
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
