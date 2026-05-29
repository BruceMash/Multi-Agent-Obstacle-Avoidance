import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path

from runner_sac import (
    EXPERIMENT_CONFIG,
    FixedSeedEvalCallback,
    build_env,
    build_model,
    load_checkpoint,
)

checkpoint = Path(r"artifacts\20260509_003834\best_model.pt")
run_dir = checkpoint.parent
manual_eval_dir = run_dir / "manual_eval"

env = build_env(EXPERIMENT_CONFIG)
eval_env = build_env(EXPERIMENT_CONFIG)

model = build_model(env, EXPERIMENT_CONFIG)
model = load_checkpoint(model, checkpoint)

callback = FixedSeedEvalCallback(
    eval_env=eval_env,
    run_dir=manual_eval_dir,
    log_dir=manual_eval_dir / "tensorboard",
    eval_freq=1,
    eval_seeds=EXPERIMENT_CONFIG.eval_seeds,
    deterministic=True,
    save_visualizations=True,
    visualization_dir=manual_eval_dir / "visualizations",
    verbose=1,
)

callback.init_callback(model)
callback.num_timesteps = model.num_timesteps
callback._evaluate()

env.close()
