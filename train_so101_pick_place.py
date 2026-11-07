#!/usr/bin/env python3
"""
Training script for SO-101 Part B2 Pick and Place Policy.
Features:
- PPO with Stable-Baselines3.
- SubprocVecEnv across parallel CPU workers.
- Observation and return normalization via VecNormalize.
- Evaluation and checkpoint callbacks.
- Multi-phase Pick-Transport-Release-Clear reward engineering.
"""

import os
import sys

# Prevent Windows EPERM permission conflict with corrupted .keras/keras.json
sys.modules["tensorflow"] = None

import argparse
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from so101_pick_place_env import SO101PickPlaceEnv


def make_env():
    return Monitor(SO101PickPlaceEnv())


def train_so101_pick_place(
    seed=42,
    total_timesteps=1_200_000,
    n_envs=8,
    model_dir="models/so101/pick_place",
    log_dir="logs/so101_pick_place",
):
    run_model_dir = os.path.join(model_dir, f"seed_{seed}")
    run_log_dir = os.path.join(log_dir, f"seed_{seed}")
    checkpoint_dir = os.path.join(run_model_dir, "checkpoints")

    os.makedirs(run_model_dir, exist_ok=True)
    os.makedirs(run_log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("==================================================================")
    print(f"TRAINING SO-101 PICK AND PLACE POLICY (SEED {seed})")
    print(f"Parallel Envs: {n_envs}")
    print(f"Total Timesteps: {total_timesteps:,}")
    print(f"Artifact Directory: {run_model_dir}")
    print("==================================================================")

    train_env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv, seed=seed)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = make_vec_env(make_env, n_envs=1, seed=seed + 100)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env.training = False
    eval_env.norm_reward = False

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=run_model_dir,
        log_path=run_log_dir,
        eval_freq=max(15_000 // n_envs, 1),
        n_eval_episodes=20,
        deterministic=True,
        warn=True,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // n_envs, 1),
        save_path=checkpoint_dir,
        name_prefix=f"so101_pick_place_s{seed}",
        save_vecnormalize=True,
    )

    callback = CallbackList([eval_callback, checkpoint_callback])

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=run_log_dir,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.20,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        seed=seed,
        device="cpu",
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=False,
    )

    final_model_path = os.path.join(run_model_dir, "so101_pick_place_final")
    model.save(final_model_path)
    train_env.save(os.path.join(run_model_dir, "so101_pick_place_vecnorm.pkl"))

    print(f"\nTraining finished for seed {seed}.")
    print(f"Final model: {final_model_path}.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--envs", type=int, default=8)
    args = parser.parse_args()

    train_so101_pick_place(seed=args.seed, total_timesteps=args.timesteps, n_envs=args.envs)

