#!/usr/bin/env python3
"""
Single-Model Sequential Multi-Seed Training & Statistical Evaluation Suite for SO-101 Precision Reaching.

Task & Architecture:
1. High-level PPO: 21-D observation -> 5-D continuous arm joint deltas.
2. Low-level: Feetech STS3215 actuator group at 500 Hz (20 substeps @ 25 Hz policy rate).
3. 3-Zone Goal-Conditioned Target Randomization:
   - Bottom Zone: Z in [0.015, 0.065] m (table/ground plane)
   - Medium Zone: Z in [0.075, 0.180] m (mid-altitude)
   - High Zone:   Z in [0.190, 0.320] m (overhead reaches)
4. Success Criteria:
   - Euclidean distance < 10 mm (0.010 m)
   - Held stably for >= 0.5 s (13 consecutive policy steps at 25 Hz)
5. Sequential Multi-Seed Training:
   - Trains sequentially across 3 random seeds into ONE single continuous model checkpoint.
   - Preserves policy network weights, optimizer states, and VecNormalize statistics continuously.
   - Outputs ONE unified model:
       models/so101/reach/reach_model.zip
       models/so101/reach/vecnormalize.pkl
6. Post-Training Evaluation:
   - Evaluates the final trained model across held-out targets in Bottom, Medium, and High zones.
   - Computes and displays sample Mean ± Std for:
       * Overall Task Success Rate (%)
       * Euclidean Distance Error (mm)
       * Time to Reach & Hold (s)
       * Action Smoothness
       * Episodic Return
       * Zone-specific Success Rates (Bottom, Medium, High)
7. Learning Curves Visualization:
   - Extracts TensorBoard logs across all training steps.
   - Generates a 4-panel publication-grade figure:
       1. Episode Return progression (with seed boundary markers)
       2. Overall Success Rate progression (%)
       3. Euclidean Distance Error progression (mm)
       4. 3-Zone Performance Breakdown (Bottom, Medium, High)
   - Saves figures and CSV/JSON summaries to disk.
"""

from __future__ import annotations

import os
import sys
import glob
import json
import csv
import copy
import argparse
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "environments") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "environments"))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.common import configure_env
configure_env()

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
    BaseCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from tensorboard.backend.event_processing import event_accumulator

from environments.reaching_env import SO101ReachingEnv



# Training Metrics Callback

class ReachMetricsCallback(BaseCallback):
    """Tracks and logs Reaching episode metrics over rolling windows to TensorBoard."""

    def __init__(self, window_size=100, verbose=0):
        super().__init__(verbose)
        self.window_size = window_size
        self.success_history = []
        self.dist_error_history = []
        self.in_tolerance_history = []
        self.hold_steps_history = []
        self.action_delta_history = []

        self.zone_successes = {"bottom": [], "medium": [], "high": []}

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        for done, info in zip(dones, infos):
            if done:
                succ = float(info.get("success", False))
                self.success_history.append(succ)
                self.dist_error_history.append(float(info.get("dist_mm", 0.0)))
                self.in_tolerance_history.append(float(info.get("in_tolerance", False)))
                self.hold_steps_history.append(float(info.get("hold_steps", 0)))
                self.action_delta_history.append(float(info.get("action_delta", 0.0)))

                zone = info.get("target_zone")
                if zone in self.zone_successes:
                    self.zone_successes[zone].append(succ)
        return True

    def _on_rollout_end(self) -> None:
        w = self.window_size
        if len(self.success_history) > 0:
            self.logger.record("rollout/success_rate", float(np.mean(self.success_history[-w:])))
            self.logger.record("rollout/mean_final_err_mm", float(np.mean(self.dist_error_history[-w:])))
            self.logger.record("rollout/in_tolerance_rate", float(np.mean(self.in_tolerance_history[-w:])))
            self.logger.record("rollout/consec_hold_mean", float(np.mean(self.hold_steps_history[-w:])))
            self.logger.record("rollout/action_delta_mean", float(np.mean(self.action_delta_history[-w:])))

        for z in ["bottom", "medium", "high"]:
            hist = self.zone_successes[z]
            if len(hist) > 0:
                self.logger.record(f"rollout/{z}_zone_success", float(np.mean(hist[-w:])))


def make_env_fn():
    def _init():
        return Monitor(SO101ReachingEnv())
    return _init



# Training Pipeline: 3 Seeds -> 1 Single Unified Model

def train_reach_multiseed_single_model(
    seeds=(42, 123, 456),
    total_timesteps=300_000,
    n_envs=8,
    config_path="configs/ppo_so101_config.yaml",
    model_dir="models/so101/reach",
    log_dir="logs/so101_reach",
    load_checkpoint=None,
    load_vecnorm=None,
    n_eval_episodes=50,
    output_dir="reports/reach_single_model",
):
    """
    Trains Reaching sequentially across 3 seeds into ONE single continuous model,
    then evaluates the final model (Mean ± Std) and plots full training learning curves.
    """
    if not os.path.isabs(config_path) and not os.path.exists(config_path):
        config_path = str(PROJECT_ROOT / config_path)
    if not os.path.isabs(model_dir):
        model_dir = str(PROJECT_ROOT / model_dir)
    if not os.path.isabs(log_dir):
        log_dir = str(PROJECT_ROOT / log_dir)
    if not os.path.isabs(output_dir):
        output_dir = str(PROJECT_ROOT / output_dir)
    if load_checkpoint and not os.path.isabs(load_checkpoint) and not os.path.exists(load_checkpoint):
        load_checkpoint = str(PROJECT_ROOT / load_checkpoint)
    if load_vecnorm and not os.path.isabs(load_vecnorm) and not os.path.exists(load_vecnorm):
        load_vecnorm = str(PROJECT_ROOT / load_vecnorm)

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    seeds = list(seeds)
    n_seeds = len(seeds)
    timesteps_per_seed = total_timesteps // n_seeds

    print("=" * 78)
    print(" SO-101 PRECISION REACHING: SINGLE-MODEL MULTI-SEED CONTINUOUS TRAINING")
    print("=" * 78)
    print(f"Seeds ({n_seeds}):            {seeds}")
    print(f"Total Timesteps:         {total_timesteps:,}")
    print(f"Timesteps per Seed:      {timesteps_per_seed:,}")
    print(f"Parallel Envs:           {n_envs}")
    print(f"Model Save Directory:    {model_dir}")
    print(f"TensorBoard Log Dir:     {log_dir}")
    print(f"Post-Train Report Dir:   {output_dir}")
    print("=" * 78)

    # Load hyperparameters
    ppo_cfg = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            full_config = yaml.safe_load(f)
        ppo_cfg = full_config.get("hyperparameters", full_config.get("ppo", {}))

    learning_rate = float(ppo_cfg.get("learning_rate", 3e-4))
    n_steps = int(ppo_cfg.get("n_steps", 1024))
    batch_size = int(ppo_cfg.get("batch_size", 256))
    n_epochs = int(ppo_cfg.get("n_epochs", 10))
    gamma = float(ppo_cfg.get("gamma", 0.99))
    gae_lambda = float(ppo_cfg.get("gae_lambda", 0.95))
    clip_range = float(ppo_cfg.get("clip_range", 0.2))
    ent_coef = float(ppo_cfg.get("ent_coef", 0.005))
    max_grad_norm = float(ppo_cfg.get("max_grad_norm", 0.5))

    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    metrics_callback = ReachMetricsCallback(window_size=100)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, 25_000 // n_envs),
        save_path=checkpoint_dir,
        name_prefix="reach_single_model",
        save_vecnormalize=True,
    )

    temp_vecnorm_path = os.path.join(model_dir, "temp_reach_vecnorm_transit.pkl")
    model = None
    train_env = None
    eval_env = None
    seed_milestones = []

    for seed_idx, current_seed in enumerate(seeds):
        stage_num = seed_idx + 1
        print("\n" + "#" * 78)
        print(f"  STAGE {stage_num}/{n_seeds}: SEED {current_seed} ({timesteps_per_seed:,} timesteps)")
        print("#" * 78)

        # 1. Close old environments FIRST before allocating new ones to prevent RAM exhaustion
        has_prev_env = train_env is not None
        if has_prev_env:
            train_env.save(temp_vecnorm_path)
            train_env.close()
            eval_env.close()
            del train_env, eval_env
            train_env = None
            eval_env = None
            import gc; gc.collect()
            time.sleep(0.5)

        # 2. Allocate fresh training environments
        new_train_env = make_vec_env(
            make_env_fn(),
            n_envs=n_envs,
            seed=current_seed,
            vec_env_cls=SubprocVecEnv,
        )

        # 3. Allocate lightweight DummyVecEnv for evaluation (0 worker subprocesses)
        new_eval_env = make_vec_env(
            make_env_fn(),
            n_envs=1,
            seed=current_seed + 10_000,
            vec_env_cls=DummyVecEnv,
        )

        if has_prev_env:
            train_env = VecNormalize.load(temp_vecnorm_path, new_train_env)
            train_env.training = True
            train_env.norm_reward = False

            eval_env = VecNormalize.load(temp_vecnorm_path, new_eval_env)
            eval_env.training = False
            eval_env.norm_reward = False
        else:
            if load_vecnorm and os.path.exists(load_vecnorm):
                print(f"Loading initial VecNormalize stats from: {load_vecnorm}")
                train_env = VecNormalize.load(load_vecnorm, new_train_env)
                train_env.training = True
                train_env.norm_reward = False
                eval_env = VecNormalize.load(load_vecnorm, new_eval_env)
                eval_env.training = False
                eval_env.norm_reward = False
            else:
                train_env = VecNormalize(new_train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
                eval_env = VecNormalize(new_eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=model_dir,
            log_path=log_dir,
            eval_freq=max(1, 10_000 // n_envs),
            n_eval_episodes=20,
            deterministic=True,
            render=False,
        )

        callbacks = CallbackList([eval_callback, checkpoint_callback, metrics_callback])

        if model is None:
            if load_checkpoint and os.path.exists(load_checkpoint):
                print(f"Resuming PPO model from checkpoint: {load_checkpoint}")
                model = PPO.load(
                    load_checkpoint,
                    env=train_env,
                    device="cpu",
                    tensorboard_log=log_dir,
                    custom_objects={
                        "learning_rate": learning_rate,
                        "clip_range": clip_range,
                        "ent_coef": ent_coef,
                    },
                )
            else:
                print("Initializing fresh single PPO model...")
                model = PPO(
                    policy="MlpPolicy",
                    env=train_env,
                    learning_rate=learning_rate,
                    n_steps=n_steps,
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                    gamma=gamma,
                    gae_lambda=gae_lambda,
                    clip_range=clip_range,
                    ent_coef=ent_coef,
                    max_grad_norm=max_grad_norm,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log=log_dir,
                    verbose=1,
                    seed=current_seed,
                )
        else:
            model.set_env(train_env)

        print(f"\nStarting optimization on Seed {current_seed} for {timesteps_per_seed:,} steps...")
        model.learn(
            total_timesteps=timesteps_per_seed,
            callback=callbacks,
            reset_num_timesteps=False,
        )
        seed_milestones.append(model.num_timesteps)
        print(f"Completed Stage {stage_num}/{n_seeds} (Seed {current_seed}). Global timesteps: {model.num_timesteps:,}")

    # Clean up transit vecnorm
    if os.path.exists(temp_vecnorm_path):
        try:
            os.remove(temp_vecnorm_path)
        except OSError:
            pass

    # Save final single model and VecNormalize statistics
    final_model_path = os.path.join(model_dir, "reach_model")
    model.save(final_model_path)
    final_vecnorm_path = os.path.join(model_dir, "vecnormalize.pkl")
    train_env.save(final_vecnorm_path)

    metadata = {
        "seeds": seeds,
        "total_timesteps": model.num_timesteps,
        "timesteps_per_seed": timesteps_per_seed,
        "n_envs": n_envs,
        "seed_milestones": seed_milestones,
        "observation_dim": int(train_env.observation_space.shape[0]),
        "action_dim": int(train_env.action_space.shape[0]),
        "final_model_path": f"{final_model_path}.zip",
        "vecnormalize_path": final_vecnorm_path,
    }
    with open(os.path.join(model_dir, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 78)
    print(" ALL 3 SEEDS SUCCESSFULLY TRAINED INTO ONE SINGLE MODEL!")
    print(f" Final Model Saved:       {final_model_path}.zip")
    print(f" VecNormalize Stats:      {final_vecnorm_path}")
    print("=" * 78)

    train_env.close()
    eval_env.close()

    # -----------------------------------------------------------------------
    # Post-Training Step 1: Statistical Evaluation (Mean ± Std)
    # -----------------------------------------------------------------------
    eval_model_zip = f"{final_model_path}.zip"
    eval_stats = evaluate_single_reach_model_statistically(
        model_path=eval_model_zip,
        vecnorm_path=final_vecnorm_path,
        n_episodes=n_eval_episodes,
        output_dir=output_dir,
    )

    # -----------------------------------------------------------------------
    # Post-Training Step 2: Learning Curves Visualization
    # -----------------------------------------------------------------------
    plot_path = os.path.join(output_dir, "reach_learning_curves.png")
    plot_single_reach_learning_curves(
        log_dir=log_dir,
        output_path=plot_path,
        seed_milestones=seed_milestones,
        seeds=seeds,
        eval_stats=eval_stats,
    )

    return eval_stats



# Statistical Evaluation (Mean ± Std) Function

def evaluate_single_reach_model_statistically(
    model_path: str,
    vecnorm_path: str | None,
    n_episodes: int = 50,
    output_dir: str = "reports/reach_single_model",
    render: bool = False,
) -> dict:
    """
    Evaluates a trained single reaching model across held-out randomized episodes.
    Computes sample Mean ± Std for all primary metrics (overall & zone-specific).
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 78)
    print(" POST-TRAINING STATISTICAL EVALUATION: REACHING (MEAN ± STD)")
    print(f" Evaluating:     {model_path}")
    print(f" VecNormalize:   {vecnorm_path}")
    print(f" Total Episodes: {n_episodes}")
    print("=" * 78)

    dummy_env = DummyVecEnv([lambda: SO101ReachingEnv()])
    if vecnorm_path and os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, dummy_env)
        env.training = False
        env.norm_reward = False
    else:
        env = dummy_env

    model = PPO.load(model_path, env=env, device="cpu")

    successes = []
    final_errors_mm = []
    reach_times_s = []
    action_deltas = []
    episode_returns = []

    zone_records = {"bottom": [], "medium": [], "high": []}
    zone_names = ["bottom", "medium", "high"]

    for ep in range(n_episodes):
        ep_seed = 70_000 + ep
        zone_choice = zone_names[ep % len(zone_names)]

        raw_env = env.envs[0]
        raw_env.current_zone = zone_choice

        obs = env.env_method("reset", seed=ep_seed)[0][0]
        if isinstance(env, VecNormalize):
            obs_tensor = env.normalize_obs(np.array([obs]))
        else:
            obs_tensor = np.array([obs])

        done = False
        ep_ret = 0.0
        step = 0
        deltas = []
        prev_act = np.zeros(5, dtype=np.float32)

        while not done:
            action, _ = model.predict(obs_tensor, deterministic=True)
            obs_tensor, reward, dones, infos = env.step(action)
            info = infos[0]
            done = dones[0]
            ep_ret += float(reward[0])

            act_flat = np.asarray(action[0]).flatten()
            deltas.append(float(np.mean(np.abs(act_flat - prev_act))))
            prev_act = act_flat.copy()
            step += 1

            if render:
                raw_env.render()

        succ = 1.0 if info.get("success", False) else 0.0
        err_mm = float(info.get("dist_mm", 0.0))
        time_s = step * 0.040  # 25 Hz policy step -> 0.040s

        successes.append(succ)
        final_errors_mm.append(err_mm)
        reach_times_s.append(time_s)
        action_deltas.append(float(np.mean(deltas)) if deltas else 0.0)
        episode_returns.append(ep_ret)

        actual_zone = info.get("target_zone", zone_choice)
        if actual_zone in zone_records:
            zone_records[actual_zone].append(succ)

    env.close()

    # Calculate Mean ± Std (ddof=1)
    def calc_mean_std(data):
        arr = np.array(data, dtype=np.float64)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        return mean_val, std_val

    succ_m, succ_s = calc_mean_std(successes)
    err_m, err_s = calc_mean_std(final_errors_mm)
    time_m, time_s = calc_mean_std(reach_times_s)
    smooth_m, smooth_s = calc_mean_std(action_deltas)
    ret_m, ret_s = calc_mean_std(episode_returns)

    b_succ_m, b_succ_s = calc_mean_std(zone_records["bottom"]) if zone_records["bottom"] else (0.0, 0.0)
    m_succ_m, m_succ_s = calc_mean_std(zone_records["medium"]) if zone_records["medium"] else (0.0, 0.0)
    h_succ_m, h_succ_s = calc_mean_std(zone_records["high"]) if zone_records["high"] else (0.0, 0.0)

    stats = {
        "n_episodes": n_episodes,
        "overall_success_rate_percent": {"mean": succ_m * 100.0, "std": succ_s * 100.0},
        "euclidean_distance_error_mm": {"mean": err_m, "std": err_s},
        "time_to_reach_s": {"mean": time_m, "std": time_s},
        "action_smoothness": {"mean": smooth_m, "std": smooth_s},
        "episode_return": {"mean": ret_m, "std": ret_s},
        "bottom_zone_success_percent": {"mean": b_succ_m * 100.0, "std": b_succ_s * 100.0},
        "medium_zone_success_percent": {"mean": m_succ_m * 100.0, "std": m_succ_s * 100.0},
        "high_zone_success_percent": {"mean": h_succ_m * 100.0, "std": h_succ_s * 100.0},
    }

    # Print Table
    print("\n" + "=" * 78)
    print("         SO-101 PRECISION REACHING: FINAL STATISTICAL BENCHMARK       ")
    print("=" * 78)
    print(f" {'Metric Description':<38} | {'Sample Mean ± Std (ddof=1)':<32}")
    print("-" * 78)
    print(f" {'Overall Task Success Rate':<38} | {stats['overall_success_rate_percent']['mean']:>6.2f}% ± {stats['overall_success_rate_percent']['std']:<5.2f}%")
    print(f" {'Final Euclidean Distance Error (mm)':<38} | {stats['euclidean_distance_error_mm']['mean']:>6.2f} ± {stats['euclidean_distance_error_mm']['std']:<5.2f} mm")
    print(f" {'Time to Reach & Settle (s)':<38} | {stats['time_to_reach_s']['mean']:>6.2f} ± {stats['time_to_reach_s']['std']:<5.2f} s")
    print(f" {'Action Smoothness Delta':<38} | {stats['action_smoothness']['mean']:>6.4f} ± {stats['action_smoothness']['std']:<5.4f}")
    print(f" {'Episodic Return':<38} | {stats['episode_return']['mean']:>7.2f} ± {stats['episode_return']['std']:<6.2f}")
    print("-" * 78)
    print(" 3-ZONE STRATIFIED PERFORMANCE BREAKDOWN:")
    print(f"   * Bottom Zone Success (0.015 - 0.065 m):   {stats['bottom_zone_success_percent']['mean']:>6.2f}% ± {stats['bottom_zone_success_percent']['std']:<5.2f}%")
    print(f"   * Medium Zone Success (0.075 - 0.180 m):   {stats['medium_zone_success_percent']['mean']:>6.2f}% ± {stats['medium_zone_success_percent']['std']:<5.2f}%")
    print(f"   * High Zone Success   (0.190 - 0.320 m):   {stats['high_zone_success_percent']['mean']:>6.2f}% ± {stats['high_zone_success_percent']['std']:<5.2f}%")
    print("=" * 78 + "\n")

    # Save to JSON
    json_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # Save to CSV
    csv_path = os.path.join(output_dir, "evaluation_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Mean", "Std", "Unit"])
        writer.writerow(["Overall Success Rate", stats["overall_success_rate_percent"]["mean"], stats["overall_success_rate_percent"]["std"], "%"])
        writer.writerow(["Euclidean Distance Error", stats["euclidean_distance_error_mm"]["mean"], stats["euclidean_distance_error_mm"]["std"], "mm"])
        writer.writerow(["Time to Reach", stats["time_to_reach_s"]["mean"], stats["time_to_reach_s"]["std"], "s"])
        writer.writerow(["Action Smoothness", stats["action_smoothness"]["mean"], stats["action_smoothness"]["std"], "delta"])
        writer.writerow(["Episode Return", stats["episode_return"]["mean"], stats["episode_return"]["std"], "reward"])
        writer.writerow(["Bottom Zone Success", stats["bottom_zone_success_percent"]["mean"], stats["bottom_zone_success_percent"]["std"], "%"])
        writer.writerow(["Medium Zone Success", stats["medium_zone_success_percent"]["mean"], stats["medium_zone_success_percent"]["std"], "%"])
        writer.writerow(["High Zone Success", stats["high_zone_success_percent"]["mean"], stats["high_zone_success_percent"]["std"], "%"])

    print(f"[Saved] Evaluation JSON summary: {json_path}")
    print(f"[Saved] Evaluation CSV summary:  {csv_path}")
    return stats



# Learning Curves Visualization Function

def extract_tensorboard_scalars(log_dir: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Extracts scalar time-series from all tfevents in a directory."""
    event_files = sorted(glob.glob(os.path.join(log_dir, "**", "events.out.tfevents*"), recursive=True))
    if not event_files:
        return {}

    all_data: dict[str, list[tuple[int, float]]] = {}
    for ef in event_files:
        try:
            ea = event_accumulator.EventAccumulator(ef)
            ea.Reload()
            tags = ea.Tags().get("scalars", [])
            for t in tags:
                if t not in all_data:
                    all_data[t] = []
                for s in ea.Scalars(t):
                    all_data[t].append((s.step, s.value))
        except Exception as e:
            print(f"Warning reading {ef}: {e}")

    result = {}
    for tag, pairs in all_data.items():
        if not pairs:
            continue
        pairs.sort(key=lambda x: x[0])
        steps = [pairs[0][0]]
        vals = [pairs[0][1]]
        for st, v in pairs[1:]:
            if st == steps[-1]:
                vals[-1] = v
            else:
                steps.append(st)
                vals.append(v)
        result[tag] = (np.array(steps, dtype=np.int64), np.array(vals, dtype=np.float64))
    return result


def plot_single_reach_learning_curves(
    log_dir: str,
    output_path: str,
    seed_milestones: list[int] | None = None,
    seeds: list[int] | None = None,
    eval_stats: dict | None = None,
):
    """
    Plots a 4-panel comprehensive training learning curves figure for Reaching:
    1. Episode Return over global timesteps (with seed boundary markers).
    2. Overall Success Rate over global timesteps.
    3. Final Euclidean Distance Error (mm) progression.
    4. 3-Zone Performance Breakdown (Bottom, Medium, High).
    """
    scalars = extract_tensorboard_scalars(log_dir)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)

    # 1. Episode Return
    ax = axes[0, 0]
    if "rollout/ep_rew_mean" in scalars:
        st, val = scalars["rollout/ep_rew_mean"]
        ax.plot(st, val, color="#1f77b4", linewidth=2.0, label="Episode Return (Rolling 100)")
    ax.set_title("Training Episode Return Progression", fontsize=12, fontweight="bold")
    ax.set_xlabel("Global Timesteps")
    ax.set_ylabel("Mean Return")
    ax.grid(True, linestyle="--", alpha=0.6)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best")

    # 2. Overall Success Rate
    ax = axes[0, 1]
    if "rollout/success_rate" in scalars:
        st, val = scalars["rollout/success_rate"]
        ax.plot(st, val * 100.0, color="#2ca02c", linewidth=2.2, label="Success Rate (< 10mm & 0.5s Hold)")
    ax.set_title("Overall Reaching Success Rate (%)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Global Timesteps")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(-5, 105)
    ax.grid(True, linestyle="--", alpha=0.6)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best")

    # 3. Final Euclidean Distance Error (mm)
    ax = axes[1, 0]
    if "rollout/mean_final_err_mm" in scalars:
        st, val = scalars["rollout/mean_final_err_mm"]
        ax.plot(st, val, color="#d62728", linewidth=2.0, label="Final Error (mm)")
        ax.axhline(10.0, color="gray", linestyle=":", linewidth=1.5, label="10 mm Tolerance Gate")
    ax.set_title("Euclidean Distance Error to Target (mm)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Global Timesteps")
    ax.set_ylabel("Distance Error (mm)")
    ax.grid(True, linestyle="--", alpha=0.6)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper right")

    # 4. 3-Zone Performance Breakdown
    ax = axes[1, 1]
    zone_plotted = False
    zone_colors = {"bottom": "#1f77b4", "medium": "#ff7f0e", "high": "#2ca02c"}
    for z in ["bottom", "medium", "high"]:
        tag = f"rollout/{z}_zone_success"
        if tag in scalars:
            st, val = scalars[tag]
            ax.plot(st, val * 100.0, color=zone_colors[z], linewidth=1.8, label=f"{z.capitalize()} Zone")
            zone_plotted = True

    if not zone_plotted and eval_stats:
        labels = ["Bottom", "Medium", "High"]
        means = [
            eval_stats["bottom_zone_success_percent"]["mean"],
            eval_stats["medium_zone_success_percent"]["mean"],
            eval_stats["high_zone_success_percent"]["mean"],
        ]
        stds = [
            eval_stats["bottom_zone_success_percent"]["std"],
            eval_stats["medium_zone_success_percent"]["std"],
            eval_stats["high_zone_success_percent"]["std"],
        ]
        bars = ax.bar(labels, means, yerr=stds, capsize=5, color=["#1f77b4", "#ff7f0e", "#2ca02c"], alpha=0.85)
        ax.set_title("Stratified 3-Zone Success Rates (Mean ± Std)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Success Rate (%)")
        ax.set_ylim(-5, 105)
    else:
        ax.set_title("3-Zone Success Progression (%)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Global Timesteps")
        ax.set_ylabel("Success Rate (%)")
        ax.set_ylim(-5, 105)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.6)

    # Add vertical dashed lines indicating transitions between seeds
    if seed_milestones and seeds and len(seed_milestones) > 1:
        for idx in range(len(seed_milestones) - 1):
            ms = seed_milestones[idx]
            from_seed = seeds[idx]
            to_seed = seeds[idx + 1]
            for a in axes.flat:
                a.axvline(ms, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)
                a.text(
                    ms, a.get_ylim()[1] * 0.92,
                    f" Seed {from_seed} -> {to_seed}",
                    color="gray", fontsize=8, rotation=90, verticalalignment="top"
                )

    plt.suptitle("SO-101 Precision Reaching: Single Continuous Model Training Across 3 Random Seeds", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Saved] Publication learning curves figure: {output_path}")



# Compatibility Wrappers for Legacy Imports

def train_reach_seed(
    seed=42,
    total_timesteps=300_000,
    n_envs=8,
    config_path="configs/ppo_so101_config.yaml",
    model_dir="models/so101/reach",
    log_dir="logs/so101_reach",
    load_checkpoint=None,
    load_vecnorm=None,
):
    """Compatibility trainer for an individual seed."""
    run_model_dir = os.path.join(model_dir, f"seed_{seed}")
    run_log_dir = os.path.join(log_dir, f"seed_{seed}")
    train_reach_multiseed_single_model(
        seeds=[seed],
        total_timesteps=total_timesteps,
        n_envs=n_envs,
        config_path=config_path,
        model_dir=run_model_dir,
        log_dir=run_log_dir,
        load_checkpoint=load_checkpoint,
        load_vecnorm=load_vecnorm,
    )


train_so101_reach = train_reach_seed



# Command Line Interface

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train & Evaluate SO-101 Precision Reaching across 3 random seeds into ONE single model"
    )
    parser.add_argument("--timesteps", type=int, default=300_000, help="Total timesteps across all 3 seeds")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456], help="3 seeds to train sequentially")
    parser.add_argument("--random-seeds", action="store_true", help="Randomly generate 3 seeds")
    parser.add_argument("--envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--model-dir", type=str, default="models/so101/reach", help="Directory to save the single model")
    parser.add_argument("--log-dir", type=str, default="logs/so101_reach", help="Directory for TensorBoard logs")
    parser.add_argument("--eval-episodes", type=int, default=50, help="Number of evaluation episodes for Mean ± Std benchmark")
    parser.add_argument("--output-dir", type=str, default="reports/reach_single_model", help="Directory to save plots and reports")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and run statistical evaluation + plot curves")
    parser.add_argument("--render", action="store_true", help="Render simulation live in MuJoCo viewer during evaluation")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .zip to resume from")
    parser.add_argument("--vecnorm", type=str, default=None, help="Path to vecnormalize .pkl")
    args = parser.parse_args()

    chosen_seeds = args.seeds
    if args.random_seeds:
        chosen_seeds = list(np.random.randint(10, 9999, size=3))

    if args.eval_only:
        final_model = os.path.join(args.model_dir, "reach_model.zip")
        if not os.path.exists(final_model):
            candidates = [
                os.path.join(args.model_dir, "best_model.zip"),
                str(PROJECT_ROOT / "models/so101/reach/seed_456/best_model.zip"),
                str(PROJECT_ROOT / "models/so101/reach/seed_123/best_model.zip"),
                str(PROJECT_ROOT / "models/so101/reach/final_reach/best_model.zip"),
            ]
            for c in candidates:
                if os.path.exists(c):
                    final_model = c
                    break

        vecnorm_file = os.path.join(args.model_dir, "vecnormalize.pkl")
        if not os.path.exists(vecnorm_file):
            for vc in [
                str(PROJECT_ROOT / "models/so101/reach/seed_456/so101_reach_vecnorm.pkl"),
                str(PROJECT_ROOT / "models/so101/reach/seed_456/vecnormalize.pkl"),
                str(PROJECT_ROOT / "models/so101/reach/final_reach/vecnormalize.pkl"),
            ]:
                if os.path.exists(vc):
                    vecnorm_file = vc
                    break

        eval_stats = evaluate_single_reach_model_statistically(
            model_path=final_model,
            vecnorm_path=vecnorm_file if os.path.exists(vecnorm_file) else None,
            n_episodes=args.eval_episodes,
            output_dir=args.output_dir,
            render=args.render,
        )
        plot_single_reach_learning_curves(
            log_dir=args.log_dir if os.path.exists(args.log_dir) else str(PROJECT_ROOT / "logs/so101_reach"),
            output_path=os.path.join(args.output_dir, "reach_learning_curves.png"),
            eval_stats=eval_stats,
        )
    else:
        train_reach_multiseed_single_model(
            seeds=chosen_seeds,
            total_timesteps=args.timesteps,
            n_envs=args.envs,
            model_dir=args.model_dir,
            log_dir=args.log_dir,
            load_checkpoint=args.resume,
            load_vecnorm=args.vecnorm,
            n_eval_episodes=args.eval_episodes,
            output_dir=args.output_dir,
        )
