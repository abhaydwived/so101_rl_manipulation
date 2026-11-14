#!/usr/bin/env python3
"""
Single-Model Sequential Multi-Seed Training & Statistical Evaluation Suite for SO-101 Pick-and-Lift.

Architecture & Workflow:
1. High-level PPO: 26-D observation -> 6-D continuous action deltas.
2. Low-level: Feetech STS3215 actuator group at 500 Hz (20 substeps @ 25 Hz policy rate).
3. Trains across the given seeds sequentially within ONE model checkpoint. Network weights,
   optimizer state and VecNormalize statistics carry over between stages.
   NOTE: the env distribution is identical in every stage, so this behaves like one long run
   with periodic env restarts. The +- reported at the end is variation across held-out
   EPISODES of that one model, not variation across independent training seeds. For true
   seed-to-seed variance, train each seed from scratch (see train_pick_lift_seed).
4. Post-Training Evaluation (curriculum rehearsal disabled):
   - Success / lift-target / grasp rates, lift height, hold duration, return
   - Decomposed rewards that sum exactly to the episode return
5. Learning Curves: 4-panel plot from TensorBoard logs, saved with CSV/JSON summaries.
"""

from __future__ import annotations

import os
import sys
import glob
import json
import csv
import argparse
import time
import warnings
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

from environments.pick_and_lift_env import SO101PickLiftEnv

try:
    _temp_env = SO101PickLiftEnv()
    OBS_DIM = int(_temp_env.observation_space.shape[0])
    _temp_env.close()
    del _temp_env
except Exception:
    OBS_DIM = 32

ACT_DIM = 6

REWARD_TAGS = (
    "reaching", "grasp", "lift_progress", "lift_milestone",
    "holding", "failure_penalty", "success",
)


class ObsAdapterWrapper(gym.ObservationWrapper):
    """Adapts observation dimensions bidirectionally for cross-checkpoint evaluation."""
    def __init__(self, env, target_dim):
        super().__init__(env)
        self.target_dim = target_dim
        base_dim = env.observation_space.shape[0]
        if target_dim == base_dim:
            self.observation_space = env.observation_space
        elif target_dim < base_dim:
            low = env.observation_space.low[:target_dim]
            high = env.observation_space.high[:target_dim]
            self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        else:
            diff = target_dim - base_dim
            low = np.pad(env.observation_space.low, (0, diff), constant_values=-10.0)
            high = np.pad(env.observation_space.high, (0, diff), constant_values=10.0)
            self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs):
        base_dim = len(obs)
        if self.target_dim == base_dim:
            return obs
        if self.target_dim < base_dim:
            return obs[:self.target_dim].astype(np.float32)
        pad = np.zeros(self.target_dim - base_dim, dtype=np.float32)
        return np.concatenate([obs, pad]).astype(np.float32)



# Helpers

def decompose_reward(info: dict) -> tuple:
    """
    Splits one step's reward into 7 buckets using the env's per-component info keys.
    The buckets cover all 14 env components, so they sum to the step reward.
    Order matches REWARD_TAGS.
    """
    reach = float(info.get("reward_reaching", 0.0))
    grasp = float(
        info.get("reward_grasp", 0.0)
        + info.get("reward_touch", 0.0)
        + info.get("reward_proximity", 0.0)
        + info.get("reward_grasp_hold", 0.0)
    )
    lift_prog = float(info.get("reward_lift", 0.0))
    lift_mile = float(info.get("reward_lift_bonus", 0.0))
    hold = float(info.get("reward_lift_hold", 0.0))
    fail = float(
        info.get("reward_drop", 0.0)
        + info.get("reward_step_penalty", 0.0)
        + info.get("reward_lift_overshoot", 0.0)
        + info.get("reward_action_reg", 0.0)
        + info.get("reward_escape", 0.0)
    )
    succ = float(info.get("reward_success", 0.0))
    return reach, grasp, lift_prog, lift_mile, hold, fail, succ


class EvalOptionsWrapper(gym.Wrapper):
    """Forces evaluation=True on every reset so grasp-rehearsal starts are never used."""

    def reset(self, **kwargs):
        options = dict(kwargs.pop("options", None) or {})
        options["evaluation"] = True
        return self.env.reset(options=options, **kwargs)


def make_env_fn():
    def _init():
        return Monitor(SO101PickLiftEnv())
    return _init


def make_eval_env_fn():
    def _init():
        return Monitor(EvalOptionsWrapper(SO101PickLiftEnv()))
    return _init



# Training Metrics Callback

class PickLiftMetricsCallback(BaseCallback):
    """Tracks and logs Pick-and-Lift episode metrics and decomposed reward components."""

    def __init__(self, window_size=100, verbose=0):
        super().__init__(verbose)
        self.window_size = window_size

        self.success_history = []
        self.lift_target_history = []
        self.grasp_history = []
        self.lift_height_history = []
        self.hold_steps_history = []
        self.velocity_history = []

        # Per-env, per-episode accumulators (same definitions as the final evaluation)
        self.cur_comp = [dict() for _ in REWARD_TAGS]
        self.comp_history = [[] for _ in REWARD_TAGS]
        self.ep_max_lift = {}
        self.ep_lift_hit = {}
        self.ep_max_hold = {}

        # Why lifted episodes fail to count as a valid hold (reset every rollout)
        self._reset_hold_fail_counters()

    def _reset_hold_fail_counters(self):
        self.hf_steps = 0
        self.hf_lin = 0
        self.hf_ang = 0
        self.hf_joint = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        for i, (done, info) in enumerate(zip(dones, infos)):
            comps = decompose_reward(info)
            for acc, v in zip(self.cur_comp, comps):
                acc[i] = acc.get(i, 0.0) + v

            self.ep_max_lift[i] = max(self.ep_max_lift.get(i, 0.0), float(info.get("lift_height", 0.0)))
            self.ep_lift_hit[i] = self.ep_lift_hit.get(i, False) or bool(info.get("lift_target_achieved", False))
            self.ep_max_hold[i] = max(self.ep_max_hold.get(i, 0), int(info.get("consec_lift_hold_steps", 0)))

            if info.get("lift_target_achieved", False):
                self.hf_steps += 1
                self.hf_lin += int(bool(info.get("hold_fail_cube_lin_vel", False)))
                self.hf_ang += int(bool(info.get("hold_fail_cube_ang_vel", False)))
                self.hf_joint += int(bool(info.get("hold_fail_arm_joint_vel", False)))

            if done:
                self.success_history.append(float(info.get("success", False)))
                self.lift_target_history.append(float(self.ep_lift_hit.pop(i, False)))
                self.grasp_history.append(float(info.get("has_grasped_once", False)))
                self.lift_height_history.append(self.ep_max_lift.pop(i, 0.0))
                self.hold_steps_history.append(float(self.ep_max_hold.pop(i, 0)))
                self.velocity_history.append(float(info.get("cube_linear_velocity", 0.0)))
                for acc, hist in zip(self.cur_comp, self.comp_history):
                    hist.append(acc.pop(i, 0.0))
        return True

    def _on_rollout_end(self) -> None:
        w = self.window_size
        if len(self.success_history) > 0:
            self.logger.record("rollout/success_rate", float(np.mean(self.success_history[-w:])))
            self.logger.record("rollout/lift_target_rate", float(np.mean(self.lift_target_history[-w:])))
            self.logger.record("rollout/grasp_rate", float(np.mean(self.grasp_history[-w:])))
            self.logger.record("rollout/max_lift_height_mean", float(np.mean(self.lift_height_history[-w:])))
            self.logger.record("rollout/max_consec_hold_mean", float(np.mean(self.hold_steps_history[-w:])))
            self.logger.record("rollout/cube_velocity_mean", float(np.mean(self.velocity_history[-w:])))

        if len(self.comp_history[0]) > 0:
            for tag, hist in zip(REWARD_TAGS, self.comp_history):
                self.logger.record(f"reward/mean_{tag}", float(np.mean(hist[-w:])))

        # Fraction of "cube is >= 5 cm and grasped" steps that fail each stability gate
        if self.hf_steps > 0:
            self.logger.record("hold/fail_cube_lin_vel_frac", self.hf_lin / self.hf_steps)
            self.logger.record("hold/fail_cube_ang_vel_frac", self.hf_ang / self.hf_steps)
            self.logger.record("hold/fail_arm_joint_vel_frac", self.hf_joint / self.hf_steps)
        self._reset_hold_fail_counters()



# Training Pipeline: seeds -> 1 Single Unified Model

def train_pick_and_lift_multiseed_single_model(
    seeds=(42, 123, 456),
    total_timesteps=1_000_000,
    n_envs=8,
    config_path="configs/ppo_so101_config.yaml",
    model_dir="models/so101/pick_lift",
    log_dir="logs/so101_pick_lift",
    load_checkpoint=None,
    load_vecnorm=None,
    n_eval_episodes=50,
    output_dir="reports/pick_lift_single_model",
):
    """
    Trains Pick-and-Lift sequentially across the given seeds into ONE continuous model,
    then evaluates the final model (Mean ± Std over episodes) and plots learning curves.
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

    seeds = [int(s) for s in seeds]          # numpy ints are not JSON serializable
    n_seeds = len(seeds)
    timesteps_per_seed = total_timesteps // n_seeds

    print("=" * 78)
    print(" SO-101 PICK-AND-LIFT: SINGLE-MODEL MULTI-SEED CONTINUOUS TRAINING")
    print("=" * 78)
    print(f"Seeds ({n_seeds}):            {seeds}")
    print(f"Total Timesteps:         {total_timesteps:,}")
    print(f"Timesteps per Seed:      {timesteps_per_seed:,}")
    print(f"Parallel Envs:           {n_envs}")
    print(f"Observation / Action:    {OBS_DIM}-D / {ACT_DIM}-D")
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
    n_steps = int(ppo_cfg.get("n_steps", 2048))
    batch_size = int(ppo_cfg.get("batch_size", 128))
    n_epochs = int(ppo_cfg.get("n_epochs", 10))
    gamma = float(ppo_cfg.get("gamma", 0.99))
    gae_lambda = float(ppo_cfg.get("gae_lambda", 0.95))
    clip_range = float(ppo_cfg.get("clip_range", 0.2))
    ent_coef = float(ppo_cfg.get("ent_coef", 0.005))
    max_grad_norm = float(ppo_cfg.get("max_grad_norm", 0.5))

    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    metrics_callback = PickLiftMetricsCallback(window_size=100)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, 25_000 // n_envs),
        save_path=checkpoint_dir,
        name_prefix="pick_lift_single_model",
        save_vecnormalize=True,
    )

    temp_vecnorm_path = os.path.join(model_dir, "temp_vecnorm_transit.pkl")
    model = None
    train_env = None
    eval_env = None
    seed_milestones = []
    best_so_far = -np.inf     # carried across stages so best_model.zip is the true overall best

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

        # 2. Fresh training environments (grasp-rehearsal curriculum active)
        new_train_env = make_vec_env(
            make_env_fn(),
            n_envs=n_envs,
            seed=current_seed,
            vec_env_cls=SubprocVecEnv,
        )

        # 3. Lightweight eval env: always evaluation=True (no rehearsal starts)
        new_eval_env = make_vec_env(
            make_eval_env_fn(),
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

        assert train_env.observation_space.shape[0] == OBS_DIM, (
            f"Env observation is {train_env.observation_space.shape[0]}-D, expected {OBS_DIM}-D"
        )
        assert train_env.action_space.shape[0] == ACT_DIM, (
            f"Env action is {train_env.action_space.shape[0]}-D, expected {ACT_DIM}-D"
        )

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=model_dir,
            log_path=log_dir,
            eval_freq=max(1, 20_000 // n_envs),
            n_eval_episodes=20,
            deterministic=True,
            render=False,
        )
        eval_callback.best_mean_reward = best_so_far

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
        best_so_far = eval_callback.best_mean_reward
        seed_milestones.append(int(model.num_timesteps))
        print(f"Completed Stage {stage_num}/{n_seeds} (Seed {current_seed}). Global timesteps: {model.num_timesteps:,}")

    # Clean up transit vecnorm
    if os.path.exists(temp_vecnorm_path):
        try:
            os.remove(temp_vecnorm_path)
        except OSError:
            pass

    # Save final single model and VecNormalize statistics
    final_model_path = os.path.join(model_dir, "pick_lift_model")
    model.save(final_model_path)
    final_vecnorm_path = os.path.join(model_dir, "vecnormalize.pkl")
    train_env.save(final_vecnorm_path)

    metadata = {
        "seeds": seeds,
        "total_timesteps": int(model.num_timesteps),
        "timesteps_per_seed": int(timesteps_per_seed),
        "n_envs": int(n_envs),
        "seed_milestones": seed_milestones,
        "observation_dim": int(train_env.observation_space.shape[0]),
        "action_dim": int(train_env.action_space.shape[0]),
        "final_model_path": f"{final_model_path}.zip",
        "vecnormalize_path": final_vecnorm_path,
    }
    with open(os.path.join(model_dir, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 78)
    print(f" ALL {n_seeds} SEED STAGES TRAINED INTO ONE SINGLE MODEL")
    print(f" Final Model Saved:       {final_model_path}.zip")
    print(f" VecNormalize Stats:      {final_vecnorm_path}")
    print("=" * 78)

    train_env.close()
    eval_env.close()

    # Post-Training Step 1: Statistical Evaluation (Mean ± Std)
    eval_stats = evaluate_single_model_statistically(
        model_path=f"{final_model_path}.zip",
        vecnorm_path=final_vecnorm_path,
        n_episodes=n_eval_episodes,
        output_dir=output_dir,
    )

    # Post-Training Step 2: Learning Curves Visualization
    plot_path = os.path.join(output_dir, "pick_lift_learning_curves.png")
    plot_single_model_learning_curves(
        log_dir=log_dir,
        output_path=plot_path,
        seed_milestones=seed_milestones,
        seeds=seeds,
        eval_stats=eval_stats,
    )

    return eval_stats



# Statistical Evaluation (Mean ± Std) Function

def evaluate_single_model_statistically(
    model_path: str,
    vecnorm_path: str | None,
    n_episodes: int = 50,
    output_dir: str = "reports/pick_lift_single_model",
    render: bool = False,
) -> dict:
    """
    Evaluates a trained model across held-out randomized episodes (rehearsal disabled).
    Computes sample Mean ± Std for all primary task metrics & decomposed reward components.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 78)
    print(" POST-TRAINING STATISTICAL EVALUATION (MEAN ± STD)")
    print(f" Evaluating:     {model_path}")
    print(f" VecNormalize:   {vecnorm_path}")
    print(f" Total Episodes: {n_episodes}")
    print("=" * 78)

    target_dim = OBS_DIM
    if vecnorm_path and os.path.exists(vecnorm_path):
        import pickle
        with open(vecnorm_path, "rb") as f:
            vn_data = pickle.load(f)
            target_dim = int(vn_data.observation_space.shape[0])
    elif model_path and os.path.exists(model_path):
        try:
            m_temp = PPO.load(model_path, device="cpu")
            target_dim = int(m_temp.observation_space.shape[0])
            del m_temp
        except Exception:
            pass

    def _make_eval_env():
        base = SO101PickLiftEnv()
        if target_dim != base.observation_space.shape[0]:
            return ObsAdapterWrapper(base, target_dim=target_dim)
        return base

    dummy_env = DummyVecEnv([_make_eval_env])
    if vecnorm_path and os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, dummy_env)
        env.training = False
        env.norm_reward = False
    else:
        env = dummy_env

    # SB3 raises a clear error here if the model's observation space is not 26-D.
    model = PPO.load(model_path, env=env, device="cpu")

    successes, lift_targets, grasps = [], [], []
    lift_heights_cm, hold_durations_s = [], []
    episode_returns, cube_target_errors_mm = [], []
    comp_per_episode = [[] for _ in REWARD_TAGS]
    decomposition_mismatch = 0.0

    for ep in range(n_episodes):
        ep_seed = 60_000 + ep
        obs = env.env_method("reset", seed=ep_seed, options={"evaluation": True})[0][0]
        if isinstance(env, VecNormalize):
            obs_tensor = env.normalize_obs(np.array([obs]))
        else:
            obs_tensor = np.array([obs])

        done = False
        ep_ret = 0.0
        max_h = 0.0
        max_hold = 0
        valid_lift_target = False
        ep_comp = np.zeros(len(REWARD_TAGS), dtype=np.float64)
        final_err_mm = 0.0
        info = {}

        while not done:
            action, _ = model.predict(obs_tensor, deterministic=True)
            obs_tensor, reward, dones, infos = env.step(action)
            info = infos[0]
            done = bool(dones[0])
            ep_ret += float(reward[0])
            ep_comp += np.array(decompose_reward(info), dtype=np.float64)

            max_h = max(max_h, float(info.get("lift_height", 0.0)))
            max_hold = max(max_hold, int(info.get("consec_lift_hold_steps", 0)))
            if info.get("lift_target_achieved", False):
                valid_lift_target = True
            final_err_mm = float(info.get("cube_target_err_mm", 0.0))

            if render:
                env.envs[0].render()

        decomposition_mismatch = max(decomposition_mismatch, abs(float(ep_comp.sum()) - ep_ret))

        dt = float(info.get("policy_dt", 0.04))
        successes.append(1.0 if info.get("success", False) else 0.0)
        lift_targets.append(1.0 if valid_lift_target else 0.0)
        grasps.append(1.0 if info.get("has_grasped_once", False) else 0.0)
        lift_heights_cm.append(max_h * 100.0)
        hold_durations_s.append(max_hold * dt)
        episode_returns.append(ep_ret)
        cube_target_errors_mm.append(final_err_mm)
        for hist, v in zip(comp_per_episode, ep_comp):
            hist.append(float(v))

    env.close()

    if decomposition_mismatch > 1e-3:
        warnings.warn(
            f"Reward components do not sum to the episode return "
            f"(max gap {decomposition_mismatch:.4f}). The env logs a component the trainer "
            "does not read, or VecNormalize reward scaling is on."
        )
    else:
        print(f"[Check] Reward components sum to the episode return (max gap {decomposition_mismatch:.2e}).")

    # Sample Mean ± Std (ddof=1)
    def calc_mean_std(data):
        arr = np.array(data, dtype=np.float64)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        return mean_val, std_val

    succ_mean, succ_std = calc_mean_std(successes)
    lift_mean, lift_std = calc_mean_std(lift_targets)
    grasp_mean, grasp_std = calc_mean_std(grasps)
    height_mean, height_std = calc_mean_std(lift_heights_cm)
    hold_mean, hold_std = calc_mean_std(hold_durations_s)
    ret_mean, ret_std = calc_mean_std(episode_returns)
    err_mean, err_std = calc_mean_std(cube_target_errors_mm)

    (r_reach_m, r_reach_s), (r_grasp_m, r_grasp_s), (r_lprog_m, r_lprog_s), \
        (r_lmile_m, r_lmile_s), (r_hold_m, r_hold_s), (r_fail_m, r_fail_s), \
        (r_succ_m, r_succ_s) = [calc_mean_std(h) for h in comp_per_episode]

    stats = {
        "n_episodes": n_episodes,
        "success_rate_percent": {"mean": succ_mean * 100.0, "std": succ_std * 100.0},
        "lift_target_rate_percent": {"mean": lift_mean * 100.0, "std": lift_std * 100.0},
        "grasp_rate_percent": {"mean": grasp_mean * 100.0, "std": grasp_std * 100.0},
        "lift_height_cm": {"mean": height_mean, "std": height_std},
        "hold_duration_s": {"mean": hold_mean, "std": hold_std},
        "episode_return": {"mean": ret_mean, "std": ret_std},
        "cube_target_error_mm": {"mean": err_mean, "std": err_std},
        "reward_reaching": {"mean": r_reach_m, "std": r_reach_s},
        "reward_grasp": {"mean": r_grasp_m, "std": r_grasp_s},
        "reward_lift_progress": {"mean": r_lprog_m, "std": r_lprog_s},
        "reward_lift_milestone": {"mean": r_lmile_m, "std": r_lmile_s},
        "reward_holding": {"mean": r_hold_m, "std": r_hold_s},
        "reward_failure_penalty": {"mean": r_fail_m, "std": r_fail_s},
        "reward_success": {"mean": r_succ_m, "std": r_succ_s},
    }

    # Print Table
    print("\n" + "=" * 78)
    print("           SO-101 PICK-AND-LIFT: FINAL STATISTICAL BENCHMARK          ")
    print("=" * 78)
    print(f" {'Metric Description':<38} | {'Sample Mean ± Std (ddof=1)':<32}")
    print("-" * 78)
    print(f" {'Task Success Rate':<38} | {stats['success_rate_percent']['mean']:>6.2f}% ± {stats['success_rate_percent']['std']:<5.2f}%")
    print(f" {'Lift Milestone Attainment (>= 5cm)':<38} | {stats['lift_target_rate_percent']['mean']:>6.2f}% ± {stats['lift_target_rate_percent']['std']:<5.2f}%")
    print(f" {'Bilateral Grasp Rate':<38} | {stats['grasp_rate_percent']['mean']:>6.2f}% ± {stats['grasp_rate_percent']['std']:<5.2f}%")
    print(f" {'Max Lift Height Gain (cm)':<38} | {stats['lift_height_cm']['mean']:>6.2f} ± {stats['lift_height_cm']['std']:<5.2f} cm")
    print(f" {'Max Steady Hold Duration (s)':<38} | {stats['hold_duration_s']['mean']:>6.2f} ± {stats['hold_duration_s']['std']:<5.2f} s")
    print(f" {'Cube Target Error (mm)':<38} | {stats['cube_target_error_mm']['mean']:>6.2f} ± {stats['cube_target_error_mm']['std']:<5.2f} mm")
    print(f" {'Episodic Return':<38} | {stats['episode_return']['mean']:>7.2f} ± {stats['episode_return']['std']:<6.2f}")
    print("-" * 78)
    print(" DECOMPOSED REWARD COMPONENT CONTRIBUTIONS (Mean ± Std):")
    print(f"   * Reaching Reward:             {stats['reward_reaching']['mean']:>6.2f} ± {stats['reward_reaching']['std']:<5.2f}")
    print(f"   * Grasp Reward:                {stats['reward_grasp']['mean']:>6.2f} ± {stats['reward_grasp']['std']:<5.2f}")
    print(f"   * Lift Progress Reward:        {stats['reward_lift_progress']['mean']:>6.2f} ± {stats['reward_lift_progress']['std']:<5.2f}")
    print(f"   * Lift Milestone Reward:       {stats['reward_lift_milestone']['mean']:>6.2f} ± {stats['reward_lift_milestone']['std']:<5.2f}")
    print(f"   * Steady Holding Reward:       {stats['reward_holding']['mean']:>6.2f} ± {stats['reward_holding']['std']:<5.2f}")
    print(f"   * Penalties (step/drop/etc.):  {stats['reward_failure_penalty']['mean']:>6.2f} ± {stats['reward_failure_penalty']['std']:<5.2f}")
    print(f"   * Terminal Success Bonus:      {stats['reward_success']['mean']:>6.2f} ± {stats['reward_success']['std']:<5.2f}")
    print("=" * 78 + "\n")

    # Save to JSON
    json_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # Save to CSV
    csv_path = os.path.join(output_dir, "evaluation_summary.csv")
    rows = [
        ("Success Rate", "success_rate_percent", "%"),
        ("Lift Target Rate", "lift_target_rate_percent", "%"),
        ("Grasp Rate", "grasp_rate_percent", "%"),
        ("Lift Height", "lift_height_cm", "cm"),
        ("Hold Duration", "hold_duration_s", "s"),
        ("Cube Target Error", "cube_target_error_mm", "mm"),
        ("Episode Return", "episode_return", "reward"),
        ("Reward Reaching", "reward_reaching", "reward"),
        ("Reward Grasp", "reward_grasp", "reward"),
        ("Reward Lift Progress", "reward_lift_progress", "reward"),
        ("Reward Lift Milestone", "reward_lift_milestone", "reward"),
        ("Reward Holding", "reward_holding", "reward"),
        ("Reward Penalties", "reward_failure_penalty", "reward"),
        ("Reward Success Bonus", "reward_success", "reward"),
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Mean", "Std", "Unit"])
        for label, key, unit in rows:
            writer.writerow([label, stats[key]["mean"], stats[key]["std"], unit])

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


def plot_single_model_learning_curves(
    log_dir: str,
    output_path: str,
    seed_milestones: list[int] | None = None,
    seeds: list[int] | None = None,
    eval_stats: dict | None = None,
):
    """
    Plots a 4-panel training learning curves figure:
    1. Episode return, 2. Success rate, 3. Grasp & lift milestone rates,
    4. Decomposed reward component trajectories.
    """
    scalars = extract_tensorboard_scalars(log_dir)
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

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

    # 2. Success Rate
    ax = axes[0, 1]
    if "rollout/success_rate" in scalars:
        st, val = scalars["rollout/success_rate"]
        ax.plot(st, val * 100.0, color="#2ca02c", linewidth=2.2, label="Success Rate (Lift >= 5cm & 1.0s Hold)")
    ax.set_title("Training Success Rate (%)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Global Timesteps")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(-5, 105)
    ax.grid(True, linestyle="--", alpha=0.6)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best")

    # 3. Grasp & Lift Milestone Rates
    ax = axes[1, 0]
    if "rollout/grasp_rate" in scalars:
        st, val = scalars["rollout/grasp_rate"]
        ax.plot(st, val * 100.0, color="#ff7f0e", linewidth=2.0, label="Grasp Rate (Bilateral Contact)")
    if "rollout/lift_target_rate" in scalars:
        st, val = scalars["rollout/lift_target_rate"]
        ax.plot(st, val * 100.0, color="#9467bd", linewidth=2.0, linestyle="--", label="Lift Target Rate (ever >= 5cm)")
    ax.set_title("Grasp & Lift Milestone Progression (%)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Global Timesteps")
    ax.set_ylabel("Attainment Rate (%)")
    ax.set_ylim(-5, 105)
    ax.grid(True, linestyle="--", alpha=0.6)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="lower right")

    # 4. Decomposed Reward Breakdown
    ax = axes[1, 1]
    decomp_tags = [
        ("reward/mean_reaching", "Reaching", "#1f77b4"),
        ("reward/mean_grasp", "Grasp Milestone", "#ff7f0e"),
        ("reward/mean_lift_progress", "Lift Progress", "#2ca02c"),
        ("reward/mean_lift_milestone", "Lift Milestone", "#8c564b"),
        ("reward/mean_holding", "Holding Steady", "#d62728"),
        ("reward/mean_success", "Success Bonus", "#9467bd"),
    ]
    has_decomp = False
    for tag, label, col in decomp_tags:
        if tag in scalars:
            st, val = scalars[tag]
            ax.plot(st, val, label=label, color=col, linewidth=1.8)
            has_decomp = True
    if not has_decomp and eval_stats:
        labels = ["Reach", "Grasp", "LiftProg", "LiftMile", "Hold", "Success"]
        keys = ["reward_reaching", "reward_grasp", "reward_lift_progress",
                "reward_lift_milestone", "reward_holding", "reward_success"]
        ax.bar(
            labels,
            [eval_stats[k]["mean"] for k in keys],
            yerr=[eval_stats[k]["std"] for k in keys],
            capsize=4, color="#17becf", alpha=0.8,
        )
        ax.set_title("Decomposed Reward Components (Mean ± Std)", fontsize=12, fontweight="bold")
    else:
        ax.set_title("Decomposed Reward Component Dynamics", fontsize=12, fontweight="bold")
        ax.set_xlabel("Global Timesteps")
        ax.set_ylabel("Reward Component Mean")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.6)

    # Vertical dashed lines marking transitions between seed stages
    if seed_milestones and seeds and len(seed_milestones) > 1:
        for idx in range(len(seed_milestones) - 1):
            ms = seed_milestones[idx]
            for a in axes.flat:
                a.axvline(ms, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)
                a.text(
                    ms, a.get_ylim()[1] * 0.92,
                    f" Seed {seeds[idx]} -> {seeds[idx + 1]}",
                    color="gray", fontsize=8, rotation=90, verticalalignment="top",
                )

    plt.suptitle("SO-101 Pick-and-Lift: Single Continuous Model Training", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Saved] Learning curves figure: {output_path}")



# Compatibility Wrappers for Legacy Imports

def train_pick_lift_seed(
    seed=42,
    total_timesteps=1_000_000,
    n_envs=8,
    config_path="configs/ppo_so101_config.yaml",
    model_dir="models/so101/hierarchical",
    log_dir="logs/so101_hierarchical",
    load_checkpoint=None,
    load_vecnorm=None,
):
    """Trains ONE seed from scratch into its own model (use this for independent per-seed runs)."""
    run_model_dir = os.path.join(model_dir, f"seed_{seed}")
    run_log_dir = os.path.join(log_dir, f"seed_{seed}")
    return train_pick_and_lift_multiseed_single_model(
        seeds=[seed],
        total_timesteps=total_timesteps,
        n_envs=n_envs,
        config_path=config_path,
        model_dir=run_model_dir,
        log_dir=run_log_dir,
        load_checkpoint=load_checkpoint,
        load_vecnorm=load_vecnorm,
        output_dir=os.path.join("reports", "pick_lift_per_seed", f"seed_{seed}"),
    )


train_hierarchical = train_pick_lift_seed



# Command Line Interface

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train & Evaluate SO-101 Pick-and-Lift (26-D obs) across seeds into ONE single model"
    )
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Total timesteps across all seeds")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456], help="Seeds to train sequentially")
    parser.add_argument("--random-seeds", action="store_true", help="Randomly generate 3 seeds")
    parser.add_argument("--envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--model-dir", type=str, default="models/so101/pick_lift", help="Directory to save the single model")
    parser.add_argument("--log-dir", type=str, default="logs/so101_pick_lift", help="Directory for TensorBoard logs")
    parser.add_argument("--eval-episodes", type=int, default=200, help="Number of evaluation episodes for the Mean ± Std benchmark")
    parser.add_argument("--output-dir", type=str, default="reports/pick_lift_single_model", help="Directory to save plots and reports")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and run statistical evaluation + plot curves")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .zip to resume from")
    parser.add_argument("--vecnorm", type=str, default=None, help="Path to vecnormalize .pkl")
    args = parser.parse_args()

    chosen_seeds = [int(s) for s in args.seeds]
    if args.random_seeds:
        chosen_seeds = [int(s) for s in np.random.randint(10, 9999, size=3)]

    if args.eval_only:
        final_model = os.path.join(args.model_dir, "pick_lift_model.zip")
        if not os.path.exists(final_model):
            candidates = [
                os.path.join(args.model_dir, "best_model.zip"),
                str(PROJECT_ROOT / "models/so101/hierarchical/seed_123/best_model.zip"),
                str(PROJECT_ROOT / "models/so101/hierarchical/seed_42/best_model.zip"),
            ]
            for c in candidates:
                if os.path.exists(c):
                    final_model = c
                    break

        vecnorm_file = os.path.join(args.model_dir, "vecnormalize.pkl")
        if not os.path.exists(vecnorm_file):
            for vc in [
                str(PROJECT_ROOT / "models/so101/hierarchical/seed_123/vecnormalize.pkl"),
                str(PROJECT_ROOT / "models/so101/hierarchical/seed_42/vecnormalize.pkl"),
            ]:
                if os.path.exists(vc):
                    vecnorm_file = vc
                    break

        eval_stats = evaluate_single_model_statistically(
            model_path=final_model,
            vecnorm_path=vecnorm_file if os.path.exists(vecnorm_file) else None,
            n_episodes=args.eval_episodes,
            output_dir=args.output_dir,
        )
        plot_single_model_learning_curves(
            log_dir=args.log_dir if os.path.exists(args.log_dir) else str(PROJECT_ROOT / "logs/archive_so101_hierarchical"),
            output_path=os.path.join(args.output_dir, "pick_lift_learning_curves.png"),
            eval_stats=eval_stats,
        )
    else:
        train_pick_and_lift_multiseed_single_model(
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