#!/usr/bin/env python3
"""
Official Held-Out Evaluation Suite for SO-101 Pick-and-Lift Task (>= 200 Episodes).

Task Objectives & Acceptance Gate:
1. Evaluates the policy deterministically on unseen episodes with fixed evaluation seeds.
2. Full Domain Randomization active across all held-out scenarios:
   - Cube size randomized by ±20% (0.012 to 0.018 m half-size)
   - Cube mass randomized (25 to 45 g)
   - Surface friction randomized (1.8 to 3.2)
   - Cube table pose & random yaw orientation
3. Strict Success Gate:
   - Cube elevated >= 5 cm above initial table resting height (lift_height >= 0.050 m)
   - Bilateral gripper jaw contact maintained throughout
   - Dynamic stability: linear vel <= 0.08 m/s, angular vel <= 1.0 rad/s, joint vel <= 1.5 rad/s
   - Held continuously for >= 1.0 s (25 consecutive steps @ 25 Hz)
4. Comprehensive Quantitative Reporting (Sample Mean ± Std, ddof=1):
   - Task Success Rate (%)
   - Lift Target Attainment Rate (>= 5 cm) (%)
   - Bilateral Grasp Confirmation Rate (%)
   - Max Lift Height Gain (cm)
   - Max Steady Hold Duration (s)
   - Action Smoothness: mean |Δa| per step
   - Episodic Return
   - Full Decomposed Rewards: Reaching, Grasp, Lift Progress, Lift Milestone, Holding, Penalty, Success
   - Failure Mode Taxonomy
5. Live 3D Simulation Rendering (`--render`):
   - Paced to normal real-time playback speed (25 Hz policy dt = 40 ms)
   - Avoids hypersonic CPU simulation speed for clear visual inspection
   - Configurable speed multiplier via `--speed` (e.g. 1.0 for 1x real-time)
"""

from __future__ import annotations

import os
import sys
import json
import csv
import argparse
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common import configure_env
configure_env()

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environments.pick_and_lift_env import SO101PickLiftEnv


class ObsAdapterWrapper(gym.ObservationWrapper):
    """Adapts observation dimensions (25-D, 26-D, 30-D, 31-D) bidirectionally for cross-checkpoint evaluation."""
    def __init__(self, env, target_dim=26):
        super().__init__(env)
        self.target_dim = target_dim
        base_dim = env.observation_space.shape[0]

        if target_dim == base_dim:
            self.observation_space = env.observation_space
        elif target_dim == 25:
            low = env.observation_space.low[:25]
            high = env.observation_space.high[:25]
            self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        elif target_dim == 26:
            if base_dim == 25:
                low = np.concatenate([env.observation_space.low, [0.0]])
                high = np.concatenate([env.observation_space.high, [1.0]])
            else:
                low = env.observation_space.low[:26]
                high = env.observation_space.high[:26]
            self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        elif target_dim == 30:
            base_low = env.observation_space.low[:25]
            base_high = env.observation_space.high[:25]
            low = np.concatenate([base_low, [-1.0, -1.0], -2.0 * np.ones(3)])
            high = np.concatenate([base_high, [1.0, 1.0], 2.0 * np.ones(3)])
            self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        elif target_dim == 31:
            base_low = env.observation_space.low[:25]
            base_high = env.observation_space.high[:25]
            low = np.concatenate([base_low, -5.0 * np.ones(3), [0.0, 0.0, 0.0]])
            high = np.concatenate([base_high, 5.0 * np.ones(3), [1.0, 1.0, 1.0]])
            self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        else:
            self.observation_space = env.observation_space

    def observation(self, obs):
        base_dim = len(obs)
        if self.target_dim == base_dim:
            return obs
        if self.target_dim == 25:
            return obs[:25].astype(np.float32)
        if self.target_dim == 26:
            if base_dim == 25:
                return np.concatenate([obs, [0.0]]).astype(np.float32)
            return obs[:26].astype(np.float32)
        if self.target_dim == 30:
            try:
                adr = self.env.unwrapped._object_qpos_address()
                quat = self.env.unwrapped.data.qpos[adr + 3 : adr + 7]
                yaw = np.arctan2(2.0 * (quat[0] * quat[3] + quat[1] * quat[2]), 1.0 - 2.0 * (quat[2]**2 + quat[3]**2))
            except Exception:
                yaw = 0.0
            tail = obs[28:31] if base_dim >= 31 else np.zeros(3, dtype=np.float32)
            return np.concatenate([obs[:25], [np.sin(yaw), np.cos(yaw)], tail]).astype(np.float32)
        if self.target_dim == 31:
            try:
                ang_vel = self.env.unwrapped._get_cube_ang_vel_vec()
                bilateral = float(self.env.unwrapped._check_contacts()[2])
                grasped = float(self.env.unwrapped.has_grasped_once)
                hold_steps = getattr(self.env.unwrapped, "consec_lift_hold_steps", 0)
                max_hold = getattr(self.env.unwrapped, "hold_duration_steps", 25)
                hold_prog = float(hold_steps) / float(max_hold)
                extra = np.concatenate([ang_vel, [bilateral, grasped, hold_prog]]).astype(np.float32)
            except Exception:
                extra = np.zeros(6, dtype=np.float32)
            return np.concatenate([obs[:25], extra]).astype(np.float32)
        return obs


def find_default_pick_lift_model() -> tuple[str, str | None]:
    """Auto-detects the best available pick-and-lift checkpoint and normalization stats."""
    candidates = [
        ("models/so101/pick_lift/pick_lift_model.zip", "models/so101/pick_lift/vecnormalize.pkl")
    ]
    for model_rel, vecnorm_rel in candidates:
        m_path = str(PROJECT_ROOT / model_rel) if not os.path.isabs(model_rel) else model_rel
        if os.path.exists(m_path):
            v_path = (str(PROJECT_ROOT / vecnorm_rel) if not os.path.isabs(vecnorm_rel) else vecnorm_rel) if vecnorm_rel else None
            v_path = v_path if (v_path and os.path.exists(v_path)) else None
            return m_path, v_path
    raise FileNotFoundError("No trained pick-and-lift model found in models/so101/.")


def evaluate_pick_and_lift_held_out(
    model_path: str | None = None,
    vecnorm_path: str | None = None,
    n_episodes: int = 200,
    seed: int = 50000,
    render: bool = False,
    playback_speed: float = 1.0,
    output_dir: str = "reports/evaluation_pick_and_lift",
    randomize_cube_place: bool = True,
    cube_x_range: tuple[float, float] = (0.24, 0.30),
    cube_y_range: tuple[float, float] = (-0.09, 0.09),
) -> dict:
    """
    Evaluates pick-and-lift policy deterministically across >= 200 held-out episodes
    with full cube place randomization across table coordinates and orientations.
    """
    if n_episodes < 1:
        raise ValueError("n_episodes must be >= 1.")

    if model_path is not None and not os.path.isabs(model_path) and not os.path.exists(model_path):
        model_path = str(PROJECT_ROOT / model_path)
    if vecnorm_path is not None and not os.path.isabs(vecnorm_path) and not os.path.exists(vecnorm_path):
        vecnorm_path = str(PROJECT_ROOT / vecnorm_path)

    if model_path is None or not os.path.exists(model_path):
        detected_model, detected_vn = find_default_pick_lift_model()
        model_path = model_path or detected_model
        vecnorm_path = vecnorm_path or detected_vn

    if not os.path.isabs(output_dir):
        output_dir = str(PROJECT_ROOT / output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 82)
    print(" SO-101 MANIPULATOR: HELD-OUT PICK-AND-LIFT BENCHMARK (>= 200 EPISODES)")
    print("=" * 82)
    print(f" Model Checkpoint:   {model_path}")
    print(f" VecNormalize Stats: {vecnorm_path or 'None (Unnormalized obs)'}")
    print(f" Total Episodes:     {n_episodes}")
    print(f" Base Scenario Seed: {seed} (Scenarios: {seed} .. {seed + n_episodes - 1})")
    print(f" Cube Place Random:  {'Active [X: ' + str(cube_x_range) + ', Y: ' + str(cube_y_range) + ']' if randomize_cube_place else 'Disabled'}")
    print(f" 3D Rendering:       {'Enabled (Real-time Pacing ' + str(playback_speed) + 'x)' if render else 'Disabled (Maximum CPU Speed)'}")
    print(f" Output Directory:   {output_dir}")
    print("=" * 82 + "\n")

    # Detect observation space dimension expected by model / vecnorm (default: 26-D)
    target_dim = 26
    if vecnorm_path and os.path.exists(vecnorm_path):
        import pickle
        with open(vecnorm_path, "rb") as f:
            vn_data = pickle.load(f)
            target_dim = int(vn_data.observation_space.shape[0])
    elif os.path.exists(model_path):
        try:
            m_temp = PPO.load(model_path, device="cpu")
            target_dim = int(m_temp.observation_space.shape[0])
            del m_temp
        except Exception:
            pass

    render_mode = "human" if render else None

    def _make_env():
        base = SO101PickLiftEnv(
            scene_path=str(PROJECT_ROOT / "so101" / "pick_scene.xml"),
            config_path=str(PROJECT_ROOT / "configs" / "so101_config.yaml"),
            render_mode=render_mode,
        )
        if target_dim != base.observation_space.shape[0]:
            return ObsAdapterWrapper(base, target_dim=target_dim)
        return base

    dummy_env = DummyVecEnv([_make_env])
    if vecnorm_path and os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, dummy_env)
        env.training = False
        env.norm_reward = False
    else:
        env = dummy_env

    raw_env = dummy_env.envs[0].unwrapped
    model = PPO.load(model_path, env=env, device="cpu")

    policy_dt = float(raw_env.policy_dt)  # 20 * 0.002 = 0.040s (25 Hz)
    target_step_dt = policy_dt / max(0.1, playback_speed)

    results = []
    failure_counts = {}

    start_eval_time = time.time()

    for idx in range(n_episodes):
        ep_seed = seed + idx

        obs = env.env_method(
            "reset", seed=ep_seed, options={"evaluation": True, "disable_curriculum": True}
        )[0][0]

        if randomize_cube_place:
            ep_rng = np.random.default_rng(ep_seed)
            cube_x = float(ep_rng.uniform(cube_x_range[0], cube_x_range[1]))
            cube_y = float(ep_rng.uniform(cube_y_range[0], cube_y_range[1]))
            cube_yaw = float(ep_rng.uniform(-np.pi, np.pi))

            raw_env._set_cube_pose(
                [cube_x, cube_y, raw_env.table_top_z + raw_env.cube_half_size],
                cube_yaw,
            )
            if raw_env.target_mocap_id >= 0:
                raw_env.data.mocap_pos[raw_env.target_mocap_id] = [
                    cube_x, cube_y, raw_env.table_top_z + raw_env.cube_half_size + 0.05
                ]
            mujoco.mj_forward(raw_env.model, raw_env.data)
            raw_env.prev_distance = raw_env._distance_to_cube()
            raw_env.prev_clearance = raw_env._cube_clearance()
            obs = raw_env._observation()
            if target_dim != len(obs):
                obs = ObsAdapterWrapper(raw_env, target_dim=target_dim).observation(obs)
        else:
            cube_pos = raw_env._cube_position()
            cube_x = float(cube_pos[0])
            cube_y = float(cube_pos[1])
            cube_yaw = 0.0

        if isinstance(env, VecNormalize):
            obs_tensor = env.normalize_obs(np.array([obs]))
        else:
            obs_tensor = np.array([obs])

        done = False
        ep_ret = 0.0
        step = 0
        max_h = 0.0
        max_hold = 0
        valid_lift_reached = False
        action_deltas = []
        prev_act = np.zeros(6, dtype=np.float32)

        ep_reach = 0.0
        ep_grasp = 0.0
        ep_lift_prog = 0.0
        ep_lift_mile = 0.0
        ep_hold = 0.0
        ep_fail = 0.0
        ep_succ = 0.0

        while not done:
            t_step_start = time.perf_counter()

            action, _ = model.predict(obs_tensor, deterministic=True)
            obs_tensor, reward, dones, infos = env.step(action)
            info = infos[0]
            done = dones[0]
            ep_ret += float(reward[0])

            act_flat = np.asarray(action[0], dtype=np.float32).flatten()
            step_delta = float(np.mean(np.abs(act_flat - prev_act)))
            action_deltas.append(step_delta)
            prev_act = act_flat.copy()
            step += 1

            # Accumulate decomposed reward components
            ep_reach += float(info.get("reward_reaching", 0.0))
            ep_grasp += float(
                info.get("reward_grasp", 0.0)
                + info.get("reward_touch", 0.0)
                + info.get("reward_proximity", 0.0)
                + info.get("reward_grasp_hold", 0.0)
            )
            ep_lift_prog += float(info.get("reward_lift", info.get("reward_lift_progress", 0.0)))
            ep_lift_mile += float(info.get("reward_lift_bonus", info.get("reward_lift_target", 0.0)))
            ep_hold += float(info.get("reward_lift_hold", 0.0))
            ep_fail += float(
                info.get("reward_drop", 0.0)
                + info.get("reward_drop_penalty", 0.0)
                + info.get("reward_step_penalty", 0.0)
                + info.get("reward_lift_overshoot", 0.0)
                + info.get("reward_action_reg", 0.0)
                + info.get("reward_escape", 0.0)
            )
            ep_succ += float(info.get("reward_success", 150.0 if info.get("success", False) else 0.0))

            h = float(info.get("lift_height", 0.0))
            if info.get("bilateral_contact", False) and info.get("has_grasped_once", False):
                if h > max_h:
                    max_h = h
            if info.get("lift_target_achieved", False):
                valid_lift_reached = True

            hold_steps = int(info.get("consec_lift_hold_steps", 0))
            if hold_steps > max_hold:
                max_hold = hold_steps

            if render:
                raw_env.render()
                t_elapsed = time.perf_counter() - t_step_start
                t_sleep = target_step_dt - t_elapsed
                if t_sleep > 0:
                    time.sleep(t_sleep)

        success = bool(info.get("success", False))
        has_grasped = bool(info.get("has_grasped_once", False))
        hold_duration_s = float(max_hold * policy_dt)
        max_lift_height_cm = float(max_h * 100.0)
        mean_delta_a = float(np.mean(action_deltas)) if action_deltas else 0.0

        # Classify failure reason if unsuccessful
        fail_reason = ""
        if not success:
            fail_reason = info.get("failure_reason", "")
            if not fail_reason:
                if not info.get("has_touched_once", False):
                    fail_reason = "timeout_no_touch"
                elif not has_grasped:
                    fail_reason = "timeout_touched_no_grasp"
                elif not valid_lift_reached:
                    fail_reason = "timeout_grasped_not_lifted"
                else:
                    fail_reason = "timeout_lifted_not_held"
            failure_counts[fail_reason] = failure_counts.get(fail_reason, 0) + 1

        rec = {
            "episode": idx + 1,
            "seed": ep_seed,
            "cube_place_x": round(cube_x, 4),
            "cube_place_y": round(cube_y, 4),
            "cube_place_z": round(float(raw_env.table_top_z + raw_env.cube_half_size), 4),
            "cube_place_yaw_deg": round(float(np.degrees(cube_yaw)), 2),
            "success": success,
            "lift_target_achieved": valid_lift_reached,
            "has_grasped": has_grasped,
            "max_lift_height_cm": max_lift_height_cm,
            "hold_duration_s": hold_duration_s,
            "consec_hold_steps": max_hold,
            "action_smoothness": mean_delta_a,
            "steps": step,
            "return": ep_ret,
            "failure_reason": fail_reason,
            "reward_reaching": ep_reach,
            "reward_grasp": ep_grasp,
            "reward_lift_progress": ep_lift_prog,
            "reward_lift_milestone": ep_lift_mile,
            "reward_holding": ep_hold,
            "reward_failure_penalty": ep_fail,
            "reward_success": ep_succ,
            "cube_half_size_mm": float(raw_env.cube_half_size * 1000.0),
        }
        results.append(rec)

        if n_episodes <= 10 or render or (idx + 1) % 25 == 0 or (idx + 1) == n_episodes:
            status_str = "SUCCESS" if success else f"FAIL ({fail_reason})"
            print(
                f" [Ep {idx + 1:3d}/{n_episodes:3d}] Seed: {ep_seed} | "
                f"Cube Place: ({cube_x:.3f}, {cube_y:.3f}) yaw: {np.degrees(cube_yaw):5.1f}° | "
                f"Lift: {max_lift_height_cm:4.1f}cm | Hold: {hold_duration_s:4.2f}s | {status_str}"
            )

    env.close()
    total_eval_wall_time = time.time() - start_eval_time


    # Compute Statistical Metrics (Mean ± Std, ddof=1)

    def calc_stats(data: list[float] | np.ndarray) -> tuple[float, float]:
        arr = np.asarray(data, dtype=np.float64)
        m = float(np.mean(arr))
        s = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        return m, s

    all_succ = [r["success"] for r in results]
    all_lift = [r["lift_target_achieved"] for r in results]
    all_grasp = [r["has_grasped"] for r in results]
    all_heights = [r["max_lift_height_cm"] for r in results]
    all_holds = [r["hold_duration_s"] for r in results]
    all_smooth = [r["action_smoothness"] for r in results]
    all_returns = [r["return"] for r in results]
    all_cube_x = [r["cube_place_x"] for r in results]
    all_cube_y = [r["cube_place_y"] for r in results]
    all_cube_yaw = [r["cube_place_yaw_deg"] for r in results]

    succ_m, succ_s = calc_stats(all_succ)
    lift_m, lift_s = calc_stats(all_lift)
    grasp_m, grasp_s = calc_stats(all_grasp)
    h_m, h_s = calc_stats(all_heights)
    hold_m, hold_s = calc_stats(all_holds)
    smooth_m, smooth_s = calc_stats(all_smooth)
    ret_m, ret_s = calc_stats(all_returns)
    cx_m, cx_s = calc_stats(all_cube_x)
    cy_m, cy_s = calc_stats(all_cube_y)
    cyaw_m, cyaw_s = calc_stats(all_cube_yaw)

    r_reach_m, r_reach_s = calc_stats([r["reward_reaching"] for r in results])
    r_grasp_m, r_grasp_s = calc_stats([r["reward_grasp"] for r in results])
    r_lprog_m, r_lprog_s = calc_stats([r["reward_lift_progress"] for r in results])
    r_lmile_m, r_lmile_s = calc_stats([r["reward_lift_milestone"] for r in results])
    r_hold_m, r_hold_s = calc_stats([r["reward_holding"] for r in results])
    r_fail_m, r_fail_s = calc_stats([r["reward_failure_penalty"] for r in results])
    r_succ_m, r_succ_s = calc_stats([r["reward_success"] for r in results])

    metrics = {
        "benchmark": "SO-101 Pick-and-Lift Held-Out Deterministic Evaluation",
        "n_episodes": n_episodes,
        "base_seed": seed,
        "model_path": str(model_path),
        "vecnorm_path": str(vecnorm_path) if vecnorm_path else None,
        "total_eval_wall_time_s": total_eval_wall_time,
        "cube_placement": {
            "randomized": bool(randomize_cube_place),
            "x": {"mean": cx_m, "std": cx_s, "min": float(np.min(all_cube_x)), "max": float(np.max(all_cube_x))},
            "y": {"mean": cy_m, "std": cy_s, "min": float(np.min(all_cube_y)), "max": float(np.max(all_cube_y))},
            "yaw_deg": {"mean": cyaw_m, "std": cyaw_s, "min": float(np.min(all_cube_yaw)), "max": float(np.max(all_cube_yaw))},
        },
        "metrics": {
            "success_rate_percent": {"mean": succ_m * 100.0, "std": succ_s * 100.0},
            "lift_target_rate_percent": {"mean": lift_m * 100.0, "std": lift_s * 100.0},
            "grasp_rate_percent": {"mean": grasp_m * 100.0, "std": grasp_s * 100.0},
            "max_lift_height_cm": {"mean": h_m, "std": h_s},
            "max_hold_duration_s": {"mean": hold_m, "std": hold_s},
            "action_smoothness": {"mean": smooth_m, "std": smooth_s},
            "episodic_return": {"mean": ret_m, "std": ret_s},
        },
        "decomposed_rewards": {
            "reward_reaching": {"mean": r_reach_m, "std": r_reach_s},
            "reward_grasp": {"mean": r_grasp_m, "std": r_grasp_s},
            "reward_lift_progress": {"mean": r_lprog_m, "std": r_lprog_s},
            "reward_lift_milestone": {"mean": r_lmile_m, "std": r_lmile_s},
            "reward_holding": {"mean": r_hold_m, "std": r_hold_s},
            "reward_failure_penalty": {"mean": r_fail_m, "std": r_fail_s},
            "reward_success": {"mean": r_succ_m, "std": r_succ_s},
        },
        "failure_modes": failure_counts,
    }


    # Terminal Summary Table

    print("\n" + "=" * 82)
    print("      SO-101 PICK-AND-LIFT: OFFICIAL HELD-OUT BENCHMARK REPORT (>= 200 EPS)     ")
    print("=" * 82)
    print(f" {'Metric Description':<38} | {'Sample Mean ± Std (ddof=1)':<36}")
    print("-" * 82)
    print(f" {'Task Success Rate (Lift >= 5cm & Hold 1s)':<38} | {metrics['metrics']['success_rate_percent']['mean']:>6.2f}% ± {metrics['metrics']['success_rate_percent']['std']:<5.2f}%")
    print(f" {'Lift Milestone Attainment (>= 5 cm)':<38} | {metrics['metrics']['lift_target_rate_percent']['mean']:>6.2f}% ± {metrics['metrics']['lift_target_rate_percent']['std']:<5.2f}%")
    print(f" {'Bilateral Grasp Confirmation Rate':<38} | {metrics['metrics']['grasp_rate_percent']['mean']:>6.2f}% ± {metrics['metrics']['grasp_rate_percent']['std']:<5.2f}%")
    print(f" {'Max Lift Height Gain (cm)':<38} | {metrics['metrics']['max_lift_height_cm']['mean']:>6.2f} ± {metrics['metrics']['max_lift_height_cm']['std']:<5.2f} cm")
    print(f" {'Max Steady Hold Duration (s)':<38} | {metrics['metrics']['max_hold_duration_s']['mean']:>6.2f} ± {metrics['metrics']['max_hold_duration_s']['std']:<5.2f} s")
    print(f" {'Action Smoothness (mean |Δa|)':<38} | {metrics['metrics']['action_smoothness']['mean']:>6.4f} ± {metrics['metrics']['action_smoothness']['std']:<5.4f}")
    print(f" {'Mean Episodic Return':<38} | {metrics['metrics']['episodic_return']['mean']:>7.2f} ± {metrics['metrics']['episodic_return']['std']:<6.2f}")
    print("-" * 82)
    print(" CUBE PLACEMENT RANDOMIZATION SUMMARY:")
    print(f"   * Table X Coordinate:          {cx_m:>6.3f} ± {cx_s:<5.3f} m (range: [{np.min(all_cube_x):.3f}, {np.max(all_cube_x):.3f}])")
    print(f"   * Table Y Coordinate:          {cy_m:>6.3f} ± {cy_s:<5.3f} m (range: [{np.min(all_cube_y):.3f}, {np.max(all_cube_y):.3f}])")
    print(f"   * Table Yaw Rotation:          {cyaw_m:>6.1f}° ± {cyaw_s:<5.1f}° (range: [{np.min(all_cube_yaw):.1f}°, {np.max(all_cube_yaw):.1f}°])")
    print("-" * 82)
    print(" DECOMPOSED REWARD COMPONENT CONTRIBUTIONS (Mean ± Std):")
    print(f"   * Reaching Reward:             {metrics['decomposed_rewards']['reward_reaching']['mean']:>6.2f} ± {metrics['decomposed_rewards']['reward_reaching']['std']:<5.2f}")
    print(f"   * Grasp Milestone Reward:      {metrics['decomposed_rewards']['reward_grasp']['mean']:>6.2f} ± {metrics['decomposed_rewards']['reward_grasp']['std']:<5.2f}")
    print(f"   * Dense Lift Progress Reward:  {metrics['decomposed_rewards']['reward_lift_progress']['mean']:>6.2f} ± {metrics['decomposed_rewards']['reward_lift_progress']['std']:<5.2f}")
    print(f"   * Lift Milestone Reward:       {metrics['decomposed_rewards']['reward_lift_milestone']['mean']:>6.2f} ± {metrics['decomposed_rewards']['reward_lift_milestone']['std']:<5.2f}")
    print(f"   * Steady Holding Reward:       {metrics['decomposed_rewards']['reward_holding']['mean']:>6.2f} ± {metrics['decomposed_rewards']['reward_holding']['std']:<5.2f}")
    print(f"   * Failure & Step Penalty:      {metrics['decomposed_rewards']['reward_failure_penalty']['mean']:>6.2f} ± {metrics['decomposed_rewards']['reward_failure_penalty']['std']:<5.2f}")
    print(f"   * Terminal Success Bonus:      {metrics['decomposed_rewards']['reward_success']['mean']:>6.2f} ± {metrics['decomposed_rewards']['reward_success']['std']:<5.2f}")
    print("-" * 82)
    print(" FAILURE MODE ANALYSIS:")
    if failure_counts:
        for reason, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
            pct = 100.0 * count / n_episodes
            print(f"   * {reason:<32}: {count:4d} episodes ({pct:5.1f}%)")
    else:
        print("   * None! 100% of evaluation episodes succeeded.")
    print("=" * 82 + "\n")


    # Save CSV and JSON Artifacts

    # 1. Per-episode Detailed CSV
    detailed_csv_path = os.path.join(output_dir, f"pick_and_lift_eval_{n_episodes}_episodes.csv")
    with open(detailed_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    # 2. Summary CSV
    summary_csv_path = os.path.join(output_dir, "pick_and_lift_eval_summary.csv")
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Category", "Metric", "Mean", "Std", "Unit"])
        writer.writerow(["Task", "Success Rate", metrics["metrics"]["success_rate_percent"]["mean"], metrics["metrics"]["success_rate_percent"]["std"], "%"])
        writer.writerow(["Task", "Lift Target Rate", metrics["metrics"]["lift_target_rate_percent"]["mean"], metrics["metrics"]["lift_target_rate_percent"]["std"], "%"])
        writer.writerow(["Task", "Grasp Rate", metrics["metrics"]["grasp_rate_percent"]["mean"], metrics["metrics"]["grasp_rate_percent"]["std"], "%"])
        writer.writerow(["Task", "Max Lift Height", metrics["metrics"]["max_lift_height_cm"]["mean"], metrics["metrics"]["max_lift_height_cm"]["std"], "cm"])
        writer.writerow(["Task", "Hold Duration", metrics["metrics"]["max_hold_duration_s"]["mean"], metrics["metrics"]["max_hold_duration_s"]["std"], "s"])
        writer.writerow(["Task", "Action Smoothness", metrics["metrics"]["action_smoothness"]["mean"], metrics["metrics"]["action_smoothness"]["std"], "delta"])
        writer.writerow(["Task", "Episodic Return", metrics["metrics"]["episodic_return"]["mean"], metrics["metrics"]["episodic_return"]["std"], "reward"])
        writer.writerow(["Cube Placement", "X Position", metrics["cube_placement"]["x"]["mean"], metrics["cube_placement"]["x"]["std"], "m"])
        writer.writerow(["Cube Placement", "Y Position", metrics["cube_placement"]["y"]["mean"], metrics["cube_placement"]["y"]["std"], "m"])
        writer.writerow(["Cube Placement", "Yaw Angle", metrics["cube_placement"]["yaw_deg"]["mean"], metrics["cube_placement"]["yaw_deg"]["std"], "deg"])
        writer.writerow(["Reward", "Reaching", metrics["decomposed_rewards"]["reward_reaching"]["mean"], metrics["decomposed_rewards"]["reward_reaching"]["std"], "reward"])
        writer.writerow(["Reward", "Grasp", metrics["decomposed_rewards"]["reward_grasp"]["mean"], metrics["decomposed_rewards"]["reward_grasp"]["std"], "reward"])
        writer.writerow(["Reward", "Lift Progress", metrics["decomposed_rewards"]["reward_lift_progress"]["mean"], metrics["decomposed_rewards"]["reward_lift_progress"]["std"], "reward"])
        writer.writerow(["Reward", "Lift Milestone", metrics["decomposed_rewards"]["reward_lift_milestone"]["mean"], metrics["decomposed_rewards"]["reward_lift_milestone"]["std"], "reward"])
        writer.writerow(["Reward", "Holding", metrics["decomposed_rewards"]["reward_holding"]["mean"], metrics["decomposed_rewards"]["reward_holding"]["std"], "reward"])
        writer.writerow(["Reward", "Failure Penalty", metrics["decomposed_rewards"]["reward_failure_penalty"]["mean"], metrics["decomposed_rewards"]["reward_failure_penalty"]["std"], "reward"])
        writer.writerow(["Reward", "Success Bonus", metrics["decomposed_rewards"]["reward_success"]["mean"], metrics["decomposed_rewards"]["reward_success"]["std"], "reward"])

    # 3. JSON Export
    json_path = os.path.join(output_dir, "pick_and_lift_eval_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


    # Publication-Grade 4-Panel Distribution Figures

    fig_path = os.path.join(output_dir, "pick_and_lift_eval_distributions.png")
    plot_pick_lift_distributions(results, metrics, fig_path)

    print(f"[Saved] Detailed per-episode CSV:  {detailed_csv_path}")
    print(f"[Saved] Summary evaluation CSV:    {summary_csv_path}")
    print(f"[Saved] JSON evaluation metrics:   {json_path}")
    print(f"[Saved] Distribution figures plot: {fig_path}")

    return metrics


def plot_pick_lift_distributions(results: list[dict], metrics: dict, output_path: str):
    """
    Plots a 4-panel publication-grade evaluation figure:
    1. Max Lift Height Gain (cm) distribution with 5.0 cm milestone line.
    2. Consecutive Hold Duration (s) distribution with 1.0 s threshold line.
    3. Decomposed Reward Component contributions bar chart with error bars.
    4. Progression Funnel: Touch -> Grasp -> Lift >= 5cm -> Successful 1.0s Hold.
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)

    # 1. Lift Height Gain Distribution
    ax = axes[0, 0]
    heights = [r["max_lift_height_cm"] for r in results]
    ax.hist(heights, bins=30, color="#1f77b4", alpha=0.75, edgecolor="black", density=True, label="Lift Height (cm)")
    ax.axvline(5.0, color="#d62728", linestyle="--", linewidth=2.0, label="5.0 cm Milestone Gate")
    ax.axvline(metrics["metrics"]["max_lift_height_cm"]["mean"], color="#2ca02c", linestyle="-", linewidth=2.0,
               label=f"Mean: {metrics['metrics']['max_lift_height_cm']['mean']:.2f} cm")
    ax.set_title("Max Lift Height Gain Distribution (cm)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Elevation Gain above Resting Height (cm)")
    ax.set_ylabel("Probability Density")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.6)

    # 2. Hold Duration Distribution
    ax = axes[0, 1]
    holds = [r["hold_duration_s"] for r in results]
    ax.hist(holds, bins=25, color="#2ca02c", alpha=0.75, edgecolor="black", density=True, label="Hold Duration (s)")
    ax.axvline(1.0, color="#d62728", linestyle="--", linewidth=2.0, label="1.0 s Success Threshold")
    ax.axvline(metrics["metrics"]["max_hold_duration_s"]["mean"], color="#1f77b4", linestyle="-", linewidth=2.0,
               label=f"Mean: {metrics['metrics']['max_hold_duration_s']['mean']:.2f} s")
    ax.set_title("Consecutive Steady Hold Duration Distribution (s)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Stillness Hold Duration at >= 5 cm (s)")
    ax.set_ylabel("Probability Density")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.6)

    # 3. Decomposed Rewards Bar Chart
    ax = axes[1, 0]
    rew_keys = ["reward_reaching", "reward_grasp", "reward_lift_progress", "reward_lift_milestone", "reward_holding", "reward_failure_penalty", "reward_success"]
    labels = ["Reach", "Grasp", "LiftProg", "LiftMile", "Hold", "FailPen", "Success"]
    means = [metrics["decomposed_rewards"][k]["mean"] for k in rew_keys]
    stds = [metrics["decomposed_rewards"][k]["std"] for k in rew_keys]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#d62728", "#17becf"]
    ax.bar(labels, means, yerr=stds, capsize=4, color=colors, alpha=0.85, edgecolor="black")
    ax.set_title("Decomposed Reward Components (Mean ± Std)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Reward Return")
    ax.grid(True, linestyle="--", alpha=0.6)

    # 4. Milestone Conversion Funnel
    ax = axes[1, 1]
    total_n = len(results)
    n_touched = sum(1 for r in results if r["reward_reaching"] > 0.0 or r["has_grasped"])
    n_grasped = sum(1 for r in results if r["has_grasped"])
    n_lifted = sum(1 for r in results if r["lift_target_achieved"])
    n_success = sum(1 for r in results if r["success"])

    funnel_labels = ["Approach / Touch", "Bilateral Grasp", "Lift >= 5 cm", "1.0s Steady Hold (Success)"]
    funnel_rates = [
        100.0 * n_touched / max(1, total_n),
        100.0 * n_grasped / max(1, total_n),
        100.0 * n_lifted / max(1, total_n),
        100.0 * n_success / max(1, total_n),
    ]
    f_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    bars = ax.barh(funnel_labels, funnel_rates, color=f_colors, alpha=0.85, edgecolor="black")
    ax.invert_yaxis()
    for bar, val in zip(bars, funnel_rates):
        ax.text(val + 1.5, bar.get_y() + bar.get_height() / 2.0, f"{val:.1f}%", va="center", fontweight="bold")
    ax.set_title("Pick-and-Lift Task Milestone Conversion Funnel", fontsize=12, fontweight="bold")
    ax.set_xlabel("Conversion Rate (%)")
    ax.set_xlim(0, 115)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.suptitle(
        f"SO-101 Pick-and-Lift: Held-Out Benchmark ({metrics['n_episodes']} Episodes, Success: {metrics['metrics']['success_rate_percent']['mean']:.1f}%)",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()



# Command Line Interface

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Official Held-Out Evaluation Suite for SO-101 Pick-and-Lift (>= 200 Episodes)"
    )
    parser.add_argument(
        "--n-episodes", "--episodes",
        type=int,
        default=200,
        dest="n_episodes",
        help="Number of randomized held-out episodes to evaluate (default: 200)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained PPO pick-and-lift model .zip (auto-detects if omitted)",
    )
    parser.add_argument(
        "--vecnorm",
        type=str,
        default=None,
        help="Path to VecNormalize .pkl file (auto-detects if omitted)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=50000,
        help="Base evaluation scenario seed (default: 50000)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render 3D interactive simulation live in MuJoCo viewer at normal 1x real-time speed",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier when --render is active (default: 1.0 for real-time)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/evaluation_pick_and_lift",
        help="Directory to save CSV logs, JSON metrics, and figures",
    )
    parser.add_argument(
        "--randomize-cube-place",
        action="store_true",
        default=True,
        help="Randomize cube placement (X, Y, Yaw) across the table on every episode (default: True)",
    )
    parser.add_argument(
        "--no-randomize-cube-place",
        action="store_false",
        dest="randomize_cube_place",
        help="Disable cube placement randomization during evaluation",
    )
    parser.add_argument(
        "--cube-x-min",
        type=float,
        default=0.24,
        help="Minimum table X spawn coordinate for cube placement (default: 0.24)",
    )
    parser.add_argument(
        "--cube-x-max",
        type=float,
        default=0.30,
        help="Maximum table X spawn coordinate for cube placement (default: 0.30)",
    )
    parser.add_argument(
        "--cube-y-min",
        type=float,
        default=-0.09,
        help="Minimum table Y spawn coordinate for cube placement (default: -0.09)",
    )
    parser.add_argument(
        "--cube-y-max",
        type=float,
        default=0.09,
        help="Maximum table Y spawn coordinate for cube placement (default: 0.09)",
    )
    args = parser.parse_args()

    evaluate_pick_and_lift_held_out(
        model_path=args.model,
        vecnorm_path=args.vecnorm,
        n_episodes=args.n_episodes,
        seed=args.seed,
        render=args.render,
        playback_speed=args.speed,
        output_dir=args.output_dir,
        randomize_cube_place=args.randomize_cube_place,
        cube_x_range=(args.cube_x_min, args.cube_x_max),
        cube_y_range=(args.cube_y_min, args.cube_y_max),
    )

