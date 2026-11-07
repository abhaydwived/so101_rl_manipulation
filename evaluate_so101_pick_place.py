#!/usr/bin/env python3
"""
Evaluation script for SO-101 Part B2 Pick-and-Place Policy.
Satisfies Part B2 and Part C mandatory requirements:
- Evaluates at least 200 held-out randomized episodes.
- Tests randomized cube locations and randomized target zones.
- Enforces strict success condition:
    1. Cube center within 2 cm (0.02 m) of target zone.
    2. Cube successfully released.
    3. Arm clear of the placed cube (> 6 cm away).
- Reports:
    - Overall Success Rate (%)
    - Grasp Rate (%)
    - Release Rate (%)
    - Clearance Rate (%)
    - Mean and Std Final Placement Error (mm)
    - Mean and Std Episode Duration (seconds)
    - Action Smoothness (mean |Delta a| per step)
"""

import os
import sys

# Prevent Windows EPERM permission conflict with corrupted .keras/keras.json
sys.modules["tensorflow"] = None

import csv
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "environments"))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from environments.pick_place_env import SO101PickPlaceEnv


def evaluate_so101_pick_place(
    model_path="models/so101/pick_place/seed_42/so101_pick_place_final.zip",
    vecnorm_path="models/so101/pick_place/seed_42/so101_pick_place_vecnorm.pkl",
    n_episodes=200,
    seed=789,
    output_dir="reports/evaluation_pick_place",
):
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(seed)

    print("==================================================================")
    print("STARTING HELD-OUT EVALUATION: PART B2 PICK AND PLACE TASK")
    print(f"Model: {model_path}")
    print(f"Held-Out Episodes: {n_episodes}")
    print(f"Success criteria: Cube <= 20.0 mm of target, released, and arm clear")
    print("==================================================================")

    raw_env = SO101PickPlaceEnv(render_mode=None)

    has_model = os.path.exists(model_path)
    if has_model:
        model = PPO.load(model_path, device="cpu")
        if os.path.exists(vecnorm_path):
            eval_env = DummyVecEnv([lambda: raw_env])
            eval_env = VecNormalize.load(vecnorm_path, eval_env)
            eval_env.training = False
            eval_env.norm_reward = False
        else:
            eval_env = None
    else:
        print(f"Notice: Model checkpoint {model_path} not found. Running baseline policy.")
        model = None
        eval_env = None

    results = []

    for ep in range(n_episodes):
        # Sample held-out randomized cube and target positions
        cube_xy = np.array([rng.uniform(0.20, 0.32), rng.uniform(0.04, 0.16)])
        target_xy = np.array([rng.uniform(0.20, 0.32), rng.uniform(-0.16, -0.04)])

        obs, _ = raw_env.reset(options={"cube_xy": cube_xy, "target_xy": target_xy})
        done = False
        step = 0
        action_deltas = []
        prev_act = np.zeros(6)

        while not done:
            if has_model:
                if eval_env is not None:
                    norm_obs = eval_env.normalize_obs(obs)
                    action, _ = model.predict(norm_obs, deterministic=True)
                else:
                    action, _ = model.predict(obs, deterministic=True)
            else:
                action = raw_env.action_space.sample()

            obs, reward, terminated, truncated, info = raw_env.step(action)
            action_deltas.append(float(np.mean(np.abs(action - prev_act))))
            prev_act = action.copy()
            step += 1
            done = terminated or truncated

        success = bool(info["success"])
        grasped = bool(info["is_grasped"])
        released = bool(info["is_released"])
        cleared = bool(info["is_clear"])
        err_mm = float(info["cube_target_err_mm"])
        time_s = step * 0.040
        mean_delta_a = float(np.mean(action_deltas)) if action_deltas else 0.0

        results.append({
            "episode": ep + 1,
            "cube_x": cube_xy[0],
            "cube_y": cube_xy[1],
            "target_x": target_xy[0],
            "target_y": target_xy[1],
            "success": success,
            "grasped": grasped,
            "released": released,
            "cleared": cleared,
            "final_err_mm": err_mm,
            "duration_s": time_s,
            "mean_action_delta": mean_delta_a,
            "steps": step,
        })

        if (ep + 1) % 50 == 0 or (ep + 1) == n_episodes:
            current_sr = 100.0 * np.mean([r["success"] for r in results])
            print(f"Evaluated {ep + 1}/{n_episodes} | Pick-and-Place Success Rate: {current_sr:.1f}%")

    csv_path = os.path.join(output_dir, "b2_pick_place_eval_200.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    successes = np.array([r["success"] for r in results])
    grasps = np.array([r["grasped"] for r in results])
    releases = np.array([r["released"] for r in results])
    clears = np.array([r["cleared"] for r in results])
    errors = np.array([r["final_err_mm"] for r in results])
    durations = np.array([r["duration_s"] for r in results])
    smoothnesses = np.array([r["mean_action_delta"] for r in results])

    sr = 100.0 * np.mean(successes)
    gr = 100.0 * np.mean(grasps)
    rr = 100.0 * np.mean(releases)
    cr = 100.0 * np.mean(clears)
    mean_err = np.mean(errors)
    std_err = np.std(errors)
    mean_dur = np.mean(durations[successes]) if np.any(successes) else np.mean(durations)
    std_dur = np.std(durations[successes]) if np.any(successes) else np.std(durations)
    mean_smooth = np.mean(smoothnesses)

    print("\n==================================================================")
    print("PART B2 OFFICIAL BENCHMARK RESULTS (200 HELD-OUT EPISODES)")
    print(f"Overall Task Success Rate:  {sr:.2f}% (criteria: <= 20 mm, released, clear)")
    print(f"Grasp Completion Rate:      {gr:.2f}%")
    print(f"Release Completion Rate:    {rr:.2f}%")
    print(f"Arm Clearance Rate:         {cr:.2f}%")
    print(f"Mean Final Placement Error: {mean_err:.2f} ± {std_err:.2f} mm")
    print(f"Mean Completion Duration:   {mean_dur:.2f} ± {std_dur:.2f} s")
    print(f"Action Smoothness |Δa|:     {mean_smooth:.4f}")
    print(f"Detailed logs saved:        {csv_path}")
    print("==================================================================")

    # Plot placement accuracy histogram
    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=25, color='teal', edgecolor='black', alpha=0.8)
    plt.axvline(20.0, color='red', linestyle='--', linewidth=2, label="20 mm Threshold")
    plt.title(f"Part B2 Placement Error Distribution (Mean: {mean_err:.2f} mm)")
    plt.xlabel("Distance from Cube Center to Target Zone Center (mm)")
    plt.ylabel("Episode Count")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "b2_placement_error_distribution.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Figure saved: {plot_path}")

    return {
        "success_rate": sr,
        "grasp_rate": gr,
        "release_rate": rr,
        "clearance_rate": cr,
        "mean_final_err_mm": mean_err,
        "std_final_err_mm": std_err,
        "mean_duration_s": mean_dur,
        "std_duration_s": std_dur,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/so101/pick_place/seed_42/so101_pick_place_final.zip")
    parser.add_argument("--vecnorm", type=str, default="models/so101/pick_place/seed_42/so101_pick_place_vecnorm.pkl")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=789)
    args = parser.parse_args()

    evaluate_so101_pick_place(
        model_path=args.model,
        vecnorm_path=args.vecnorm,
        n_episodes=args.episodes,
        seed=args.seed,
    )

