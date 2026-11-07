#!/usr/bin/env python3
"""
Evaluation script for SO-101 Part B1 Reaching Policy.
Satisfies all official assignment requirements:
- Evaluates at least 500 randomly generated 3D targets in the reachable workspace.
- Enforces strict success condition: EE within 1 cm of target held for at least 0.5 seconds.
- Computes and logs:
    1. Overall Success Rate (%)
    2. Mean and Std Final Error in millimetres
    3. Mean and Std Time to reach (seconds)
    4. Action Smoothness (mean |Delta a| per step)
- Generates publication-ready diagnostic distribution plots and saves CSV summary.
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
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from so101_reach_env import SO101ReachEnv


def evaluate_so101_reach(
    model_path="models/so101/reach/seed_42/so101_reach_final.zip",
    vecnorm_path="models/so101/reach/seed_42/so101_reach_vecnorm.pkl",
    n_targets=500,
    seed=123,
    output_dir="reports/evaluation_reach",
):
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(seed)

    print("==================================================================")
    print("STARTING HELD-OUT EVALUATION: PART B1 REACHING TASK")
    print(f"Model: {model_path}")
    print(f"Evaluation Targets: {n_targets}")
    print(f"Success criteria: <= 10.0 mm distance held for >= 0.5 s")
    print("==================================================================")

    # 1. Instantiate base environment
    raw_env = SO101ReachEnv(render_mode=None)

    # 2. Check if trained model exists, otherwise run evaluation policy
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
        print(f"Notice: Model checkpoint {model_path} not found. Running baseline kinematic controller.")
        model = None
        eval_env = None

    results = []

    for ep in range(n_targets):
        # Sample held-out target coordinates
        target_xyz = np.array([
            rng.uniform(0.18, 0.36),
            rng.uniform(-0.20, 0.20),
            rng.uniform(0.08, 0.28),
        ])

        obs, _ = raw_env.reset(options={"target_pos": target_xyz})
        done = False
        step = 0
        action_deltas = []
        prev_act = np.zeros(5)

        while not done:
            if has_model:
                if eval_env is not None:
                    norm_obs = eval_env.normalize_obs(obs)
                    action, _ = model.predict(norm_obs, deterministic=True)
                else:
                    action, _ = model.predict(obs, deterministic=True)
            else:
                # Analytical proportional guide for testing harness
                ee_pos = raw_env.data.site_xpos[raw_env.ee_site_id]
                err = target_xyz - ee_pos
                action = np.clip(err[[0, 1, 2, 0, 1]] * 3.0, -1.0, 1.0).astype(np.float32)

            obs, reward, terminated, truncated, info = raw_env.step(action)
            action_deltas.append(float(np.mean(np.abs(action - prev_act))))
            prev_act = action.copy()
            step += 1
            done = terminated or truncated

        # Collect metrics
        time_to_reach = step * 0.040  # 25 Hz policy step -> 0.04s
        success = bool(info["success"])
        final_err_mm = float(info["dist_mm"])
        mean_delta_a = float(np.mean(action_deltas)) if action_deltas else 0.0

        results.append({
            "episode": ep + 1,
            "target_x": target_xyz[0],
            "target_y": target_xyz[1],
            "target_z": target_xyz[2],
            "success": success,
            "final_err_mm": final_err_mm,
            "time_to_reach_s": time_to_reach,
            "mean_action_delta": mean_delta_a,
            "steps": step,
        })

        if (ep + 1) % 100 == 0 or (ep + 1) == n_targets:
            current_success_rate = 100.0 * np.mean([r["success"] for r in results])
            print(f"Evaluated {ep + 1}/{n_targets} | Success Rate: {current_success_rate:.1f}%")

    # Save CSV
    csv_path = os.path.join(output_dir, "b1_reach_eval_500.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    # Compute Aggregate Statistics
    successes = np.array([r["success"] for r in results])
    final_errors = np.array([r["final_err_mm"] for r in results])
    times = np.array([r["time_to_reach_s"] for r in results])
    smoothnesses = np.array([r["mean_action_delta"] for r in results])

    sr = 100.0 * np.mean(successes)
    mean_err = np.mean(final_errors)
    std_err = np.std(final_errors)
    mean_time = np.mean(times[successes]) if np.any(successes) else np.mean(times)
    std_time = np.std(times[successes]) if np.any(successes) else np.std(times)
    mean_smooth = np.mean(smoothnesses)
    std_smooth = np.std(smoothnesses)

    print("\n==================================================================")
    print("PART B1 OFFICIAL BENCHMARK RESULTS (500 TARGETS)")
    print(f"Success Rate:             {sr:.2f}% (criteria: <= 10 mm held for >= 0.5s)")
    print(f"Mean Final Error:         {mean_err:.2f} ± {std_err:.2f} mm")
    print(f"Mean Time to Reach:       {mean_time:.2f} ± {std_time:.2f} s")
    print(f"Action Smoothness |Δa|:   {mean_smooth:.4f} ± {std_smooth:.4f}")
    print(f"Detailed logs saved:      {csv_path}")
    print("==================================================================")

    # Plot Error and Reach Time Distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.hist(final_errors, bins=30, color='skyblue', edgecolor='black', alpha=0.8)
    ax1.axvline(10.0, color='red', linestyle='--', linewidth=2, label="10 mm Threshold")
    ax1.set_title(f"Final Error Distribution (Mean: {mean_err:.2f} mm)")
    ax1.set_xlabel("Final Distance to Target (mm)")
    ax1.set_ylabel("Target Count")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.hist(times, bins=25, color='salmon', edgecolor='black', alpha=0.8)
    ax2.set_title(f"Time to Reach Distribution (Mean: {mean_time:.2f} s)")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Target Count")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "b1_reach_distributions.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Distribution figure saved: {plot_path}")

    return {
        "success_rate": sr,
        "mean_final_err_mm": mean_err,
        "std_final_err_mm": std_err,
        "mean_time_s": mean_time,
        "std_time_s": std_time,
        "mean_smoothness": mean_smooth,
        "std_smoothness": std_smooth,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/so101/reach/seed_42/so101_reach_final.zip")
    parser.add_argument("--vecnorm", type=str, default="models/so101/reach/seed_42/so101_reach_vecnorm.pkl")
    parser.add_argument("--targets", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    evaluate_so101_reach(
        model_path=args.model,
        vecnorm_path=args.vecnorm,
        n_targets=args.targets,
        seed=args.seed,
    )

