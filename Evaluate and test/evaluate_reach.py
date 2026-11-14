#!/usr/bin/env python3
"""
Dedicated Held-Out Evaluation Suite for SO-101 Precision Reaching Task.

Benchmark Specifications:
1. Evaluates over >= 500 randomized target positions across 3 height zones:
   - Bottom Zone: Z in [0.015, 0.065] m (near ground/table plane)
   - Medium Zone: Z in [0.075, 0.180] m (mid-altitude reach)
   - High Zone:   Z in [0.190, 0.320] m (overhead reach)
2. Precision & Stability Gate:
   - Euclidean error to target < 10 mm (0.010 m)
   - Held stably for >= 0.5 s (13 consecutive policy steps at 25 Hz)
3. Quantitative Reports (Sample Mean ± Std, ddof=1):
   - Overall & Per-Zone Success Rate (%)
   - Final Euclidean Distance Error (mm)
   - Time-to-Reach & Settle (s)
   - Action Smoothness: mean |Δa| per step across joints
   - Episodic Return
4. Live 3D Simulation Rendering (`--render`):
   - Paced to normal real-time playback speed (25 Hz policy dt = 40 ms)
   - Avoids hypersonic CPU simulation speed for clear visual inspection
   - Configurable speed multiplier via `--speed` (e.g. 1.0 for 1x real-time)
5. Comprehensive Artifact Generation:
   - Detailed per-target CSV log
   - Summary statistics CSV & JSON
   - Publication-quality distribution plot (Error & Reach Time across zones)
6. Commands:
   Full 500-Target Benchmark:
   - python evaluate_reach.py --n-targets 500
   - python evaluate_reach.py --n-targets 500 --seed 123
   Simulation Visualization:
   - python evaluate_reach.py --render  
   - python evaluate_reach.py --n-targets 10 --render --speed 1.5
   Evaluating a Specific Model Checkpoint:
   - python evaluate_reach.py --model models/so101/reach/reach_model.zip --vecnorm models/so101/reach/vecnormalize.pkl --n-targets 500
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

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environments.reaching_env import SO101ReachingEnv


def find_default_reach_model() -> tuple[str, str | None]:
    """Auto-detects the best trained reaching policy and normalization stats."""
    candidates = [
        ("models/so101/reach/reach_model.zip", "models/so101/reach/vecnormalize.pkl")
    ]
    for model_rel, vecnorm_rel in candidates:
        m_path = str(PROJECT_ROOT / model_rel) if not os.path.isabs(model_rel) else model_rel
        if os.path.exists(m_path):
            v_path = (str(PROJECT_ROOT / vecnorm_rel) if not os.path.isabs(vecnorm_rel) else vecnorm_rel) if vecnorm_rel else None
            v_path = v_path if (v_path and os.path.exists(v_path)) else None
            return m_path, v_path
    raise FileNotFoundError("No trained reaching model found in models/so101/reach/.")


def evaluate_reach_held_out(
    model_path: str | None = None,
    vecnorm_path: str | None = None,
    n_targets: int = 500,
    seed: int = 2026,
    render: bool = False,
    playback_speed: float = 1.0,
    output_dir: str = "reports/evaluation_reach",
) -> dict:
    """
    Evaluates reaching policy across >= 500 held-out randomized target scenarios.
    """
    if n_targets < 1:
        raise ValueError("n_targets must be >= 1.")

    if model_path is not None and not os.path.isabs(model_path) and not os.path.exists(model_path):
        model_path = str(PROJECT_ROOT / model_path)
    if vecnorm_path is not None and not os.path.isabs(vecnorm_path) and not os.path.exists(vecnorm_path):
        vecnorm_path = str(PROJECT_ROOT / vecnorm_path)

    if model_path is None or not os.path.exists(model_path):
        detected_model, detected_vn = find_default_reach_model()
        model_path = model_path or detected_model
        vecnorm_path = vecnorm_path or detected_vn

    if not os.path.isabs(output_dir):
        output_dir = str(PROJECT_ROOT / output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 82)
    print(" SO-101 MANIPULATOR: HELD-OUT PRECISION REACHING BENCHMARK (>= 500 TARGETS)")
    print("=" * 82)
    print(f" Model Checkpoint:   {model_path}")
    print(f" VecNormalize Stats: {vecnorm_path or 'None (Unnormalized obs)'}")
    print(f" Total Targets:      {n_targets}")
    print(f" Base Scenario Seed: {seed}")
    print(f" 3D Rendering:       {'Enabled (Real-time Pacing ' + str(playback_speed) + 'x)' if render else 'Disabled (Maximum CPU Speed)'}")
    print(f" Output Directory:   {output_dir}")
    print("=" * 82 + "\n")

    # Construct single environment with optional human render mode
    render_mode = "human" if render else None

    def _make_env():
        return SO101ReachingEnv(
            scene_path=str(PROJECT_ROOT / "so101" / "reach_scene.xml"),
            config_path=str(PROJECT_ROOT / "configs" / "so101_config.yaml"),
            render_mode=render_mode,
            max_episode_steps=350,
            control_substeps=20,
            success_dist=0.010,
            hold_time_sec=0.50,
            randomize_init_pose=True,
        )

    dummy_env = DummyVecEnv([_make_env])
    if vecnorm_path and os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, dummy_env)
        env.training = False
        env.norm_reward = False
    else:
        env = dummy_env

    raw_env: SO101ReachingEnv = env.envs[0]
    model = PPO.load(model_path, env=env, device="cpu")

    policy_dt = float(raw_env.control_substeps * raw_env.model.opt.timestep)  # 20 * 0.002s = 0.040s (25 Hz)
    target_step_dt = policy_dt / max(0.1, playback_speed)

    results = []
    zone_names = ["bottom", "medium", "high"]

    start_eval_time = time.time()

    for idx in range(n_targets):
        # Evenly cycle across Bottom, Medium, High zones for comprehensive stratified coverage
        zone_name = zone_names[idx % len(zone_names)]
        ep_seed = seed + idx

        raw_env.current_zone = zone_name
        obs = env.env_method("reset", seed=ep_seed)[0][0]

        if isinstance(env, VecNormalize):
            obs_tensor = env.normalize_obs(np.array([obs]))
        else:
            obs_tensor = np.array([obs])

        done = False
        step = 0
        ep_ret = 0.0
        action_deltas = []
        prev_act = np.zeros(5, dtype=np.float32)

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

            if render:
                raw_env.render()
                # Pacing to normal real-time speed
                t_elapsed = time.perf_counter() - t_step_start
                t_sleep = target_step_dt - t_elapsed
                if t_sleep > 0:
                    time.sleep(t_sleep)

        success = bool(info.get("success", False))
        final_err_mm = float(info.get("dist_mm", 0.0))
        time_to_reach_s = float(step * policy_dt)
        mean_delta_a = float(np.mean(action_deltas)) if action_deltas else 0.0
        actual_zone = str(info.get("target_zone", zone_name))
        target_xyz = raw_env.target_pos.tolist()
        final_ee_xyz = raw_env.get_ee_xpos().tolist()

        res_record = {
            "target_idx": idx + 1,
            "seed": ep_seed,
            "target_zone": actual_zone,
            "success": success,
            "final_err_mm": final_err_mm,
            "time_to_reach_s": time_to_reach_s,
            "action_smoothness": mean_delta_a,
            "steps": step,
            "return": ep_ret,
            "target_x": target_xyz[0],
            "target_y": target_xyz[1],
            "target_z": target_xyz[2],
            "final_ee_x": final_ee_xyz[0],
            "final_ee_y": final_ee_xyz[1],
            "final_ee_z": final_ee_xyz[2],
        }
        results.append(res_record)

        if (idx + 1) % 50 == 0 or (idx + 1) == n_targets:
            curr_succs = [r["success"] for r in results]
            curr_sr = 100.0 * np.mean(curr_succs)
            curr_err = np.mean([r["final_err_mm"] for r in results])
            curr_time = np.mean([r["time_to_reach_s"] for r in results])
            print(
                f" Progress: [{idx + 1:4d}/{n_targets:4d}] targets | "
                f"Running Success: {curr_sr:5.1f}% | "
                f"Mean Err: {curr_err:4.2f} mm | "
                f"Mean Time: {curr_time:4.2f} s"
            )

    env.close()
    total_eval_wall_time = time.time() - start_eval_time


    # Compute Rigorous Statistical Metrics (Mean ± Std, ddof=1)

    def calc_stats(data: list[float] | np.ndarray) -> tuple[float, float]:
        arr = np.asarray(data, dtype=np.float64)
        m = float(np.mean(arr))
        s = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        return m, s

    all_successes = np.array([r["success"] for r in results], dtype=bool)
    all_errors = np.array([r["final_err_mm"] for r in results], dtype=np.float64)
    all_times = np.array([r["time_to_reach_s"] for r in results], dtype=np.float64)
    all_smoothness = np.array([r["action_smoothness"] for r in results], dtype=np.float64)
    all_returns = np.array([r["return"] for r in results], dtype=np.float64)

    # Time-to-reach is conventionally evaluated on successful acquisitions
    eval_times = all_times[all_successes] if np.any(all_successes) else all_times

    succ_mean, succ_std = calc_stats(all_successes)
    err_mean, err_std = calc_stats(all_errors)
    time_mean, time_std = calc_stats(eval_times)
    smooth_mean, smooth_std = calc_stats(all_smoothness)
    ret_mean, ret_std = calc_stats(all_returns)

    # Zone-stratified performance
    zone_stats = {}
    for z in ["bottom", "medium", "high"]:
        z_recs = [r for r in results if r["target_zone"] == z]
        if z_recs:
            z_succ = np.array([r["success"] for r in z_recs], dtype=bool)
            z_errs = np.array([r["final_err_mm"] for r in z_recs], dtype=np.float64)
            z_times = np.array([r["time_to_reach_s"] for r in z_recs if r["success"]], dtype=np.float64)
            if len(z_times) == 0:
                z_times = np.array([r["time_to_reach_s"] for r in z_recs], dtype=np.float64)
            z_smooth = np.array([r["action_smoothness"] for r in z_recs], dtype=np.float64)

            zs_m, zs_s = calc_stats(z_succ)
            ze_m, ze_s = calc_stats(z_errs)
            zt_m, zt_s = calc_stats(z_times)
            zsm_m, zsm_s = calc_stats(z_smooth)

            zone_stats[z] = {
                "n_targets": len(z_recs),
                "success_rate_percent": {"mean": zs_m * 100.0, "std": zs_s * 100.0},
                "final_err_mm": {"mean": ze_m, "std": ze_s},
                "time_to_reach_s": {"mean": zt_m, "std": zt_s},
                "action_smoothness": {"mean": zsm_m, "std": zsm_s},
            }

    overall_metrics = {
        "benchmark": "SO-101 Precision Reaching Held-Out Evaluation",
        "n_targets": n_targets,
        "base_seed": seed,
        "model_path": str(model_path),
        "vecnorm_path": str(vecnorm_path) if vecnorm_path else None,
        "total_eval_wall_time_s": total_eval_wall_time,
        "overall": {
            "success_rate_percent": {"mean": succ_mean * 100.0, "std": succ_std * 100.0},
            "final_err_mm": {"mean": err_mean, "std": err_std},
            "time_to_reach_s": {"mean": time_mean, "std": time_std},
            "action_smoothness": {"mean": smooth_mean, "std": smooth_std},
            "episodic_return": {"mean": ret_mean, "std": ret_std},
        },
        "zones": zone_stats,
    }


    # Terminal Summary Table

    print("\n" + "=" * 82)
    print("       SO-101 PRECISION REACHING: OFFICIAL HELD-OUT BENCHMARK REPORT      ")
    print("=" * 82)
    print(f" {'Metric Description':<38} | {'Sample Mean ± Std (ddof=1)':<36}")
    print("-" * 82)
    print(f" {'Overall Task Success Rate':<38} | {overall_metrics['overall']['success_rate_percent']['mean']:>6.2f}% ± {overall_metrics['overall']['success_rate_percent']['std']:<5.2f}%")
    print(f" {'Mean Final Error (mm)':<38} | {overall_metrics['overall']['final_err_mm']['mean']:>6.2f} ± {overall_metrics['overall']['final_err_mm']['std']:<5.2f} mm")
    print(f" {'Time-to-Reach & Hold (s)':<38} | {overall_metrics['overall']['time_to_reach_s']['mean']:>6.2f} ± {overall_metrics['overall']['time_to_reach_s']['std']:<5.2f} s")
    print(f" {'Action Smoothness (mean |Δa|)':<38} | {overall_metrics['overall']['action_smoothness']['mean']:>6.4f} ± {overall_metrics['overall']['action_smoothness']['std']:<5.4f}")
    print(f" {'Mean Episodic Return':<38} | {overall_metrics['overall']['episodic_return']['mean']:>7.2f} ± {overall_metrics['overall']['episodic_return']['std']:<6.2f}")
    print("-" * 82)
    print(" STRATIFIED 3-ZONE BREAKDOWN:")
    for z in ["bottom", "medium", "high"]:
        if z in zone_stats:
            zs = zone_stats[z]
            print(
                f"   * {z.capitalize():<6} Zone ({zs['n_targets']} targets): "
                f"Success: {zs['success_rate_percent']['mean']:>5.1f}% ± {zs['success_rate_percent']['std']:<4.1f}% | "
                f"Err: {zs['final_err_mm']['mean']:4.2f} ± {zs['final_err_mm']['std']:4.2f} mm | "
                f"Time: {zs['time_to_reach_s']['mean']:4.2f} ± {zs['time_to_reach_s']['std']:4.2f} s | "
                f"Smooth: {zs['action_smoothness']['mean']:.4f}"
            )
    print("=" * 82 + "\n")


    # Save CSV and JSON Artifacts

    # 1. Per-target Detailed CSV
    detailed_csv_path = os.path.join(output_dir, f"reach_eval_{n_targets}_targets.csv")
    with open(detailed_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    # 2. Summary CSV
    summary_csv_path = os.path.join(output_dir, "reach_eval_summary.csv")
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Category", "Metric", "Mean", "Std", "Unit"])
        writer.writerow(["Overall", "Success Rate", overall_metrics["overall"]["success_rate_percent"]["mean"], overall_metrics["overall"]["success_rate_percent"]["std"], "%"])
        writer.writerow(["Overall", "Mean Final Error", overall_metrics["overall"]["final_err_mm"]["mean"], overall_metrics["overall"]["final_err_mm"]["std"], "mm"])
        writer.writerow(["Overall", "Time-to-Reach", overall_metrics["overall"]["time_to_reach_s"]["mean"], overall_metrics["overall"]["time_to_reach_s"]["std"], "s"])
        writer.writerow(["Overall", "Action Smoothness (|Delta a|)", overall_metrics["overall"]["action_smoothness"]["mean"], overall_metrics["overall"]["action_smoothness"]["std"], "delta"])
        writer.writerow(["Overall", "Episodic Return", overall_metrics["overall"]["episodic_return"]["mean"], overall_metrics["overall"]["episodic_return"]["std"], "reward"])
        for z in ["bottom", "medium", "high"]:
            if z in zone_stats:
                zs = zone_stats[z]
                writer.writerow([f"Zone_{z.capitalize()}", "Success Rate", zs["success_rate_percent"]["mean"], zs["success_rate_percent"]["std"], "%"])
                writer.writerow([f"Zone_{z.capitalize()}", "Final Error", zs["final_err_mm"]["mean"], zs["final_err_mm"]["std"], "mm"])
                writer.writerow([f"Zone_{z.capitalize()}", "Time-to-Reach", zs["time_to_reach_s"]["mean"], zs["time_to_reach_s"]["std"], "s"])
                writer.writerow([f"Zone_{z.capitalize()}", "Action Smoothness", zs["action_smoothness"]["mean"], zs["action_smoothness"]["std"], "delta"])

    # 3. JSON Export
    json_path = os.path.join(output_dir, "reach_eval_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(overall_metrics, f, indent=2)


    # Publication-Grade Distribution Figures

    fig_path = os.path.join(output_dir, "reach_eval_distributions.png")
    plot_evaluation_distributions(results, overall_metrics, fig_path)

    print(f"[Saved] Per-target evaluation CSV: {detailed_csv_path}")
    print(f"[Saved] Summary evaluation CSV:    {summary_csv_path}")
    print(f"[Saved] JSON evaluation metrics:   {json_path}")
    print(f"[Saved] Distribution figures plot: {fig_path}")

    return overall_metrics


def plot_evaluation_distributions(results: list[dict], metrics: dict, output_path: str):
    """
    Plots publication-grade 4-panel distribution figures:
    1. Final Euclidean Distance Error histogram & KDE with 10 mm tolerance line.
    2. Time-to-Reach distribution violin/box plot across zones.
    3. Action Smoothness (|Δa|) distribution across zones.
    4. 3-Zone Success Rate bar chart with error bars.
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)

    zone_colors = {"bottom": "#1f77b4", "medium": "#ff7f0e", "high": "#2ca02c"}

    # 1. Error Distribution
    ax = axes[0, 0]
    errors = [r["final_err_mm"] for r in results]
    ax.hist(errors, bins=35, color="#1f77b4", alpha=0.75, edgecolor="black", density=True, label="Target Error (mm)")
    ax.axvline(10.0, color="#d62728", linestyle="--", linewidth=2.0, label="10 mm Precision Threshold")
    ax.axvline(metrics["overall"]["final_err_mm"]["mean"], color="#2ca02c", linestyle="-", linewidth=2.0,
               label=f"Mean: {metrics['overall']['final_err_mm']['mean']:.2f} mm")
    ax.set_title("Final Euclidean Distance Error Distribution", fontsize=12, fontweight="bold")
    ax.set_xlabel("Euclidean Distance Error (mm)")
    ax.set_ylabel("Probability Density")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.6)

    # 2. Time-to-Reach across Zones
    ax = axes[0, 1]
    zone_times = [
        [r["time_to_reach_s"] for r in results if r["target_zone"] == z and r["success"]]
        for z in ["bottom", "medium", "high"]
    ]
    bp = ax.boxplot(zone_times, patch_artist=True, tick_labels=["Bottom", "Medium", "High"])
    for patch, z in zip(bp["boxes"], ["bottom", "medium", "high"]):
        patch.set_facecolor(zone_colors[z])
        patch.set_alpha(0.7)
    ax.set_title("Time-to-Reach & Hold across Target Zones", fontsize=12, fontweight="bold")
    ax.set_xlabel("Target Height Zone")
    ax.set_ylabel("Time to Reach & Hold (s)")
    ax.grid(True, linestyle="--", alpha=0.6)

    # 3. Action Smoothness (|Δa|) across Zones
    ax = axes[1, 0]
    zone_smooth = [
        [r["action_smoothness"] for r in results if r["target_zone"] == z]
        for z in ["bottom", "medium", "high"]
    ]
    bp2 = ax.boxplot(zone_smooth, patch_artist=True, tick_labels=["Bottom", "Medium", "High"])
    for patch, z in zip(bp2["boxes"], ["bottom", "medium", "high"]):
        patch.set_facecolor(zone_colors[z])
        patch.set_alpha(0.7)
    ax.set_title("Action Smoothness (|Δa| per step) across Zones", fontsize=12, fontweight="bold")
    ax.set_xlabel("Target Height Zone")
    ax.set_ylabel("Mean |Δa| per Step")
    ax.grid(True, linestyle="--", alpha=0.6)

    # 4. Stratified 3-Zone Success Rates
    ax = axes[1, 1]
    zones = ["bottom", "medium", "high"]
    means = [metrics["zones"].get(z, {}).get("success_rate_percent", {}).get("mean", 0.0) for z in zones]
    stds = [metrics["zones"].get(z, {}).get("success_rate_percent", {}).get("std", 0.0) for z in zones]
    bars = ax.bar(["Bottom", "Medium", "High"], means, yerr=stds, capsize=5,
                  color=[zone_colors[z] for z in zones], alpha=0.85, edgecolor="black")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2.0, m + 2.0, f"{m:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_title("3-Zone Stratified Success Rate (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(-5, 115)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.suptitle(
        f"SO-101 Precision Reaching: Held-Out Benchmark ({metrics['n_targets']} Targets, Overall Success: {metrics['overall']['success_rate_percent']['mean']:.1f}%)",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# ---------------------------------------------------------------------------
# Command Line Interface
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Official Held-Out Evaluation Suite for SO-101 Precision Reaching (>= 500 Targets)"
    )
    parser.add_argument(
        "--n-targets", "--episodes",
        type=int,
        default=500,
        dest="n_targets",
        help="Number of randomized held-out targets to evaluate (default: 500)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained PPO reaching model .zip (auto-detects if omitted)",
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
        default=2026,
        help="Deterministic scenario evaluation seed (default: 2026)",
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
        default="reports/evaluation_reach",
        help="Directory to save CSV logs, JSON metrics, and figures",
    )
    args = parser.parse_args()

    evaluate_reach_held_out(
        model_path=args.model,
        vecnorm_path=args.vecnorm,
        n_targets=args.n_targets,
        seed=args.seed,
        render=args.render,
        playback_speed=args.speed,
        output_dir=args.output_dir,
    )
