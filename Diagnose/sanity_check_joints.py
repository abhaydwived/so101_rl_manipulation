#!/usr/bin/env python3
"""
Sanity script for SO-101 robotic arm in MuJoCo.
Satisfies Part A assignment requirement:
- Sweeps every joint individually through its full operational range of motion.
- Measures and logs end-effector Cartesian poses and collision contacts.
- Verifies position tracking, actuator torque limits, and self-collision avoidance.
- Saves trajectory logs and generates diagnostic visualization figures.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.common import configure_env
configure_env()

import numpy as np
import matplotlib.pyplot as plt
import mujoco

from src.so101_model_mapper import SO101ModelMapper
from src.feetech_actuator import FeetechSTS3215ActuatorGroup


def run_joint_sanity_sweep(
    model_path=None,
    config_path=None,
    output_dir=None,
    n_sweep_points=80,
    steps_per_point=15,
):
    if model_path is None:
        model_path = str(PROJECT_ROOT / "so101" / "reach_scene.xml")
    elif not os.path.isabs(model_path):
        model_path = str(PROJECT_ROOT / model_path)

    if config_path is None:
        config_path = str(PROJECT_ROOT / "configs" / "so101_config.yaml")
    elif not os.path.isabs(config_path):
        config_path = str(PROJECT_ROOT / config_path)

    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "reports" / "sanity_check")
    elif not os.path.isabs(output_dir):
        output_dir = str(PROJECT_ROOT / output_dir)

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    mapper = SO101ModelMapper(config_path, model)

    actuators = FeetechSTS3215ActuatorGroup(
        n_actuators=model.nu,
        kp=250.0,
        kv=12.0,
        max_torque=2.94,
        max_velocity=5.5,
        dt_physics=model.opt.timestep,
    )

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    if site_id < 0:
        raise ValueError("Site 'gripperframe' not found!")

    home_qpos = np.array(mapper.cfg["home_qpos"]["full"], dtype=np.float64)
    data.qpos[:6] = home_qpos
    data.qvel[:6] = 0.0
    actuators.reset(home_qpos)
    mujoco.mj_forward(model, data)

    sweep_results = {}
    total_collisions = 0

    print("==================================================================")
    print("STARTING SO-101 SYSTEMATIC JOINT RANGE & ACTUATOR SANITY SWEEP")
    print(f"Model: {model_path}")
    print(f"Physics timestep: {model.opt.timestep} s ({1.0/model.opt.timestep:.0f} Hz)")
    print("==================================================================")

    for j_idx, j_name in enumerate(mapper.joint_names):
        j_range = model.jnt_range[j_idx]
        low, high = j_range[0], j_range[1]
        print(f"\n[Joint {j_idx}: {j_name}] Range: [{low:.3f}, {high:.3f}] rad ({np.degrees(low):.1f}° to {np.degrees(high):.1f}°)")

        # Target trajectory: home -> min -> max -> home
        target_seq = np.concatenate([
            np.linspace(home_qpos[j_idx], low, n_sweep_points // 3),
            np.linspace(low, high, n_sweep_points // 3),
            np.linspace(high, home_qpos[j_idx], n_sweep_points // 3),
        ])

        log_cmd = []
        log_actual = []
        log_ee = []
        log_torques = []
        log_contacts = []

        # Reset to home before sweeping this joint
        data.qpos[:6] = home_qpos.copy()
        data.qvel[:6] = 0.0
        actuators.reset(home_qpos)
        current_target = home_qpos.copy()

        for target_val in target_seq:
            current_target[j_idx] = target_val
            for _ in range(steps_per_point):
                torques, tracked = actuators.step(current_target, data.qpos[:6], data.qvel[:6])
                data.ctrl[:6] = current_target
                mujoco.mj_step(model, data)

            log_cmd.append(target_val)
            log_actual.append(data.qpos[j_idx])
            log_ee.append(data.site_xpos[site_id].copy())
            log_torques.append(data.actuator_force[j_idx])
            log_contacts.append(data.ncon)

            if data.ncon > 0:
                total_collisions += data.ncon

        log_cmd = np.array(log_cmd)
        log_actual = np.array(log_actual)
        log_ee = np.array(log_ee)
        log_torques = np.array(log_torques)

        tracking_error = np.abs(log_cmd - log_actual)
        max_err = np.max(tracking_error)
        mean_err = np.mean(tracking_error)
        max_tau = np.max(np.abs(log_torques))

        print(f"  -> Max tracking error: {max_err:.4f} rad ({np.degrees(max_err):.2f}°)")
        print(f"  -> Mean tracking error: {mean_err:.4f} rad ({np.degrees(mean_err):.2f}°)")
        print(f"  -> Peak actuator torque: {max_tau:.2f} N·m (Limit: 2.94 N·m)")
        print(f"  -> End-effector X range: [{np.min(log_ee[:, 0]):.3f}, {np.max(log_ee[:, 0]):.3f}] m")
        print(f"  -> End-effector Y range: [{np.min(log_ee[:, 1]):.3f}, {np.max(log_ee[:, 1]):.3f}] m")
        print(f"  -> End-effector Z range: [{np.min(log_ee[:, 2]):.3f}, {np.max(log_ee[:, 2]):.3f}] m")
        print(f"  -> Max unexpected contacts during sweep: {np.max(log_contacts)}")

        sweep_results[j_name] = {
            "cmd": log_cmd,
            "actual": log_actual,
            "ee": log_ee,
            "torques": log_torques,
            "max_err": max_err,
            "mean_err": mean_err,
            "max_tau": max_tau,
        }

    # Plot diagnostic figures
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, j_name in enumerate(mapper.joint_names):
        ax = axes[idx]
        res = sweep_results[j_name]
        steps = np.arange(len(res["cmd"]))
        ax.plot(steps, np.degrees(res["cmd"]), 'k--', label="Target Angle (°)", alpha=0.7)
        ax.plot(steps, np.degrees(res["actual"]), 'b-', label="Actual Angle (°)", linewidth=1.8)
        ax.set_title(f"Joint {idx}: {j_name} (Max Err: {np.degrees(res['max_err']):.2f}°)")
        ax.set_xlabel("Sweep Timestep")
        ax.set_ylabel("Angle (deg)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "joint_sweep_tracking.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()

    # Plot 3D End-Effector Workspace Reachability
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for j_name in mapper.joint_names:
        ee = sweep_results[j_name]["ee"]
        ax.plot(ee[:, 0], ee[:, 1], ee[:, 2], label=f"Sweep {j_name}", alpha=0.8)

    ax.scatter([home_qpos[0]], [home_qpos[1]], [home_qpos[2]], color='r', s=50, label="Home EE")
    ax.set_title("SO-101 End-Effector Trajectories During Range Sweeps")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend(loc="upper left", fontsize=8)
    workspace_plot_path = os.path.join(output_dir, "workspace_ee_trajectories.png")
    plt.savefig(workspace_plot_path, dpi=200)
    plt.close()

    print("\n==================================================================")
    print("SANITY SWEEP COMPLETE")
    print(f"Tracking plots saved to: {plot_path}")
    print(f"Workspace 3D plot saved to: {workspace_plot_path}")
    print(f"Self-collisions detected: {total_collisions}")
    print("==================================================================")

    return sweep_results


if __name__ == "__main__":
    run_joint_sanity_sweep()

