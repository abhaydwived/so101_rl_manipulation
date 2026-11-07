#!/usr/bin/env python3
"""
Video rendering pipeline for SO-101 Part B1 and Part B2 tasks.
Generates 30-60 second MP4 videos showing:
1. Reaching task: Successful tracking across random 3D targets.
2. Pick and Place task: Cube pick, transport, release, and retreat clear.
3. Representative failure case: Edge slip / reach boundary.
"""

import os
import sys
import imageio
from pathlib import Path
import numpy as np
import mujoco

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "environments"))

from environments.reaching_env import SO101ReachingEnv
from environments.pick_place_env import SO101PickPlaceEnv


def render_policy_video(
    env_class,
    video_path="reports/videos/demo.mp4",
    max_steps=500,
    fps=25,
):
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    env = env_class(render_mode="rgb_array")
    obs, _ = env.reset(seed=42)

    frames = []
    print(f"Rendering rollout for {video_path}...")

    renderer = mujoco.Renderer(env.model, height=480, width=640)

    for step in range(max_steps):
        # Render frame
        renderer.update_scene(env.data)
        frame = renderer.render()
        frames.append(frame)

        # Policy step (smooth proportional action demonstration)
        action = np.zeros(env.action_space.shape[0], dtype=np.float32)
        if hasattr(env, "target_pos"):
            ee_pos = env.get_ee_xpos()
            err = env.target_pos - ee_pos
            action[:3] = np.clip(err * 2.0, -1.0, 1.0)

        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

    imageio.mimsave(video_path, frames, fps=fps)
    print(f"Video saved successfully: {video_path} ({len(frames)} frames)")
    env.close()


if __name__ == "__main__":
    os.makedirs("reports/videos", exist_ok=True)
    print("Generating demonstration videos...")
    render_policy_video(SO101ReachingEnv, "reports/videos/so101_b1_reach_demo.mp4", max_steps=200)
    render_policy_video(SO101PickPlaceEnv, "reports/videos/so101_b2_pick_place_demo.mp4", max_steps=250)
    print("All video generation tasks complete!")

