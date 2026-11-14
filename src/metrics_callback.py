#!/usr/bin/env python3
"""
Custom Callbacks for Comprehensive Training Metrics Tracking in TensorBoard.

PickPlaceMetricsCallback:
    Tracks two primary metrics:
      1. Stage Success Rate: Whether the policy completed the current stage subtask.
      2. Full-Task Success Rate: Whether the policy achieved end-to-end task completion.
    Also logs per-stage subtask rates, placement error, action smoothness,
    episode reward, and episode length.

CurriculumCallback:
    Re-exported from src.curriculum_callback for backwards compatibility.
"""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

# Re-export CurriculumCallback for backward compatibility
from src.curriculum_callback import CurriculumCallback


class PickPlaceMetricsCallback(BaseCallback):
    """
    Logs staged task metrics extracted from environment infos to TensorBoard.
    """

    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq

        # Primary requested metrics buffers
        self.episode_stage_successes = []
        self.episode_full_task_successes = []

        # Rolling episode subtask metric buffers
        self.episode_successes = []
        self.episode_grasps = []
        self.episode_lifts = []
        self.episode_places = []
        self.episode_releases = []
        self.episode_clears = []
        self.episode_errors = []
        self.episode_smoothness = []
        self.episode_lengths = []
        self.episode_rewards = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [True] * len(infos))
        for done, info in zip(dones, infos):
            if "action_delta" in info:
                self.episode_smoothness.append(float(info["action_delta"]))

            if not done:
                continue

            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])

            # Two explicit separate metrics (episode-level)
            if "stage_success" in info:
                self.episode_stage_successes.append(float(info["stage_success"]))
            if "full_task_success" in info:
                self.episode_full_task_successes.append(float(info["full_task_success"]))

            if "success" in info:
                self.episode_successes.append(float(info["success"]))
            if "stage_grasped" in info:
                self.episode_grasps.append(float(info["stage_grasped"]))
            if "stage_lifted" in info:
                self.episode_lifts.append(float(info["stage_lifted"]))
            if "stage_placed" in info:
                self.episode_places.append(float(info["stage_placed"]))
            if "stage_released" in info:
                self.episode_releases.append(float(info["stage_released"]))
            if "stage_cleared" in info:
                self.episode_clears.append(float(info["stage_cleared"]))
            if "cube_target_err_mm" in info:
                self.episode_errors.append(float(info["cube_target_err_mm"]))

        # Periodic logging to TensorBoard
        if self.n_calls % self.log_freq == 0 and len(self.episode_successes) > 0:
            # Two primary separate metrics
            if self.episode_stage_successes:
                self.logger.record("custom/stage_success_rate", np.mean(self.episode_stage_successes[-100:]))
            if self.episode_full_task_successes:
                self.logger.record("custom/full_task_success_rate", np.mean(self.episode_full_task_successes[-100:]))

            self.logger.record("custom/success_rate", np.mean(self.episode_successes[-100:]))
            self.logger.record("custom/grasp_rate", np.mean(self.episode_grasps[-100:]))
            self.logger.record("custom/lift_rate", np.mean(self.episode_lifts[-100:]))
            self.logger.record("custom/placement_rate", np.mean(self.episode_places[-100:]))
            self.logger.record("custom/release_rate", np.mean(self.episode_releases[-100:]))
            self.logger.record("custom/clearance_rate", np.mean(self.episode_clears[-100:]))

            if self.episode_errors:
                self.logger.record("custom/mean_placement_error_mm", np.mean(self.episode_errors[-100:]))
            if self.episode_smoothness:
                self.logger.record("custom/action_smoothness", np.mean(self.episode_smoothness[-500:]))
            if self.episode_rewards:
                self.logger.record("custom/mean_reward", np.mean(self.episode_rewards[-100:]))
            if self.episode_lengths:
                self.logger.record("custom/mean_episode_length", np.mean(self.episode_lengths[-100:]))

        return True
