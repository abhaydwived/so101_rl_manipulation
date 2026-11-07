#!/usr/bin/env python3
"""
Comprehensive Gymnasium API and Space Validation Suite for SO-101.
Validates:
1. Environment instantiation and Gymnasium EnvChecker compliance.
2. Exact observation and action space dimensions.
3. Proper scaling, absence of NaN/Inf values, and safe joint limit clipping.
4. Feetech STS3215 actuator mapping and torque boundedness.
5. Deterministic reset reproducibility across seeds.
"""

import sys
import os
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from gymnasium.utils.env_checker import check_env
from environments.reaching_env import SO101ReachingEnv
from environments.pick_place_env import SO101PickPlaceEnv


def validate_environment(env_class, name, expected_obs_dim, expected_act_dim):
    print("==================================================================")
    print(f"VALIDATING ENVIRONMENT: {name}")
    print("==================================================================")

    env = env_class()

    # 1. Check dimensions
    assert env.observation_space.shape == (expected_obs_dim,), (
        f"Obs shape mismatch: expected ({expected_obs_dim},), got {env.observation_space.shape}"
    )
    assert env.action_space.shape == (expected_act_dim,), (
        f"Action shape mismatch: expected ({expected_act_dim},), got {env.action_space.shape}"
    )
    print(f"[PASSED] Observation Space: {env.observation_space}")
    print(f"[PASSED] Action Space:      {env.action_space}")

    # 2. Gymnasium API Checker
    check_env(env.unwrapped, skip_render_check=True)
    print("[PASSED] Gymnasium check_env() API compliance passed.")

    # 3. Test Reset & Determinism
    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)
    assert np.allclose(obs1, obs2), "Reset failed deterministic seeding check!"
    print("[PASSED] Deterministic seed reproducibility confirmed.")

    # 4. Check for NaNs and Infs over 50 random steps
    for step in range(50):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)

        assert not np.isnan(obs).any(), f"NaN detected in observation at step {step}"
        assert not np.isinf(obs).any(), f"Inf detected in observation at step {step}"
        assert not np.isnan(reward), f"NaN detected in reward at step {step}"
        assert not np.isinf(reward), f"Inf detected in reward at step {step}"

        # Verify joint position bounds
        qpos_arm = env.data.qpos[:5]
        assert np.all(qpos_arm >= env.arm_low - 0.05), f"Joint lower limit violated: {qpos_arm} vs {env.arm_low}"
        assert np.all(qpos_arm <= env.arm_high + 0.05), f"Joint upper limit violated: {qpos_arm} vs {env.arm_high}"

        # Verify actuator torque saturation (Feetech STS3215: 2.94 Nm)
        torques = env.data.actuator_force[:6]
        assert np.all(np.abs(torques) <= 2.95), f"Torque limit exceeded 2.94 Nm: {torques}"

        if term or trunc:
            obs, _ = env.reset()

    print("[PASSED] 50-step simulation sanity, no NaNs/Infs, limits strictly enforced.")
    env.close()
    print(f"[SUCCESS] {name} passed all validation tests!\n")


if __name__ == "__main__":
    validate_environment(SO101ReachingEnv, "SO101ReachingEnv (Part B1)", expected_obs_dim=21, expected_act_dim=5)
    validate_environment(SO101PickPlaceEnv, "SO101PickPlaceEnv (Part B2)", expected_obs_dim=30, expected_act_dim=6)
    print("ALL ENVIRONMENT AND SPACE VALIDATIONS COMPLETED SUCCESSFULLY!")
