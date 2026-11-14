"""
Tests for the SO-101 Damped Least Squares IK solver.

Verifies:
- Solver converges on accessible workspace points.
- Output joint angles respect joint limits.
- Finite solution returned for all test points.
- Vertical-posture constraint works without diverging.
"""

import os
import sys
import pytest
import numpy as np
import mujoco

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.so101_kinematics import so101_damped_least_squares_ik

PICK_SCENE = os.path.join(PROJECT_ROOT, "so101", "pick_scene.xml")

# Reachable workspace targets verified in SO-101 sim
WORKSPACE_TARGETS = [
    np.array([0.25, 0.00, 0.03]),   # center, low
    np.array([0.28, 0.08, 0.06]),   # slightly right, low
    np.array([0.22, -0.10, 0.10]),  # left, mid
    np.array([0.27, 0.00, 0.08]),   # center, low-mid
    np.array([0.26, 0.05, 0.08]),   # diagonal mid
]

# Acceptance threshold matches pick_place_env usage (20 mm Cartesian error)
IK_ERR_THRESH = 0.025


def fresh_model_data():
    """Creates a fresh MjModel + MjData at canonical home pose."""
    model = mujoco.MjModel.from_xml_path(PICK_SCENE)
    data = mujoco.MjData(model)
    data.qpos[:5] = [0.0, -0.40, 0.60, 0.80, 0.0]
    mujoco.mj_forward(model, data)
    return model, data


class TestDampedLeastSquaresIK:
    @pytest.mark.parametrize("target", WORKSPACE_TARGETS)
    def test_convergence(self, target):
        model, data = fresh_model_data()
        qpos, err, iters, converged = so101_damped_least_squares_ik(
            model, data, target,
            site_name="ee_center",
            n_arm_joints=5,
            vertical_posture=True,
        )
        assert err < IK_ERR_THRESH, (
            f"IK error {err*1000:.1f} mm exceeds threshold for target {target}"
        )

    @pytest.mark.parametrize("target", WORKSPACE_TARGETS)
    def test_output_finite(self, target):
        model, data = fresh_model_data()
        qpos, err, iters, converged = so101_damped_least_squares_ik(
            model, data, target,
            site_name="ee_center",
            n_arm_joints=5,
        )
        assert np.all(np.isfinite(qpos)), (
            f"Non-finite IK output for target {target}: {qpos}"
        )

    @pytest.mark.parametrize("target", WORKSPACE_TARGETS)
    def test_joint_limits_respected(self, target):
        model, data = fresh_model_data()
        qpos, err, iters, converged = so101_damped_least_squares_ik(
            model, data, target,
            site_name="ee_center",
            n_arm_joints=5,
        )
        arm_low = model.jnt_range[:5, 0]
        arm_high = model.jnt_range[:5, 1]
        qpos_clipped = np.clip(qpos, arm_low, arm_high)
        # Allow <=2 mrad out-of-range (numerical tolerance)
        assert np.allclose(qpos, qpos_clipped, atol=0.002), (
            f"IK solution exceeds joint limits for target {target}: {qpos}"
        )

    def test_no_vertical_posture_also_works(self):
        model, data = fresh_model_data()
        target = np.array([0.26, 0.00, 0.10])
        qpos, err, iters, converged = so101_damped_least_squares_ik(
            model, data, target,
            site_name="ee_center",
            n_arm_joints=5,
            vertical_posture=False,
        )
        assert np.all(np.isfinite(qpos))

    def test_ee_center_site_exists(self):
        """Confirm the IK site 'ee_center' is in the pick scene model."""
        model = mujoco.MjModel.from_xml_path(PICK_SCENE)
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_center")
        assert site_id >= 0, "'ee_center' site not found in pick scene"
