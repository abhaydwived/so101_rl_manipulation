"""
Tests for MuJoCo model loading and joint/actuator validation.

Verifies that:
- Both scene XMLs load without errors.
- The correct number of joints and actuators exist.
- The end-effector site 'gripperframe' is present.
- SO101ModelMapper validates joint ordering, limits, and force ranges.
"""

import os
import sys
import pytest
import mujoco
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.so101_model_mapper import SO101ModelMapper

REACH_SCENE = os.path.join(PROJECT_ROOT, "so101", "reach_scene.xml")
PICK_SCENE = os.path.join(PROJECT_ROOT, "so101", "pick_scene.xml")
CONFIG = os.path.join(PROJECT_ROOT, "configs", "so101_config.yaml")


@pytest.fixture(scope="module")
def reach_model():
    return mujoco.MjModel.from_xml_path(REACH_SCENE)


@pytest.fixture(scope="module")
def pick_model():
    return mujoco.MjModel.from_xml_path(PICK_SCENE)


class TestModelLoading:
    def test_reach_scene_loads(self, reach_model):
        assert reach_model is not None

    def test_pick_scene_loads(self, pick_model):
        assert pick_model is not None

    def test_reach_scene_6_joints(self, reach_model):
        assert reach_model.njnt == 6, f"Expected 6 joints, got {reach_model.njnt}"

    def test_pick_scene_joints_minimum(self, pick_model):
        # Pick scene has robot joints + cube free joint = at least 7
        assert pick_model.njnt >= 7

    def test_reach_scene_6_actuators(self, reach_model):
        assert reach_model.nu == 6, f"Expected 6 actuators, got {reach_model.nu}"

    def test_pick_scene_6_actuators(self, pick_model):
        assert pick_model.nu == 6, f"Expected 6 actuators, got {pick_model.nu}"

    def test_gripperframe_site_reach(self, reach_model):
        site_id = mujoco.mj_name2id(reach_model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        assert site_id >= 0, "'gripperframe' site not found in reach_scene"

    def test_gripperframe_site_pick(self, pick_model):
        site_id = mujoco.mj_name2id(pick_model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        assert site_id >= 0, "'gripperframe' site not found in pick_scene"

    def test_ee_center_site_pick(self, pick_model):
        site_id = mujoco.mj_name2id(pick_model, mujoco.mjtObj.mjOBJ_SITE, "ee_center")
        assert site_id >= 0, "'ee_center' site not found in pick_scene"

    def test_target_body_reach(self, reach_model):
        body_id = mujoco.mj_name2id(reach_model, mujoco.mjtObj.mjOBJ_BODY, "target")
        assert body_id >= 0, "'target' body not found in reach_scene"

    def test_cube_body_pick(self, pick_model):
        body_id = mujoco.mj_name2id(pick_model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        assert body_id >= 0, "'cube' body not found in pick_scene"


class TestModelMapper:
    def test_mapper_validates_reach_model(self, reach_model):
        mapper = SO101ModelMapper(CONFIG, reach_model)
        assert mapper is not None

    def test_mapper_validates_pick_model(self, pick_model):
        mapper = SO101ModelMapper(CONFIG, pick_model)
        assert mapper is not None

    def test_mapper_joint_names(self, reach_model):
        mapper = SO101ModelMapper(CONFIG, reach_model)
        expected = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        assert mapper.joint_names == expected

    def test_mapper_gripper_config(self, reach_model):
        mapper = SO101ModelMapper(CONFIG, reach_model)
        assert mapper.gripper_cfg["closed_qpos"] == pytest.approx(-0.10, abs=1e-3)
        assert mapper.gripper_cfg["open_qpos"] == pytest.approx(1.00, abs=1e-3)

    def test_mapper_home_qpos_length(self, reach_model):
        mapper = SO101ModelMapper(CONFIG, reach_model)
        assert len(mapper.cfg["home_qpos"]["full"]) == 6

    def test_joint_limits_nonzero(self, reach_model):
        for i in range(5):
            lo, hi = reach_model.jnt_range[i]
            assert hi > lo, f"Joint {i} has invalid range: [{lo}, {hi}]"

