import os
import yaml
import numpy as np
import mujoco


class SO101ModelMapper:
    """
    Centralized mapping and validation interface for SO-101 manipulator.
    Validates MuJoCo model joint IDs, actuator IDs, kinematics, limits,
    and action-space bounds against configs/so101_config.yaml.
    """

    def __init__(self, config_path="configs/so101_config.yaml", model=None):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.robot_cfg = self.cfg["robot"]
        self.joint_cfg = self.cfg["joints"]
        self.actuator_cfg = self.cfg["actuators"]
        self.frames_cfg = self.cfg["frames"]
        self.gripper_cfg = self.cfg["gripper"]

        self.joint_names = self.joint_cfg["names"]
        self.n_joints = len(self.joint_names)
        self.arm_joint_names = self.joint_names[:5]
        self.gripper_joint_name = self.joint_names[5]

        # Explicit name-resolved MuJoCo indices (populated in validate_model / resolve_indices)
        self.arm_joint_ids = np.arange(5, dtype=np.int32)
        self.gripper_joint_id = 5
        self.robot_joint_ids = np.arange(6, dtype=np.int32)
        self.arm_qpos_indices = np.arange(5, dtype=np.int32)
        self.gripper_qpos_index = 5
        self.robot_qpos_indices = np.arange(6, dtype=np.int32)
        self.arm_dof_indices = np.arange(5, dtype=np.int32)
        self.gripper_dof_index = 5
        self.robot_dof_indices = np.arange(6, dtype=np.int32)
        self.arm_actuator_ids = np.arange(5, dtype=np.int32)
        self.gripper_actuator_id = 5
        self.robot_actuator_ids = np.arange(6, dtype=np.int32)

        if model is not None:
            self.validate_model(model)

    def resolve_indices(self, model: mujoco.MjModel) -> None:
        """Resolve joint, qpos, dof, and actuator indices explicitly by name."""
        jnt_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.joint_names]
        act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in self.joint_names]
        if any(j < 0 for j in jnt_ids):
            missing = [n for n, j in zip(self.joint_names, jnt_ids) if j < 0]
            raise ValueError(f"Missing joints in MuJoCo model: {missing}")
        if any(a < 0 for a in act_ids):
            missing = [n for n, a in zip(self.joint_names, act_ids) if a < 0]
            raise ValueError(f"Missing actuators in MuJoCo model: {missing}")

        self.robot_joint_ids = np.asarray(jnt_ids, dtype=np.int32)
        self.arm_joint_ids = self.robot_joint_ids[:5]
        self.gripper_joint_id = int(self.robot_joint_ids[5])

        self.robot_qpos_indices = np.asarray([int(model.jnt_qposadr[j]) for j in jnt_ids], dtype=np.int32)
        self.arm_qpos_indices = self.robot_qpos_indices[:5]
        self.gripper_qpos_index = int(self.robot_qpos_indices[5])

        self.robot_dof_indices = np.asarray([int(model.jnt_dofadr[j]) for j in jnt_ids], dtype=np.int32)
        self.arm_dof_indices = self.robot_dof_indices[:5]
        self.gripper_dof_index = int(self.robot_dof_indices[5])

        self.robot_actuator_ids = np.asarray(act_ids, dtype=np.int32)
        self.arm_actuator_ids = self.robot_actuator_ids[:5]
        self.gripper_actuator_id = int(self.robot_actuator_ids[5])

    def validate_model(self, model: mujoco.MjModel):
        """
        Validates the loaded MuJoCo model against the configuration.
        Ensures joint and actuator ordering match exactly with zero assumptions.
        """
        self.resolve_indices(model)

        # 1. Validate Joints
        for expected_idx, name in enumerate(self.joint_names):
            jnt_id = int(self.robot_joint_ids[expected_idx])
            if jnt_id != expected_idx:
                raise ValueError(
                    f"Joint '{name}' ID mismatch: expected {expected_idx}, got {jnt_id}"
                )

            # Check limits
            expected_range = self.joint_cfg["limits"][name]
            actual_range = model.jnt_range[jnt_id]
            if not np.allclose(expected_range, actual_range, atol=1e-3):
                raise ValueError(
                    f"Joint '{name}' range mismatch: expected {expected_range}, got {actual_range}"
                )

        # 2. Validate Actuators
        for expected_idx, name in enumerate(self.joint_names):
            act_id = int(self.robot_actuator_ids[expected_idx])
            if act_id != expected_idx:
                raise ValueError(
                    f"Actuator '{name}' ID mismatch: expected {expected_idx}, got {act_id}"
                )

            # Check actuator force limits (Feetech STS3215: 2.94 Nm)
            force_range = model.actuator_forcerange[act_id]
            expected_force = self.actuator_cfg["force_ranges"][name]
            if not np.allclose(expected_force, force_range, atol=1e-2):
                raise ValueError(
                    f"Actuator '{name}' force range mismatch: expected {expected_force}, got {force_range}"
                )

        # 3. Validate Sites
        ee_site = self.frames_cfg["end_effector_site"]
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site)
        if site_id < 0:
            raise ValueError(f"End-effector site '{ee_site}' not found in model!")

        return True

    def get_joint_limits(self, model: mujoco.MjModel):
        """Returns (low, high) arrays for arm joints and gripper using name-resolved IDs."""
        arm_low = model.jnt_range[self.arm_joint_ids, 0].copy()
        arm_high = model.jnt_range[self.arm_joint_ids, 1].copy()
        gripper_low = float(model.jnt_range[self.gripper_joint_id, 0])
        gripper_high = float(model.jnt_range[self.gripper_joint_id, 1])
        return arm_low, arm_high, gripper_low, gripper_high

    def action_to_control(self, action: np.ndarray, current_targets: np.ndarray, max_arm_delta=0.03):
        """
        Maps normalized policy action [-1, 1] to target joint positions.
        action[:5]: delta arm positions in rad
        action[5]: absolute/delta gripper command
        """
        action = np.clip(action, -1.0, 1.0)
        target_ctrl = current_targets.copy()

        # Arm delta target
        target_ctrl[:5] += action[:5] * max_arm_delta

        # Gripper: > 0 means close (-0.10 rad), < 0 means open (1.0 rad)
        if action[5] > 0.0:
            # closing
            target_ctrl[5] = self.gripper_cfg["closed_qpos"]
        else:
            # opening
            target_ctrl[5] = self.gripper_cfg["open_qpos"]

        return target_ctrl


if __name__ == "__main__":
    import sys
    model_path = "so101/so101.xml"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        sys.exit(1)

    model = mujoco.MjModel.from_xml_path(model_path)
    mapper = SO101ModelMapper("configs/so101_config.yaml", model)
    print("SUCCESS: SO101ModelMapper successfully validated all 6 joints, actuators, and limits!")

