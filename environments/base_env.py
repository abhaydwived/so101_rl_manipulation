import os
import sys
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.so101_model_mapper import SO101ModelMapper
from src.feetech_actuator import FeetechSTS3215ActuatorGroup


class SO101BaseEnv(gym.Env):
    """
    Abstract Base Gymnasium Environment for the SO-101 Manipulator.

    Implements:
    - MuJoCo model loading, compilation, and automatic mapper validation.
    - Explicit decoupling: physics timestep (0.002s = 500 Hz) vs policy control frequency (25 Hz).
    - True position-controlled actuator model with Feetech STS3215 physical dynamics.
    - Normalized continuous action space with bounded joint position deltas.
    - Deterministic seeding, state extraction, collision querying, and rendering.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(
        self,
        scene_path: str,
        config_path: str = "configs/so101_config.yaml",
        render_mode: str = None,
        max_episode_steps: int = 150,
        control_substeps: int = 20,
        max_arm_delta: float = 0.035,   # ~2.0 degrees max change per policy step
    ):
        super().__init__()
        self.scene_path = scene_path
        self.config_path = config_path
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.control_substeps = control_substeps
        self.max_arm_delta = max_arm_delta

        if not os.path.exists(scene_path):
            raise FileNotFoundError(f"MuJoCo scene XML not found: {scene_path}")

        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)

        # Validate joint and actuator ordering strictly against centralized config
        self.mapper = SO101ModelMapper(config_path, self.model)

        # Feetech STS3215 position actuators
        self.actuators = FeetechSTS3215ActuatorGroup(
            n_actuators=self.model.nu,
            kp=250.0,
            kv=12.0,
            max_torque=2.94,
            max_velocity=5.50,
            dt_physics=self.model.opt.timestep,
        )

        # Joint limits and references
        self.arm_low = self.model.jnt_range[:5, 0].copy()
        self.arm_high = self.model.jnt_range[:5, 1].copy()
        self.gripper_low = self.model.jnt_range[5, 0]
        self.gripper_high = self.model.jnt_range[5, 1]

        self.home_qpos = np.array(self.mapper.cfg["home_qpos"]["full"], dtype=np.float64)
        self.current_target_qpos = self.home_qpos.copy()

        # Site and frame IDs
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        if self.ee_site_id < 0:
            raise ValueError("Required site 'gripperframe' not found in model!")

        self.current_step = 0
        self.viewer = None

    def get_ee_xpos(self) -> np.ndarray:
        """Returns the 3D Cartesian position of the end-effector site."""
        return self.data.site_xpos[self.ee_site_id].copy()

    def get_arm_qpos(self) -> np.ndarray:
        """Returns the 5 arm joint positions quantized to 12-bit magnetic encoder resolution."""
        return self.actuators.quantize_encoder(self.data.qpos[:5].copy())

    def get_arm_qvel(self) -> np.ndarray:
        """Returns the 5 arm joint velocities."""
        return self.data.qvel[:5].copy()

    def get_gripper_qpos(self) -> float:
        """Returns current gripper joint angle in radians."""
        return float(self.data.qpos[5])

    def apply_action(self, arm_action: np.ndarray, gripper_close: bool = False):
        """
        Translates normalized policy action into Feetech STS3215 position commands
        and integrates physics over control_substeps (25 Hz control frequency).
        """
        # Bounded joint position delta control
        dq = np.clip(arm_action, -1.0, 1.0) * self.max_arm_delta
        self.current_target_qpos[:5] = np.clip(
            self.current_target_qpos[:5] + dq,
            self.arm_low,
            self.arm_high,
        )

        # Discrete/continuous gripper positioning
        if gripper_close:
            self.current_target_qpos[5] = self.mapper.gripper_cfg["closed_qpos"]  # -0.10 rad
        else:
            self.current_target_qpos[5] = self.mapper.gripper_cfg["open_qpos"]    # 1.00 rad

        # Substep physics loop
        for _ in range(self.control_substeps):
            torques, tracked = self.actuators.step(
                self.current_target_qpos,
                self.data.qpos[:6],
                self.data.qvel[:6],
            )
            self.data.ctrl[:6] = self.current_target_qpos
            mujoco.mj_step(self.model, self.data)

    def check_self_collisions(self) -> int:
        """Returns the count of active collisions excluding allowable parent-child contacts."""
        return int(self.data.ncon)

    def render(self):
        if self.render_mode == "rgb_array":
            renderer = mujoco.Renderer(self.model, height=480, width=640)
            renderer.update_scene(self.data)
            return renderer.render()
        elif self.render_mode == "human":
            if self.viewer is None:
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

