import os
import sys
import time
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.common import configure_env
configure_env()

from src.so101_model_mapper import SO101ModelMapper
from src.feetech_actuator import FeetechSTS3215ActuatorGroup


class SO101ReachingEnv(gym.Env):
    """
    Unified Goal-Conditioned Reaching Environment for the SO-101 Manipulator.

    Includes full low-level MuJoCo model management and Feetech STS3215 actuator dynamics:
    - 3-Zone Target Randomization:
        1. Bottom (near ground/table plane): Z in [0.015, 0.065] m
        2. Medium height:                   Z in [0.075, 0.180] m
        3. High height (overhead reach):    Z in [0.190, 0.320] m
    - Diverse & Natural Starting Pose Randomization:
        Picks from 5 distinct realistic arm archetypes (center ready, left-angled,
        right-angled, compact rest, elevated) with continuous joint jitter.
    - Stable Vibration-Free Reward Function:
        Smooth exponential precision gradient, near-target velocity damping,
        monotonic in-tolerance hold progression, zero-action reward, slip penalty,
        and action smoothness regulation.
    - Success Condition:
        End-effector within 1 cm (0.01 m) of the 3D target, held stably
        for at least 0.5 seconds (13 consecutive policy steps at 25 Hz).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(
        self,
        scene_path: str = "so101/reach_scene.xml",
        config_path: str = "configs/so101_config.yaml",
        render_mode: str = None,
        max_episode_steps: int = 350,   # 14.0 s at 25 Hz policy rate
        control_substeps: int = 20,     # 25 Hz policy control frequency
        success_dist: float = 0.010,    # 1 cm requirement
        hold_time_sec: float = 0.50,    # 0.5 second hold requirement
        randomize_init_pose: bool = True,
    ):
        super().__init__()
        self.scene_path = scene_path
        self.config_path = config_path
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.control_substeps = control_substeps
        self.max_arm_delta = 0.030      # ~1.7 deg/step max change per policy step
        self.success_dist = success_dist
        self.required_hold_steps = int(np.ceil(hold_time_sec / (control_substeps * 0.002)))  # 13 steps
        self.randomize_init_pose = randomize_init_pose

        # Real-time pacing & rendering
        self.realtime = False
        self._last_render_time = None
        self.renderer = None
        self.viewer = None

        if not os.path.exists(scene_path):
            raise FileNotFoundError(f"MuJoCo scene XML not found: {scene_path}")

        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)

        # Validate joint and actuator ordering strictly against centralized config
        self.mapper = SO101ModelMapper(config_path, self.model)

        # Name-resolved MuJoCo indices (arm joints, gripper joint, actuators)
        self.arm_joint_ids    = self.mapper.arm_joint_ids
        self.gripper_joint_id = self.mapper.gripper_joint_id
        self.arm_qpos_idx     = self.mapper.arm_qpos_indices
        self.gripper_qpos_idx = self.mapper.gripper_qpos_index
        self.robot_qpos_idx   = self.mapper.robot_qpos_indices
        self.arm_dof_idx      = self.mapper.arm_dof_indices
        self.gripper_dof_idx  = self.mapper.gripper_dof_index
        self.robot_dof_idx    = self.mapper.robot_dof_indices
        self.arm_act_idx      = self.mapper.arm_actuator_ids
        self.gripper_act_idx  = self.mapper.gripper_actuator_id
        self.robot_act_idx    = self.mapper.robot_actuator_ids

        # Feetech STS3215 position actuators
        self.actuators = FeetechSTS3215ActuatorGroup(
            n_actuators=len(self.robot_act_idx),
            kp=250.0,
            kv=12.0,
            max_torque=2.94,
            max_velocity=5.50,
            dt_physics=self.model.opt.timestep,
        )

        # Joint limits and references resolved by joint ID
        self.arm_low, self.arm_high, self.gripper_low, self.gripper_high = (
            self.mapper.get_joint_limits(self.model)
        )

        self.home_qpos = np.array(self.mapper.cfg["home_qpos"]["full"], dtype=np.float64)
        self.current_target_qpos = self.home_qpos.copy()

        # Site and frame IDs
        ee_site_name = self.mapper.frames_cfg["end_effector_site"]
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
        if self.ee_site_id < 0:
            raise ValueError(f"Required site '{ee_site_name}' not found in model!")

        self.target_mocap_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self.target_mocap_idx = self.model.body_mocapid[self.target_mocap_id]

        # 3 Distinct Height Zones (all kinematically verified with 100% reachability)
        self.zones = {
            "bottom": {"x": [0.22, 0.34], "y": [-0.16, 0.16], "z": [0.015, 0.065]},  # near ground / plane
            "medium": {"x": [0.22, 0.34], "y": [-0.16, 0.16], "z": [0.075, 0.180]},  # mid-altitude
            "high":   {"x": [0.22, 0.32], "y": [-0.15, 0.15], "z": [0.190, 0.320]},  # overhead reaching
        }
        self.zone_names = ["bottom", "medium", "high"]

        # 5-D Action Space: continuous position deltas for the 5 arm joints
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )

        # 21-D Observation Space
        # qpos(5) + qvel(5) + ee_pos(3) + target_pos(3) + ee_to_tgt(3) + dist(1) + hold_ratio(1)
        obs_dim = 5 + 5 + 3 + 3 + 3 + 1 + 1
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )

        self.target_pos = np.array([0.28, 0.0, 0.12], dtype=np.float64)
        self.current_zone = "medium"
        self.consecutive_hold_steps = 0
        self.prev_dist = None
        self.prev_action = np.zeros(5, dtype=np.float32)
        self.current_step = 0

    # --------------------------- State Queries ---------------------------
    def get_ee_xpos(self) -> np.ndarray:
        """Returns the 3D Cartesian position of the end-effector site."""
        return self.data.site_xpos[self.ee_site_id].copy()

    def get_arm_qpos(self) -> np.ndarray:
        """Returns the 5 arm joint positions quantized to 12-bit magnetic encoder resolution."""
        return self.actuators.quantize_encoder(self.data.qpos[self.arm_qpos_idx].copy())

    def get_arm_qvel(self) -> np.ndarray:
        """Returns the 5 arm joint velocities."""
        return self.data.qvel[self.arm_dof_idx].copy()

    def get_gripper_qpos(self) -> float:
        """Returns current gripper joint angle in radians."""
        return float(self.data.qpos[self.gripper_qpos_idx])

    def _physics_substep(self, n_substeps: int | None = None) -> None:
        """
        Integrates internal Feetech STS3215 actuator dynamics and advances MuJoCo physics.
        Computes motor torques via Feetech actuator model and writes them to data.ctrl.
        """
        n = self.control_substeps if n_substeps is None else n_substeps
        for _ in range(n):
            torques, _ = self.actuators.step(
                self.current_target_qpos,
                self.data.qpos[self.robot_qpos_idx],
                self.data.qvel[self.robot_dof_idx],
            )
            self.data.ctrl[self.robot_act_idx] = torques
            mujoco.mj_step(self.model, self.data)

    def apply_action(self, arm_action: np.ndarray, gripper_close: bool = False, speed_scale: float = 1.0):
        """
        Translates normalized policy action into Feetech STS3215 position commands
        and integrates physics over control_substeps (25 Hz control frequency).
        """
        dq = np.clip(arm_action, -1.0, 1.0) * (self.max_arm_delta * float(speed_scale))
        self.current_target_qpos[:5] = np.clip(
            self.current_target_qpos[:5] + dq,
            self.arm_low,
            self.arm_high,
        )

        if gripper_close:
            self.current_target_qpos[5] = self.mapper.gripper_cfg["closed_qpos"]
        else:
            self.current_target_qpos[5] = self.mapper.gripper_cfg["open_qpos"]

        self._physics_substep()

    def check_self_collisions(self) -> int:
        """Returns the count of active collisions."""
        return int(self.data.ncon)

    def _get_obs(self) -> np.ndarray:
        qpos = self.get_arm_qpos()
        qvel = self.get_arm_qvel()
        ee_pos = self.get_ee_xpos()
        ee_to_tgt = self.target_pos - ee_pos
        dist = float(np.linalg.norm(ee_to_tgt))
        hold_ratio = float(self.consecutive_hold_steps) / float(self.required_hold_steps)

        obs = np.concatenate([
            qpos,
            qvel,
            ee_pos,
            self.target_pos,
            ee_to_tgt,
            [dist],
            [hold_ratio],
        ]).astype(np.float32)

        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _sample_target(self) -> np.ndarray:
        zone = self.np_random.choice(self.zone_names)
        self.current_zone = zone
        bounds = self.zones[zone]

        for _ in range(50):
            x = float(self.np_random.uniform(bounds["x"][0], bounds["x"][1]))
            y = float(self.np_random.uniform(bounds["y"][0], bounds["y"][1]))
            z = float(self.np_random.uniform(bounds["z"][0], bounds["z"][1]))

            r = np.hypot(x, y)
            r3d = np.sqrt(x**2 + y**2 + z**2)
            if 0.18 <= r <= 0.35 and r3d <= 0.38:
                return np.array([x, y, z], dtype=np.float64)

        return np.array([0.26, 0.0, bounds["z"][0] + 0.03], dtype=np.float64)

    def _sample_initial_arm_pose(self) -> np.ndarray:
        if not self.randomize_init_pose:
            return self.home_qpos[:5].copy()

        archetypes = [
            np.array([ 0.00, -0.45,  0.70,  0.50,  0.00]),   # Standard centered ready
            np.array([ 0.35, -0.30,  0.60,  0.45, -0.25]),   # Angled left
            np.array([-0.35, -0.30,  0.60,  0.45,  0.25]),   # Angled right
            np.array([ 0.00, -0.65,  0.95,  0.30,  0.00]),   # Compact tucked rest
            np.array([ 0.00, -0.15,  0.40,  0.65,  0.00]),   # High-reach posture
        ]
        chosen = archetypes[self.np_random.integers(0, len(archetypes))]
        jitter = self.np_random.uniform(-0.06, 0.06, size=5)
        pose = chosen + jitter
        return np.clip(pose, self.arm_low, self.arm_high)

    # Gym API 
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        arm_pose = self._sample_initial_arm_pose()
        self.data.qpos[self.arm_qpos_idx] = arm_pose
        self.data.qpos[self.gripper_qpos_idx] = self.mapper.gripper_cfg["open_qpos"]
        self.data.qvel[self.robot_dof_idx] = 0.0

        self.current_target_qpos[:5] = arm_pose.copy()
        self.current_target_qpos[5] = self.mapper.gripper_cfg["open_qpos"]

        self.actuators.reset(self.current_target_qpos)

        # Sample and set target mocap position
        self.target_pos = self._sample_target()
        self.data.mocap_pos[self.target_mocap_idx] = self.target_pos.copy()

        mujoco.mj_forward(self.model, self.data)

        self.current_step = 0
        self.consecutive_hold_steps = 0
        self.prev_dist = float(np.linalg.norm(self.target_pos - self.get_ee_xpos()))
        self.prev_action = np.zeros(5, dtype=np.float32)

        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (5,):
            raise ValueError(f"Expected action shape (5,), got {action.shape}")
        action = np.clip(action, -1.0, 1.0)

        self.current_step += 1

        dist_before = float(np.linalg.norm(self.target_pos - self.get_ee_xpos()))

        # Precision velocity scaling & goal stabilization:
        # 1. Full transit speed at distance (> 4 cm).
        # 2. Smooth deceleration approaching target (1 to 4 cm).
        # 3. Inside goal tolerance (<= 1 cm): quadratic damping and action deadband to eliminate micro-jittering.
        if dist_before <= self.success_dist:
            speed_scale = float(np.clip((dist_before / self.success_dist) ** 2, 0.0, 0.20))
            arm_action = np.where(np.abs(action) < 0.10, 0.0, action).astype(np.float32)
        elif dist_before < 0.040:
            speed_scale = float(np.clip(dist_before / 0.040, 0.20, 1.0))
            arm_action = action
        else:
            speed_scale = 1.0
            arm_action = action

        self.apply_action(arm_action, gripper_close=False, speed_scale=speed_scale)

        ee_pos = self.get_ee_xpos()
        dist = float(np.linalg.norm(self.target_pos - ee_pos))
        in_tolerance = dist <= self.success_dist
        was_in_tolerance = self.consecutive_hold_steps > 0

        if in_tolerance:
            self.consecutive_hold_steps += 1
        else:
            self.consecutive_hold_steps = 0

        success = self.consecutive_hold_steps >= self.required_hold_steps

        # Reward Function
        dist_reward = -8.0 * dist
        progress = (self.prev_dist - dist) * 20.0
        precision_bonus = 6.0 * float(np.exp(-dist / 0.025))

        arm_v_norm = float(np.linalg.norm(self.get_arm_qvel()))

        if dist < 0.050:
            vel_damping = -0.50 * (arm_v_norm ** 2) * max(0.0, 1.0 - (dist / 0.050))
        else:
            vel_damping = 0.0

        if in_tolerance:
            hold_ratio = float(self.consecutive_hold_steps) / float(self.required_hold_steps)
            stillness = 12.0 / (1.0 + 60.0 * arm_v_norm)
            hold_reward = 12.0 + 15.0 * hold_ratio + stillness
            action_penalty = -1.0 * float(np.sum(action ** 2))
        else:
            hold_reward = 0.0
            action_penalty = 0.0

        slip_penalty = -5.0 if (was_in_tolerance and not in_tolerance and self.current_step > 1) else 0.0
        smoothness_penalty = -0.04 * float(np.sum((action - self.prev_action) ** 2))

        reward = (
            dist_reward
            + progress
            + precision_bonus
            + vel_damping
            + hold_reward
            + action_penalty
            + slip_penalty
            + smoothness_penalty
        )

        if success:
            reward += 300.0

        self.prev_dist = dist
        self.prev_action = action.copy()

        terminated = bool(success)
        truncated = bool(self.current_step >= self.max_episode_steps)

        info = {
            "success": success,
            "dist_mm": dist * 1000.0,
            "in_tolerance": in_tolerance,
            "hold_steps": self.consecutive_hold_steps,
            "target_zone": self.current_zone,
            "action_delta": float(np.mean(np.abs(action - self.prev_action))),
            "current_step": self.current_step,
        }

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            self.renderer.update_scene(self.data)
            return self.renderer.render()
        elif self.render_mode == "human":
            if self.viewer is None:
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self._last_render_time = time.perf_counter()
            step_duration = self.control_substeps * self.model.opt.timestep  # 0.04 s at 25 Hz
            if self._last_render_time is not None:
                elapsed = time.perf_counter() - self._last_render_time
                sleep_for = step_duration - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)
            self._last_render_time = time.perf_counter()
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
