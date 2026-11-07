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


class SO101ReachEnv(gym.Env):
    """
    Gymnasium Environment for SO-101 Part B1 Reaching Task.

    Task Objectives:
    - Train an RL policy to move the gripper tip towards randomly generated 3D targets.
    - Success condition: End-effector within 1 cm (0.01 m) of the target,
      maintained for at least 0.5 seconds (13 consecutive policy steps at 25 Hz).
    - Control architecture: PPO position deltas -> Feetech STS3215 position actuators.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(
        self,
        scene_path="so101/reach_scene.xml",
        config_path="configs/so101_config.yaml",
        render_mode=None,
        max_episode_steps=120,   # ~4.8 seconds per episode
        control_substeps=20,     # 20 substeps * 0.002s = 0.04s (25 Hz control frequency)
        success_dist=0.010,      # 1 cm threshold
        hold_time_sec=0.50,      # 0.5 second hold requirement
    ):
        super().__init__()
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.control_substeps = control_substeps
        self.success_dist = success_dist

        # 0.5 s / 0.04 s per step = 12.5 steps -> 13 steps required
        self.required_hold_steps = int(np.ceil(hold_time_sec / (control_substeps * 0.002)))

        if not os.path.exists(scene_path):
            raise FileNotFoundError(f"Scene XML not found: {scene_path}")

        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)

        self.mapper = SO101ModelMapper(config_path, self.model)
        self.actuators = FeetechSTS3215ActuatorGroup(
            n_actuators=self.model.nu,
            kp=250.0,
            kv=12.0,
            max_torque=2.94,
            max_velocity=5.50,
            dt_physics=self.model.opt.timestep,
        )

        self.ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe"
        )
        self.target_mocap_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target"
        )
        self.target_mocap_index = self.model.body_mocapid[self.target_mocap_id]

        self.home_qpos = np.array(self.mapper.cfg["home_qpos"]["full"], dtype=np.float64)
        self.current_target_qpos = self.home_qpos.copy()

        # Arm joint limits for clipping
        self.arm_low = self.model.jnt_range[:5, 0].copy()
        self.arm_high = self.model.jnt_range[:5, 1].copy()

        # Workspace sampling bounds
        self.ws_x = [0.18, 0.36]
        self.ws_y = [-0.20, 0.20]
        self.ws_z = [0.08, 0.28]

        # Action: 5 arm joint position deltas in [-1, 1]
        self.max_arm_delta = 0.035  # ~2.0 degrees per policy step
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )

        # Observation:
        # qpos[:5] (5) + qvel[:5] (5) + target_pos (3) + ee_pos (3) +
        # ee_to_target_vec (3) + ee_to_target_dist (1) + consecutive_hold_ratio (1) = 21 dims
        obs_dim = 5 + 5 + 3 + 3 + 3 + 1 + 1
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )

        self.target_pos = np.array([0.25, 0.0, 0.15], dtype=np.float64)
        self.consecutive_hold_steps = 0
        self.current_step = 0
        self.prev_dist = None
        self.prev_action = np.zeros(5, dtype=np.float32)
        self.viewer = None

    def _get_obs(self):
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        ee_to_target_vec = self.target_pos - ee_pos
        dist = float(np.linalg.norm(ee_to_target_vec))
        hold_ratio = float(self.consecutive_hold_steps) / float(self.required_hold_steps)

        # Quantize joint position feedback to 12-bit magnetic encoder resolution
        quantized_qpos = self.actuators.quantize_encoder(self.data.qpos[:5])

        obs = np.concatenate([
            quantized_qpos,
            self.data.qvel[:5],
            self.target_pos,
            ee_pos,
            ee_to_target_vec,
            [dist],
            [hold_ratio],
        ]).astype(np.float32)

        return np.clip(obs, -10.0, 10.0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Reset robot to home configuration
        self.data.qpos[:6] = self.home_qpos.copy()
        self.data.qvel[:6] = 0.0
        self.current_target_qpos = self.home_qpos.copy()
        self.actuators.reset(self.home_qpos)

        # Sample randomized 3D reach target
        if options and "target_pos" in options:
            self.target_pos = np.asarray(options["target_pos"], dtype=np.float64)
        else:
            self.target_pos = np.array([
                self.np_random.uniform(self.ws_x[0], self.ws_x[1]),
                self.np_random.uniform(self.ws_y[0], self.ws_y[1]),
                self.np_random.uniform(self.ws_z[0], self.ws_z[1]),
            ], dtype=np.float64)

        # Update mocap body in MuJoCo
        self.data.mocap_pos[self.target_mocap_index] = self.target_pos.copy()
        self.data.ctrl[:6] = self.home_qpos.copy()

        mujoco.mj_forward(self.model, self.data)

        self.current_step = 0
        self.consecutive_hold_steps = 0
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        self.prev_dist = float(np.linalg.norm(self.target_pos - ee_pos))
        self.prev_action = np.zeros(5, dtype=np.float32)

        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action = np.clip(action, -1.0, 1.0)

        # Update desired joint positions via policy delta commands
        dq = action * self.max_arm_delta
        self.current_target_qpos[:5] = np.clip(
            self.current_target_qpos[:5] + dq,
            self.arm_low,
            self.arm_high,
        )
        # Gripper remains at home open state for pure reaching task
        self.current_target_qpos[5] = self.home_qpos[5]

        # Multi-substep physics execution with Feetech STS3215 dynamics
        for _ in range(self.control_substeps):
            torques, tracked = self.actuators.step(
                self.current_target_qpos,
                self.data.qpos[:6],
                self.data.qvel[:6],
            )
            # MuJoCo position actuator ctrl
            self.data.ctrl[:6] = self.current_target_qpos
            mujoco.mj_step(self.model, self.data)

        # Compute observations and task metrics
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        dist = float(np.linalg.norm(self.target_pos - ee_pos))
        self.current_step += 1

        # Check success condition (dist <= 1 cm)
        in_tolerance = dist <= self.success_dist
        if in_tolerance:
            self.consecutive_hold_steps += 1
        else:
            self.consecutive_hold_steps = 0

        success = self.consecutive_hold_steps >= self.required_hold_steps

        # -------------------------------------------------------------
        # Reward Engineering:
        # 1. Potential progress reward: (prev_dist - dist)
        # 2. Dense distance bonus: exp(-20 * dist)
        # 3. Holding reward: proportional to consecutive hold steps
        # 4. Action smoothness penalty: -0.01 * ||a_t - a_{t-1}||^2
        # 5. Terminal success bonus: +100.0
        # -------------------------------------------------------------
        progress = (self.prev_dist - dist) * 15.0
        dense_near = float(np.exp(-25.0 * dist))
        hold_reward = 0.5 * float(self.consecutive_hold_steps)
        smoothness_penalty = -0.02 * float(np.sum((action - self.prev_action) ** 2))

        reward = progress + dense_near + hold_reward + smoothness_penalty
        if success:
            reward += 100.0

        self.prev_dist = dist
        self.prev_action = action.copy()

        terminated = bool(success)
        truncated = bool(self.current_step >= self.max_episode_steps)

        info = {
            "success": success,
            "dist_mm": dist * 1000.0,
            "in_tolerance": in_tolerance,
            "hold_steps": self.consecutive_hold_steps,
            "action_delta": float(np.mean(np.abs(action - self.prev_action))),
            "current_step": self.current_step,
        }

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), float(reward), terminated, truncated, info

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
