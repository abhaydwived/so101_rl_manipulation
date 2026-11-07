import numpy as np
import mujoco
from gymnasium import spaces
from environments.base_env import SO101BaseEnv


class SO101ReachingEnv(SO101BaseEnv):
    """
    Part B1: Required Reaching Task Environment for SO-101 Manipulator.

    Observation Space (21-D continuous):
    - 5 arm joint positions (quantized to 12-bit magnetic encoder resolution)
    - 5 arm joint velocities
    - 3 end-effector 3D position
    - 3 target 3D position
    - 3 relative vector (target - end_effector)
    - 1 Euclidean distance to target
    - 1 hold progress ratio (consecutive_hold_steps / required_hold_steps)

    Action Space (5-D continuous):
    - 5 normalized arm joint position deltas in [-1, 1], scaled by max_arm_delta.

    Success Condition:
    - End-effector within 1 cm (0.01 m) of the random target.
    - Maintained for at least 0.5 seconds (13 consecutive policy steps at 25 Hz).
    """

    def __init__(
        self,
        scene_path: str = "so101/reach_scene.xml",
        config_path: str = "configs/so101_config.yaml",
        render_mode: str = None,
        max_episode_steps: int = 200,   # 8.0 seconds at 25 Hz allows steady approach + 0.5s hold + margin
        control_substeps: int = 20,     # 25 Hz policy control frequency
        success_dist: float = 0.010,    # 1 cm requirement
        hold_time_sec: float = 0.50,    # 0.5 second hold requirement
    ):
        super().__init__(
            scene_path=scene_path,
            config_path=config_path,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            control_substeps=control_substeps,
            max_arm_delta=0.035,        # 0.035 rad (~2.0 deg) provides optimal speed to reach in ~2.5s and settle
        )

        self.success_dist = success_dist
        self.required_hold_steps = int(np.ceil(hold_time_sec / (control_substeps * 0.002)))  # 13 steps

        self.target_mocap_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self.target_mocap_idx = self.model.body_mocapid[self.target_mocap_id]

        # Workspace sampling boundaries
        self.ws_x = [0.18, 0.36]
        self.ws_y = [-0.20, 0.20]
        self.ws_z = [0.08, 0.28]

        # Define 5-D Action Space
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )

        # Define 21-D Observation Space
        obs_dim = 5 + 5 + 3 + 3 + 3 + 1 + 1
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )

        print(f"[SO101ReachingEnv] Initialized successfully.")
        print(f"  -> Observation Space: Box{self.observation_space.shape} (21 dimensions)")
        print(f"  -> Action Space:      Box{self.action_space.shape} (5 arm joint deltas)")
        print(f"  -> Success Criteria:  <= {self.success_dist*100:.1f} cm held for >= {hold_time_sec} s ({self.required_hold_steps} steps)")

        self.target_pos = np.array([0.25, 0.0, 0.15], dtype=np.float64)
        self.consecutive_hold_steps = 0
        self.prev_dist = None
        self.prev_action = np.zeros(5, dtype=np.float32)

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

        return np.clip(obs, -10.0, 10.0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Reset arm to canonical home configuration
        self.data.qpos[:6] = self.home_qpos.copy()
        self.data.qvel[:6] = 0.0
        self.current_target_qpos = self.home_qpos.copy()
        self.actuators.reset(self.home_qpos)

        # Sample randomized 3D target coordinates
        if options and "target_pos" in options:
            self.target_pos = np.asarray(options["target_pos"], dtype=np.float64)
        else:
            self.target_pos = np.array([
                self.np_random.uniform(self.ws_x[0], self.ws_x[1]),
                self.np_random.uniform(self.ws_y[0], self.ws_y[1]),
                self.np_random.uniform(self.ws_z[0], self.ws_z[1]),
            ], dtype=np.float64)

        # Update visual mocap target in MuJoCo
        self.data.mocap_pos[self.target_mocap_idx] = self.target_pos.copy()
        self.data.ctrl[:6] = self.home_qpos.copy()
        mujoco.mj_forward(self.model, self.data)

        self.current_step = 0
        self.consecutive_hold_steps = 0
        self.prev_dist = float(np.linalg.norm(self.target_pos - self.get_ee_xpos()))
        self.prev_action = np.zeros(5, dtype=np.float32)

        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action = np.clip(action, -1.0, 1.0)

        # Apply position-controlled action (gripper remains open)
        self.apply_action(arm_action=action, gripper_close=False)
        self.current_step += 1

        ee_pos = self.get_ee_xpos()
        dist = float(np.linalg.norm(self.target_pos - ee_pos))

        # Check Part B1 success condition
        in_tolerance = dist <= self.success_dist
        if in_tolerance:
            self.consecutive_hold_steps += 1
        else:
            self.consecutive_hold_steps = 0

        success = self.consecutive_hold_steps >= self.required_hold_steps

        # -------------------------------------------------------------
        # Re-engineered Stable & Steady Reward Design:
        # 1. Linear distance penalty: -10 * dist (dense, non-saturating gradient everywhere)
        # 2. Continuous progress reward: 25 * (prev_dist - dist)
        # 3. Precision zone shaping: 5.0 / (1.0 + 100.0 * dist)
        # 4. Success holding bonus: +5.0 per step within 1 cm
        # 5. Settling damping: penalize velocity only when close to prevent overshoot
        # 6. Action smoothness penalty: -0.02 * ||a_t - a_{t-1}||^2
        # 7. Terminal success bonus: +150.0
        # -------------------------------------------------------------
        dist_penalty = -10.0 * dist
        progress = (self.prev_dist - dist) * 25.0
        precision_bonus = 5.0 / (1.0 + 100.0 * dist)

        # Holding bonus scaled by step count within 1 cm
        hold_reward = 5.0 * float(self.consecutive_hold_steps) if in_tolerance else 0.0

        # Gentle speed damping when within 3 cm to settle into hold
        arm_v_norm = float(np.linalg.norm(self.get_arm_qvel()))
        speed_penalty = -0.05 * arm_v_norm if dist < 0.030 else 0.0

        # Action smoothness penalty
        smoothness_penalty = -0.02 * float(np.sum((action - self.prev_action) ** 2))

        reward = dist_penalty + progress + precision_bonus + hold_reward + speed_penalty + smoothness_penalty
        if success:
            reward += 150.0

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

