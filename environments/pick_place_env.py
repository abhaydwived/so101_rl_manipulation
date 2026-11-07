import numpy as np
import mujoco
from gymnasium import spaces
from environments.base_env import SO101BaseEnv


class SO101PickPlaceEnv(SO101BaseEnv):
    """
    Part B2: Advanced Pick and Place Task Environment for SO-101 Manipulator.

    Observation Space (30-D continuous):
    - 5 arm joint positions (quantized to 12-bit magnetic encoder resolution)
    - 5 arm joint velocities
    - 1 gripper joint position
    - 3 end-effector 3D position
    - 3 cube 3D position
    - 4 cube orientation quaternion [w, x, y, z]
    - 3 target zone 3D position
    - 3 relative vector (cube - end_effector)
    - 3 relative vector (target - cube)

    Action Space (6-D continuous):
    - 5 normalized arm joint position deltas in [-1, 1], scaled by max_arm_delta.
    - 1 normalized gripper command: > 0 closes jaws (-0.10 rad), <= 0 opens jaws (1.00 rad).

    Success Condition:
    - Cube center within 2 cm (0.02 m) of target zone.
    - Cube successfully released (gripper open, not gripping).
    - Arm clear of the placed cube (gripper tip > 6 cm away).
    """

    def __init__(
        self,
        scene_path: str = "so101/pick_scene.xml",
        config_path: str = "configs/so101_config.yaml",
        render_mode: str = None,
        max_episode_steps: int = 200,   # 8.0 seconds at 25 Hz
        control_substeps: int = 20,     # 25 Hz policy control frequency
    ):
        super().__init__(
            scene_path=scene_path,
            config_path=config_path,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            control_substeps=control_substeps,
            max_arm_delta=0.035,
        )

        self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.target_mocap_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_zone")
        self.target_mocap_idx = self.model.body_mocapid[self.target_mocap_id]

        # Physical thresholds
        self.lift_height_thresh = 0.045     # lifted if cube_z > 4.5 cm
        self.target_radius_thresh = 0.020   # 2.0 cm placement accuracy
        self.arm_clear_thresh = 0.060       # 6.0 cm retreat clearance

        # Define 6-D Action Space: 5 arm deltas + 1 gripper command
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )

        # Define 30-D Observation Space
        obs_dim = 5 + 5 + 1 + 3 + 3 + 4 + 3 + 3 + 3
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )

        print(f"[SO101PickPlaceEnv] Initialized successfully.")
        print(f"  -> Observation Space: Box{self.observation_space.shape} (30 dimensions)")
        print(f"  -> Action Space:      Box{self.action_space.shape} (5 arm deltas + 1 gripper)")
        print(f"  -> Success Criteria:  Cube <= {self.target_radius_thresh*100:.1f} cm of target, released, and arm clear >= {self.arm_clear_thresh*100:.1f} cm")

        self.target_pos = np.array([0.28, -0.15, 0.015], dtype=np.float64)
        self.gripper_closed = False
        self.has_grasped_once = False
        self.has_reached_target_once = False
        self.has_released_once = False
        self.prev_ee_to_cube = None
        self.prev_cube_to_target = None
        self.prev_cube_z = None
        self.prev_action = np.zeros(6, dtype=np.float32)

    def _get_cube_xpos(self) -> np.ndarray:
        return self.data.xpos[self.cube_body_id].copy()

    def _get_cube_quat(self) -> np.ndarray:
        cube_jnt = int(self.model.body("cube").jntadr[0])
        adr = int(self.model.jnt_qposadr[cube_jnt])
        return self.data.qpos[adr + 3 : adr + 7].copy()

    def _get_obs(self) -> np.ndarray:
        qpos = self.get_arm_qpos()
        qvel = self.get_arm_qvel()
        gripper_q = self.get_gripper_qpos()
        ee_pos = self.get_ee_xpos()
        cube_pos = self._get_cube_xpos()
        cube_quat = self._get_cube_quat()
        ee_to_cube = cube_pos - ee_pos
        cube_to_tgt = self.target_pos - cube_pos

        obs = np.concatenate([
            qpos,
            qvel,
            [gripper_q],
            ee_pos,
            cube_pos,
            cube_quat,
            self.target_pos,
            ee_to_cube,
            cube_to_tgt,
        ]).astype(np.float32)

        return np.clip(obs, -10.0, 10.0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[:6] = self.home_qpos.copy()
        self.data.qvel[:6] = 0.0
        self.current_target_qpos = self.home_qpos.copy()
        self.actuators.reset(self.home_qpos)

        # 1. Randomize Cube Placement
        if options and "cube_xy" in options:
            cube_x, cube_y = options["cube_xy"]
        else:
            cube_x = self.np_random.uniform(0.20, 0.32)
            cube_y = self.np_random.uniform(0.04, 0.16)

        cube_jnt_adr = int(self.model.jnt_qposadr[self.model.body("cube").jntadr[0]])
        self.data.qpos[cube_jnt_adr : cube_jnt_adr + 3] = [cube_x, cube_y, 0.015]
        self.data.qpos[cube_jnt_adr + 3 : cube_jnt_adr + 7] = [1.0, 0.0, 0.0, 0.0]

        # 2. Randomize Target Zone Placement
        if options and "target_xy" in options:
            tgt_x, tgt_y = options["target_xy"]
        else:
            tgt_x = self.np_random.uniform(0.20, 0.32)
            tgt_y = self.np_random.uniform(-0.16, -0.04)

        self.target_pos = np.array([tgt_x, tgt_y, 0.015], dtype=np.float64)
        self.data.mocap_pos[self.target_mocap_idx] = np.array([tgt_x, tgt_y, 0.001])

        self.gripper_closed = False
        self.has_grasped_once = False
        self.has_reached_target_once = False
        self.has_released_once = False
        self.current_step = 0

        # Adaptive Reset Curriculum:
        roll = self.np_random.random()
        if roll < 0.25:
            # Stage 3 Curriculum: Spawn held above target zone
            self.data.qpos[cube_jnt_adr : cube_jnt_adr + 3] = [tgt_x, tgt_y, 0.07]
            self.data.qpos[:5] = [np.arctan2(tgt_y, tgt_x), 0.30, -0.50, 0.20, 0.0]
            self.data.qpos[5] = -0.10
            self.current_target_qpos[:6] = self.data.qpos[:6].copy()
            self.gripper_closed = True
            self.has_grasped_once = True
            self.has_reached_target_once = True
        elif roll < 0.55:
            # Stage 2 Curriculum: Spawn hovering right over cube
            self.data.qpos[:5] = [np.arctan2(cube_y, cube_x), 0.35, -0.55, 0.20, 0.0]
            self.data.qpos[5] = 1.0
            self.current_target_qpos[:6] = self.data.qpos[:6].copy()
            self.gripper_closed = False

        self.data.ctrl[:6] = self.current_target_qpos.copy()
        mujoco.mj_forward(self.model, self.data)

        ee_pos = self.get_ee_xpos()
        cube_pos = self._get_cube_xpos()
        self.prev_ee_to_cube = float(np.linalg.norm(cube_pos - ee_pos))
        self.prev_cube_to_target = float(np.linalg.norm(self.target_pos[:2] - cube_pos[:2]))
        self.prev_cube_z = float(cube_pos[2])
        self.prev_action = np.zeros(6, dtype=np.float32)

        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action = np.clip(action, -1.0, 1.0)

        arm_action = action[:5]
        gripper_close = bool(action[5] > 0.0)
        self.gripper_closed = gripper_close

        # Apply position-controlled action
        self.apply_action(arm_action=arm_action, gripper_close=gripper_close)
        self.current_step += 1

        ee_pos = self.get_ee_xpos()
        cube_pos = self._get_cube_xpos()
        ee_to_cube = float(np.linalg.norm(cube_pos - ee_pos))
        cube_to_tgt_xy = float(np.linalg.norm(self.target_pos[:2] - cube_pos[:2]))
        cube_z = float(cube_pos[2])

        # State evaluations
        is_lifted = cube_z > self.lift_height_thresh
        is_grasped = bool(self.gripper_closed and ee_to_cube < 0.040 and is_lifted)

        if is_grasped:
            self.has_grasped_once = True

        near_target = cube_to_tgt_xy <= self.target_radius_thresh
        if near_target and is_grasped:
            self.has_reached_target_once = True

        is_released = bool(
            self.has_grasped_once
            and (not self.gripper_closed)
            and near_target
            and cube_z < 0.035
        )
        if is_released:
            self.has_released_once = True

        is_clear = bool(self.has_released_once and ee_to_cube > self.arm_clear_thresh)
        success = bool(self.has_released_once and near_target and is_clear)

        # Multi-Phase Reward Calculation
        reward = 0.0

        # Phase 1: Reaching progress
        if not self.has_grasped_once:
            reach_progress = (self.prev_ee_to_cube - ee_to_cube) * 15.0
            reward += reach_progress
            if ee_to_cube < 0.035:
                reward += 1.0

        # Phase 2: Grasp and Lift progress
        if self.gripper_closed and ee_to_cube < 0.040:
            lift_gain = max(0.0, cube_z - self.prev_cube_z) * 30.0
            reward += lift_gain
            if is_lifted and not self.has_grasped_once:
                reward += 10.0

        # Phase 3: Transport progress (only when actively held)
        if is_grasped:
            transport_progress = (self.prev_cube_to_target - cube_to_tgt_xy) * 25.0
            reward += transport_progress
            if near_target:
                reward += 15.0

        # Phase 4: Settle & Release reward
        if self.has_reached_target_once and not self.gripper_closed:
            if near_target:
                reward += 25.0

        # Phase 5: Arm Clearance Retreat reward
        if self.has_released_once:
            clearance_gain = (ee_to_cube - self.prev_ee_to_cube) * 20.0
            reward += clearance_gain

        # Action smoothness penalty
        smoothness_penalty = -0.015 * float(np.sum((action - self.prev_action) ** 2))
        reward += smoothness_penalty

        # Terminal Success
        if success:
            reward += 120.0

        # Catastrophic departure
        if cube_z < -0.05 or np.linalg.norm(cube_pos[:2]) > 0.8:
            reward -= 20.0
            terminated = True
        else:
            terminated = bool(success)

        truncated = bool(self.current_step >= self.max_episode_steps)

        self.prev_ee_to_cube = ee_to_cube
        self.prev_cube_to_target = cube_to_tgt_xy
        self.prev_cube_z = cube_z
        self.prev_action = action.copy()

        info = {
            "success": success,
            "is_grasped": is_grasped,
            "is_released": self.has_released_once,
            "is_clear": is_clear,
            "cube_target_err_mm": cube_to_tgt_xy * 1000.0,
            "ee_cube_dist_mm": ee_to_cube * 1000.0,
            "cube_z_mm": cube_z * 1000.0,
            "action_delta": float(np.mean(np.abs(action - self.prev_action))),
            "current_step": self.current_step,
        }

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), float(reward), terminated, truncated, info
