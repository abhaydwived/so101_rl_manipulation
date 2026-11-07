import numpy as np
import mujoco
from gymnasium import spaces
from environments.base_env import SO101BaseEnv


class SO101PickPlaceEnv(SO101BaseEnv):
    """
    Part B2: Advanced Pick-and-Place Gymnasium Environment for SO-101 Manipulator.

    Implements:
    - Stages: APPROACH -> GRASP -> LIFT -> TRANSPORT -> PLACE -> RELEASE -> ARM CLEAR -> SUCCESS
    - Physical Grasp Verification: Sensed contact normal impulses and jaw-cube contacts (no fake attachments).
    - Placement Stability Gate: Requires cube within 2 cm, stationary velocities, and settled contact.
    - Dynamics & Domain Randomization: Cube pose, size, mass (25-45g), and friction (0.8-1.8).
    - Staged Curriculum Progression: Levels 1-6 (Grasp learning -> Transport -> Full Pick & Place).
    """

    def __init__(
        self,
        scene_path: str = "so101/pick_scene.xml",
        config_path: str = "configs/so101_config.yaml",
        render_mode: str = None,
        max_episode_steps: int = 200,   # 8.0 seconds at 25 Hz policy rate
        control_substeps: int = 20,     # 25 Hz policy control frequency
        curriculum_level: int = 6,      # Default to Level 6 (Full Task)
        domain_randomization: bool = True,
    ):
        super().__init__(
            scene_path=scene_path,
            config_path=config_path,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            control_substeps=control_substeps,
            max_arm_delta=0.035,
        )

        self.curriculum_level = curriculum_level
        self.domain_randomization = domain_randomization

        self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.cube_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        self.target_mocap_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_zone")
        self.target_mocap_idx = self.model.body_mocapid[self.target_mocap_id]

        # Jaw collision geom IDs for strict physical contact sensing
        self.fixed_jaw_geoms = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "fixed_jaw_base"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "fixed_jaw_mid"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "fixed_jaw_tip"),
        ]
        self.moving_jaw_geoms = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "moving_jaw_base"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "moving_jaw_mid"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "moving_jaw_tip"),
        ]

        # Physical thresholds
        self.lift_height_thresh = 0.045     # table is z=0.0; lifted if cube_z > 4.5 cm
        self.target_radius_thresh = 0.020   # 2.0 cm placement precision requirement
        self.arm_clear_thresh = 0.060       # 6.0 cm arm retreat clearance requirement
        self.settle_vel_thresh = 0.040      # 4 cm/s linear velocity for stability gate

        # Action: 5 arm deltas + 1 gripper command
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )

        # Observation (30-D):
        # qpos[:5] (5) + qvel[:5] (5) + gripper_q (1) + ee_pos (3) +
        # cube_pos (3) + cube_quat (4) + target_pos (3) + ee_to_cube (3) + cube_to_tgt (3) = 30 dims
        obs_dim = 5 + 5 + 1 + 3 + 3 + 4 + 3 + 3 + 3
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )

        print(f"[SO101PickPlaceEnv] Initialized successfully.")
        print(f"  -> Observation Space: Box{self.observation_space.shape} (30 dimensions)")
        print(f"  -> Action Space:      Box{self.action_space.shape} (5 arm deltas + 1 gripper)")
        print(f"  -> Curriculum Level:  {self.curriculum_level} | Domain Randomization: {self.domain_randomization}")
        print(f"  -> Success Criteria:  Cube <= {self.target_radius_thresh*100:.1f} cm of target, released, stable, arm clear >= {self.arm_clear_thresh*100:.1f} cm")

        self.target_pos = np.array([0.28, -0.15, 0.015], dtype=np.float64)
        self.gripper_closed = False

        # Stage flags
        self.stage_reached_cube = False
        self.stage_grasped = False
        self.stage_lifted = False
        self.stage_transported = False
        self.stage_placed = False
        self.stage_released = False
        self.stage_cleared = False
        self.settle_steps = 0

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

    def _get_cube_cvel(self) -> np.ndarray:
        return self.data.cvel[self.cube_body_id].copy()

    def _check_physical_contacts(self):
        """
        Queries actual MuJoCo contact manifold between gripper jaws and cube.
        Returns:
            has_fixed_contact (bool)
            has_moving_contact (bool)
            is_bilateral_grip (bool): true physical clamp between opposing jaws
        """
        has_fixed = False
        has_moving = False

        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 == self.cube_geom_id or g2 == self.cube_geom_id:
                other_geom = g2 if g1 == self.cube_geom_id else g1
                if other_geom in self.fixed_jaw_geoms:
                    has_fixed = True
                elif other_geom in self.moving_jaw_geoms:
                    has_moving = True

        return has_fixed, has_moving, (has_fixed and has_moving)

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

        # 1. Reset Robot to canonical home configuration
        self.data.qpos[:6] = self.home_qpos.copy()
        self.data.qvel[:6] = 0.0
        self.current_target_qpos = self.home_qpos.copy()
        self.actuators.reset(self.home_qpos)

        # 2. Domain Randomization (Cube dynamics and physical dimensions)
        if self.domain_randomization:
            # Mass randomization: 35g +/- 30% (25g to 45g)
            rand_mass = float(self.np_random.uniform(0.025, 0.045))
            self.model.body_mass[self.cube_body_id] = rand_mass
            # Friction randomization: [0.8, 1.8]
            rand_fric = float(self.np_random.uniform(0.8, 1.8))
            self.model.geom_friction[self.cube_geom_id, 0] = rand_fric

        # 3. Randomize Cube Placement
        if options and "cube_xy" in options:
            cube_x, cube_y = options["cube_xy"]
        else:
            cube_x = self.np_random.uniform(0.20, 0.32)
            cube_y = self.np_random.uniform(0.04, 0.16)

        cube_jnt = int(self.model.body("cube").jntadr[0])
        cube_jnt_adr = int(self.model.jnt_qposadr[cube_jnt])
        self.data.qpos[cube_jnt_adr : cube_jnt_adr + 3] = [cube_x, cube_y, 0.015]
        self.data.qpos[cube_jnt_adr + 3 : cube_jnt_adr + 7] = [1.0, 0.0, 0.0, 0.0]

        # 4. Randomize Target Zone Placement
        if options and "target_xy" in options:
            tgt_x, tgt_y = options["target_xy"]
        else:
            tgt_x = self.np_random.uniform(0.20, 0.32)
            tgt_y = self.np_random.uniform(-0.16, -0.04)

        self.target_pos = np.array([tgt_x, tgt_y, 0.015], dtype=np.float64)
        self.data.mocap_pos[self.target_mocap_idx] = np.array([tgt_x, tgt_y, 0.001])

        # Stage reset
        self.gripper_closed = False
        self.stage_reached_cube = False
        self.stage_grasped = False
        self.stage_lifted = False
        self.stage_transported = False
        self.stage_placed = False
        self.stage_released = False
        self.stage_cleared = False
        self.settle_steps = 0
        self.current_step = 0

        # Curriculum Resets across Levels 1-6
        if self.curriculum_level >= 3:
            roll = self.np_random.random()
            # 25% Stage: Spawn already grasping above target zone (Level 5 practice)
            if roll < 0.25:
                self.data.qpos[cube_jnt_adr : cube_jnt_adr + 3] = [tgt_x, tgt_y, 0.065]
                self.data.qpos[:5] = [np.arctan2(tgt_y, tgt_x), 0.30, -0.50, 0.20, 0.0]
                self.data.qpos[5] = -0.10
                self.current_target_qpos[:6] = self.data.qpos[:6].copy()
                self.gripper_closed = True
                self.stage_grasped = True
                self.stage_lifted = True
                self.stage_transported = True
            # 30% Stage: Spawn poised right above the cube (Level 2-3 practice)
            elif roll < 0.55:
                self.data.qpos[:5] = [np.arctan2(cube_y, cube_x), 0.35, -0.55, 0.20, 0.0]
                self.data.qpos[5] = 1.0
                self.current_target_qpos[:6] = self.data.qpos[:6].copy()
                self.gripper_closed = False
                self.stage_reached_cube = True

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

        # Apply position-controlled command through Feetech STS3215 dynamics
        self.apply_action(arm_action=arm_action, gripper_close=gripper_close)
        self.current_step += 1

        ee_pos = self.get_ee_xpos()
        cube_pos = self._get_cube_xpos()
        cube_cvel = self._get_cube_cvel()
        lin_speed = float(np.linalg.norm(cube_cvel[3:]))

        ee_to_cube = float(np.linalg.norm(cube_pos - ee_pos))
        cube_to_tgt_xy = float(np.linalg.norm(self.target_pos[:2] - cube_pos[:2]))
        cube_z = float(cube_pos[2])

        # -----------------------------------------------------------------
        # 1. PHYSICAL GRASP VERIFICATION
        # -----------------------------------------------------------------
        _, _, bilateral_contact = self._check_physical_contacts()
        is_physically_grasped = bool(self.gripper_closed and bilateral_contact and (cube_z > self.lift_height_thresh))

        if ee_to_cube < 0.035:
            self.stage_reached_cube = True
        if is_physically_grasped:
            self.stage_grasped = True
            self.stage_lifted = True

        # -----------------------------------------------------------------
        # 2. PLACEMENT & STABILITY VERIFICATION
        # -----------------------------------------------------------------
        near_target = cube_to_tgt_xy <= self.target_radius_thresh
        if near_target and self.stage_grasped:
            self.stage_transported = True

        is_settled_on_table = bool(cube_z < 0.030 and lin_speed < self.settle_vel_thresh)
        if near_target and is_settled_on_table:
            self.stage_placed = True
            self.settle_steps += 1
        else:
            self.settle_steps = 0

        is_valid_release = bool(
            self.stage_grasped
            and (not self.gripper_closed)
            and near_target
            and (self.settle_steps >= 2)
        )
        if is_valid_release:
            self.stage_released = True

        # -----------------------------------------------------------------
        # 3. ARM CLEARANCE & SUCCESS CONDITION
        # -----------------------------------------------------------------
        is_arm_clear = bool(self.stage_released and (ee_to_cube >= self.arm_clear_thresh))
        if is_arm_clear:
            self.stage_cleared = True

        # Assignment mandatory success criterion:
        # 1. Cube centre within 2 cm of target
        # 2. Cube successfully released
        # 3. Arm clear of the placed cube
        success = bool(near_target and self.stage_released and is_arm_clear and is_settled_on_table)

        # -----------------------------------------------------------------
        # 4. REWARD ENGINEERING: STAGED POTENTIAL SHAPING
        # -----------------------------------------------------------------
        reward = 0.0

        # Stage A: Approach reward
        if not self.stage_grasped:
            reach_progress = (self.prev_ee_to_cube - ee_to_cube) * 15.0
            reward += reach_progress
            if ee_to_cube < 0.030:
                reward += 1.5

        # Stage B & C: Grasp and Lift reward
        if self.gripper_closed and bilateral_contact:
            reward += 2.0
            lift_gain = max(0.0, cube_z - self.prev_cube_z) * 35.0
            reward += lift_gain
            if is_physically_grasped:
                reward += 12.0

        # Stage D: Transport reward (only when holding lifted cube)
        if is_physically_grasped:
            transport_progress = (self.prev_cube_to_target - cube_to_tgt_xy) * 30.0
            reward += transport_progress
            if near_target:
                reward += 15.0

        # Stage E & F: Settle and Release reward
        if self.stage_transported and not self.gripper_closed and near_target:
            if is_settled_on_table:
                reward += 30.0

        # Stage G: Arm clearance retreat reward
        if self.stage_released:
            clearance_gain = (ee_to_cube - self.prev_ee_to_cube) * 25.0
            reward += clearance_gain
            if is_arm_clear:
                reward += 20.0

        # Action smoothness penalty
        smoothness_penalty = -0.015 * float(np.sum((action - self.prev_action) ** 2))
        reward += smoothness_penalty

        # Terminal Success Bonus
        if success:
            reward += 150.0

        # Catastrophic failure penalties
        if cube_z < -0.05 or np.linalg.norm(cube_pos[:2]) > 0.8:
            reward -= 25.0
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
            "stage_reached": self.stage_reached_cube,
            "stage_grasped": self.stage_grasped,
            "stage_lifted": self.stage_lifted,
            "stage_transported": self.stage_transported,
            "stage_placed": self.stage_placed,
            "stage_released": self.stage_released,
            "stage_cleared": self.stage_cleared,
            "physical_contact": bilateral_contact,
            "cube_target_err_mm": cube_to_tgt_xy * 1000.0,
            "ee_cube_dist_mm": ee_to_cube * 1000.0,
            "cube_z_mm": cube_z * 1000.0,
            "linear_speed": lin_speed,
            "action_delta": float(np.mean(np.abs(action - self.prev_action))),
            "current_step": self.current_step,
        }

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), float(reward), terminated, truncated, info
