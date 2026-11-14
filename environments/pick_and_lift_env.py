"""
Simple SO-101 pick-and-lift Gymnasium environment (v1.1 - hold fix).

Task:
  1. Reach the cube.
  2. Close the gripper until both jaws contact the cube.
  3. Lift the cube so its bottom is at least 5 cm above the table.
  4. Keep bilateral jaw contact and maintain that height for 1 second.

Changes vs v1 (all aimed at the "arm keeps going up, never holds" failure):
  A. Lift progress is CAPPED at the target height, so rising further pays nothing.
  B. Overshoot penalty above a hold band (5-7 cm) so "up and up" is actively bad.
  C. Hold reward is much larger (1.0/step) and is paid only while held.
  D. Stillness: a stronger action penalty once the cube is in the hold zone, so
     "stop moving" becomes a learnable, rewarded behaviour.
  E. Contact grace (3 policy steps): a 1-frame contact flicker no longer resets
     the 1 s hold counter or triggers the drop penalty.
  F. Gripper is latched closed while holding (exploration noise cannot open it).
  G. Observation gets hold progress (hold_steps / hold_steps_required) -> 26-D.
     Old checkpoints / VecNormalize stats are NOT compatible; retrain.

The reach reward is still disabled permanently once bilateral contact is confirmed.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

# This file is intended to live at <project>/environments/pick_and_lift_env.py.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common import configure_env
configure_env()

from src.so101_model_mapper import SO101ModelMapper
from src.feetech_actuator import FeetechSTS3215ActuatorGroup


# ------------------------------- Task setup -------------------------------
CONTROL_SUBSTEPS = 20                 # 20 * 0.002 s = 0.04 s policy interval
MAX_EPISODE_STEPS = 400
MAX_JOINT_DELTA = 0.035                # rad per policy step
HOLD_SECONDS = 1.0
LIFT_HEIGHT = 0.050                    # cube-bottom clearance above table
HOLD_BAND_TOP = 0.063                  # above this, overshoot penalty kicks in
PROGRESS_CAP = 0.060                   # lift progress reward stops here
GRASP_CONFIRM_STEPS = 2                # avoids a one-frame contact flicker
CONTACT_DISTANCE_TOL = 0.0015          # 1.5 mm; no contact-normal filtering
CONTACT_GRACE_STEPS = 3                # tolerated policy steps without contact
GRASP_MIN_UPRIGHT = 0.95               # |z-component| of the cube's most vertical axis
GRASP_MAX_EE_DIST = 0.04               # m, cube centre to EE site for a real grasp

# Small, interpretable rewards.
REACH_PROGRESS_SCALE = 10.0
GRASP_REWARD = 6.0
LIFT_PROGRESS_SCALE = 20.0
LIFT_SUCCESS_REWARD = 10.0
HOLD_STEP_REWARD = 1.0                 # was 0.25: hold must beat "keep climbing"
OVERSHOOT_SCALE = 30.0                 # per metre above HOLD_BAND_TOP, per step
SUCCESS_REWARD = 50.0
DROP_PENALTY = -2.0
STEP_PENALTY = -0.005
ACTION_PENALTY = 0.001
HOLD_ACTION_PENALTY = 0.5              # stillness incentive in the hold zone (was 0.05)
ACTION_RATE_PENALTY = 0.05             # penalty on (a_t - a_{t-1})^2, arm joints (was 0.01)
HOLD_RATE_PENALTY = 0.5                # same, inside the hold zone

# Anti-jitter physics (fix J).
ARMATURE_MIN = 0.01                    # kg m^2, floor on joint armature
DOF_DAMPING_MIN = 0.3                  # N m s/rad, floor on joint damping
ACT_KP = 250.0
ACT_KV = 20.0                          # was 12: more damping in the PD loop
GRIP_SQUEEZE_RAD = 0.01                # carrying: target = gripper q - this
CONTACT_IMPRATIO = 10.0                # stiffer friction cone -> less chatter

# Inference-time stabilizer (only used when stabilize_hold=True).
SMOOTH_ALPHA = 0.4                     # low-pass: 1.0 = no smoothing
SMOOTH_DEADZONE = 0.08                 # arm actions below this become exactly 0
FAILURE_PENALTY = -10.0
MAX_CUBE_SPEED = 2.0

REWARD_COMPONENTS = (
    "step_penalty", "reach", "proximity", "touch", "grasp", "grasp_hold",
    "lift", "lift_overshoot", "lift_bonus", "lift_hold", "drop",
    "action_reg", "success", "escape",
)


class SO101PickLiftEnv(gym.Env):
    """A deliberately simple, two-stage SO-101 pick-and-lift environment."""

    metadata = {"render_modes": ["human"], "render_fps": 25}

    def __init__(
        self,
        scene_path: str = "so101/pick_scene.xml",
        config_path: str = "configs/so101_config.yaml",
        render_mode: str | None = None,
        randomize_table_friction: bool = False,
        stabilize_hold: bool = False,
        freeze_in_hold: bool = True,
    ):
        super().__init__()

        if not os.path.isfile(scene_path):
            raise FileNotFoundError(
                f"MuJoCo scene not found: {scene_path}. Run from the project root "
                "or pass scene_path explicitly."
            )
        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"SO-101 config not found: {config_path}. The mapper needs this YAML "
                "for joint names, limits, home pose, and gripper open/closed positions."
            )

        self.scene_path = scene_path
        self.config_path = config_path
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        self.model.opt.impratio = CONTACT_IMPRATIO
        self.randomize_table_friction = bool(randomize_table_friction)
        self.stabilize_hold = bool(stabilize_hold)   # smooth + deadzone; eval/demo only
        self.freeze_in_hold = bool(freeze_in_hold)   # zero arm action while holding

        self.mapper = SO101ModelMapper(config_path, self.model)
        self.arm_qpos_idx = self.mapper.arm_qpos_indices
        self.gripper_qpos_idx = int(self.mapper.gripper_qpos_index)
        self.robot_qpos_idx = self.mapper.robot_qpos_indices
        self.arm_dof_idx = self.mapper.arm_dof_indices
        self.robot_dof_idx = self.mapper.robot_dof_indices
        self.robot_act_idx = self.mapper.robot_actuator_ids

        # Armature + damping floors on the robot joints (cube free joint untouched).
        self.model.dof_armature[self.robot_dof_idx] = np.maximum(
            self.model.dof_armature[self.robot_dof_idx], ARMATURE_MIN
        )
        self.model.dof_damping[self.robot_dof_idx] = np.maximum(
            self.model.dof_damping[self.robot_dof_idx], DOF_DAMPING_MIN
        )

        self.site_id = self._id(mujoco.mjtObj.mjOBJ_SITE,
                                self.mapper.frames_cfg["end_effector_site"])
        self.cube_body_id = self._id(mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.cube_geom_id = self._id(mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        cube_joint_id = int(self.model.body_jntadr[self.cube_body_id])
        if cube_joint_id < 0 or self.model.jnt_type[cube_joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("The cube body must have a free joint.")

        self.cube_qpos_adr = int(self.model.jnt_qposadr[cube_joint_id])
        self.cube_dof_adr = int(self.model.jnt_dofadr[cube_joint_id])

        self.table_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "table_top"
        )
        if self.table_geom_id < 0:
            raise ValueError("Scene must contain a geom named 'table_top'.")

        self.target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target_zone"
        )
        self.target_mocap_id = (
            int(self.model.body_mocapid[self.target_body_id])
            if self.target_body_id >= 0 else -1
        )

        self.fixed_jaw_geoms = self._geom_ids(
            ["fixed_jaw_base", "fixed_jaw_mid", "fixed_jaw_tip"]
        )
        self.moving_jaw_geoms = self._geom_ids(
            ["moving_jaw_base", "moving_jaw_mid", "moving_jaw_tip"]
        )
        self.fixed_jaw_set = set(self.fixed_jaw_geoms)
        self.moving_jaw_set = set(self.moving_jaw_geoms)

        mujoco.mj_forward(self.model, self.data)
        self.table_top_z = self._table_top_height()

        self.arm_low, self.arm_high, self.gripper_low, self.gripper_high = (
            self.mapper.get_joint_limits(self.model)
        )
        self.home_qpos = np.asarray(
            self.mapper.cfg["home_qpos"]["full"], dtype=np.float64
        ).copy()
        self.gripper_open_q = float(
            self.mapper.gripper_cfg.get("open_qpos", 1.0)
        )
        self.gripper_closed_q = float(
            self.mapper.gripper_cfg.get("closed_qpos", -0.1)
        )

        self.actuators = FeetechSTS3215ActuatorGroup(
            n_actuators=len(self.robot_act_idx),
            kp=ACT_KP,
            kv=ACT_KV,
            max_torque=2.94,
            max_velocity=5.5,
            dt_physics=float(self.model.opt.timestep),
        )

        self.policy_dt = float(self.model.opt.timestep * CONTROL_SUBSTEPS)
        self.hold_steps_required = max(1, int(np.ceil(HOLD_SECONDS / self.policy_dt)))
        self.max_episode_steps = MAX_EPISODE_STEPS
        self.render_mode = render_mode
        self.viewer = None

        # 26-D observation:
        # q(5), dq(5), EE xyz(3), cube xyz(3), cube linear velocity(3),
        # gripper q(1), EE-to-cube vector(3), phase(1), cube-bottom clearance(1),
        # hold progress(1).
        obs_low = np.concatenate([
            self.arm_low, -10*np.ones(5), -2*np.ones(3), -2*np.ones(3),
            -5*np.ones(3), [self.gripper_low], -2*np.ones(3), [0.0], [-0.05], [0.0]
        ]).astype(np.float32)
        obs_high = np.concatenate([
            self.arm_high, 10*np.ones(5), 2*np.ones(3), 2*np.ones(3),
            5*np.ones(3), [self.gripper_high], 2*np.ones(3), [1.0], [0.5], [1.0]
        ]).astype(np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)

        self.max_delta = MAX_JOINT_DELTA
        self._reset_task_state()

    def _id(self, obj_type, name):
        idx = mujoco.mj_name2id(self.model, obj_type, name)
        if idx < 0:
            raise ValueError(f"Required MuJoCo object '{name}' not found.")
        return int(idx)

    def _geom_ids(self, names):
        return [self._id(mujoco.mjtObj.mjOBJ_GEOM, name) for name in names]

    def _reset_task_state(self):
        self.current_step = 0
        self.phase = 0                   # 0 = reach/grasp, 1 = lift/hold
        self.gripper_closed = False
        self.bilateral_steps = 0
        self.has_grasped_once = False
        self.has_reached_lift = False
        self.hold_steps = 0
        self.contact_lost_steps = 0      # consecutive steps without bilateral contact
        self.was_held = False
        self._smooth_prev = np.zeros(5, dtype=np.float32)
        self._freeze_arm = False
        self.prev_distance = 0.0
        self.prev_clearance = 0.0
        self.prev_action = np.zeros(6, dtype=np.float32)

    def _table_top_height(self):
        pos = self.data.geom_xpos[self.table_geom_id]
        rot = self.data.geom_xmat[self.table_geom_id].reshape(3, 3)
        half = self.model.geom_size[self.table_geom_id]
        return float(pos[2] + np.dot(np.abs(rot[2, :]), half))

    def _cube_clearance(self):
        pos = self.data.geom_xpos[self.cube_geom_id]
        rot = self.data.geom_xmat[self.cube_geom_id].reshape(3, 3)
        half = self.model.geom_size[self.cube_geom_id]
        vertical_half_extent = float(np.dot(np.abs(rot[2, :]), half))
        return float(pos[2] - vertical_half_extent - self.table_top_z)

    def _cube_upright(self):
        """1.0 when any cube face is flat to the table, ~0.58 at worst tilt."""
        rot = self.data.xmat[self.cube_body_id].reshape(3, 3)
        return float(np.max(np.abs(rot[2, :])))

    def _cube_position(self):
        return self.data.xpos[self.cube_body_id].copy()

    def _cube_linear_velocity(self):
        return self.data.qvel[self.cube_dof_adr:self.cube_dof_adr + 3].copy()

    def _cube_speed(self):
        return float(np.linalg.norm(self._cube_linear_velocity()))

    def _ee_position(self):
        return self.data.site_xpos[self.site_id].copy()

    def _distance_to_cube(self):
        return float(np.linalg.norm(self._cube_position() - self._ee_position()))

    def _check_bilateral_contact(self):
        """True when each jaw has at least one contact with the cube."""
        fixed = False
        moving = False
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.dist > CONTACT_DISTANCE_TOL:
                continue
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if g1 == self.cube_geom_id:
                other = g2
            elif g2 == self.cube_geom_id:
                other = g1
            else:
                continue

            if other in self.fixed_jaw_set:
                fixed = True
            elif other in self.moving_jaw_set:
                moving = True

            if fixed and moving:
                return True
        return False

    def _set_cube_pose(self, xyz, yaw):
        self.data.qpos[self.cube_qpos_adr:self.cube_qpos_adr + 3] = xyz
        self.data.qpos[self.cube_qpos_adr + 3:self.cube_qpos_adr + 7] = [
            np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)
        ]

    def _randomize_cube(self):
        # 2-3 cm edge length (nominal 2.5 cm, +/-20%).
        half = float(self.np_random.uniform(0.010, 0.015))
        mass = float(self.np_random.uniform(0.025, 0.045))
        self.cube_half_size = half
        self.cube_mass = mass
        self.cube_friction = float(self.np_random.uniform(1.8, 3.2))

        self.model.geom_size[self.cube_geom_id] = np.array([half, half, half])
        self.model.geom_rbound[self.cube_geom_id] = np.sqrt(3.0) * half
        self.model.geom_aabb[self.cube_geom_id] = np.array([0, 0, 0, half, half, half])
        self.model.body_mass[self.cube_body_id] = mass
        edge = 2.0 * half
        self.model.body_inertia[self.cube_body_id] = np.full(
            3, mass * edge * edge / 6.0
        )
        self.model.geom_friction[self.cube_geom_id][0] = self.cube_friction

        if self.randomize_table_friction:
            self.table_friction = float(self.np_random.uniform(0.8, 1.6))
            self.model.geom_friction[self.table_geom_id][0] = self.table_friction
        else:
            self.table_friction = float(self.model.geom_friction[self.table_geom_id][0])

        mujoco.mj_setConst(self.model, self.data)

    def _observation(self):
        ee = self._ee_position()
        cube = self._cube_position()
        rel = cube - ee
        hold_progress = min(1.0, self.hold_steps / float(self.hold_steps_required))
        obs = np.concatenate([
            self.data.qpos[self.arm_qpos_idx],
            self.data.qvel[self.arm_dof_idx],
            ee,
            cube,
            self._cube_linear_velocity(),
            [self.data.qpos[self.gripper_qpos_idx]],
            rel,
            [float(self.phase)],
            [self._cube_clearance()],
            [hold_progress],
        ]).astype(np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _physics_step(self):
        for _ in range(CONTROL_SUBSTEPS):
            q = self.data.qpos[self.robot_qpos_idx]
            qd = self.data.qvel[self.robot_dof_idx]
            torque, _ = self.actuators.step(self.desired_qpos, q, qd)
            self.data.ctrl[self.robot_act_idx] = torque
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._randomize_cube()

        self.data.qpos[self.robot_qpos_idx] = self.home_qpos
        self.data.qvel[self.robot_dof_idx] = 0.0
        self.desired_qpos = self.home_qpos.copy()
        self.desired_qpos[5] = self.gripper_open_q
        self.data.qpos[self.gripper_qpos_idx] = self.gripper_open_q
        self.gripper_closed = False

        # Randomized reachable table positions and yaw.
        x = float(self.np_random.uniform(0.24, 0.30))
        y = float(self.np_random.uniform(-0.09, 0.09))
        self._set_cube_pose(
            [x, y, self.table_top_z + self.cube_half_size],
            float(self.np_random.uniform(-np.pi, np.pi)),
        )

        if self.target_mocap_id >= 0:
            self.data.mocap_pos[self.target_mocap_id] = [
                x, y, self.table_top_z + self.cube_half_size + LIFT_HEIGHT
            ]

        mujoco.mj_forward(self.model, self.data)
        self.actuators.reset(self.data.qpos[self.robot_qpos_idx].copy())
        self._reset_task_state()
        self.prev_distance = self._distance_to_cube()
        self.prev_clearance = self._cube_clearance()
        return self._observation(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (6,):
            raise ValueError(f"Expected action shape (6,), received {action.shape}.")
        action = np.clip(action, -1.0, 1.0)

        # Optional inference-time stabilizer: smooth + deadzone the arm action and
        # freeze the arm while the cube is held at/above the lift height. A zero
        # arm action leaves desired_qpos unchanged, so the arm stays still.
        if self.stabilize_hold:
            a = SMOOTH_ALPHA * action[:5] + (1.0 - SMOOTH_ALPHA) * self._smooth_prev
            a = np.where(np.abs(a) < SMOOTH_DEADZONE, 0.0, a)
            self._smooth_prev = a.astype(np.float32)
            action[:5] = self._smooth_prev

        # Hard freeze: while the cube is held at/above the lift height the arm
        # target stops moving, so the 1 s hold cannot be broken by action jitter.
        # It releases automatically if the cube drops below the lift height or
        # contact is lost beyond the grace window.
        if self.freeze_in_hold and self._freeze_arm:
            action[:5] = 0.0
            self._smooth_prev[:] = 0.0

        # Arm action: small joint target increments.
        self.desired_qpos[:5] = np.clip(
            self.desired_qpos[:5] + action[:5] * self.max_delta,
            self.arm_low, self.arm_high
        )

        # Gripper: stateful command. While carrying (phase 1 with contact), the
        # gripper is latched closed so exploration noise cannot drop the cube.
        # If contact has been lost for longer than the grace window, it is freed
        # again so the policy can re-open / re-grasp.
        carrying = self.phase == 1 and self.contact_lost_steps <= CONTACT_GRACE_STEPS
        if carrying:
            self.gripper_closed = True
        elif action[5] > 0.15:
            self.gripper_closed = True
        elif action[5] < -0.15:
            self.gripper_closed = False
        grip_target = (
            self.gripper_closed_q if self.gripper_closed else self.gripper_open_q
        )
        if carrying:
            # Bounded squeeze: command just beyond the current jaw position instead
            # of the fully closed target, so grip force stays steady, not saturated.
            grip_target = max(
                self.gripper_closed_q,
                float(self.data.qpos[self.gripper_qpos_idx]) - GRIP_SQUEEZE_RAD,
            )
        self.desired_qpos[5] = grip_target

        self._physics_step()
        reward, terminated, info = self._calculate_reward(action)
        self._freeze_arm = bool(
            info["held_with_grace"] and info["lift_height_threshold_reached"]
        )

        self.current_step += 1
        truncated = self.current_step >= self.max_episode_steps
        if self.render_mode == "human":
            self.render()

        self.prev_action = action.copy()
        return self._observation(), float(reward), bool(terminated), bool(truncated), info

    def _calculate_reward(self, action):
        reward_parts = {name: 0.0 for name in REWARD_COMPONENTS}
        reward_parts["step_penalty"] = STEP_PENALTY

        distance = self._distance_to_cube()
        clearance = self._cube_clearance()
        cube_pos = self._cube_position()
        cube_speed = self._cube_speed()
        bilateral = self._check_bilateral_contact()
        upright = self._cube_upright()
        # A real grasp: both jaws on the cube, cube not tilted/wedged, cube centred
        # near the EE site. Used only to enter phase 1 (not for the hold grace).
        proper_grasp = bool(
            bilateral and upright >= GRASP_MIN_UPRIGHT and distance <= GRASP_MAX_EE_DIST
        )

        # Contact bookkeeping with a short grace window (fix E).
        if bilateral:
            self.contact_lost_steps = 0
        else:
            self.contact_lost_steps += 1
        held = self.contact_lost_steps <= CONTACT_GRACE_STEPS

        in_hold_zone = False

        # Stage 1: dense reaching reward. It stops permanently at first bilateral contact.
        if self.phase == 0:
            if proper_grasp:
                self.bilateral_steps += 1
            else:
                reward_parts["reach"] = REACH_PROGRESS_SCALE * (
                    self.prev_distance - distance
                )
                self.bilateral_steps = 0

            if self.bilateral_steps >= GRASP_CONFIRM_STEPS:
                self.phase = 1
                self.has_grasped_once = True
                reward_parts["grasp"] = GRASP_REWARD
                self.hold_steps = 0
        else:
            # Stage 2: no reaching reward is ever paid again.
            if held:
                # (A) Lift progress counts only up to PROGRESS_CAP, so going higher
                # than needed yields nothing.
                capped_now = min(clearance, PROGRESS_CAP)
                capped_prev = min(self.prev_clearance, PROGRESS_CAP)
                reward_parts["lift"] = LIFT_PROGRESS_SCALE * (capped_now - capped_prev)

                if clearance >= LIFT_HEIGHT and not self.has_reached_lift:
                    self.has_reached_lift = True
                    reward_parts["lift_bonus"] = LIFT_SUCCESS_REWARD

                if clearance >= LIFT_HEIGHT:
                    in_hold_zone = True
                    self.hold_steps += 1
                    # (C) Strong per-step hold reward.
                    reward_parts["lift_hold"] = HOLD_STEP_REWARD
                    # (B) Climbing beyond the hold band is penalised.
                    if clearance > HOLD_BAND_TOP:
                        reward_parts["lift_overshoot"] = -OVERSHOOT_SCALE * (
                            clearance - HOLD_BAND_TOP
                        )
                else:
                    self.hold_steps = 0
            else:
                self.hold_steps = 0
                # One small penalty on losing the grasp, not every subsequent step.
                if self.was_held:
                    reward_parts["drop"] = DROP_PENALTY

        # (D) Tiny regularizer normally; stronger once the cube is in the hold zone
        # so that "stop moving the arm" is rewarded.
        act_pen = HOLD_ACTION_PENALTY if in_hold_zone else ACTION_PENALTY
        rate_pen = HOLD_RATE_PENALTY if in_hold_zone else ACTION_RATE_PENALTY
        # (H) Action-rate penalty: prev_action is still last step's action here.
        action_rate = float(np.sum((action[:5] - self.prev_action[:5]) ** 2))
        reward_parts["action_reg"] = (
            -act_pen * float(np.sum(action[:5] ** 2))
            - rate_pen * action_rate
        )

        success = self.hold_steps >= self.hold_steps_required
        terminated = False
        if success:
            reward_parts["success"] = SUCCESS_REWARD
            terminated = True

        # Basic safety termination for a cube that is flung away or falls far below
        # the table. This is not part of the grasp/hold success condition.
        escaped = (
            cube_pos[2] < self.table_top_z - 0.10
            or np.linalg.norm(cube_pos[:2] - np.array([0.25, 0.0])) > 0.70
            or cube_speed > MAX_CUBE_SPEED
        )
        if escaped and not success:
            reward_parts["escape"] = FAILURE_PENALTY
            terminated = True

        self.prev_distance = distance
        self.prev_clearance = clearance
        self.was_held = bool(self.phase == 1 and held)

        total_reward = float(sum(reward_parts.values()))
        info = {
            "success": bool(success),
            "task_stage": int(self.phase + 1),
            "bilateral_contact": bool(bilateral),
            "proper_grasp": bool(proper_grasp),
            "cube_upright": float(upright),
            "has_grasped_once": bool(self.has_grasped_once),
            "is_holding": bool(bilateral),
            "held_with_grace": bool(self.phase == 1 and held),
            "contact_lost_steps": int(self.contact_lost_steps),
            "lift_height": float(clearance),
            "cube_bottom_clearance": float(clearance),
            "lift_target_achieved": bool(clearance >= LIFT_HEIGHT and bilateral),
            "lift_height_threshold_reached": bool(clearance >= LIFT_HEIGHT),
            "stable_hold": bool(clearance >= LIFT_HEIGHT and bilateral),
            "consec_lift_hold_steps": int(self.hold_steps),
            "hold_duration_steps": int(self.hold_steps_required),
            "cube_linear_velocity": float(cube_speed),
            "cube_angular_velocity": float(np.linalg.norm(
                self.data.qvel[self.cube_dof_adr + 3:self.cube_dof_adr + 6]
            )),
            "arm_joint_velocity_norm": float(np.linalg.norm(self.data.qvel[self.arm_dof_idx])),
            "hold_fail_cube_lin_vel": False,
            "hold_fail_cube_ang_vel": False,
            "hold_fail_arm_joint_vel": False,
            "policy_dt": float(self.policy_dt),
            "cube_half_size": float(self.cube_half_size),
            "cube_mass": float(self.cube_mass),
            "cube_friction": float(self.cube_friction),
            "table_friction": float(self.table_friction),
            "distance_ee_to_obj": float(distance),
            "obj_height": float(cube_pos[2]),
            "cube_target_err_mm": float(max(0.0, (LIFT_HEIGHT - clearance) * 1000.0)),
            "stage_lifted": bool(clearance >= LIFT_HEIGHT and bilateral),
            "cube_pos": cube_pos.tolist(),
            "current_step": int(self.current_step),
            "gripper_qpos": float(self.data.qpos[self.gripper_qpos_idx]),
            "gripper_closed": bool(self.gripper_closed),
            "gripper_command": bool(self.gripper_closed),
            "reward_total": total_reward,
        }

        # Keep the info keys expected by the supplied training/evaluation callback.
        info["reward_reaching"] = reward_parts["reach"]
        info["reward_grasp"] = reward_parts["grasp"]
        info["reward_touch"] = reward_parts["touch"]
        info["reward_proximity"] = reward_parts["proximity"]
        info["reward_grasp_hold"] = reward_parts["grasp_hold"]
        info["reward_lift"] = reward_parts["lift"]
        info["reward_lift_bonus"] = reward_parts["lift_bonus"]
        info["reward_lift_hold"] = reward_parts["lift_hold"]
        info["reward_drop"] = reward_parts["drop"]
        info["reward_step_penalty"] = reward_parts["step_penalty"]
        info["reward_lift_overshoot"] = reward_parts["lift_overshoot"]
        info["reward_action_reg"] = reward_parts["action_reg"]
        info["reward_escape"] = reward_parts["escape"]
        info["reward_success"] = reward_parts["success"]
        return total_reward, terminated, info

    def render(self):
        if self.render_mode != "human":
            return
        import mujoco.viewer
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None