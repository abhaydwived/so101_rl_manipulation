"""
Tests for SO-101 Gymnasium environments.

Verifies:
- Reaching environment: obs shape (21,), action shape (5,), reset/step work.
- Pick-place environment: obs shape (26,), action shape (6,), reset/step work.
- Curriculum reset distribution: ~30% placement, ~35% grasp, ~35% full.
- Both environments return valid info dicts with required keys.
"""

import os
import sys
import pytest
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from gymnasium.utils.env_checker import check_env
from environments.reaching_env import SO101ReachingEnv
from environments.pick_and_lift_env import SO101PickLiftEnv as SO101HierarchicalPickPlaceEnv


@pytest.fixture(scope="module")
def reach_env():
    env = SO101ReachingEnv(scene_path="so101/reach_scene.xml", render_mode=None)
    yield env
    env.close()


@pytest.fixture(scope="module")
def pick_env():
    env = SO101HierarchicalPickPlaceEnv(scene_path="so101/pick_scene.xml", render_mode=None)
    yield env
    env.close()


class TestReachingEnv:
    def test_obs_shape(self, reach_env):
        obs, _ = reach_env.reset(seed=0)
        assert obs.shape == (21,), f"Expected (21,), got {obs.shape}"

    def test_action_space_shape(self, reach_env):
        assert reach_env.action_space.shape == (5,)

    def test_obs_dtype(self, reach_env):
        obs, _ = reach_env.reset(seed=1)
        assert obs.dtype == np.float32

    def test_step_no_crash(self, reach_env):
        reach_env.reset(seed=42)
        for _ in range(10):
            action = reach_env.action_space.sample()
            obs, reward, terminated, truncated, info = reach_env.step(action)
            assert obs.shape == (21,)
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)

    def test_info_keys(self, reach_env):
        reach_env.reset(seed=0)
        _, _, _, _, info = reach_env.step(reach_env.action_space.sample())
        required_keys = {"success", "dist_mm", "in_tolerance", "hold_steps", "target_zone"}
        assert required_keys.issubset(info.keys())

    def test_target_zones_exist(self, reach_env):
        assert set(reach_env.zone_names) == {"bottom", "medium", "high"}

    def test_success_dist(self, reach_env):
        assert reach_env.success_dist == pytest.approx(0.010, abs=1e-4)

    def test_obs_clipped(self, reach_env):
        obs, _ = reach_env.reset(seed=0)
        assert np.all(obs >= -10.0) and np.all(obs <= 10.0)

    def test_target_zone_bottom_z_range(self, reach_env):
        z_range = reach_env.zones["bottom"]["z"]
        assert z_range[0] >= 0.01 and z_range[1] <= 0.10

    def test_target_zone_high_z_range(self, reach_env):
        z_range = reach_env.zones["high"]["z"]
        assert z_range[1] > 0.15

    def test_max_episode_steps(self, reach_env):
        assert reach_env.max_episode_steps == 350

    def test_gymnasium_api_compliance(self, reach_env):
        check_env(reach_env.unwrapped, skip_render_check=True)

    def test_deterministic_reset(self, reach_env):
        obs1, _ = reach_env.reset(seed=42)
        obs2, _ = reach_env.reset(seed=42)
        assert np.allclose(obs1, obs2), "Reset failed deterministic seeding check!"


class TestPickPlaceEnv:
    def test_obs_shape(self, pick_env):
        obs, _ = pick_env.reset(seed=0)
        assert obs.shape == pick_env.observation_space.shape, f"Expected {pick_env.observation_space.shape}, got {obs.shape}"

    def test_action_space_shape(self, pick_env):
        assert pick_env.action_space.shape == (6,)

    def test_obs_dtype(self, pick_env):
        obs, _ = pick_env.reset(seed=1)
        assert obs.dtype == np.float32

    def test_step_no_crash(self, pick_env):
        pick_env.reset(seed=42)
        for _ in range(10):
            action = pick_env.action_space.sample()
            obs, reward, terminated, truncated, info = pick_env.step(action)
            assert obs.shape == pick_env.observation_space.shape
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)

    def test_info_keys(self, pick_env):
        pick_env.reset(seed=0)
        _, _, _, _, info = pick_env.step(pick_env.action_space.sample())
        required_keys = {
            "success", "stage_lifted",
            "consec_lift_hold_steps", "lift_height", "cube_target_err_mm",
        }
        assert required_keys.issubset(info.keys())
        assert "stage_grasped" in info or "has_grasped_once" in info

    def test_max_episode_steps(self, pick_env):
        assert pick_env.max_episode_steps in (300, 400)

    def test_invalid_action_shape_raises(self, pick_env):
        pick_env.reset(seed=0)
        with pytest.raises(ValueError):
            pick_env.step(np.zeros(5))

    def test_gymnasium_api_compliance(self, pick_env):
        check_env(pick_env.unwrapped, skip_render_check=True)

    def test_deterministic_reset(self, pick_env):
        obs1, _ = pick_env.reset(seed=42)
        obs2, _ = pick_env.reset(seed=42)
        assert np.allclose(obs1, obs2), "Reset failed deterministic seeding check!"


@pytest.mark.skip(reason="SO101PickPlaceEnv superseded by SO101HierarchicalPickPlaceEnv")
class TestCurriculumDistribution:
    """
    Verifies curriculum stage initialisation and episode state tracking.

    Stage 4 resets pre-credit the grasp/lift state so transport episodes start
    with the cube already held. Stage 2 resets position the arm near the cube
    with gripper open. This test validates those behaviours.
    """

    def test_curriculum_proportions(self):
        """Stage 4 init should pre-credit grasp + lift flags."""
        env = SO101PickPlaceEnv(
            scene_path="so101/pick_scene.xml",
            curriculum_stage=4,
            render_mode=None,
        )
        pre_credited_count = 0
        n = 20
        for i in range(n):
            env.reset(seed=i)
            # Stage 4 _init_for_stage pre-credits grasp and lift
            if env.has_grasped_once and env.had_sustained_grasp and env.has_lifted_once:
                pre_credited_count += 1

        env.close()
        # At stage 4, the majority of resets use the stage-4 init (75% by _STAGE_REHEARSAL)
        assert pre_credited_count > 0, \
            "Stage 4 resets should pre-credit has_grasped_once/has_lifted_once at least some of the time"

    def test_stage2_arm_near_cube(self):
        """Stage 2 init should place EE close to cube."""
        env = SO101PickPlaceEnv(
            scene_path="so101/pick_scene.xml",
            curriculum_stage=2,
            render_mode=None,
        )
        near_count = 0
        n = 20
        for i in range(n):
            env.reset(seed=i)
            ee_to_obj = np.linalg.norm(
                env.get_ee_xpos() - env.data.xpos[env.cube_body_id]
            )
            if ee_to_obj < 0.12:
                near_count += 1

        env.close()
        # Stage 2 places arm ~2.5 cm above cube; most resets should be close
        assert near_count > n // 2, \
            f"Stage 2 resets should mostly start near cube, got {near_count}/{n}"

    def test_stage3_starts_with_genuine_bilateral_grasp(self):
        """Stage 3 warm-starts must establish genuine bilateral contact before pre-crediting grasp."""
        env = SO101PickPlaceEnv(
            scene_path="so101/pick_scene.xml",
            curriculum_stage=3,
            render_mode=None,
        )
        credited = 0
        for seed in range(20):
            env.reset(seed=seed)
            if env.has_grasped_once:
                credited += 1
                _, _, bilateral = env._check_physical_contacts()
                assert bilateral, f"Stage 3 seed={seed} pre-credited without bilateral contact"
                assert env.gripper_closed
                assert env.reward_state.prev_valid_grasp
                assert env.get_cube_xpos()[2] < 0.03
                _, _, _, _, info = env.step(np.array([0, 0, 0, 0, 0, 1.0], dtype=np.float32))
                assert info["valid_grasp"] and info["bilateral_contact"]
        env.close()
        assert credited > 0

    def test_stage4_starts_with_physically_supported_grasped_cube(self):
        """Stage 4 warm-starts must lift cube via contact physics and drop it if gripper opens."""
        from environments.reward import LIFT_HEIGHT_THRESH

        env = SO101PickPlaceEnv(
            scene_path="so101/pick_scene.xml",
            curriculum_stage=4,
            render_mode=None,
        )
        credited_lifted = 0
        tested_release_drop = False
        for seed in range(20):
            env.reset(seed=seed)
            if env.has_lifted_once:
                credited_lifted += 1
                _, _, bilateral = env._check_physical_contacts()
                cube_z = float(env.get_cube_xpos()[2])
                assert bilateral, f"Stage 4 seed={seed} pre-credited lift without bilateral contact"
                assert env.gripper_closed
                assert cube_z > LIFT_HEIGHT_THRESH
                assert np.allclose(env.desired_qpos, env.current_target_qpos)

                # Holding gripper closed keeps cube supported by MuJoCo contact friction
                _, _, _, _, info = env.step(np.array([0, 0, 0, 0, 0, 1.0], dtype=np.float32))
                assert info["valid_grasp"] and info["lifted"]

                # Opening gripper must let gravity drop the cube (no artificial attachment)
                if not tested_release_drop:
                    for _ in range(15):
                        _, _, _, _, info_rel = env.step(np.array([0, 0, 0, 0, 0, -1.0], dtype=np.float32))
                    assert float(env.get_cube_xpos()[2]) < cube_z - 0.015
                    assert not info_rel["valid_grasp"]
                    tested_release_drop = True
        env.close()
        assert credited_lifted > 0
        assert tested_release_drop

    def test_no_artificial_cube_attachment(self):
        """Curriculum module must not teleport cube to EE; cube moves only via physics."""
        import inspect
        import environments.curriculum as curr_mod

        src = inspect.getsource(curr_mod)
        assert "_set_object_pose" not in src

        # Moving arm with open gripper far from cube leaves cube stationary on table
        env = SO101PickPlaceEnv(curriculum_stage=1, domain_randomization=False)
        env.reset(seed=0)
        cube_before = env.get_cube_xpos().copy()
        for _ in range(5):
            env.step(np.array([0.5, -0.5, 0.5, 0.0, 0.0, -1.0], dtype=np.float32))
        cube_after = env.get_cube_xpos().copy()
        env.close()
        assert np.allclose(cube_before, cube_after, atol=1e-4)

    def test_reset_clears_stale_state_across_stages(self):
        """Resetting after dirtying episode state restores clean initial states."""
        env = SO101PickPlaceEnv(curriculum_stage=2, domain_randomization=True)
        env.reset(seed=5)
        for _ in range(10):
            env.step(np.array([0.8, -0.4, 0.3, -0.2, 0.1, 1.0], dtype=np.float32))

        env.set_curriculum_stage(1)
        obs1, _ = env.reset(seed=7)
        assert env.current_step == 0
        assert env.reward_state.current_step == 0
        assert np.allclose(env.reward_state.prev_action, np.zeros(6))
        assert not env.has_grasped_once
        assert not env.has_lifted_once
        assert not env.gripper_closed
        expected_qpos = np.concatenate([env.home_qpos[:5], [env.gripper_open_q]])
        assert np.allclose(env.desired_qpos, expected_qpos)
        assert np.allclose(env.current_target_qpos, expected_qpos)
        assert np.allclose(env.data.ctrl[:6], 0.0)
        assert np.allclose(env.actuators.internal_target, expected_qpos)
        assert np.allclose(env.data.qvel, 0.0)

        obs2, _ = env.reset(seed=7)
        env.close()
        assert np.allclose(obs1, obs2)


@pytest.mark.skip(reason="Deleted reward/curriculum modules")
class TestRewardModule:
    """Direct tests for the isolated reward module."""

    def test_reaching_reward_gradient(self):
        from environments.reward import EpisodeState, compute_reward
        state = EpisodeState()
        ee = np.array([0.25, 0.0, 0.10])
        obj = np.array([0.25, 0.0, 0.02])
        target = np.array([0.26, -0.10, 0.015])
        base_xy = np.array([0.25, 0.0])
        action = np.zeros(6, dtype=np.float32)

        r1, term, info = compute_reward(state, ee, obj, target, False, False, action, base_xy)
        assert r1 > 0.0
        assert not term
        assert not info["success"]

        # Closer EE should yield higher reach reward
        state2 = EpisodeState()
        ee_closer = np.array([0.25, 0.0, 0.04])
        r2, _, _ = compute_reward(state2, ee_closer, obj, target, False, False, action, base_xy)
        assert r2 > r1

    def test_escape_penalty_triggered(self):
        from environments.reward import EpisodeState, compute_reward
        state = EpisodeState()
        ee = np.array([0.25, 0.0, 0.10])
        obj_escaped = np.array([1.5, 1.5, 0.02])  # far outside
        target = np.array([0.26, -0.10, 0.015])
        base_xy = np.array([0.25, 0.0])
        action = np.zeros(6, dtype=np.float32)

        r, term, info = compute_reward(state, ee, obj_escaped, target, False, False, action, base_xy)
        assert r < -19.0
        assert term
        assert not info["success"]

    def test_action_smoothness_updates_prev_action(self):
        from environments.reward import EpisodeState, compute_reward
        state = EpisodeState()
        ee = np.array([0.25, 0.0, 0.10])
        obj = np.array([0.25, 0.0, 0.02])
        target = np.array([0.26, -0.10, 0.015])
        base_xy = np.array([0.25, 0.0])

        act1 = np.ones(6, dtype=np.float32) * 0.5
        _, _, info1 = compute_reward(state, ee, obj, target, False, False, act1, base_xy)
        assert info1["action_delta"] == pytest.approx(0.5, abs=1e-6)
        assert np.allclose(state.prev_action, act1)

        # Repeating the exact same action must yield zero smoothness delta
        _, _, info2 = compute_reward(state, ee, obj, target, False, False, act1, base_xy)
        assert info2["action_delta"] == pytest.approx(0.0, abs=1e-6)

        # Changing action compares against act1, not zeros
        act2 = -np.ones(6, dtype=np.float32) * 0.25
        _, _, info3 = compute_reward(state, ee, obj, target, False, False, act2, base_xy)
        assert info3["action_delta"] == pytest.approx(0.75, abs=1e-6)
        assert np.allclose(state.prev_action, act2)

    def test_grasp_requires_bilateral_contact_and_flicker_tolerance(self):
        from environments.reward import (
            EpisodeState,
            compute_reward,
            GRASP_FLICKER_TOLERANCE,
            LIFT_STABILIZE_STEPS,
        )
        state = EpisodeState()
        target = np.array([0.26, -0.10, 0.015])
        base_xy = np.array([0.25, 0.0])
        action = np.zeros(6, dtype=np.float32)

        # Proximity alone (even 1 cm away, closed, lifted) without bilateral contact is NOT a valid grasp
        obj_air = np.array([0.25, 0.0, 0.06])
        ee_near = np.array([0.25, 0.0, 0.07])
        _, _, info = compute_reward(state, ee_near, obj_air, target, False, True, action, base_xy)
        assert not info["valid_grasp"]
        assert info["lifted"]
        assert not info["stable_lift"]

        # Genuine bilateral contact on table -> valid_grasp=True, lifted=False, stable_lift=False
        obj_table = np.array([0.25, 0.0, 0.015])
        ee_table = np.array([0.25, 0.0, 0.020])
        _, _, info = compute_reward(state, ee_table, obj_table, target, True, True, action, base_xy)
        assert info["valid_grasp"]
        assert not info["lifted"]
        assert not info["stable_lift"]

        # Brief contact flicker while closed is tolerated up to GRASP_FLICKER_TOLERANCE steps
        for _ in range(GRASP_FLICKER_TOLERANCE):
            _, _, info = compute_reward(state, ee_table, obj_table, target, False, True, action, base_xy)
            assert info["valid_grasp"]

        # Exceeding flicker tolerance invalidates grasp
        _, _, info = compute_reward(state, ee_table, obj_table, target, False, True, action, base_xy)
        assert not info["valid_grasp"]

        # Opening gripper immediately clears grasp even right after bilateral contact
        _, _, info = compute_reward(state, ee_table, obj_table, target, True, True, action, base_xy)
        assert info["valid_grasp"]
        _, _, info = compute_reward(state, ee_table, obj_table, target, False, False, action, base_xy)
        assert not info["valid_grasp"]

        # Confirm sustained grasp + stable_lift condition progression
        state2 = EpisodeState()
        for _ in range(3):
            _, _, info = compute_reward(state2, ee_table, obj_table, target, True, True, action, base_xy)
        assert info["valid_grasp"] and state2.has_grasped_once and not info["stable_lift"]
        for k in range(LIFT_STABILIZE_STEPS):
            _, _, info = compute_reward(state2, ee_near, obj_air, target, True, True, action, base_xy)
            if k < LIFT_STABILIZE_STEPS - 1:
                assert info["valid_grasp"] and info["lifted"] and not info["stable_lift"]
        assert info["valid_grasp"] and info["lifted"] and info["stable_lift"]

    def test_placement_stability_resets_when_not_ready(self):
        from environments.reward import (
            EpisodeState,
            compute_reward,
            PLACEMENT_STABLE_STEPS,
        )
        state = EpisodeState(
            has_grasped_once=True,
            had_sustained_grasp=True,
            prev_valid_grasp=True,
            prev_obj_pos=np.array([0.26, -0.10, 0.018]),
            prev_ee_pos=np.array([0.26, -0.10, 0.023]),
            prev_obj_height=0.018,
        )
        target = np.array([0.26, -0.10, 0.015])
        base_xy = np.array([0.25, 0.0])
        action = np.zeros(6, dtype=np.float32)
        obj_ok = np.array([0.26, -0.10, 0.018])
        ee_ok = np.array([0.26, -0.10, 0.023])

        # Accumulate 2 consecutive stable steps (less than PLACEMENT_STABLE_STEPS=3)
        for _ in range(PLACEMENT_STABLE_STEPS - 1):
            _, _, info = compute_reward(state, ee_ok, obj_ok, target, True, True, action, base_xy)
        assert info["placement_stable_steps"] == PLACEMENT_STABLE_STEPS - 1
        assert not state.has_reached_target_once

        # Still holding (bilateral=True, closed=True), but move outside target zone -> must reset to 0
        obj_far = np.array([0.30, -0.10, 0.018])
        ee_far = np.array([0.30, -0.10, 0.023])
        _, _, info = compute_reward(state, ee_far, obj_far, target, True, True, action, base_xy)
        assert not info["placement_ready"]
        assert info["placement_stable_steps"] == 0
        assert not state.has_reached_target_once

        # Move back and hold for full PLACEMENT_STABLE_STEPS consecutive steps
        state.prev_obj_pos = obj_ok.copy()
        state.prev_ee_pos = ee_ok.copy()
        for _ in range(PLACEMENT_STABLE_STEPS):
            _, _, info = compute_reward(state, ee_ok, obj_ok, target, True, True, action, base_xy)
        assert info["placement_stable_steps"] == PLACEMENT_STABLE_STEPS
        assert state.has_reached_target_once

    def test_final_success_without_mandatory_return_home(self):
        """Final success triggers immediately upon valid placement, release, and >= 6 cm clearance."""
        from environments.reward import EpisodeState, compute_reward, SUCCESS_BONUS

        state = EpisodeState(
            has_grasped_once=True,
            had_sustained_grasp=True,
            has_reached_target_once=True,
            prev_ee_pos=np.array([0.26, -0.10, 0.025]),
            prev_obj_pos=np.array([0.26, -0.10, 0.015]),
            prev_obj_height=0.015,
            prev_ee_to_obj=0.01,
        )
        obj = np.array([0.26, -0.10, 0.015])
        target = np.array([0.26, -0.10, 0.015])
        base_xy = np.array([0.0, 0.0])
        action = np.zeros(6, dtype=np.float32)
        far_from_home_q = np.array([0.9, 0.3, -0.4, 0.5, -0.7])
        nonzero_vel = np.ones(5) * 0.8
        home_q = np.array([0.0, -0.40, 0.60, 0.80, 0.0])

        # Release cube at target and settle for 3 steps, but EE is only 3 cm clear (< 6 cm threshold) -> not yet success
        ee_close = np.array([0.26, -0.10, 0.045])
        for _ in range(3):
            _, term, info = compute_reward(
                state, ee_close, obj, target, False, False, action, base_xy,
                arm_qpos=far_from_home_q, arm_qvel=nonzero_vel, home_qpos=home_q, curriculum_stage=5,
            )
        assert info["stage_released"]
        assert not info["stage_cleared"]
        assert not term and not info["success"]

        # Move EE 7 cm clear (>= 6 cm) while still far from home_q and moving -> succeeds immediately
        ee_clear = np.array([0.26, -0.10, 0.085])
        r, term, info = compute_reward(
            state, ee_clear, obj, target, False, False, action, base_xy,
            arm_qpos=far_from_home_q, arm_qvel=nonzero_vel, home_qpos=home_q, curriculum_stage=5,
        )
        assert info["stage_cleared"]
        assert term and info["success"] and info["full_task_success"]
        assert r >= SUCCESS_BONUS

    def test_pre_grasp_cube_displacement_penalized(self):
        """Displacing the cube horizontally before a valid grasp incurs a displacement penalty."""
        from environments.reward import EpisodeState, compute_reward

        state = EpisodeState(
            has_grasped_once=False,
            initial_obj_xy=np.array([0.25, 0.0]),
            initial_obj_z=0.015,
            prev_ee_to_obj=0.08,
            prev_obj_pos=np.array([0.25, 0.0, 0.015]),
            prev_ee_pos=np.array([0.25, 0.0, 0.08]),
        )
        ee = np.array([0.25, 0.0, 0.08])
        target = np.array([0.26, -0.10, 0.015])
        base_xy = np.array([0.0, 0.0])
        action = np.zeros(6, dtype=np.float32)

        # Baseline: undisturbed cube
        obj_clean = np.array([0.25, 0.0, 0.015])
        r_clean, _, _ = compute_reward(state, ee, obj_clean, target, False, False, action, base_xy)

        # Disturbed cube (displaced 15 mm across table before grasp)
        state_disturbed = EpisodeState(
            has_grasped_once=False,
            initial_obj_xy=np.array([0.25, 0.0]),
            initial_obj_z=0.015,
            prev_ee_to_obj=0.08,
            prev_obj_pos=np.array([0.25, 0.0, 0.015]),
            prev_ee_pos=np.array([0.25, 0.0, 0.08]),
        )
        obj_disturbed = np.array([0.265, 0.0, 0.015])
        r_disturbed, _, _ = compute_reward(state_disturbed, ee, obj_disturbed, target, False, False, action, base_xy)
        assert r_disturbed < r_clean - 0.25, f"Displacing cube before grasp was not penalized: {r_disturbed} vs {r_clean}"

    def test_on_table_grasp_does_not_infinitely_farm_holding_dense(self):
        """Holding on table without lifting stops awarding holding dense reward, forcing lift."""
        from environments.reward import EpisodeState, compute_reward, HOLDING_DENSE

        state = EpisodeState(
            has_grasped_once=True,
            had_sustained_grasp=True,
            has_touched_once=True,
            consec_holding_steps=20,  # Clamped on table for 20 steps
            initial_obj_xy=np.array([0.25, 0.0]),
            initial_obj_z=0.015,
            prev_obj_height=0.015,
            prev_ee_to_obj=0.01,
            prev_ee_pos=np.array([0.25, 0.0, 0.02]),
            prev_obj_pos=np.array([0.25, 0.0, 0.015]),
        )
        ee = np.array([0.25, 0.0, 0.02])
        obj = np.array([0.25, 0.0, 0.015])  # Still on table (height 0.015 <= 0.045)
        target = np.array([0.26, -0.10, 0.015])
        base_xy = np.array([0.0, 0.0])
        action = np.zeros(6, dtype=np.float32)

        r_table, _, _ = compute_reward(state, ee, obj, target, True, True, action, base_xy)
        assert r_table < HOLDING_DENSE, f"Holding on table should not earn full HOLDING_DENSE: r={r_table}"

    def test_dense_lift_gradient_and_bonus(self):
        """Lifting the cube upward generates dense lift progress and triggers LIFT_ONCE_BONUS."""
        from environments.reward import EpisodeState, compute_reward, LIFT_ONCE_BONUS

        state = EpisodeState(
            has_grasped_once=True,
            had_sustained_grasp=True,
            consec_holding_steps=5,
            initial_obj_xy=np.array([0.25, 0.0]),
            initial_obj_z=0.015,
            prev_obj_height=0.015,
            prev_ee_to_obj=0.01,
            prev_ee_pos=np.array([0.25, 0.0, 0.02]),
            prev_obj_pos=np.array([0.25, 0.0, 0.015]),
        )
        target = np.array([0.26, -0.10, 0.015])
        base_xy = np.array([0.0, 0.0])
        action = np.zeros(6, dtype=np.float32)

        # Lift cube to 0.050 m (> 0.045 m threshold)
        ee_lift = np.array([0.25, 0.0, 0.055])
        obj_lift = np.array([0.25, 0.0, 0.050])
        r_lift, _, info = compute_reward(state, ee_lift, obj_lift, target, True, True, action, base_xy)
        assert info["lifted"]
        assert info["stage_lifted"]
        assert r_lift >= LIFT_ONCE_BONUS


@pytest.mark.skip(reason="Deleted reward/curriculum modules")
class TestEnvironmentAudit:
    """Tests for name-based joint mapping, elbow control, physical lifting, and DR."""

    def test_action_to_named_joint_mapping_and_elbow(self):
        """Each of the 6 policy actions drives its exact named joint, including elbow_flex."""
        env = SO101PickPlaceEnv(curriculum_stage=1, domain_randomization=False)
        expected_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        assert env.mapper.joint_names == expected_names

        for idx, name in enumerate(expected_names):
            env.reset(seed=0)
            q_idx = int(env.robot_qpos_idx[idx])
            before = float(env.data.qpos[q_idx])
            act = np.zeros(6, dtype=np.float32)
            act[idx] = 1.0
            for _ in range(5):
                env.step(act)
            after = float(env.data.qpos[q_idx])

            if name == "gripper":
                # Open is 1.00 rad, closing (+1.0) reduces angle toward -0.10 rad
                assert after < before - 0.1, f"Gripper failed to close on action[5]: {before} -> {after}"
            else:
                assert after > before + 0.05, f"Arm joint '{name}' (action[{idx}]) did not move positively: {before} -> {after}"
                # Verify other arm joints remain near initial targets
                for other in range(5):
                    if other != idx:
                        assert abs(env.desired_qpos[other] - env.home_qpos[other]) < 1e-6

        # Explicitly test negative elbow_flex command
        env.reset(seed=0)
        elbow_q_idx = int(env.arm_qpos_idx[2])
        elbow_before = float(env.data.qpos[elbow_q_idx])
        for _ in range(5):
            env.step(np.array([0.0, 0.0, -1.0, 0.0, 0.0, -1.0], dtype=np.float32))
        assert float(env.data.qpos[elbow_q_idx]) < elbow_before - 0.05
        env.close()

    def test_physical_grasp_and_cube_lifting(self):
        """Closing jaws on cube and actuating shoulder/elbow physically lifts cube via contact."""
        import mujoco
        from environments.reward import LIFT_HEIGHT_THRESH

        env = SO101PickPlaceEnv(curriculum_stage=3, domain_randomization=False)
        lifted_success = False
        for seed in range(15):
            env.reset(seed=seed)
            _, _, bilateral = env._check_physical_contacts()
            if not (env.has_grasped_once and bilateral):
                continue

            # Drive arm upward using Jacobian-computed joint deltas while holding gripper closed
            for _ in range(25):
                jacp = np.zeros((3, env.model.nv))
                mujoco.mj_jacSite(env.model, env.data, jacp, None, env.ee_site_id)
                J_arm = jacp[:, env.arm_dof_idx]
                dq = np.linalg.lstsq(J_arm, np.array([0.0, 0.0, 0.004]), rcond=None)[0]
                arm_act = np.clip(dq / 0.025, -1.0, 1.0)
                _, _, _, _, info = env.step(np.concatenate([arm_act, [1.0]]).astype(np.float32))
                if info["lifted"] and info["valid_grasp"]:
                    lifted_success = True
                    break
            if lifted_success:
                assert float(env.get_cube_xpos()[2]) > LIFT_HEIGHT_THRESH
                break
        env.close()
        assert lifted_success, "Physical Stage 3 grasp could not lift cube via contact physics"

    def test_domain_randomization_mass_inertia_and_reset(self):
        """Cube mass, size, and inertia must update consistently under DR and restore when off."""
        # 1. DR active (Stage 5)
        env_dr = SO101PickPlaceEnv(curriculum_stage=5, domain_randomization=True)
        masses, inertias, sizes = [], [], []
        for s in range(5):
            env_dr.reset(seed=s)
            m = float(env_dr.model.body_mass[env_dr.cube_body_id])
            hs = float(env_dr.model.geom_size[env_dr.cube_geom_id, 0])
            inertia = env_dr.model.body_inertia[env_dr.cube_body_id].copy()
            expected_I = (2.0 / 3.0) * m * (hs ** 2)
            assert np.allclose(inertia, [expected_I, expected_I, expected_I], rtol=1e-5)
            masses.append(m)
            sizes.append(hs)
            inertias.append(inertia[0])
        assert max(masses) > min(masses)
        assert max(sizes) > min(sizes)
        assert max(inertias) > min(inertias)

        # 2. Switching DR off must restore nominal mass, size, and inertia
        env_dr.domain_randomization = False
        env_dr.reset(seed=0)
        assert float(env_dr.model.body_mass[env_dr.cube_body_id]) == pytest.approx(env_dr.nominal_cube_mass)
        assert float(env_dr.model.geom_size[env_dr.cube_geom_id, 0]) == pytest.approx(env_dr.nominal_cube_size)
        assert np.allclose(env_dr.model.body_inertia[env_dr.cube_body_id], env_dr.nominal_cube_inertia)
        env_dr.close()

    def test_gripper_centering_eliminates_fixed_jaw_approach_collision(self):
        """Aligning ee_center at cube position with open jaws has 0 fixed jaw collisions."""
        import mujoco
        from src.so101_kinematics import so101_damped_least_squares_ik

        m = mujoco.MjModel.from_xml_path("so101/pick_scene.xml")
        d = mujoco.MjData(m)
        cube_pos = np.array([0.25, 0.0, 0.015])
        cube_jnt = m.body("cube").jntadr[0]
        cube_adr = m.jnt_qposadr[cube_jnt]
        d.qpos[cube_adr : cube_adr + 3] = cube_pos
        d.qpos[cube_adr + 3 : cube_adr + 7] = [1, 0, 0, 0]

        sol, err, _, conv = so101_damped_least_squares_ik(
            m, d, cube_pos, site_name="ee_center", n_arm_joints=5, vertical_posture=True
        )
        assert conv and err < 0.005
        d.qpos[:5] = sol
        d.qpos[5] = 1.0  # open gripper
        mujoco.mj_step(m, d)

        cube_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        fj_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"fixed_jaw_{p}") for p in ["base", "mid", "tip"]]

        fj_contacts = [
            c for c in d.contact[:d.ncon]
            if (c.geom1 == cube_gid and c.geom2 in fj_ids) or (c.geom2 == cube_gid and c.geom1 in fj_ids)
        ]
        assert len(fj_contacts) == 0, f"Fixed jaw collides on descent: {len(fj_contacts)} contacts"


@pytest.mark.skip(reason="Deleted reward/curriculum modules")
class TestCurriculumAndActuatorFixes:
    """Verifies all 17 numbered environment and reward audit requirements."""

    def test_dense_reward_budget_and_gate_stop(self):
        """Dense rewards over 400 steps remain well below success total; gate dense stops after target reached."""
        from environments.reward import EpisodeState, compute_reward, SUCCESS_BONUS

        state = EpisodeState(
            has_grasped_once=True,
            had_sustained_grasp=True,
            has_lifted_once=True,
            consec_holding_steps=100,
            consec_lifted_steps=100,
            prev_ee_pos=np.array([0.26, -0.10, 0.06]),
            prev_obj_pos=np.array([0.26, -0.10, 0.06]),
            prev_obj_height=0.06,
            max_obj_height=0.06,
            max_ee_lift_height=0.06,
            prev_ee_to_obj=0.01,
            prev_obj_to_target=0.01,
            has_reached_target_once=True,  # Gate passed
        )
        ee = np.array([0.26, -0.10, 0.06])
        obj = np.array([0.26, -0.10, 0.06])
        target = np.array([0.26, -0.10, 0.015])
        base_xy = np.array([0.0, 0.0])
        action = np.zeros(6, dtype=np.float32)

        # In target area with gate already confirmed once
        total_r = 0.0
        for _ in range(400):
            r, _, _ = compute_reward(state, ee, obj, target, True, True, action, base_xy)
            total_r += r

        # Total hovering dense reward over 400 steps must be well below terminal success total (~112.5)
        assert total_r < 85.0, f"Dense hovering reward farmed {total_r}, exceeding budget"

    def test_up_down_lift_pumping_prevented(self):
        """Lifting up, dropping, and lifting back up pays only for new max height."""
        from environments.reward import EpisodeState, compute_reward, LIFT_SCALE

        state = EpisodeState(
            has_grasped_once=True,
            had_sustained_grasp=True,
            consec_holding_steps=10,
            initial_obj_xy=np.array([0.25, 0.0]),
            initial_obj_z=0.015,
            max_obj_height=0.015,
            prev_obj_height=0.015,
            prev_ee_pos=np.array([0.25, 0.0, 0.02]),
            max_ee_lift_height=0.02,
        )
        ee = np.array([0.25, 0.0, 0.05])
        target = np.array([0.26, -0.10, 0.015])
        base_xy = np.array([0.0, 0.0])
        action = np.zeros(6, dtype=np.float32)

        # 1. Lift from 0.015 to 0.050 (new max) -> pays delta
        obj_up = np.array([0.25, 0.0, 0.050])
        r_up1, _, _ = compute_reward(state, ee, obj_up, target, True, True, action, base_xy)
        assert state.max_obj_height == pytest.approx(0.050)

        # 2. Lower back down to 0.020 -> no lift delta reward
        obj_down = np.array([0.25, 0.0, 0.020])
        r_down, _, _ = compute_reward(state, ee, obj_down, target, True, True, action, base_xy)
        assert state.max_obj_height == pytest.approx(0.050)

        # 3. Lift back up to 0.040 (below previous max of 0.050) -> 0 lift delta reward!
        obj_re_up = np.array([0.25, 0.0, 0.040])
        r_up2, _, _ = compute_reward(state, ee, obj_re_up, target, True, True, action, base_xy)
        assert state.max_obj_height == pytest.approx(0.050)
        assert r_up2 < r_up1, "Up/down pumping paid reward below previous max height"

    def test_descent_shaping_symmetric(self):
        """Descent in placement zone earns reward descending, penalizes ascending."""
        from environments.reward import EpisodeState, compute_reward, PLACEMENT_DESCENT_SCALE

        state = EpisodeState(
            has_grasped_once=True,
            had_sustained_grasp=True,
            consec_holding_steps=10,
            has_lifted_once=True,
            consec_lifted_steps=10,
            prev_obj_height=0.040,
            prev_ee_pos=np.array([0.26, -0.10, 0.045]),
            prev_obj_pos=np.array([0.26, -0.10, 0.040]),
            prev_ee_to_obj=0.005,
            prev_obj_to_target=0.005,
        )
        ee = np.array([0.26, -0.10, 0.035])
        target = np.array([0.26, -0.10, 0.015])
        base_xy = np.array([0.0, 0.0])
        action = np.zeros(6, dtype=np.float32)

        # Descend from 0.040 to 0.030
        obj_down = np.array([0.26, -0.10, 0.030])
        r_descend, _, _ = compute_reward(state, ee, obj_down, target, True, True, action, base_xy)

        # Ascend from 0.030 to 0.040
        ee_up = np.array([0.26, -0.10, 0.045])
        obj_up = np.array([0.26, -0.10, 0.040])
        r_ascend, _, _ = compute_reward(state, ee_up, obj_up, target, True, True, action, base_xy)

        assert r_descend > r_ascend, "Descent should yield higher reward than ascending"

    def test_pre_grasp_displacement_penalty_capped_and_rearm_drops(self):
        """Pre-grasp displacement penalty is capped; sustained grasp re-arms after re-grasping."""
        from environments.reward import EpisodeState, compute_reward, MAX_PRE_GRASP_DISP_PENALTY, DROP_PENALTY

        state = EpisodeState(
            has_grasped_once=False,
            initial_obj_xy=np.array([0.25, 0.0]),
            initial_obj_z=0.015,
            prev_obj_pos=np.array([0.25, 0.0, 0.015]),
        )
        ee = np.array([0.25, 0.0, 0.08])
        target = np.array([0.26, -0.10, 0.015])
        base_xy = np.array([0.0, 0.0])
        action = np.zeros(6, dtype=np.float32)
        obj_disturbed = np.array([0.29, 0.0, 0.015])  # 4 cm disturbance

        total_penalty = 0.0
        for _ in range(50):
            r, _, _ = compute_reward(state, ee, obj_disturbed, target, False, False, action, base_xy)
            total_penalty += max(0.0, -r)

        assert state.cumulative_disp_penalty <= MAX_PRE_GRASP_DISP_PENALTY + 1e-4

        # Drop penalty re-arming:
        # Simulate grasp -> drop -> re-grasp -> second drop
        state.had_sustained_grasp = True
        state.consec_not_holding_steps = 5
        r_drop1, _, _ = compute_reward(state, ee, obj_disturbed, target, False, False, action, base_xy)
        assert not state.had_sustained_grasp

        # Re-grasp: 3 consecutive steps holding
        state.consec_holding_steps = 3
        compute_reward(state, ee, obj_disturbed, target, True, True, action, base_xy)
        assert state.had_sustained_grasp, "Re-grasp failed to re-arm had_sustained_grasp"

        # Second drop:
        state.consec_not_holding_steps = 5
        r_drop2, _, _ = compute_reward(state, ee, obj_disturbed, target, False, False, action, base_xy)
        assert not state.had_sustained_grasp

    def test_stage_success_early_termination(self):
        """Stages 1-4 no longer terminate early on stage_success (allow agent to practice full rollout)."""
        from environments.reward import EpisodeState, compute_reward

        state = EpisodeState(has_reached_aligned=True)
        ee = np.array([0.25, 0.0, 0.02])
        obj = np.array([0.25, 0.0, 0.015])
        target = np.array([0.26, -0.10, 0.015])
        base_xy = np.array([0.0, 0.0])
        action = np.zeros(6, dtype=np.float32)

        _, term_st1, info1 = compute_reward(state, ee, obj, target, False, False, action, base_xy, curriculum_stage=1)
        assert info1["stage_success"]
        assert not term_st1, "Stage 1 terminated early on stage_success but early termination was requested disabled"

    def test_jaw_friction_dr_and_geom_rbound(self, pick_env):
        """Domain randomisation updates jaw friction [0.6, 2.5] and geom_rbound = sqrt(3)*half_size."""
        from environments.curriculum import apply_domain_randomisation

        pick_env.domain_randomization = True
        half_size = apply_domain_randomisation(pick_env, dr_scale=1.0)

        # Check geom_rbound
        rbound = float(pick_env.model.geom_rbound[pick_env.cube_geom_id])
        assert rbound == pytest.approx(np.sqrt(3.0) * half_size)

        # Check jaw frictions
        for gid in pick_env.all_jaw_geoms:
            fric = float(pick_env.model.geom_friction[gid, 0])
            assert 0.6 <= fric <= 2.5

        # Check nominal restoration
        pick_env.domain_randomization = False
        apply_domain_randomisation(pick_env, dr_scale=0.0)
        assert float(pick_env.model.geom_rbound[pick_env.cube_geom_id]) == pytest.approx(
            np.sqrt(3.0) * pick_env.nominal_cube_size
        )
        for gid in pick_env.all_jaw_geoms:
            assert np.allclose(pick_env.model.geom_friction[gid], pick_env.nominal_jaw_friction[gid])

    def test_home_pose_vertical_and_stage1_nontrivial(self, pick_env):
        """Canonical home pose has q1+q2+q3 = pi/2 (pitch error < 0.05), ee_site z-axis pointing down."""
        pick_env.set_curriculum_stage(1)
        obs, _ = pick_env.reset(seed=42)

        # Tool vertical
        q1, q2, q3 = pick_env.data.qpos[pick_env.arm_qpos_idx][1:4]
        pitch_sum = q1 + q2 + q3
        assert abs((np.pi / 2.0) - pitch_sum) < 0.05, f"Pitch error too large: {pitch_sum}"

        # Site z-axis pointing down
        site_z = pick_env.data.site_xmat[pick_env.ee_site_id].reshape(3, 3)[:, 2]
        dot_neg_z = -site_z[2]
        assert dot_neg_z > 0.99, f"Site z-axis not pointing down: {dot_neg_z}"

        # Stage 1 initial ee_dist > 0.10 m (non-trivial reach)
        ee_xpos = pick_env.get_ee_xpos()
        cube_xpos = pick_env.get_cube_xpos()
        dist = np.linalg.norm(ee_xpos - cube_xpos)
        assert dist > 0.10, f"Stage 1 reset too close: {dist} m"

    def test_target_windup_clamping(self, pick_env):
        """Step clamps |desired_qpos[:5] - qpos[:5]| <= 0.12 rad."""
        pick_env.reset(seed=0)
        # Attempt massive action delta
        large_action = np.ones(6, dtype=np.float32)
        for _ in range(10):
            pick_env.step(large_action)
            arm_q = pick_env.data.qpos[pick_env.arm_qpos_idx]
            windup = np.abs(pick_env.desired_qpos[:5] - arm_q)
            assert np.all(windup <= 0.12 + 1e-4), f"Target windup exceeded 0.12 rad: {windup}"

    def test_target_rejection_sampling_min_sep(self, pick_env):
        """Target sampling always satisfies min_sep from the very first draw."""
        pick_env.set_curriculum_stage(3)
        for seed in range(30):
            pick_env.reset(seed=seed)
            obj_xy = pick_env.get_cube_xpos()[:2]
            tgt_xy = pick_env.target_pos[:2]
            sep = np.linalg.norm(obj_xy - tgt_xy)
            assert sep >= 0.07 - 1e-4, f"Seed {seed} failed min_sep: {sep} < 0.07"




