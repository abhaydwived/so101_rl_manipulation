"""
Stability diagnostic for SO101PickLiftEnv. Run from the project root or Diagnose/:

    python Diagnose/diagnose_stability.py                      # test A only (no policy)
    python Diagnose/diagnose_stability.py --model path/to/model.zip \
        [--vecnorm path/to/vecnormalize.pkl] [--episodes 3]

Test A (no policy): zero arm action, gripper closed on nothing and then on the cube.
   If the arm shakes here, the cause is physics / actuator tuning, not the policy.
Test B (policy): deterministic rollouts. Reports per-joint action jitter, joint
   velocity, contact flicker, grasp quality and max cube height.
"""
import argparse
import pickle
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "environments") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "environments"))

try:
    from src.common import configure_env
    configure_env()
except Exception:
    pass

from environments.pick_and_lift_env import SO101PickLiftEnv


class ObsAdapterWrapper(gym.ObservationWrapper):
    """Adapts observation dimensions bidirectionally for cross-checkpoint compatibility."""
    def __init__(self, env, target_dim):
        super().__init__(env)
        self.target_dim = target_dim
        base_dim = env.observation_space.shape[0]
        if target_dim == base_dim:
            self.observation_space = env.observation_space
        elif target_dim < base_dim:
            low = env.observation_space.low[:target_dim]
            high = env.observation_space.high[:target_dim]
            self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        else:
            diff = target_dim - base_dim
            low = np.pad(env.observation_space.low, (0, diff), constant_values=-10.0)
            high = np.pad(env.observation_space.high, (0, diff), constant_values=10.0)
            self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs):
        base_dim = len(obs)
        if self.target_dim == base_dim:
            return obs
        if self.target_dim < base_dim:
            return obs[:self.target_dim].astype(np.float32)
        pad = np.zeros(self.target_dim - base_dim, dtype=np.float32)
        return np.concatenate([obs, pad]).astype(np.float32)


def unwrap(venv):
    while hasattr(venv, "venv"):
        venv = venv.venv
    return venv.envs[0].unwrapped


def test_a():
    print("=== TEST A: zero action, no policy ===")
    env = SO101PickLiftEnv()
    for label, grip in (("gripper open", -1.0), ("gripper closed", 1.0)):
        env.reset(seed=0)
        qv, pd_err, q_hist = [], [], []
        for t in range(200):
            a = np.zeros(6, np.float32)
            a[5] = grip
            env.step(a)
            if t >= 100:
                qv.append(env.data.qvel[env.robot_dof_idx].copy())
                q = env.data.qpos[env.robot_qpos_idx].copy()
                q_hist.append(q)
                pd_err.append(env.desired_qpos - q)
        qv, pd_err, q_hist = map(np.array, (qv, pd_err, q_hist))
        print(f"[{label}] last 100 steps")
        print("  qvel std   :", qv.std(0).round(4))
        print("  qpos std   :", q_hist.std(0).round(5))
        print("  PD err mean:", pd_err.mean(0).round(3))
        print("  PD err std :", pd_err.std(0).round(4))
    env.close()


def test_b(model_path, vecnorm_path, episodes):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    print("\n=== TEST B: policy rollouts (deterministic) ===")

    if model_path is not None and not Path(model_path).is_absolute() and not Path(model_path).exists():
        model_path = str(PROJECT_ROOT / model_path)
    if vecnorm_path is not None and not Path(vecnorm_path).is_absolute() and not Path(vecnorm_path).exists():
        vecnorm_path = str(PROJECT_ROOT / vecnorm_path)

    # Detect observation dimension expected by checkpoint/vecnorm
    target_dim = None
    if vecnorm_path and Path(vecnorm_path).exists():
        with open(vecnorm_path, "rb") as f:
            vn_data = pickle.load(f)
            target_dim = int(vn_data.observation_space.shape[0])
    elif model_path and Path(model_path).exists():
        m_temp = PPO.load(model_path, device="cpu")
        target_dim = int(m_temp.observation_space.shape[0])
        del m_temp

    def _make_env():
        base = SO101PickLiftEnv()
        if target_dim is not None and target_dim != base.observation_space.shape[0]:
            return ObsAdapterWrapper(base, target_dim=target_dim)
        return base

    venv = DummyVecEnv([_make_env])
    if vecnorm_path and Path(vecnorm_path).exists():
        venv = VecNormalize.load(vecnorm_path, venv)
        venv.training = False
        venv.norm_reward = False

    model = PPO.load(model_path, env=venv, device="cpu")
    env = unwrap(venv)

    for ep in range(episodes):
        obs = venv.reset()
        acts, qvs, flick, proper, bil, clr = [], [], 0, 0, 0, []
        steps = 0
        for _ in range(env.max_episode_steps):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, done, infos = venv.step(a)
            info = infos[0]
            steps += 1
            acts.append(np.asarray(a)[0].copy())
            qvs.append(env.data.qvel[env.robot_dof_idx].copy())
            flick += int(info.get("contact_lost_steps", 0) > 0 and info.get("task_stage", 1) == 2)
            proper += int(info.get("proper_grasp", info.get("has_grasped_once", False)))
            bil += int(info.get("bilateral_contact", False))
            clr.append(info.get("cube_bottom_clearance", info.get("lift_height", 0.0)))
            if done[0]:
                break
        acts, qvs = np.array(acts), np.array(qvs)
        k = min(60, len(acts))
        max_clr_val = max(clr) if clr else 0.0
        print(f"\nEpisode {ep}: steps={steps} success={info.get('success', False)} "
              f"stage={info.get('task_stage', 1)} max_clearance_cm={max_clr_val*100:.1f}")
        print(f"  steps bilateral={bil} proper_grasp={proper} contact-lost(in stage2)={flick}")
        print("  action std (last %d steps) arm:" % k, acts[-k:, :5].std(0).round(3))
        print("  mean |action| last %d      arm:" % k, np.abs(acts[-k:, :5]).mean(0).round(3))
        print("  step-to-step |da|          arm:",
              np.abs(np.diff(acts[-k:, :5], axis=0)).mean(0).round(3))
        print("  gripper action last %d: mean=%.2f std=%.2f" % (k, acts[-k:, 5].mean(), acts[-k:, 5].std()))
        print("  qvel std last %d     robot:" % k, qvs[-k:].std(0).round(3))
    venv.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--vecnorm")
    ap.add_argument("--episodes", type=int, default=3)
    args = ap.parse_args()
    test_a()
    if args.model:
        test_b(args.model, args.vecnorm, args.episodes)

