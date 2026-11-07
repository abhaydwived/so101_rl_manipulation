import numpy as np
import mujoco


def so101_forward_kinematics(model, data, qpos, site_name="gripperframe"):
    """
    Computes forward kinematics for the SO-101 manipulator.
    Returns 3D Cartesian coordinates of the requested site.
    """
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"Site '{site_name}' not found in model.")

    data.qpos[:len(qpos)] = qpos
    mujoco.mj_forward(model, data)
    return data.site_xpos[site_id].copy()


def so101_get_jacobian(model, data, site_name="gripperframe"):
    """
    Returns 3xnv translational Jacobian for the requested site.
    """
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    return jacp


def so101_damped_least_squares_ik(
    model,
    data,
    target_pos,
    site_name="gripperframe",
    n_arm_joints=5,
    damping=0.03,
    max_iters=150,
    tol=1e-3,
    step_scale=0.8,
):
    """
    Damped Least-Squares (DLS) Inverse Kinematics for SO-101 5-DOF arm.
    Operates on a scratch MjData copy to avoid disturbing simulation state.
    Returns (qpos_arm, final_error, n_iters, converged).
    """
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos[:]
    scratch.qvel[:] = 0.0

    arm_limits_low = model.jnt_range[:n_arm_joints, 0]
    arm_limits_high = model.jnt_range[:n_arm_joints, 1]

    for it in range(max_iters):
        mujoco.mj_forward(model, scratch)
        ee_pos = scratch.site_xpos[site_id]
        error = target_pos - ee_pos
        err_norm = np.linalg.norm(error)

        if err_norm < tol:
            return scratch.qpos[:n_arm_joints].copy(), err_norm, it, True

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, scratch, jacp, jacr, site_id)
        J = jacp[:, :n_arm_joints]

        JJt = J @ J.T
        lam_sq = damping ** 2
        dq = J.T @ np.linalg.solve(JJt + lam_sq * np.eye(3), error)

        scratch.qpos[:n_arm_joints] += step_scale * dq
        # Enforce joint limits
        scratch.qpos[:n_arm_joints] = np.clip(
            scratch.qpos[:n_arm_joints], arm_limits_low, arm_limits_high
        )

    mujoco.mj_forward(model, scratch)
    final_err = np.linalg.norm(target_pos - scratch.site_xpos[site_id])
    converged = final_err < (tol * 2.0)
    return scratch.qpos[:n_arm_joints].copy(), final_err, max_iters, converged
