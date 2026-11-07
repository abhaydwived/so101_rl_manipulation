# Observation and Action Space Design Document: SO-101 Manipulator

## 1. Action Space Design & Comparison

### Selected Action Representation: Normalized Joint Position Deltas ($\Delta q$)
Rather than commanding raw actuator torques or absolute joint angles directly from the neural policy, the SO-101 RL framework uses **normalized bounded joint position deltas**:
$$a_t \in [-1, 1]^d \implies \Delta q_t = a_t \cdot \Delta q_{max}$$
$$q_{target, t} = \text{clip}(q_{target, t-1} + \Delta q_t, q_{low}, q_{high})$$

### Comparative Tradeoff Analysis: Absolute vs. Delta Position Targets

| Dimension | Absolute Joint Position Targets ($q_{abs}$) | Joint Position Deltas ($\Delta q$) [Selected] |
| :--- | :--- | :--- |
| **Exploration Safety** | Large random jumps during early exploration cause severe actuator torque saturation ($> 2.94\text{ N}\cdot\text{m}$) and instability. | Smooth local exploration; joint transitions are strictly bounded by $\Delta q_{max} = 0.035\text{ rad}$ ($\sim 2.0^\circ$). |
| **Sim-to-Real Transfer** | Discontinuous jumps trigger servo internal velocity limits and high tracking error in Feetech STS3215 firmware. | Commands inherently respect servo tracking bandwidth, minimizing mechanical jerk and gear chatter. |
| **Action Smoothness** | Policy frequently outputs oscillating distant setpoints, resulting in high $|a_t - a_{t-1}|$ action jerk. | Smooth trajectory generation with explicit action difference penalties: $r_{smooth} = -\lambda \|\Delta a_t\|^2$. |
| **Policy Invariance** | Non-Markovian jump hazards; requires memorizing absolute global angles. | Relative differential actions naturally form smooth velocity-like trajectories. |

### Action Dimensions:
- **Part B1 (Reaching)**: **5-D continuous Box($-1, 1$)** representing position deltas for `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll]`.
- **Part B2 (Pick and Place)**: **6-D continuous Box($-1, 1$)** representing 5 arm position deltas + 1 continuous/thresholded gripper command:
  - $a[5] > 0.0 \implies q_{gripper} = -0.10\text{ rad}$ (Firm grip on 3 cm cube).
  - $a[5] \le 0.0 \implies q_{gripper} = +1.00\text{ rad}$ (Wide open jaws).

---

## 2. Observation Space Design & Justification

The old Franka Panda 29-D observation vector included hardcoded 7-DOF joints, two prismatic finger positions, and Panda-specific attachment site vectors. It has been redesigned into **task-specific compact state representations**:

### A. Part B1 Reaching Task Observation Space (21 Dimensions)
The reaching policy requires only state variables necessary to localize the end-effector relative to the target and assess hold stability:

| Component | Dimensions | Description & Physical Meaning |
| :--- | :---: | :--- |
| Arm Joint Positions | 5 | $q \in \mathbb{R}^5$, quantized to 12-bit magnetic encoder resolution ($0.088^\circ$). |
| Arm Joint Velocities | 5 | $\dot{q} \in \mathbb{R}^5$, servo joint velocity feedback. |
| End-Effector Position | 3 | $p_{ee} = [x, y, z] \in \mathbb{R}^3$, Cartesian coordinates of `gripperframe`. |
| Target Position | 3 | $p_{tgt} = [x, y, z] \in \mathbb{R}^3$, 3D coordinates of randomized goal marker. |
| Relative Vector | 3 | $p_{tgt} - p_{ee}$, direct translation error vector for linear guidance. |
| Euclidean Distance | 1 | $\|p_{tgt} - p_{ee}\|$, scalar distance for value function monotonic estimation. |
| Hold Progress Ratio | 1 | $N_{hold} / 13.0$, ratio indicating progress toward the $0.5\text{ s}$ stability requirement. |
| **Total** | **21** | Continuous Box($-10.0, 10.0$) |

### B. Part B2 Pick-and-Place Task Observation Space (30 Dimensions)
The pick-and-place policy requires tracking both the arm and the manipulated cube, plus the target drop zone:

| Component | Dimensions | Description & Physical Meaning |
| :--- | :---: | :--- |
| Arm Joint Positions | 5 | $q \in \mathbb{R}^5$, quantized to 12-bit magnetic encoder resolution. |
| Arm Joint Velocities | 5 | $\dot{q} \in \mathbb{R}^5$, joint velocity feedback. |
| Gripper Joint Position | 1 | $q_{grip} \in [-0.17, 1.74]\text{ rad}$, actual jaw opening angle. |
| End-Effector Position | 3 | $p_{ee} = [x, y, z] \in \mathbb{R}^3$, Cartesian tip position. |
| Cube Position | 3 | $p_{cube} = [x, y, z] \in \mathbb{R}^3$, 3D center of the manipulated block. |
| Cube Orientation | 4 | $q_{cube} = [w, x, y, z] \in \mathbb{R}^4$, unit quaternion representing cube rotation. |
| Target Zone Position | 3 | $p_{zone} = [x, y, z] \in \mathbb{R}^3$, 3D coordinates of target drop zone. |
| EE to Cube Vector | 3 | $p_{cube} - p_{ee}$, relative vector for approach, alignment, and grasp. |
| Cube to Target Vector | 3 | $p_{zone} - p_{cube}$, relative vector for transport and placement. |
| **Total** | **30** | Continuous Box($-10.0, 10.0$) |

---

## 3. Observation Normalization Methodology

To ensure numerical stability across heterogeneous physical units (radians, rad/s, meters, quaternions):
1. **Running Statistics Normalization (`VecNormalize`)**:
   $$x_{norm} = \text{clip}\left(\frac{x - \mu_{running}}{\sqrt{\sigma_{running}^2 + 10^{-8}}}, -10.0, 10.0\right)$$
   Maintained across parallel vectorized environments during training and serialized to `so101_*_vecnorm.pkl`.
2. **Deterministic Evaluation**:
   During held-out deterministic benchmarking, running statistics updating is frozen (`training=False, norm_reward=False`), ensuring strictly stationary input normalization.
