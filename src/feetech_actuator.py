import numpy as np


class FeetechSTS3215ActuatorGroup:
    """
    Simulates the internal microcontroller and motor dynamics of a group of
    Feetech STS3215 serial bus smart servos.

    Features:
    - Commanded position targets (rad)
    - Substep velocity limiting (max ~5.5 rad/s)
    - First-order lag (internal low-pass filter / motor inductance)
    - Internal PD tracking equivalent: tau = Kp*(target - q) - Kv*qvel
    - Torque saturation at stall limit (2.94 Nm)
    - Quantization to 12-bit magnetic encoder resolution (0.088 deg = 0.00153 rad)
    """

    def __init__(
        self,
        n_actuators=6,
        kp=250.0,
        kv=12.0,
        max_torque=2.94,
        max_velocity=5.50,
        time_constant=0.025,
        dt_physics=0.002,
    ):
        self.n_actuators = n_actuators
        self.kp = np.full(n_actuators, kp, dtype=np.float64)
        self.kv = np.full(n_actuators, kv, dtype=np.float64)
        self.max_torque = np.full(n_actuators, max_torque, dtype=np.float64)
        self.max_velocity = np.full(n_actuators, max_velocity, dtype=np.float64)
        self.time_constant = time_constant
        self.dt = dt_physics

        # Encoder resolution (12-bit: 4096 steps per 2*pi)
        self.encoder_resolution = (2.0 * np.pi) / 4096.0  # ~0.00153 rad

        # Filtered internal target position
        self.internal_target = np.zeros(n_actuators, dtype=np.float64)

    def reset(self, initial_qpos):
        self.internal_target[:] = np.asarray(initial_qpos, dtype=np.float64)

    def quantize_encoder(self, qpos):
        """Quantizes continuous angle to 12-bit magnetic encoder resolution."""
        steps = np.round(qpos / self.encoder_resolution)
        return steps * self.encoder_resolution

    def step(self, commanded_qpos, current_qpos, current_qvel):
        """
        Executes one physics substep of internal servo response.
        Returns:
            actuator_torques: Torques computed by servo MCU, clipped to stall limits.
            filtered_target: The actual tracked internal servo position.
        """
        commanded = np.asarray(commanded_qpos, dtype=np.float64)
        current_q = np.asarray(current_qpos, dtype=np.float64)
        current_v = np.asarray(current_qvel, dtype=np.float64)

        # 1. First-order servo response / command lag
        alpha = self.dt / (self.time_constant + self.dt)
        filtered_cmd = self.internal_target + alpha * (commanded - self.internal_target)

        # 2. Rate limiting (velocity clamp on target trajectory)
        max_delta = self.max_velocity * self.dt
        delta = np.clip(filtered_cmd - self.internal_target, -max_delta, max_delta)
        self.internal_target += delta

        # 3. Internal PD position loop
        err = self.internal_target - current_q
        torques = self.kp * err - self.kv * current_v

        # 4. Physical torque saturation
        clipped_torques = np.clip(torques, -self.max_torque, self.max_torque)

        return clipped_torques, self.internal_target.copy()

