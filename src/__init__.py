"""
SO-101 Source Utilities Package.

Exposes:
    FeetechSTS3215ActuatorGroup — servo actuator model at 500 Hz physics
    SO101ModelMapper            — config-driven joint/actuator validation
    so101_damped_least_squares_ik — damped-least-squares IK solver

Note: PickPlaceMetricsCallback is intentionally NOT imported here to avoid
pulling in stable_baselines3 and TensorFlow at module-load time.
Import it directly: from src.metrics_callback import PickPlaceMetricsCallback
"""

from src.common import configure_env
from src.feetech_actuator import FeetechSTS3215ActuatorGroup
from src.so101_model_mapper import SO101ModelMapper
from src.so101_kinematics import so101_damped_least_squares_ik

__all__ = [
    "configure_env",
    "FeetechSTS3215ActuatorGroup",
    "SO101ModelMapper",
    "so101_damped_least_squares_ik",
]
