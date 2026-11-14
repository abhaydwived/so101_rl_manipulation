"""
SO-101 Gymnasium Environments Package.

Exposes the active task environments:
    SO101ReachingEnv           — Part B1: goal-conditioned 3-zone reaching task
    SO101PickLiftEnv — Part B2: hierarchical pick-and-place with 4-stage state machine
"""

from environments.reaching_env import SO101ReachingEnv
from environments.pick_and_lift_env import SO101PickLiftEnv

__all__ = ["SO101ReachingEnv", "SO101PickLiftEnv"]
