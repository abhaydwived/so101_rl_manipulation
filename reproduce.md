```bash
# Evaluate 3-Zone Goal-Conditioned Precision Reaching (500 targets)
python Evaluate_and_test/evaluate_reach.py --episodes 500

# Evaluate Contact-Rich Pick-and-Lift with Random Cube Placement (200 episodes)
python Evaluate_and_test/evaluate_pick_and_lift.py --episodes 200 --randomize-cube-place

# Mujoco Simulation Visualization (Paced to 1.0x Real-Time Speed)
python Evaluate_and_test/evaluate_reach.py --render --episodes 10 
python Evaluate_and_test/evaluate_pick_and_lift.py --render --episodes 10 --randomize-cube-place
```