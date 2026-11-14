# SO-101 Robot Arm Master Evaluation Summary

- **Execution Timestamp:** 2026-11-14 00:25:13
- **Evaluation Episodes per Benchmark:** 2
- **Seeds:** [42, 123, 456]
- **Total Runtime:** 11.6 s

## 1. Part A: Joint Sanity Sweep & Actuator Limits
- **Status:** PASSED
- **Actuators:** Feetech STS3215 Actuator Group (500 Hz physics, 25 Hz policy)

## 2. Part B1: 3-Zone Goal-Conditioned Precision Reaching

| Metric | Mean | Std | Unit |
|---|---|---|---|
| Success Rate | 33.33 | 57.74 | % |
| Euclidean Distance Error | 12.19 | 10.38 | mm |
| Reaching Time | 10.13 | 6.70 | s |

## 3. Part B2: SO-101 Pick-and-Lift Task Benchmark

| Metric | Mean | Std | Unit |
|---|---|---|---|
| Task Success Rate (Lift >= 5cm & 1.0s Hold) | 0.00 | 0.00 | % |
| Lift Milestone Attainment (>= 5cm) | 0.00 | 0.00 | % |
| Bilateral Grasp Rate | 33.33 | 28.87 | % |
| Max Lift Height Gain | 0.11 | 0.12 | cm |
| Max Steady Hold Duration | 0.00 | 0.00 | s |
| Episodic Return | 2.20 | 5.48 | reward |

