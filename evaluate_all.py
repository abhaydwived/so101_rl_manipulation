#!/usr/bin/env python3
"""
Master one-command evaluation script for MUKS Robotics take-home deliverables.
Runs:
1. Part A: Joint sanity sweep and limits verification.
2. Part B1: 500-target reaching held-out benchmark.
3. Part B2: 200-episode pick-and-place held-out benchmark.
4. Sim-to-real readiness and failure mode synthesis.
"""

import os
import sys

# Prevent Windows EPERM permission conflict with corrupted .keras/keras.json
sys.modules["tensorflow"] = None

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from sanity_check_joints import run_joint_sanity_sweep
from evaluate_so101_reach import evaluate_so101_reach
from evaluate_so101_pick_place import evaluate_so101_pick_place


def run_all_evaluations(render=False):
    print("##################################################################")
    print("   MUKS ROBOTICS SO-101 RL TAKE-HOME: COMPLETE EVALUATION SUITE   ")
    print("##################################################################\n")

    # Step 1: Joint Sanity Sweep
    print(">>> STEP 1/3: RUNNING PART A JOINT SANITY SWEEP...")
    run_joint_sanity_sweep(
        model_path="so101/reach_scene.xml",
        config_path="configs/so101_config.yaml",
        output_dir="reports/sanity_check",
    )

    # Step 2: Part B1 500 Targets Evaluation
    print("\n>>> STEP 2/3: RUNNING PART B1 REACHING BENCHMARK (500 TARGETS)...")
    evaluate_so101_reach(
        model_path="models/so101/reach/seed_42/so101_reach_final.zip",
        vecnorm_path="models/so101/reach/seed_42/so101_reach_vecnorm.pkl",
        n_targets=500,
        output_dir="reports/evaluation_reach",
        render=render,
    )

    # Step 3: Part B2 200 Episodes Pick and Place Evaluation
    print("\n>>> STEP 3/3: RUNNING PART B2 PICK AND PLACE BENCHMARK (200 EPISODES)...")
    evaluate_so101_pick_place(
        model_path="models/so101/pick_place/seed_42/so101_pick_place_final.zip",
        vecnorm_path="models/so101/pick_place/seed_42/so101_pick_place_vecnorm.pkl",
        n_episodes=200,
        output_dir="reports/evaluation_pick_place",
        render=render,
    )

    print("\n##################################################################")
    print("ALL EVALUATIONS SUCCESSFULLY COMPLETED!")
    print("Generated Deliverables:")
    print("  - Part A Reports: reports/sanity_check/")
    print("  - Part B1 CSV & Figures: reports/evaluation_reach/")
    print("  - Part B2 CSV & Figures: reports/evaluation_pick_place/")
    print("##################################################################")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="Render simulation live in MuJoCo viewer during evaluation")
    args = parser.parse_args()

    run_all_evaluations(render=args.render)

