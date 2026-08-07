#!/usr/bin/env python3
"""Diagnostic-only, no model/critic/motion: reads the live proprio state
exactly the way inference_online.py's _read_camera_and_state builds it
(r_ee_tr + quaternion + thresholded gripper flag), and compares it against
the real training dataset's observation.state distribution (per-dim min/max/
mean from meta/stats.json) to catch state-vector bugs (wrong frame, wrong
units, wrong gripper convention/direction) that a pure image check can't.

Usage: uv run scripts/debug_check_proprio.py [--data-root /home/fabian/kobo_cube]
"""
import argparse
import json

import numpy as np
import rospy
import tf
import tf.transformations as tft


def main(data_root: str):
    rospy.init_node('debug_check_proprio', anonymous=True)

    joint_state_pos = {"value": None}

    def _cb(msg):
        joint_state_pos["value"] = np.array(msg.position)

    rospy.Subscriber('/panda_dual/joint_states', __import__('sensor_msgs.msg', fromlist=['JointState']).JointState, _cb, queue_size=1)

    listener = tf.TransformListener()
    rospy.loginfo("Waiting for TF + joint_states...")
    rospy.sleep(2)

    WORLD_FRAME = 'base_link'
    r_ee_tr, r_ee_rot = listener.lookupTransform(WORLD_FRAME, 'panda_right_hand', rospy.Time(0))
    print(f"\n  raw quaternion from tf (before hemisphere fix): {r_ee_rot}")
    # Same canonicalization as inference_online.py's _read_camera_and_state
    # (added 2026-07-31 after this exact script caught the sign-flip bug).
    if r_ee_rot[3] > 0.0:
        r_ee_rot = tuple(-c for c in r_ee_rot)
    qx, qy, qz, qw = r_ee_rot

    if joint_state_pos["value"] is None:
        raise RuntimeError("Never received a /panda_dual/joint_states message.")
    jsp = joint_state_pos["value"]

    # Same convention as inference_online.py's _read_camera_and_state.
    finger_pos = jsp[7]  # panda_right_finger_joint1
    gripper_width_raw = 2.0 * finger_pos
    gripper_open = gripper_width_raw > 0.04
    gripper_flag = 1.0 if gripper_open else 0.0

    state_8 = np.array([*r_ee_tr, qx, qy, qz, qw, gripper_flag], dtype=np.float32)

    print("\n=== Live proprio (as fed to the model) ===")
    labels = ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper_flag"]
    for label, val in zip(labels, state_8):
        print(f"  {label:>13}: {val:.4f}")
    print(f"\n  raw finger joint pos (panda_right_finger_joint1): {finger_pos:.4f}")
    print(f"  raw gripper_width (2x finger pos): {gripper_width_raw:.4f}  -> thresholded flag: {gripper_flag}")

    stats_path = f"{data_root}/meta/stats.json"
    with open(stats_path) as f:
        stats = json.load(f)
    state_stats = stats["observation.state"]
    print(f"\n=== Training dataset's observation.state stats (from {stats_path}) ===")
    for i, label in enumerate(labels):
        lo, hi = state_stats["min"][i], state_stats["max"][i]
        mean = state_stats["mean"][i]
        val = state_8[i]
        in_range = lo <= val <= hi
        flag = "OK" if in_range else "*** OUT OF TRAINING RANGE ***"
        print(f"  {label:>13}: train_min={lo:.4f} train_max={hi:.4f} train_mean={mean:.4f}  |  live={val:.4f}  {flag}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="/home/fabian/kobo_cube")
    args = parser.parse_args()
    main(args.data_root)
