#!/usr/bin/env python3
"""Diagnostic-only, no critic, no motion: direct test of whether the model's
sample_actions output actually reacts to the CUBE specifically being present
vs absent (as opposed to debug_vision_sensitivity.py's cruder real-vs-blank-
image test). Captures one real frame, pauses for you to add/remove the cube
by hand (arm should stay still in between), captures a second real frame,
then compares sample_actions on (state, frame_A) vs (state, frame_B) using
the SAME proprio state for both (frame_A's) so the image is the only thing
that differs.

Usage: uv run scripts/debug_cube_sensitivity.py [--checkpoint-exp-name full_lora_test] [--checkpoint-step 30000]
"""
import argparse
import dataclasses
import struct
import mmap

import cv2
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import posix_ipc
import rospy
import sensor_msgs.msg
import tf

import openpi.models.model as _model
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.sharding as sharding
import openpi.transforms as _transforms

from train_end_to_end import init_train_state

WIDTH, HEIGHT = 1280, 720
CHANNELS = 3
IMG_BYTES = WIDTH * HEIGHT * CHANNELS
DEPTH_BYTES = WIDTH * HEIGHT * 4
CAM_SET_SIZE = (IMG_BYTES * 3) + (DEPTH_BYTES * 3)
HEADER_SIZE = 168


def read_one_frame():
    memory = posix_ipc.SharedMemory("zed_shm")
    map_file = mmap.mmap(memory.fd, memory.size)
    mv = memoryview(map_file)
    frame_count, active_buf = struct.unpack('Qi', mv[:12])
    prev = -1
    while frame_count == prev:
        frame_count, active_buf = struct.unpack('Qi', mv[:12])
    slot_offset = HEADER_SIZE + (((active_buf - 1) % 3) * CAM_SET_SIZE)
    rgb_data = mv[slot_offset + IMG_BYTES: slot_offset + (2 * IMG_BYTES)]
    rgb_data_external = mv[slot_offset + 2 * IMG_BYTES: slot_offset + (3 * IMG_BYTES)]
    img_np = np.frombuffer(rgb_data, dtype=np.uint8).reshape(HEIGHT, WIDTH, CHANNELS).copy()
    img_np_external = np.frombuffer(rgb_data_external, dtype=np.uint8).reshape(HEIGHT, WIDTH, CHANNELS).copy()
    img_np = cv2.flip(cv2.flip(img_np, 0), 1).transpose(2, 0, 1)
    img_np_external = cv2.flip(cv2.flip(img_np_external, 0), 1).transpose(2, 0, 1)
    return img_np, img_np_external


def get_live_state_8(listener, joint_state_pos):
    r_ee_tr, r_ee_rot = listener.lookupTransform('base_link', 'panda_right_hand', rospy.Time(0))
    if r_ee_rot[3] > 0.0:
        r_ee_rot = tuple(-c for c in r_ee_rot)
    qx, qy, qz, qw = r_ee_rot
    jsp = joint_state_pos["value"]
    gripper_flag = 1.0 if (2.0 * jsp[7]) > 0.04 else 0.0
    return np.array([*r_ee_tr, qx, qy, qz, qw, gripper_flag], dtype=np.float32)


def build_observation(input_transform, img_np, img_np_external, state_8, prompt):
    example = {
        "observation/image": img_np,
        "observation/external_image": img_np_external,
        "observation/state": state_8,
        "prompt": np.array([prompt]),
    }
    data = input_transform(dict(example))
    data = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], data)
    return _model.Observation.from_dict(data)


def capture(tag: str):
    """Mode A: just capture the current camera frame + live proprio state and
    save to /tmp/cube_sensitivity_<tag>.npz. No model loading -- fast, so the
    time between capture-A and capture-B (during which you physically change
    the scene) doesn't need to hold a GPU/model resident."""
    rospy.init_node('debug_cube_sensitivity_capture', anonymous=True)
    joint_state_pos = {"value": None}

    def _cb(msg):
        joint_state_pos["value"] = np.array(msg.position)

    rospy.Subscriber('/panda_dual/joint_states', sensor_msgs.msg.JointState, _cb, queue_size=1)
    listener = tf.TransformListener()
    rospy.sleep(2)

    img, img_ext = read_one_frame()
    state_8 = get_live_state_8(listener, joint_state_pos)
    out_path = f"/tmp/cube_sensitivity_{tag}.npz"
    np.savez(out_path, img=img, img_ext=img_ext, state_8=state_8)
    cv2.imwrite(f"/tmp/cube_sensitivity_{tag}.png", cv2.cvtColor(img.transpose(1, 2, 0), cv2.COLOR_RGB2BGR))
    print(f"Captured scene '{tag}': state_8={state_8}")
    print(f"Saved to {out_path} (and a .png preview)")


def compare(config_name: str, checkpoint_exp_name: str, checkpoint_step: int | None, tag_a: str, tag_b: str):
    """Mode B: loads both captured scenes, uses scene A's proprio state for
    BOTH observations (so the image is the only variable), and compares
    sample_actions outputs."""
    a = np.load(f"/tmp/cube_sensitivity_{tag_a}.npz")
    b = np.load(f"/tmp/cube_sensitivity_{tag_b}.npz")
    img_a, img_a_ext, state_a = a["img"], a["img_ext"], a["state_8"]
    img_b, img_b_ext = b["img"], b["img_ext"]

    config = _config.get_config(config_name)
    config = dataclasses.replace(config, exp_name=checkpoint_exp_name)
    mesh = sharding.make_mesh(1)
    rng = jax.random.key(config.seed)
    train_state_shape, _ = init_train_state(config, rng, mesh, resume=True, jepa_predictor_checkpoint=None)
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir, keep_period=config.keep_period, overwrite=False, resume=True,
    )
    assert resuming, f"No checkpoint at {config.checkpoint_dir}"
    train_state = _checkpoints.restore_state(checkpoint_manager, train_state_shape, None, step=checkpoint_step)
    print(f"Restored checkpoint step={int(train_state.step)}")
    model = nnx.merge(train_state.model_def, jax.lax.stop_gradient(train_state.params))

    data_config = config.data.create(config.assets_dirs, config.model)
    normalize = _transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm)
    input_transform = _transforms.compose(
        [_transforms.InjectDefaultPrompt(None), *data_config.data_transforms.inputs, normalize, *data_config.model_transforms.inputs]
    )
    prompt = "pick up the orange cube and place it on the red tape"

    obs_a = build_observation(input_transform, img_a, img_a_ext, state_a, prompt)
    obs_b = build_observation(input_transform, img_b, img_b_ext, state_a, prompt)

    sample_rng = jax.random.key(0)
    noise = jax.random.normal(jax.random.key(1), (1, config.model.action_horizon, config.model.action_dim))
    actions_a = model.base_model.sample_actions(sample_rng, obs_a, num_steps=10, noise=noise)
    actions_b = model.base_model.sample_actions(sample_rng, obs_b, num_steps=10, noise=noise)

    diff = np.asarray(actions_a) - np.asarray(actions_b)
    print(f"\nactions_{tag_a}[0,0,:8]: {np.asarray(actions_a)[0, 0, :8]}")
    print(f"actions_{tag_b}[0,0,:8]: {np.asarray(actions_b)[0, 0, :8]}")
    print(f"\nmax abs diff: {np.abs(diff).max():.6f}")
    print(f"mean abs diff: {np.abs(diff).mean():.6f}")
    print(f"L2 norm of {tag_a}: {np.linalg.norm(actions_a):.6f}")
    print(f"L2 norm of diff: {np.linalg.norm(diff):.6f}")
    print(f"relative diff (||diff|| / ||{tag_a}||): {np.linalg.norm(diff) / np.linalg.norm(actions_a):.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    p_capture = sub.add_parser("capture")
    p_capture.add_argument("--tag", type=str, required=True)

    p_compare = sub.add_parser("compare")
    p_compare.add_argument("--config-name", type=str, default="pi0_kobo_cube_low_mem")
    p_compare.add_argument("--checkpoint-exp-name", type=str, default="full_lora_test")
    p_compare.add_argument("--checkpoint-step", type=int, default=None)
    p_compare.add_argument("--tag-a", type=str, default="A")
    p_compare.add_argument("--tag-b", type=str, default="B")

    args = parser.parse_args()
    if args.mode == "capture":
        capture(args.tag)
    else:
        compare(args.config_name, args.checkpoint_exp_name, args.checkpoint_step, args.tag_a, args.tag_b)
