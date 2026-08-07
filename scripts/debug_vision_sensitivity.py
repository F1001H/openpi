#!/usr/bin/env python3
"""Diagnostic-only, no critic, no motion: tests whether the loaded BC/JEPA
model's sample_actions output actually depends on the image input at all.
Written in response to Fabian's observation that removing the cube from the
scene didn't change the robot's motion -- captures one real camera frame,
builds two observations that are IDENTICAL except one has the real image and
the other has an all-zero (blank) image in its place, calls sample_actions on
both with the SAME rng/noise, and reports how different the outputs are.

If the model is genuinely vision-sensitive, blanking the image should
produce a substantially different action. If the outputs are nearly
identical, that's strong evidence the model isn't actually using the image
input (a real bug worth chasing down), independent of whether removing one
object (the cube) from an otherwise-unchanged real scene would have been a
big enough visual change to matter on its own.

Usage: uv run scripts/debug_vision_sensitivity.py [--config-name ...] [--checkpoint-exp-name full_lora_test] [--checkpoint-step 30000]
"""
import argparse
import dataclasses
import struct
import mmap

import cv2
import jax
import jax.numpy as jnp
import numpy as np
import posix_ipc
import rospy
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
    prev_frame_count = -1
    frame_count, active_buf = struct.unpack('Qi', mv[:12])
    while frame_count == prev_frame_count:
        frame_count, active_buf = struct.unpack('Qi', mv[:12])
    slot_offset = HEADER_SIZE + (((active_buf - 1) % 3) * CAM_SET_SIZE)
    rgb_data = mv[slot_offset + IMG_BYTES: slot_offset + (2 * IMG_BYTES)]
    rgb_data_external = mv[slot_offset + 2 * IMG_BYTES: slot_offset + (3 * IMG_BYTES)]
    img_np = np.frombuffer(rgb_data, dtype=np.uint8).reshape(HEIGHT, WIDTH, CHANNELS).copy()
    img_np_external = np.frombuffer(rgb_data_external, dtype=np.uint8).reshape(HEIGHT, WIDTH, CHANNELS).copy()
    img_np = cv2.flip(cv2.flip(img_np, 0), 1).transpose(2, 0, 1)
    img_np_external = cv2.flip(cv2.flip(img_np_external, 0), 1).transpose(2, 0, 1)
    return img_np, img_np_external


def get_live_state_8():
    rospy.init_node('debug_vision_sensitivity', anonymous=True)
    joint_state_pos = {"value": None}

    def _cb(msg):
        joint_state_pos["value"] = np.array(msg.position)

    import sensor_msgs.msg
    rospy.Subscriber('/panda_dual/joint_states', sensor_msgs.msg.JointState, _cb, queue_size=1)
    listener = tf.TransformListener()
    rospy.sleep(2)
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


def main(config_name: str, checkpoint_exp_name: str, checkpoint_step: int | None):
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

    import flax.nnx as nnx
    model = nnx.merge(train_state.model_def, jax.lax.stop_gradient(train_state.params))

    data_config = config.data.create(config.assets_dirs, config.model)
    normalize = _transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm)
    input_transform = _transforms.compose(
        [_transforms.InjectDefaultPrompt(None), *data_config.data_transforms.inputs, normalize, *data_config.model_transforms.inputs]
    )

    img_np, img_np_external = read_one_frame()
    state_8 = get_live_state_8()
    prompt = "pick up the orange cube and place it on the red tape"

    obs_real = build_observation(input_transform, img_np, img_np_external, state_8, prompt)
    blank = np.zeros_like(img_np)
    obs_blank = build_observation(input_transform, blank, blank, state_8, prompt)

    sample_rng = jax.random.key(0)
    noise = jax.random.normal(jax.random.key(1), (1, config.model.action_horizon, config.model.action_dim))

    actions_real = model.base_model.sample_actions(sample_rng, obs_real, num_steps=10, noise=noise)
    actions_blank = model.base_model.sample_actions(sample_rng, obs_blank, num_steps=10, noise=noise)

    diff = np.asarray(actions_real) - np.asarray(actions_blank)
    print(f"\nactions_real[0,0,:8]:  {np.asarray(actions_real)[0, 0, :8]}")
    print(f"actions_blank[0,0,:8]: {np.asarray(actions_blank)[0, 0, :8]}")
    print(f"\nmax abs diff across full [horizon, action_dim]: {np.abs(diff).max():.6f}")
    print(f"mean abs diff: {np.abs(diff).mean():.6f}")
    print(f"L2 norm of real:  {np.linalg.norm(actions_real):.6f}")
    print(f"L2 norm of diff:  {np.linalg.norm(diff):.6f}")
    print(f"relative diff (||diff|| / ||real||): {np.linalg.norm(diff) / np.linalg.norm(actions_real):.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", type=str, default="pi0_kobo_cube_low_mem")
    parser.add_argument("--checkpoint-exp-name", type=str, default="full_lora_test")
    parser.add_argument("--checkpoint-step", type=int, default=None)
    args = parser.parse_args()
    main(args.config_name, args.checkpoint_exp_name, args.checkpoint_step)
