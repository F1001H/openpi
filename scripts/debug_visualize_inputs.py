#!/usr/bin/env python3
"""Diagnostic-only, no model/critic/motion involved: reads one frame from
the live ZED camera SHM, runs it through the EXACT same repack-free input
pipeline inference_online.py/create_trained_policy use (KoboInputs ->
Normalize -> model_transforms), and dumps every intermediate image to PNG so
they can be visually inspected. Written to debug "the policy doesn't seem to
actually work" -- the fastest way to rule in/out a vision-input bug (wrong
camera mapped to the wrong slot, bad flip/transpose, wrong crop/resize) is to
just look at what the model is actually being fed.

Usage: uv run scripts/debug_visualize_inputs.py [--config-name pi0_kobo_cube_low_mem] [--out-dir /tmp/vis]
"""
import argparse
import os
import struct
import mmap

import cv2
import numpy as np
import posix_ipc

import openpi.training.config as _config
import openpi.transforms as _transforms

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
    header_peek = mv[:12]
    frame_count, active_buf = struct.unpack('Qi', header_peek)
    # Wait for a genuinely new frame rather than possibly reading one
    # mid-write.
    while frame_count == prev_frame_count:
        header_peek = mv[:12]
        frame_count, active_buf = struct.unpack('Qi', header_peek)

    slot_offset = HEADER_SIZE + (((active_buf - 1) % 3) * CAM_SET_SIZE)
    rgb_data = mv[slot_offset + IMG_BYTES: slot_offset + (2 * IMG_BYTES)]
    rgb_data_external = mv[slot_offset + 2 * IMG_BYTES: slot_offset + (3 * IMG_BYTES)]
    img_np = np.frombuffer(rgb_data, dtype=np.uint8).reshape(HEIGHT, WIDTH, CHANNELS).copy()
    img_np_external = np.frombuffer(rgb_data_external, dtype=np.uint8).reshape(HEIGHT, WIDTH, CHANNELS).copy()
    return img_np, img_np_external


def main(config_name: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    img_np_raw, img_np_external_raw = read_one_frame()
    # SHM buffer is RGB (matches the model's own convention throughout this
    # codebase) -- cv2.imwrite expects BGR, so convert for every saved PNG or
    # colors come out inverted (this bit me on the first pass: an orange
    # cube/red tape rendered as blue when saved without this conversion).
    cv2.imwrite(os.path.join(out_dir, "00_raw_main_cam.png"), cv2.cvtColor(img_np_raw, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, "00_raw_external_cam.png"), cv2.cvtColor(img_np_external_raw, cv2.COLOR_RGB2BGR))

    # Exactly mirror inference_online.py's _read_camera_and_state processing.
    img_np_external = cv2.flip(img_np_external_raw, 0)
    img_np_external = cv2.flip(img_np_external, 1)
    cv2.imwrite(os.path.join(out_dir, "01_flipped_external_cam.png"), cv2.cvtColor(img_np_external, cv2.COLOR_RGB2BGR))
    img_np_external = img_np_external.transpose(2, 0, 1)

    img_np = cv2.flip(img_np_raw, 0)
    img_np = cv2.flip(img_np, 1)
    cv2.imwrite(os.path.join(out_dir, "01_flipped_main_cam.png"), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    img_np = img_np.transpose(2, 0, 1)

    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    example = {
        "observation/image": img_np,
        "observation/external_image": img_np_external,
        "observation/state": np.zeros(8, dtype=np.float32),
        "prompt": np.array(["pick up the orange cube and place it on the red tape"]),
    }
    data = dict(example)
    for t in data_config.data_transforms.inputs:
        data = t(data)

    # data["image"] is a dict of {"base_0_rgb": ..., "left_wrist_0_rgb": ..., "right_wrist_0_rgb": ...}
    # -- exactly the model's actual camera inputs, in HWC uint8, per KoboInputs' _parse_image.
    for cam_name, img in data["image"].items():
        img_bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(out_dir, f"02_model_input_{cam_name}.png"), img_bgr)
        print(f"{cam_name}: shape={np.asarray(img).shape}, dtype={np.asarray(img).dtype}, "
              f"min={np.asarray(img).min()}, max={np.asarray(img).max()}")

    print(f"\nimage_mask: {data.get('image_mask')}")
    print(f"\nWrote images to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", type=str, default="pi0_kobo_cube_low_mem")
    parser.add_argument("--out-dir", type=str, default="/tmp/vis_inputs")
    args = parser.parse_args()
    main(args.config_name, args.out_dir)
