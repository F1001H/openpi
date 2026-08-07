#!/usr/bin/env python3
"""Diagnostic-only, no model/ROS involved: estimates how much the cube's
position actually varied across kobo_cube's 100 demo episodes, by running a
simple orange-color blob detector on each episode's first frame (the
external/base_0_rgb camera, which gives a clear top-down-ish view of the
cube + red tape target). Written to test the hypothesis that the policy
"overfit to an average trajectory" because the training data didn't vary the
cube's starting position much.

Usage: uv run scripts/debug_cube_position_variance.py [--data-root /home/fabian/kobo_cube]
"""
import argparse

import cv2
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


def detect_cube_centroid(frame_rgb: np.ndarray) -> tuple[float, float] | None:
    """frame_rgb: HWC uint8, RGB. Returns (x_frac, y_frac) centroid of the
    largest orange blob, as a fraction of image width/height, or None if no
    plausible blob found."""
    h_full, w_full = frame_rgb.shape[:2]
    # Restrict to the known workspace region -- background clutter (desk
    # objects, clothing) elsewhere in frame produces false-positive orange
    # blobs that don't correspond to the actual cube (caught by spot-checking
    # "outlier" detections against their source images: several supposedly
    # extreme x/y positions turned out to have the cube in the same spot as
    # everything else, with some OTHER orange-ish object driving the
    # measurement instead). Bounds are generous around the visually-confirmed
    # cluster of correct detections.
    roi_x0, roi_x1 = int(0.30 * w_full), int(0.80 * w_full)
    roi_y0, roi_y1 = int(0.15 * h_full), int(0.60 * h_full)
    roi = frame_rgb[roi_y0:roi_y1, roi_x0:roi_x1]

    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    # Orange: hue ~5-25 in OpenCV's 0-179 scale, require decent saturation/value
    # to avoid skin tones / washed-out background.
    lower = np.array([5, 120, 100])
    upper = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # Require a real cube-sized, roughly-square blob -- a low area threshold
    # (originally 50px^2) let small orange specks/background clutter get
    # picked as "largest" in some frames, producing false low/high outliers
    # that didn't match the actual cube's visible position (caught by
    # spot-checking outlier frames against their images).
    candidates = [c for c in contours if cv2.contourArea(c) >= 800]
    if not candidates:
        return None
    largest = max(candidates, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(largest)
    aspect = bw / bh if bh > 0 else 0
    if not (0.5 <= aspect <= 2.0):
        return None
    m = cv2.moments(largest)
    if m["m00"] == 0:
        return None
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    # Convert back to full-frame fractional coordinates.
    return (roi_x0 + cx) / w_full, (roi_y0 + cy) / h_full


def main(data_root: str):
    meta = LeRobotDatasetMetadata(root=data_root, repo_id=data_root)
    ds = LeRobotDataset(root=data_root, repo_id=data_root)

    ep_table = meta.episodes
    froms = np.asarray(ep_table["dataset_from_index"])
    num_episodes = len(froms)

    centroids = []
    missing = []
    for ep_i in range(num_episodes):
        frame_idx = int(froms[ep_i])
        item = ds[frame_idx]
        img = np.asarray(item["observation.images.cam2"])  # external cam, mapped to base_0_rgb
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        centroid = detect_cube_centroid(img)
        if centroid is None:
            missing.append(ep_i)
            continue
        centroids.append(centroid)
        if ep_i < 3 or ep_i % 20 == 0:
            cv2.imwrite(f"/tmp/cube_pos_ep{ep_i}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    centroids = np.array(centroids)
    print(f"Detected cube in {len(centroids)}/{num_episodes} episodes' first frames ({len(missing)} missed)")
    if len(missing) > 0:
        print(f"  missed episode indices (first 10): {missing[:10]}")
    print(f"\nCentroid x_frac: mean={centroids[:,0].mean():.4f} std={centroids[:,0].std():.4f} "
          f"min={centroids[:,0].min():.4f} max={centroids[:,0].max():.4f}")
    print(f"Centroid y_frac: mean={centroids[:,1].mean():.4f} std={centroids[:,1].std():.4f} "
          f"min={centroids[:,1].min():.4f} max={centroids[:,1].max():.4f}")
    print(f"\nSample .png previews written to /tmp/cube_pos_ep*.png for visual spot-check.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="/home/fabian/kobo_cube")
    args = parser.parse_args()
    main(args.data_root)
