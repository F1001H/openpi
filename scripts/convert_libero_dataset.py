#!/usr/bin/env python3
"""One-time conversion of physical-intelligence/libero into a genuine local
LeRobot v3.0 dataset.

WHY THIS EXISTS: the installed `lerobot` (0.4.4) only loads codebase-version
v3.0 datasets -- anything older raises BackwardCompatibilityError. LeRobot
ships an official v2.1->v3.0 converter
(lerobot.datasets.v30.convert_dataset_v21_to_v30), but physical-intelligence/
libero on the Hub is codebase_version "v2.0", not "v2.1" -- confirmed via
meta/info.json and via api.list_repo_refs (no "v2.1" tag exists, only
"main"/"v2.0"). The official converter hardcodes revision="v2.1" for its
download step and unconditionally requires meta/episodes_stats.jsonl (a
v2.1-only per-episode stats file this v2.0 dataset doesn't have) --
patching around that safely would mean re-deriving those stats by hand, at
which point you're maintaining a shadow implementation of the converter
anyway.

This script sidesteps the incompatibility rather than patching it: it reads
the raw v2.0 parquet files directly (pandas/PIL -- no lerobot version
dependency there) and re-records them through the INSTALLED lerobot's own
supported recording API (LeRobotDataset.create() + add_frame() +
save_episode()), which by construction writes a real v3.0-format dataset
this same lerobot install can then read. Verified against a 4-episode subset
locally before running this at full scale (see conversation/session notes;
scripts/qc_label_rewards.py and scripts/train_qc_critic.py ran cleanly
against the result).

This is a ONE-TIME, CPU-only, no-GPU-needed step -- run it once (e.g. via
slurm/prepare_libero_dataset.slurm on the cluster) and point every
pi05_libero_low_mem run's --data.root at the resulting directory afterward.

NOT resumable in place: LeRobotDataset.create() requires --out-root to not
already exist (it's a fresh-write API, not an append one). --start-episode/
--num-episodes exist to let you convert a small slice for a quick sanity
check (point a throwaway --out-root at just --num-episodes=10, inspect it,
then delete it and run the real full conversion) -- if a full run dies
partway through, delete --out-root and rerun from scratch rather than trying
to resume with --start-episode into the same directory.

Usage:
    uv run scripts/convert_libero_dataset.py \
        --repo-id=physical-intelligence/libero \
        --out-root=/path/to/libero_v3 \
        [--num-episodes=N] [--start-episode=N] [--image-writer-threads=8]
"""

import argparse
import io
import json
import logging

import jsonlines
import numpy as np
import pandas as pd
import tqdm
from PIL import Image
from huggingface_hub import hf_hub_download, snapshot_download

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main(
    repo_id: str,
    out_root: str,
    start_episode: int,
    num_episodes: int | None,
    image_writer_threads: int,
    fps: int | None,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    info_path = hf_hub_download(repo_id, "meta/info.json", repo_type="dataset")
    with open(info_path) as f:
        info = json.load(f)
    total_episodes = info["total_episodes"]
    fps = fps or info["fps"]
    robot_type = info.get("robot_type")

    end_episode = total_episodes if num_episodes is None else min(total_episodes, start_episode + num_episodes)
    logging.info(f"Converting episodes [{start_episode}, {end_episode}) of {total_episodes} from {repo_id}")

    # Fetch all raw parquet files (+ tasks) needed for this episode range in
    # one batched, HF-parallelized call, rather than one hf_hub_download per
    # episode -- much faster for hundreds/thousands of small files.
    chunks_size = info.get("chunks_size", 1000)
    chunks_needed = sorted({e // chunks_size for e in range(start_episode, end_episode)})
    allow_patterns = ["meta/tasks.jsonl"] + [f"data/chunk-{c:03d}/*.parquet" for c in chunks_needed]
    logging.info(f"Downloading raw v2.x files (chunks {chunks_needed})...")
    local_dir = snapshot_download(repo_id, repo_type="dataset", allow_patterns=allow_patterns)

    with jsonlines.open(f"{local_dir}/meta/tasks.jsonl") as reader:
        task_by_index = {row["task_index"]: row["task"] for row in reader}

    # Feature dtypes/shapes are read straight off the source's own
    # info.json, minus the bookkeeping columns LeRobotDataset.create()
    # already adds itself via DEFAULT_FEATURES (timestamp/frame_index/
    # episode_index/index/task_index) -- including those again would
    # conflict. video features are unsupported here: this script assumes
    # image-only cameras (physical-intelligence/libero has none, confirmed
    # via api.dataset_info).
    _bookkeeping_keys = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    video_keys = [k for k, spec in info["features"].items() if spec["dtype"] == "video"]
    if video_keys:
        raise RuntimeError(f"Unsupported video-typed feature(s) {video_keys} -- this script assumes image-only cameras.")

    features = {}
    for key, spec in info["features"].items():
        if key in _bookkeeping_keys:
            continue
        features[key] = {"dtype": spec["dtype"], "shape": tuple(spec["shape"]), "names": spec.get("names")}
    image_keys = [k for k, v in features.items() if v["dtype"] == "image"]
    non_image_keys = [k for k in features if k not in image_keys]
    logging.info(f"image_keys={image_keys} other_keys={non_image_keys}")

    ds = LeRobotDataset.create(
        repo_id=f"local/{repo_id.split('/')[-1]}",
        fps=fps,
        features=features,
        root=out_root,
        robot_type=robot_type,
        use_videos=False,
        image_writer_threads=image_writer_threads,
    )

    chunks_size = info.get("chunks_size", 1000)
    for ep_idx in tqdm.tqdm(range(start_episode, end_episode), desc="episodes"):
        chunk_idx = ep_idx // chunks_size
        ep_path = f"{local_dir}/data/chunk-{chunk_idx:03d}/episode_{ep_idx:06d}.parquet"
        df = pd.read_parquet(ep_path)
        for _, row in df.iterrows():
            frame = {}
            for key in image_keys:
                frame[key] = np.array(Image.open(io.BytesIO(row[key]["bytes"])).convert("RGB"))
            for key in non_image_keys:
                frame[key] = np.asarray(row[key], dtype=np.float32)
            frame["task"] = task_by_index[int(row["task_index"])]
            ds.add_frame(frame)
        ds.save_episode()

    ds.finalize()
    logging.info(f"Wrote episodes [{start_episode}, {end_episode}) to {out_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, default="physical-intelligence/libero")
    parser.add_argument("--out-root", type=str, required=True)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="Convert only this many episodes starting at --start-episode (default: all remaining). "
        "Useful for a quick local check before committing to a full run.",
    )
    parser.add_argument("--image-writer-threads", type=int, default=8)
    parser.add_argument("--fps", type=int, default=None, help="Override source fps (default: read from meta/info.json).")
    args = parser.parse_args()

    main(args.repo_id, args.out_root, args.start_episode, args.num_episodes, args.image_writer_threads, args.fps)
