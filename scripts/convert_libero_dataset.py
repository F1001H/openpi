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

RESUMABLE: if --out-root already exists with a loadable partial dataset
(i.e. at least one episode's metadata was fully flushed), this script
reopens it via the plain LeRobotDataset(repo_id=..., root=...) constructor
-- which lerobot's own recording API supports resuming into, same as
resuming an interrupted robot data-collection session -- and continues from
meta.total_episodes instead of restarting at episode 0. metadata_buffer_size
is set to 1 (flush after every episode, not lerobot's default batch-of-10)
specifically so a crash never loses more than the ONE episode that was
in-progress at the time, regardless of when it happens. Re-running the exact
same command after a crash is the intended recovery path -- no flags to
change. (--start-episode/--num-episodes are a SEPARATE, unrelated knob: they
exist to let you convert a small slice for a quick sanity check by pointing
a throwaway --out-root at just --num-episodes=10.)

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
import pathlib
import shutil

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

    local_repo_id = f"local/{repo_id.split('/')[-1]}"
    out_root_path = pathlib.Path(out_root)
    ds = None
    # Check the filesystem directly BEFORE ever calling the plain
    # LeRobotDataset(repo_id=..., root=...) constructor: that constructor
    # eagerly mkdir()s root, and -- if local metadata fails to load --
    # falls into ITS OWN internal network-fallback path, which tries to
    # validate local_repo_id ("local/...", not a real Hub id) against the
    # Hub and raises RepositoryNotFoundError (a 401, NOT a clean
    # FileNotFoundError/NotADirectoryError we could catch cleanly). Same
    # failure family as this session's earlier fix for a fresh empty
    # out_root -- gate on the real, on-disk signal instead of relying on
    # the constructor's own exception type.
    if (out_root_path / "meta" / "info.json").exists():
        ds = LeRobotDataset(repo_id=local_repo_id, root=out_root)
        # metadata_buffer_size isn't a constructor kwarg on the plain
        # LeRobotDataset(...) path (only on .create()) -- it's just a plain
        # attribute read at flush-decision time (save_episode's
        # `len(self.metadata_buffer) >= self.metadata_buffer_size` check),
        # so overriding it directly here is safe and has the same effect.
        ds.meta.metadata_buffer_size = 1
        ds.start_image_writer(num_threads=image_writer_threads)
        if ds.meta.total_episodes > 0:
            logging.info(f"Resuming existing dataset at {out_root}: {ds.meta.total_episodes} episodes already done.")
            start_episode = max(start_episode, ds.meta.total_episodes)
    elif out_root_path.exists():
        # Exists but no meta/info.json -- either a stray empty dir (e.g.
        # from a run that crashed before .create() finished, or a leftover
        # from this branch on a previous attempt) or genuinely unrelated
        # contents. Nothing resumable is here either way -- .create() below
        # requires the directory to NOT exist (exist_ok=False), so clear it.
        logging.info(f"{out_root} exists but has no meta/info.json (nothing resumable) -- clearing it.")
        shutil.rmtree(out_root_path)

    if ds is None:
        ds = LeRobotDataset.create(
            repo_id=local_repo_id,
            fps=fps,
            features=features,
            root=out_root,
            robot_type=robot_type,
            use_videos=False,
            image_writer_threads=image_writer_threads,
            metadata_buffer_size=1,
        )

    if start_episode >= end_episode:
        logging.info(f"Nothing to do: {start_episode} episodes already done, target was {end_episode}.")
        ds.finalize()
        return

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
