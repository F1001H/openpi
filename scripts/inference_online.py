#!/usr/bin/env python3
"""DRAFT, UNTESTED -- Phase 2b of the Q-chunking post-training plan
(.claude/plans/velvet-whistling-gosling.md; see also memory file
project_qchunking_posttraining). Copy of scripts/inference.py (the working
BC-only real-robot deployment -- NOT modified by this file) with
policy.infer() replaced by best-of-N critic-scored action selection
(src/qc/actor.py), plus intrinsic-reward computation and replay-buffer
logging for later offline critic updates.

**DO NOT RUN THIS ON THE REAL ROBOT WITHOUT FABIAN'S LINE-BY-LINE REVIEW**,
plus a supervised dry run at a small --num-samples with a human at the
E-stop the whole time. This was written with no ROS/hardware environment to
test against -- only the underlying JAX/critic pieces (best_of_n_action_batch,
compute_intrinsic_reward, checkpoint loading) have been exercised standalone.

Loads the base BC/JEPA model directly via init_train_state +
checkpoints.restore_state (train_end_to_end.py's own pattern), NOT through
policy_config.create_trained_policy(), for two reasons: (1) that helper only
exposes the wrapped base_model, not OpenPIWithJEPA's extract_vision_latents/
jepa_predictor needed for compute_intrinsic_reward, and (2) it loads
norm_stats from the CHECKPOINT'S OWN SAVED assets -- for any checkpoint
trained before the 2026-07-27 norm_stats fix (see memory) those would be the
stale, pre-fix stats. Calling config.data.create(...) directly here always
gets the current, correct stats, and works for a post-fix checkpoint like
checkpoints/pi0_kobo_cube_low_mem/full_lora_test (the real, converged
30k-step run -- default below -- or the earlier qc_phase2_smoke_postfix
smoke-test checkpoint).

KNOWN OPEN ISSUE, flagged for Fabian, partially resolved: inference.py's live
`current_task_space_state` is built as r_ee_tr (3) + quaternion (4) = 7 dims,
but the training dataset's observation.state is 8-dim (includes a gripper
flag, see meta/info.json). This file adds a live 8th dimension (in
`_read_camera_and_state`) to match PROPRIO_DIM/the critic's training
convention. CONFIRMED 2026-07-30 against the real kobo_cube dataset: that
8th column (both observation.state and action) is a BINARY {0.0, 1.0} flag,
not a continuous width in meters -- an earlier version of this file read a
continuous `2 * joint_state_pos[7]` value here, which was wrong. Now
thresholds the same joint reading into a 0/1 flag instead. STILL UNVERIFIED:
which physical state (open vs closed) maps to 1.0 vs 0.0 -- assumed
open=1.0 (see `_read_camera_and_state`'s comment for the reasoning), not
confirmed against the actual data-collection code. Confirm this before
trusting reward/critic values that depend on it.
"""

import dataclasses
import functools
import os
import struct
import sys
import time

import actionlib
import cv2
import etils.epath as epath
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import posix_ipc
import mmap
import rospy
import tf
import tf.transformations as tft
from dual_panda_multi_mode_controllers.msg import ControlMode, SwitchControlActionGoal
from franka_gripper.msg import MoveAction, MoveGoal
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import openpi.models.model as _model
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.sharding as sharding
import openpi.transforms as _transforms

from jepa.train_step_transitions import compute_intrinsic_reward
from qc.actor import best_of_n_action_batch
from qc.checkpoint import load_critic

from train_end_to_end import init_train_state


# --- MATCH C++ CONSTANTS (identical to inference.py) ---
WIDTH, HEIGHT = 1280, 720
CHANNELS = 3
IMG_BYTES = WIDTH * HEIGHT * CHANNELS
DEPTH_BYTES = WIDTH * HEIGHT * 4
CAM_SET_SIZE = (IMG_BYTES * 3) + (DEPTH_BYTES * 3)
HEADER_SIZE = 168

# JEPA predictor embed_dim / kobo's native proprio+action dims -- must match
# whatever qc_label_rewards.py/train_qc_critic.py were run with for the
# loaded critic checkpoint (see those scripts' own module-level constants).
EMBED_DIM = 1408
PROPRIO_DIM = 8
ACTION_DIM = 8

# Per-step motion safety limits, confirmed with Fabian 2026-07-31 after a
# real motion-enabled test run moved "too far too fast" -- the raw
# model-predicted absolute pose targets were being published directly with
# no software-side motion limiting, relying entirely on the underlying
# dual_panda_multi_mode_controllers/Franka hardware limits, which turned out
# not to be tight enough for a first real test of this untested-on-hardware
# action-selection code. Clamped in run()'s per-step loop against a FRESH
# current-hand-pose TF lookup (not a stale one from before the chunk
# started), independent of what the model/critic actually predicts.
MAX_DELTA_TRANSLATION_M = 0.1
MAX_DELTA_ROTATION_RAD = np.radians(15.0)


def _clamp_target_pose(
    current_tr: np.ndarray, current_quat: np.ndarray, target_tr: np.ndarray, target_quat: np.ndarray,
    max_translation: float, max_rotation_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Clamps target_tr/target_quat to be at most max_translation meters /
    max_rotation_rad radians away from current_tr/current_quat, preserving
    direction. Quaternions are [x, y, z, w], matching tf's convention
    (assumed already same-hemisphere, e.g. via a prior hemisphere check --
    this function does not itself handle the +-q ambiguity)."""
    current_tr = np.asarray(current_tr, dtype=np.float64)
    target_tr = np.asarray(target_tr, dtype=np.float64)
    delta = target_tr - current_tr
    dist = np.linalg.norm(delta)
    if dist > max_translation and dist > 1e-9:
        target_tr = current_tr + delta / dist * max_translation

    dot = np.clip(np.abs(np.dot(current_quat, target_quat)), -1.0, 1.0)
    angle = 2.0 * np.arccos(dot)
    if angle > max_rotation_rad and angle > 1e-9:
        fraction = max_rotation_rad / angle
        target_quat = np.asarray(tft.quaternion_slerp(current_quat, target_quat, fraction))

    return target_tr, target_quat


class ReplayBuffer:
    """Minimal fixed-capacity circular buffer of realized
    (obs, action, reward, next_obs) transitions, for later offline critic
    updates (RLPD-style mixing with kobo_cube's offline data) -- NOT used for
    any live/synchronous training in this first version, per the plan's
    staged-rollout-first recommendation (collect + inspect before ever
    training online). Stores plain numpy, not Observation/jnp objects, to
    keep this dependency-free of jax/openpi internals and trivially
    picklable to disk for later offline processing."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._buffer: list[dict] = []
        self._write_idx = 0

    def add(self, obs_dict: dict, action: np.ndarray, reward: float, next_obs_dict: dict) -> None:
        item = {
            "obs": obs_dict,
            "action": np.asarray(action, dtype=np.float32),
            "reward": np.float32(reward),
            "next_obs": next_obs_dict,
        }
        if len(self._buffer) < self.capacity:
            self._buffer.append(item)
        else:
            self._buffer[self._write_idx] = item
            self._write_idx = (self._write_idx + 1) % self.capacity

    def save(self, path: str) -> None:
        np.save(path, np.asarray(self._buffer, dtype=object), allow_pickle=True)

    def __len__(self) -> int:
        return len(self._buffer)


class Pi05OnlineRosInterface:
    def __init__(
        self,
        config_name: str = "pi0_kobo_cube_low_mem",
        checkpoint_exp_name: str = "full_lora_test",
        checkpoint_step: int | None = None,
        critic_checkpoint_path: str | None = None,
        num_samples: int = 16,
        horizon_length: int = 5,
        replay_buffer_capacity: int = 10_000,
        replay_buffer_path: str = "~/openpi/qc_online_replay_buffer.npy",
        dry_run: bool = False,
        use_jepa: bool = True,
    ):
        rospy.init_node('pi05_online_inference_node')

        self.num_samples = num_samples
        self.horizon_length = horizon_length
        self.replay_buffer = ReplayBuffer(replay_buffer_capacity)
        self.replay_buffer_path = os.path.expanduser(replay_buffer_path)
        # use_jepa=False: for checkpoints trained via plain scripts/train.py
        # (e.g. bc_only_test1) -- NOT wrapped in OpenPIWithJEPA at all, so
        # they have no jepa_predictor/target_norm and their checkpoint's
        # pytree structure is fundamentally incompatible with
        # init_train_state's JEPA-wrapped template (confirmed via a real
        # crash: restore_state's orbax pytree-structure check fails outright
        # trying to load a plain-BC checkpoint against that template). This
        # flag switches to the same plain-params restore path
        # policy_config.create_trained_policy uses, skips the critic/
        # best-of-N/intrinsic-reward machinery entirely (no JEPA predictor
        # exists to compute reward from), and just runs the base model's own
        # sample_actions directly -- a pure BC ablation test.
        self.use_jepa = use_jepa
        # dry_run: validate the perception -> best-of-N action selection ->
        # intrinsic-reward pipeline against REAL live camera/state, with ZERO
        # robot motion -- no startup_procedure() movement, no cartesian
        # target publishing, no replay-buffer writes (dry-run transitions
        # aren't real rollouts and shouldn't get mixed into real training
        # data). Everything else (model/critic inference, reward computation,
        # logging) runs for real. Intended as the safe first step before any
        # supervised low-num-samples dry run WITH motion enabled.
        self.dry_run = dry_run
        if dry_run:
            rospy.logwarn("DRY RUN MODE: no robot motion will be commanded.")

        # 1. Initialize model + critic + input transform pipeline
        self.setup_model_and_critic(config_name, checkpoint_exp_name, checkpoint_step, critic_checkpoint_path)

        # 2. ROS Publishers & Action Clients (identical to inference.py)
        self.pub_dual_arm_joint_target = rospy.Publisher(
            '/panda_dual/multi_mode_controller/desired_joint_position', JointState, queue_size=10
        )

        self.subscribe_joint_states = rospy.Subscriber(
            '/panda_dual/joint_states', JointState, self.__process_joint_states, queue_size=1
        )

        self.pub_right_cartesian_target = rospy.Publisher(
            '/panda_dual/multi_mode_controller/panda_right/target_pose', PoseStamped, queue_size=0
        )
        self.pub_left_cartesian_target = rospy.Publisher(
            '/panda_dual/multi_mode_controller/panda_left/target_pose', PoseStamped, queue_size=0
        )

        self.right_move_client = actionlib.SimpleActionClient(
            '/panda_dual/panda_right/franka_gripper/move', MoveAction
        )
        self.left_move_client = actionlib.SimpleActionClient(
            '/panda_dual/panda_left/franka_gripper/move', MoveAction
        )

        rospy.loginfo("Waiting for gripper servers...")
        self.right_move_client.wait_for_server(rospy.Duration(5.0))
        self.left_move_client.wait_for_server(rospy.Duration(5.0))

        self.transform_listener = tf.TransformListener()

        # 3. Shared Memory
        self.setup_shm()

        self.rate = rospy.Rate(30)
        self.prev_frame_count = 0

        self.prompt = "pick up the orange cube and place it on the red tape"

    def setup_model_and_critic(
        self, config_name: str, checkpoint_exp_name: str, checkpoint_step: int | None, critic_checkpoint_path: str | None,
    ):
        # Persistent JIT compilation cache, matching train_end_to_end.py's
        # own convention -- without this, every process launch (and this
        # loop's own best_of_n_action_batch/compute_intrinsic_reward calls,
        # which aren't wrapped in an explicit jax.jit and so recompile
        # per-op eagerly rather than reusing one cached graph) pays full
        # compile cost from scratch. Found while debugging an intermittent
        # "ptxas exited with non-zero error code 15" crash on the second
        # loop iteration during a live dry run -- this cache reduces how
        # much compilation happens at all, which should reduce exposure to
        # that failure even if the underlying root cause (this machine's
        # swap was fully saturated during that same test) isn't directly
        # fixed by it.
        jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))
        config = _config.get_config(config_name)
        config = dataclasses.replace(config, exp_name=checkpoint_exp_name)
        self.config = config

        self.mesh = sharding.make_mesh(1)
        self.replicated_sharding = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec())

        rospy.loginfo(f"Loading model from {config.checkpoint_dir} (use_jepa={self.use_jepa})")
        rng = jax.random.key(config.seed)

        if self.use_jepa:
            # jepa_predictor_checkpoint=None: the checkpoint being restored
            # already has the co-trained predictor weights baked into its own
            # params (same reasoning as qc_label_rewards.py's label_rewards()).
            train_state_shape, _ = init_train_state(config, rng, self.mesh, resume=True, jepa_predictor_checkpoint=None)
            checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
                config.checkpoint_dir, keep_period=config.keep_period, overwrite=False, resume=True,
            )
            if not resuming:
                raise RuntimeError(f"No checkpoint found at {config.checkpoint_dir} to restore from.")
            self.train_state = _checkpoints.restore_state(checkpoint_manager, train_state_shape, None, step=checkpoint_step)
            rospy.loginfo(f"Restored checkpoint step={int(self.train_state.step)}")
            # Used as the `base_model` arg to best_of_n_action_batch (needs
            # .base_model.sample_actions / .extract_vision_latents) -- a
            # stop-gradiented merge, matching qc_label_rewards.py's _label_batch
            # convention. compute_intrinsic_reward does its own separate merge
            # from self.train_state internally; this duplication is accepted
            # elsewhere in this codebase (see that function's own docstring).
            self.model = nnx.merge(self.train_state.model_def, jax.lax.stop_gradient(self.train_state.params))
        else:
            # Plain BC checkpoint (scripts/train.py, e.g. bc_only_test1) --
            # NOT wrapped in OpenPIWithJEPA, no jepa_predictor/target_norm at
            # all. init_train_state/restore_state's JEPA-wrapped template is
            # structurally incompatible with this checkpoint's actual pytree
            # (confirmed via a real crash: orbax's restore-time structure
            # check fails outright). Mirror policy_config.create_trained_
            # policy's own restore path instead -- just the "params" item,
            # no TrainState/optimizer/EMA machinery needed for inference.
            self.train_state = None
            if checkpoint_step is None:
                steps = [int(p.name) for p in config.checkpoint_dir.iterdir() if p.name.isdigit()]
                if not steps:
                    raise RuntimeError(f"No numbered checkpoint steps found under {config.checkpoint_dir}")
                checkpoint_step = max(steps)
            params_path = config.checkpoint_dir / str(checkpoint_step) / "params"
            restored_params = _model.restore_params(params_path, dtype=jnp.bfloat16)
            self.model = config.model.load(restored_params)
            rospy.loginfo(f"Restored plain BC checkpoint step={checkpoint_step}")

        self.data_config = config.data.create(config.assets_dirs, config.model)
        self.norm_stats = self.data_config.norm_stats
        self.use_quantile_norm = self.data_config.use_quantile_norm
        self.normalize = _transforms.Normalize(self.norm_stats, use_quantiles=self.use_quantile_norm)

        # Real input pipeline, minus repack_transforms: our live ROS
        # "example" dict is already built in the "observation/image"-style
        # keys repack_transforms.structure would have remapped raw LeRobot
        # columns into -- exactly matching policy_config.create_trained_
        # policy's real transform composition, confirmed by reading that
        # function (it also omits repack_transforms by default, for the same
        # reason: its caller already hands it a "observation/image"-keyed
        # dict, same as inference.py's `example`).
        self.input_transform = _transforms.compose(
            [
                _transforms.InjectDefaultPrompt(None),  # no-op: our example dict always sets "prompt" explicitly
                *self.data_config.data_transforms.inputs,
                self.normalize,
                *self.data_config.model_transforms.inputs,
            ]
        )

        if not self.use_jepa:
            # Pure BC ablation: no critic, no best-of-N, no intrinsic reward
            # (there's no jepa_predictor to compute prediction error from).
            # Just JIT-wrap sample_actions + Unnormalize directly, same
            # reasoning as the use_jepa branch below (avoid eager per-op
            # recompilation/memory growth across loop iterations).
            self.critic = None
            self._model_graphdef, self._model_state = nnx.split(self.model)

            def _select_action_plain(model_state, rng, obs):
                model = nnx.merge(self._model_graphdef, model_state)
                actions = model.sample_actions(rng, obs, num_steps=10)
                unnormalize = _transforms.Unnormalize(self.norm_stats, use_quantiles=self.use_quantile_norm)
                actions = unnormalize({"actions": actions})["actions"]
                return actions[:, : self.horizon_length, :ACTION_DIM]

            self._select_action_plain_jit = jax.jit(_select_action_plain)
            self._rng = jax.random.key(config.seed + 1)
            return

        if critic_checkpoint_path is None:
            raise ValueError(
                "critic_checkpoint_path is required -- point it at a scripts/train_qc_critic.py "
                "checkpoint (e.g. checkpoints_qc/<run>/final) trained against a POST-2026-07-27-fix "
                "BC/JEPA checkpoint (see memory: pre-fix checkpoints are incompatible with this file's "
                "Unnormalize-based action reconstruction). As of 2026-07-29, the real trained critic "
                "against full_lora_test (the default checkpoint_exp_name) is at "
                "/home/fabian/openpi/checkpoints_qc/full_lora_test_run1/final -- converged cleanly "
                "(critic_loss ~0.001-0.003, q_mean/target_q_mean stable ~0.25-0.33, not diverging)."
            )
        rospy.loginfo(f"Loading critic from {critic_checkpoint_path}")
        # use_target=True: the EMA-smoothed target network is the more
        # stable choice for inference-time action scoring (train_step.py's
        # own convention for bootstrapping targets).
        self.critic = load_critic(
            critic_checkpoint_path, EMBED_DIM, PROPRIO_DIM, ACTION_DIM, self.horizon_length, use_target=True,
        )

        # JIT-wrap the per-step action-selection and reward calls, matching
        # qc_label_rewards.py's labeling_fn / module_jit's own convention
        # (nnx_utils.py) -- freeze model/critic state once, close over static
        # config via functools.partial, and jax.jit the rest. Found this to
        # be NECESSARY, not just a speedup, while debugging a live dry run:
        # calling best_of_n_action_batch/compute_intrinsic_reward as bare
        # Python functions runs every internal op eagerly (no jit boundary to
        # reuse across calls), and GPU memory usage grew iteration over
        # iteration until a RESOURCE_EXHAUSTED crash on the 3rd loop
        # iteration -- with these jitted, qc_label_rewards.py's own identical
        # pattern ran 20,556 batches earlier without any such growth.
        self._model_graphdef, self._model_state = nnx.split(self.model)
        self._critic_graphdef, self._critic_state = nnx.split(self.critic)

        def _select_action(model_state, critic_state, rng, obs, proprio):
            model = nnx.merge(self._model_graphdef, model_state)
            critic = nnx.merge(self._critic_graphdef, critic_state)
            return best_of_n_action_batch(
                rng, model, critic, obs, proprio,
                self.num_samples, self.horizon_length, ACTION_DIM,
                self.norm_stats, use_quantile_norm=self.use_quantile_norm,
            )

        self._select_action_jit = jax.jit(_select_action)
        self._compute_reward_jit = jax.jit(functools.partial(compute_intrinsic_reward, self.config))

        self._rng = jax.random.key(config.seed + 1)

    def setup_shm(self):
        try:
            self.memory = posix_ipc.SharedMemory("zed_shm")
            self.map_file = mmap.mmap(self.memory.fd, self.memory.size)
            self.mv = memoryview(self.map_file)
            rospy.loginfo("Connected to Shared Memory.")
        except Exception as e:
            rospy.logerr(f"SHM Connection failed: {e}")
            raise

    def __process_joint_states(self, data):
        self.joint_state_pos = np.array(data.position)

    def startup_procedure(self):
        self.right_gripper_open = True
        self.left_gripper_open = True
        rospy.sleep(1)

        rospy.loginfo('Loading Controllers')
        self.switch_controller('joint')

        init_joint_config = np.array(
            [1.622424959701404, -0.5128421328809281, -0.6169117761913098, -1.5570858432566788,
             -0.15386249067312407, 1.702816256827778, -0.677949522209834, 0.03981238603591919,
             0.03981238603591919, -1.24707628442903, -0.13936313613694162, 0.4163110730020622,
             -1.4373301961472207, -0.4435531249046325, 1.6925763938253924, -0.5369736137470078,
             0.04024578630924225, 0.04024578630924225]
        )
        rate = rospy.Rate(200)
        target_state = JointState()
        target_state.name = ['panda_right_joint1', 'panda_right_joint2', 'panda_right_joint3', 'panda_right_joint4',
                              'panda_right_joint5', 'panda_right_joint6', 'panda_right_joint7',
                              'panda_right_finger_joint1', 'panda_right_finger_joint2', 'panda_left_joint1',
                              'panda_left_joint2', 'panda_left_joint3', 'panda_left_joint4', 'panda_left_joint5',
                              'panda_left_joint6', 'panda_left_joint7', 'panda_left_finger_joint1',
                              'panda_left_finger_joint2']
        max_joint_diff = rospy.get_param("/PandaJointImpedanceController_panda_left/max_joint_diff") * np.pi / 180
        max_joint_diff /= 2

        def goto_pose(desired_joint_config):
            while np.linalg.norm(self.joint_state_pos - desired_joint_config) > 0.08 and not rospy.is_shutdown():
                delta = desired_joint_config - self.joint_state_pos
                mask = (np.abs(delta) >= max_joint_diff)
                delta[mask] = max_joint_diff * np.sign(delta[mask])
                target_state.position = self.joint_state_pos + delta
                self.pub_dual_arm_joint_target.publish(target_state)
                rate.sleep()

        rospy.loginfo('Going to initial pointcloud pose')
        goto_pose(init_joint_config)
        time.sleep(5)
        rospy.loginfo('Loading cartesian controller')
        self.switch_controller('cartesian')

    def switch_controller(self, start_controllers):
        pub = rospy.Publisher('/panda_dual/multi_mode_controller/switch_control/goal', SwitchControlActionGoal, queue_size=1)
        switcher = SwitchControlActionGoal()
        right_mode = ControlMode()
        left_mode = ControlMode()
        right_mode.ctrl_resources = ['panda_right']
        left_mode.ctrl_resources = ['panda_left']
        right_mode.ctrl_type = left_mode.ctrl_type = start_controllers
        switcher.goal.ctrl_modes.mode_list = [right_mode, left_mode]
        switcher.goal_id.stamp = switcher.header.stamp = rospy.Time.now()
        pub.publish(switcher)
        rospy.sleep(1)
        pub.publish(switcher)
        rospy.sleep(1)

    def send_gripper_commands(self, right_width, left_width):
        def create_goal(w):
            msg = MoveGoal()
            msg.width = np.clip(w * 2.0, 0, 0.08)
            msg.speed = 0.1
            return msg

        self.right_move_client.send_goal(create_goal(right_width))
        self.left_move_client.send_goal(create_goal(left_width))

    def _read_camera_and_state(self, world_frame: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Blocks until a new SHM camera frame is available, then reads the
        current right-hand pose + gripper width. Returns
        (img_np, img_np_external, state_8, r_ee_rot) -- state_8 is
        [x, y, z, qx, qy, qz, qw, gripper_width], matching the training
        dataset's observation.state layout (see this file's module docstring
        for the caveat on the gripper_width reading)."""
        while True:
            if rospy.is_shutdown():
                # Found via a live test where an external SIGTERM (timeout
                # command) triggered rospy shutdown while this loop was
                # blocked in time.sleep(0.001) -- the loop used to be
                # `while not rospy.is_shutdown(): ...`, which exits the WHILE
                # condition without ever running the body again, leaving
                # frame_count unbound at `self.prev_frame_count = frame_count`
                # below. A normal Ctrl-C/rosnode kill during real operation
                # would hit this same path, so raise a clean, expected
                # exception instead of crashing on an UnboundLocalError.
                raise rospy.ROSInterruptException("Shutdown requested while waiting for a new camera frame.")
            header_peek = self.mv[:12]
            frame_count, active_buf = struct.unpack('Qi', header_peek)
            if frame_count != self.prev_frame_count:
                break
            time.sleep(0.001)
        self.prev_frame_count = frame_count

        slot_offset = HEADER_SIZE + (((active_buf - 1) % 3) * CAM_SET_SIZE)
        rgb_data = self.mv[slot_offset + IMG_BYTES: slot_offset + (2 * IMG_BYTES)]
        rgb_data_external = self.mv[slot_offset + 2 * IMG_BYTES: slot_offset + (3 * IMG_BYTES)]
        img_np = np.frombuffer(rgb_data, dtype=np.uint8).reshape(HEIGHT, WIDTH, CHANNELS)
        img_np_external = np.frombuffer(rgb_data_external, dtype=np.uint8).reshape(HEIGHT, WIDTH, CHANNELS)
        img_np_external = cv2.flip(img_np_external, 0)
        img_np_external = cv2.flip(img_np_external, 1)
        img_np_external = img_np_external.transpose(2, 0, 1)
        img_np = cv2.flip(img_np, 0)
        img_np = cv2.flip(img_np, 1)
        img_np = img_np.transpose(2, 0, 1)

        r_ee_tr, r_ee_rot = self.transform_listener.lookupTransform(world_frame, 'panda_right_hand', rospy.Time(0))

        # Quaternion hemisphere canonicalization -- CONFIRMED 2026-07-31 via
        # scripts/debug_check_proprio.py against the real training dataset:
        # tf's lookupTransform returned a quaternion whose sign was flipped
        # on EVERY component relative to the training distribution (live
        # qx/qy/qz/qw were each within ~0.01 of the NEGATION of the training
        # mean -- e.g. live qw=+0.26 vs training mean qw=-0.19, and training's
        # qw range [-0.353,-0.004] never crosses zero across all 82,325
        # frames). q and -q represent the identical physical rotation
        # (quaternion double-cover), so this is very likely the SAME real
        # orientation as training, just resolved to the opposite antipodal
        # representative -- but Normalize does plain elementwise z-scoring
        # with no awareness of that ambiguity, so feeding the unflipped
        # quaternion as proprio produced a statistically nonsensical input to
        # the model on every single inference call, for every checkpoint
        # tested (a plausible root cause for "none of them seem to actually
        # work"). Canonicalize into training's known hemisphere (qw <= 0,
        # with clear margin in the training data) before using r_ee_rot for
        # anything -- this fixes both the state_8 proprio built below AND
        # run()'s output-side hemisphere check, which also uses this same
        # r_ee_rot as its reference.
        if r_ee_rot[3] > 0.0:
            r_ee_rot = tuple(-c for c in r_ee_rot)
        qx, qy, qz, qw = r_ee_rot

        # CONFIRMED (2026-07-30) against the real kobo_cube dataset: both
        # observation.state[...,7] and action[...,7] take ONLY the values
        # {0.0, 1.0} across all 82,325 frames (mean~0.81, i.e. mostly 1.0) --
        # this is a binary open/closed flag, NOT a continuous physical width
        # in meters. An earlier version of this file read a continuous
        # `2 * joint_state_pos[7]` value here (~0-0.08m), which was wrong in
        # both scale and semantics -- fixed to threshold into the same binary
        # convention instead. panda_right_finger_joint1 sits near its max
        # (~0.04m per finger, ~0.08m total) when open and near 0 when closed
        # (see startup_procedure's init_joint_config, which parks it open);
        # thresholding at the midpoint (0.04m total) recovers a 0/1 flag from
        # the real joint reading.
        #
        # STILL A GUESS, flagged for Fabian: which physical state maps to 1.0
        # vs 0.0 (open=1.0 assumed here, matching the dataset's mostly-1.0
        # mean and demos plausibly spending more time approaching with the
        # gripper open than closed) is not independently verified against the
        # actual data-collection code that wrote this column.
        gripper_open = (2.0 * self.joint_state_pos[7]) > 0.04 if hasattr(self, "joint_state_pos") else True
        gripper_flag = 1.0 if gripper_open else 0.0

        state_8 = np.array([*r_ee_tr, qx, qy, qz, qw, gripper_flag], dtype=np.float32)
        return img_np, img_np_external, state_8, np.asarray(r_ee_rot)

    def _build_observation(self, img_np: np.ndarray, img_np_external: np.ndarray, state_8: np.ndarray) -> _model.Observation:
        example = {
            "observation/image": img_np,
            "observation/external_image": img_np_external,
            "observation/state": state_8,
            "prompt": np.array([self.prompt]),
        }
        data = self.input_transform(dict(example))
        data = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], data)
        return _model.Observation.from_dict(data)

    def run(self):
        if self.dry_run:
            rospy.logwarn("DRY RUN: skipping startup_procedure() -- no controller switch, no motion to init pose.")
            # startup_procedure() normally gives tf.TransformListener several
            # seconds (its own rospy.sleep(1) calls plus the physical move
            # sequence) to receive the latched /tf_static transforms before
            # the first lookupTransform call below. Skipping it entirely
            # removes that incidental buffer -- without it, the first lookup
            # can race the listener's subscription and see an artificial
            # "unconnected tree" error simply because /tf_static hasn't
            # arrived yet, not because the tree is actually broken. Confirmed
            # via `rostopic echo /tf_static` while debugging: base_link is
            # the tree's root (base_link -> torso -> panda_*_link0 -> ... ->
            # panda_*_hand), fully connected once received.
            rospy.sleep(2)
        else:
            self.startup_procedure()
        WORLD_FRAME = 'base_link'

        try:
            tl, ql = self.transform_listener.lookupTransform(WORLD_FRAME, 'panda_left_link0', rospy.Time(0))
            self.T_base_l0_l = tft.concatenate_matrices(tft.translation_matrix(tl), tft.quaternion_matrix(ql))
        except Exception as e:
            rospy.logerr(f"Static TF lookup failed: {e}")
        self.T_hand_ee = tft.translation_matrix([0.0, 0.0, 0.1034])

        while not rospy.is_shutdown():
            img_np, img_np_external, state_8, r_ee_rot = self._read_camera_and_state(WORLD_FRAME)
            obs = self._build_observation(img_np, img_np_external, state_8)

            # --- Best-of-N critic-scored action selection, replacing
            # policy.infer() -- see src/qc/actor.py. proprio is RAW native
            # units (no Normalize), matching QChunkTransitionDataset's own
            # proprio_t convention (src/utils/data_loader.py), which the
            # critic was trained against. use_jepa=False skips the critic
            # entirely -- pure BC ablation, see setup_model_and_critic. ---
            self._rng, action_rng = jax.random.split(self._rng)
            if self.use_jepa:
                proprio_native = jnp.asarray(state_8[np.newaxis, :])
                best_actions = self._select_action_jit(
                    self._model_state, self._critic_state, action_rng, obs, proprio_native,
                )
            else:
                best_actions = self._select_action_plain_jit(self._model_state, action_rng, obs)
            action_chunk = np.asarray(best_actions[0])  # [horizon_length, 8], native units

            # --- Execute horizon_length steps (NOT the full action_horizon
            # =10 like inference.py) -- replan after horizon_length to match
            # the critic's own chunk length and get fresher critic-scored
            # actions sooner, per the plan's design. ---
            for idx, absolute_step in enumerate(action_chunk):
                tr, qr = self.transform_listener.lookupTransform(WORLD_FRAME, 'panda_right_link0', rospy.Time(0))
                self.T_base_l0_r = tft.concatenate_matrices(tft.translation_matrix(tr), tft.quaternion_matrix(qr))
                target_tr = absolute_step[:3]
                pred_quat = absolute_step[3:7]
                grasp_width = absolute_step[7]
                rospy.loginfo(grasp_width)

                if np.dot(r_ee_rot, pred_quat) < 0.0:
                    pred_quat = -pred_quat

                norm = np.linalg.norm(pred_quat)
                safe_quat = pred_quat / norm if norm > 0 else pred_quat

                # Clamp against the ACTUAL current hand pose (fresh lookup,
                # not the pose from before this chunk started -- the robot
                # has been moving since then) -- see MAX_DELTA_TRANSLATION_M/
                # MAX_DELTA_ROTATION_RAD's module-level comment for why this
                # exists.
                current_hand_tr, current_hand_quat = self.transform_listener.lookupTransform(
                    WORLD_FRAME, 'panda_right_hand', rospy.Time(0)
                )
                target_tr, safe_quat = _clamp_target_pose(
                    current_hand_tr, current_hand_quat, target_tr, safe_quat,
                    MAX_DELTA_TRANSLATION_M, MAX_DELTA_ROTATION_RAD,
                )

                T_world_target = tft.concatenate_matrices(
                    tft.translation_matrix(target_tr), tft.quaternion_matrix(safe_quat)
                )
                T_world_ee_target = T_world_target @ self.T_hand_ee
                T_l0_target_r = np.linalg.inv(self.T_base_l0_r) @ T_world_ee_target

                def to_msg(mat, frame):
                    m = PoseStamped()
                    m.header.stamp = rospy.Time.now()
                    m.header.frame_id = frame
                    m.pose.position.x, m.pose.position.y, m.pose.position.z = mat[:3, 3]
                    q = tft.quaternion_from_matrix(mat)
                    m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = q
                    return m

                msg = to_msg(T_l0_target_r, 'panda_right_link0')
                if self.dry_run:
                    rospy.loginfo(
                        f"DRY RUN step {idx}: would publish target_tr={target_tr}, "
                        f"quat={safe_quat}, grasp_width={grasp_width:.3f} (not sent)"
                    )
                else:
                    self.pub_right_cartesian_target.publish(msg)
                self.rate.sleep()

            if not self.use_jepa:
                # Pure BC ablation: no jepa_predictor exists to compute
                # intrinsic reward from, and there's no critic training data
                # to collect either -- nothing left to do this iteration.
                continue

            # --- Capture the resulting next observation + compute intrinsic
            # reward on the realized transition ---
            next_img_np, next_img_np_external, next_state_8, _ = self._read_camera_and_state(WORLD_FRAME)
            next_obs = self._build_observation(next_img_np, next_img_np_external, next_state_8)

            # Reconstruct the training-equivalent normalized+padded action_t
            # (compute_intrinsic_reward expects Pi0's padded action_dim,
            # z-scored units, matching train_step's action_chunk[:, 0, :]
            # convention) from the REALIZED native-unit action actually
            # executed: normalize (native dim, correct direction -- Normalize
            # runs BEFORE padding in the real pipeline, see
            # policy_config.create_trained_policy's transform composition)
            # then pad to config.model.action_dim, mirroring model_transforms'
            # own padding step.
            #
            # NOTE: this computes ONE reward for the whole horizon_length-
            # step transition (obs before chunk -> obs after chunk), using
            # only the chunk's FIRST sub-action for action_t -- NOT h
            # separate single-step rewards discount-summed the way
            # qc_label_rewards.py's OFFLINE labeling does. A deliberate
            # simplification for online collection; revisit if this diverges
            # too much in scale/semantics from the offline reward when mixing
            # online + offline data for later critic updates.
            first_action_native = action_chunk[0][np.newaxis, :]  # [1, 8]
            normed = self.normalize({"actions": first_action_native})["actions"]  # [1, 8], z-scored
            action_t = _transforms.pad_to_dim(jnp.asarray(normed), self.config.model.action_dim, axis=-1)  # [1, 32]

            reward = self._compute_reward_jit(self.train_state, obs, action_t, next_obs)
            reward_val = float(np.asarray(reward)[0])
            rospy.loginfo(f"Intrinsic reward for this transition: {reward_val:.4f}")

            if self.dry_run:
                # No real transition was executed (no motion was commanded),
                # so this observation pair isn't a genuine rollout -- don't
                # let it pollute a real replay buffer used for later critic
                # updates. The reward/action-selection values above are still
                # meaningful to inspect (that's the point of dry-run mode),
                # just not worth persisting.
                continue

            self.replay_buffer.add(
                obs_dict={"image": img_np, "external_image": img_np_external, "state": state_8},
                action=action_chunk,
                reward=reward_val,
                next_obs_dict={"image": next_img_np, "external_image": next_img_np_external, "state": next_state_8},
            )
            if len(self.replay_buffer) % 50 == 0:
                rospy.loginfo(f"Replay buffer size: {len(self.replay_buffer)}, saving to {self.replay_buffer_path}")
                self.replay_buffer.save(self.replay_buffer_path)


if __name__ == '__main__':
    import argparse

    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument("--config-name", type=str, default="pi0_kobo_cube_low_mem")
    _parser.add_argument("--checkpoint-exp-name", type=str, default="full_lora_test")
    _parser.add_argument("--checkpoint-step", type=int, default=None)
    _parser.add_argument(
        "--critic-checkpoint-path", type=str, default=None,
        help="Required unless --no-jepa is set (pure BC ablation runs need no critic).",
    )
    _parser.add_argument(
        "--no-jepa", dest="use_jepa", action="store_false", default=True,
        help="For checkpoints trained via plain scripts/train.py (not OpenPIWithJEPA-wrapped, e.g. "
             "bc_only_test1) -- skips the critic/best-of-N/intrinsic-reward machinery entirely and just "
             "runs the base model's own sample_actions. See setup_model_and_critic's use_jepa docstring.",
    )
    _parser.add_argument("--num-samples", type=int, default=16)
    _parser.add_argument("--horizon-length", type=int, default=5)
    _parser.add_argument("--replay-buffer-capacity", type=int, default=10_000)
    _parser.add_argument("--replay-buffer-path", type=str, default="~/openpi/qc_online_replay_buffer.npy")
    _parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Validate the perception -> action-selection -> reward pipeline against real live camera/state "
             "with ZERO robot motion (no startup_procedure movement, no cartesian target publishing, no "
             "replay-buffer writes). Recommended before any real supervised dry run with motion enabled.",
    )
    _args, _remaining = _parser.parse_known_args()
    sys.argv = [sys.argv[0]] + _remaining

    try:
        node = Pi05OnlineRosInterface(
            config_name=_args.config_name,
            checkpoint_exp_name=_args.checkpoint_exp_name,
            checkpoint_step=_args.checkpoint_step,
            critic_checkpoint_path=_args.critic_checkpoint_path,
            num_samples=_args.num_samples,
            horizon_length=_args.horizon_length,
            replay_buffer_capacity=_args.replay_buffer_capacity,
            replay_buffer_path=_args.replay_buffer_path,
            dry_run=_args.dry_run,
            use_jepa=_args.use_jepa,
        )
        node.run()
    except rospy.ROSInterruptException:
        pass
