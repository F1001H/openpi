#!/usr/bin/env python3
import rospy
import actionlib
import torch
import numpy as np
import posix_ipc
import mmap
import struct
import os
import time
from sensor_msgs.msg import JointState
from franka_gripper.msg import MoveAction, MoveGoal
from openpi.training import config as _config
from openpi.policies import policy_config
from geometry_msgs.msg import PoseStamped, Pose
import tf
import tf.transformations as tft
from dual_panda_multi_mode_controllers.msg import SwitchControlActionGoal, ControlMode
import cv2




# --- MATCH C++ CONSTANTS ---
WIDTH, HEIGHT = 1280, 720
CHANNELS = 3
IMG_BYTES = WIDTH * HEIGHT * CHANNELS
DEPTH_BYTES = WIDTH * HEIGHT * 4
CAM_SET_SIZE = (IMG_BYTES * 3) + (DEPTH_BYTES * 3)
HEADER_SIZE = 168 

class Pi05RosInterface:
    def __init__(self):
        rospy.init_node('pi05_inference_node')
        
        self.device = "cuda"
        self.dtype = torch.bfloat16
        
        # 1. Initialize Policy
        self.policy = self.setup_policy()
        
        # 2. ROS Publishers & Action Clients
        self.pub_dual_arm_joint_target = rospy.Publisher('/panda_dual/multi_mode_controller/desired_joint_position', JointState, queue_size=10)


        self.subscribe_joint_states = rospy.Subscriber('/panda_dual/joint_states', JointState, self.__process_joint_states, queue_size=1)


        self.pub_right_cartesian_target = rospy.Publisher('/panda_dual/multi_mode_controller/panda_right/target_pose', PoseStamped, queue_size=0)
        self.pub_left_cartesian_target = rospy.Publisher('/panda_dual/multi_mode_controller/panda_left/target_pose', PoseStamped, queue_size=0)
        
        # Note: If grippers aren't responding, ensure the MoveAction server is running
        self.right_move_client = actionlib.SimpleActionClient('/panda_dual/panda_right/franka_gripper/move', MoveAction)
        self.left_move_client = actionlib.SimpleActionClient('/panda_dual/panda_left/franka_gripper/move', MoveAction)
        
        rospy.loginfo("Waiting for gripper servers...")
        # Optional: reduce timeout if you want to test arm movement without grippers
        self.right_move_client.wait_for_server(rospy.Duration(5.0))
        self.left_move_client.wait_for_server(rospy.Duration(5.0))

        self.transform_listener = tf.TransformListener()


        # 3. Shared Memory
        self.setup_shm()
        
        self.rate = rospy.Rate(30) 
        self.prev_frame_count = 0

        self.execution_horizon = 30

    def setup_policy(self):
        config = _config.get_config("pi0_kobo_cube_low_mem")
        ckpt_path = os.path.expanduser("~/openpi/checkpoints/pi0_kobo_cube_low_mem/fifth_test_cube/20000")
        rospy.loginfo(f"Loading policy from {ckpt_path}")
        policy = policy_config.create_trained_policy(config, ckpt_path)
        rospy.loginfo("Policy loaded successfully.")
        return policy
    
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

        # open gripper first before first joint state
        self.right_gripper_open = True
        self.left_gripper_open = True
        #self.send_right_gripper_goal()
        #self.send_left_gripper_goal()
        rospy.sleep(1)

        rospy.loginfo('Loading Controllers')
        self.switch_controller('joint')

        # Take the inital joints from the output of running "rostopic echo /panda_dual/joint_states/position -n 1" if you need to reset it

        # Plate handover
        # init_joint_config = np.array(
        #     [-0.45950383006062423, -0.6033743888625315, 0.6600517126635501, -2.1307089038648104, 1.1895847103728185, 1.3960325290091433, 0.25504477253895963, 0.03897283971309662, 0.03897283971309662, -1.2549422859572372, -0.599490169073406, 0.4569275441002426, -2.219323342353101, 1.0265747843396096, 2.2989121056662665, 0.01883106500572628, 0.03898400068283081, 0.03898400068283081]
        # )

        # To get a full view pointcloud of the initial scene
        init_joint_config = np.array(
            [1.622424959701404, -0.5128421328809281, -0.6169117761913098, -1.5570858432566788, -0.15386249067312407, 1.702816256827778, -0.677949522209834, 0.03981238603591919, 0.03981238603591919, -1.24707628442903, -0.13936313613694162, 0.4163110730020622, -1.4373301961472207, -0.4435531249046325, 1.6925763938253924, -0.5369736137470078, 0.04024578630924225, 0.04024578630924225]
        )
        rate = rospy.Rate(200)
        target_state = JointState()
        target_state.name = ['panda_right_joint1', 'panda_right_joint2', 'panda_right_joint3', 'panda_right_joint4',
                           'panda_right_joint5', 'panda_right_joint6', 'panda_right_joint7',
                           'panda_right_finger_joint1', 'panda_right_finger_joint2', 'panda_left_joint1',
                           'panda_left_joint2', 'panda_left_joint3', 'panda_left_joint4', 'panda_left_joint5',
                           'panda_left_joint6', 'panda_left_joint7', 'panda_left_finger_joint1',
                           'panda_left_finger_joint2']
        max_joint_diff = rospy.get_param("/PandaJointImpedanceController_panda_left/max_joint_diff")*np.pi/180
        max_joint_diff /= 2
        
        def goto_pose(desired_joint_config):
            while np.linalg.norm(self.joint_state_pos - desired_joint_config) > 0.08 and not rospy.is_shutdown():
                delta = desired_joint_config - self.joint_state_pos
                mask = (np.abs(delta) >= max_joint_diff)
                delta[mask] = max_joint_diff*np.sign(delta[mask])
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
        pub.publish(switcher) # Sometimes publishing only once doesn't work
        rospy.sleep(1)

    def unnormalize(self, action):
        return (action * self.stds) + self.means

    def normalize_state(self, raw_state):
        return (raw_state - self.state_means) / self.state_stds
    
    def send_gripper_commands(self, right_width, left_width):
        """Sends width commands to Franka controllers"""
        def create_goal(w):
            msg = MoveGoal()
            # Franka gripper width is total distance (0.0 to 0.08m)
            # Your index 7 is usually one finger, so we double it for the full width
            msg.width = np.clip(w * 2.0, 0, 0.08)
            msg.speed = 0.1
            return msg

        self.right_move_client.send_goal(create_goal(right_width))
        self.left_move_client.send_goal(create_goal(left_width))

    def run(self):
        self.startup_procedure()
        WORLD_FRAME = 'base_link'
        # Initialize Pose trackers
        last_commanded_pose = None

        try:
        
            tl, ql = self.transform_listener.lookupTransform(WORLD_FRAME, 'panda_left_link0', rospy.Time(0))
            self.T_base_l0_l = tft.concatenate_matrices(tft.translation_matrix(tl), tft.quaternion_matrix(ql))
        except Exception as e:
            rospy.logerr(f"Static TF lookup failed: {e}")
        self.T_hand_ee = tft.translation_matrix([0.0, 0.0, 0.1034])

        while not rospy.is_shutdown():
            # --- 1. SHM SYNC & IMAGE EXTRACTION ---
            header_peek = self.mv[:12]
            frame_count, active_buf = struct.unpack('Qi', header_peek)

            if frame_count == self.prev_frame_count:
                time.sleep(0.001)
                continue
            self.prev_frame_count = frame_count

            # Extract image (matching your index logic)
            slot_offset = HEADER_SIZE + (((active_buf - 1) % 3) * CAM_SET_SIZE)
            rgb_data = self.mv[slot_offset + IMG_BYTES : slot_offset + (2 * IMG_BYTES)]
            rgb_data_external = self.mv[slot_offset + 2 * IMG_BYTES : slot_offset + (3 * IMG_BYTES)]
            img_np = np.frombuffer(rgb_data, dtype=np.uint8).reshape(HEIGHT, WIDTH, CHANNELS)
            img_np_external = np.frombuffer(rgb_data_external, dtype=np.uint8).reshape(HEIGHT, WIDTH, CHANNELS)
            img_np_external = cv2.flip(img_np_external, 0)  # Vertical flip to match ROS image orientation
            img_np_external = cv2.flip(img_np_external, 1)  # Horizontal flip to match ROS image orientation
            img_np_external = img_np_external.transpose(2, 0, 1)
            img_np = cv2.flip(img_np, 0)  # Vertical flip to match ROS image orientation
            img_np = cv2.flip(img_np, 1)  # Horizontal flip to match ROS image orientation
            img_np = img_np.transpose(2, 0, 1)  # Convert to CxHxW for PyTorch
            #cv2.imshow("ZED Camera", img_np)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break
            
            
            r_ee_tr, r_ee_rot = self.transform_listener.lookupTransform(WORLD_FRAME, 'panda_right_hand', rospy.Time(0))
            T_curr = tft.concatenate_matrices(tft.translation_matrix(r_ee_tr), tft.quaternion_matrix(r_ee_rot))


            qx, qy, qz, qw = r_ee_rot
            
            # --- QUATERNION CHECK ---
            # If your dataset was recorded using [w, x, y, z], use this:
            # orientation_payload = [qw, qx, qy, qz]
            # If your dataset strictly kept the raw ROS [x, y, z, w] array, use this:
            orientation_payload = [qx, qy, qz, qw]
            # ------------------------

            current_task_space_state = np.array(r_ee_tr + orientation_payload, dtype=np.float32)
            #print("Current task-space state:", current_task_space_state)
            dummy_action = np.zeros((10, 32), dtype=np.float32)
            example = {
                "observation/image": img_np,        # Shape: (1, H, W, 3)
                "observation/external_image": img_np_external,  # Shape: (1, H, W, 3)
                "observation/state": current_task_space_state,
                "prompt": np.array(["pick up the orange cube and place it on the red tape"]),

                "action": dummy_action
                
                }

            # Run inference using your working approach
            predictions = self.policy.infer(example)
            
            # Extract your action chunk (assuming shape is already stripped to (10, 7) or similar)
            # Adjust indexing depending on your exact workaround layout
            action_chunk = np.array(predictions["actions"]) 
            if action_chunk.ndim == 3:
                action_chunk = action_chunk[0] # Strip batch dimension if present -> Shape: (10, 7)
                
            #rospy.loginfo(f"Predicted action chunk shape: {action_chunk}")

            # --- 4. ABSOLUTE POSE INTEGRATION & EXECUTION ---
            curr_time = rospy.Time.now()

            # For your first implementation, we iterate and execute the 10-step trajectory loop
            for idx, absolute_step in enumerate(action_chunk):
                tr, qr = self.transform_listener.lookupTransform(WORLD_FRAME, 'panda_right_link0', rospy.Time(0))
                self.T_base_l0_r = tft.concatenate_matrices(tft.translation_matrix(tr), tft.quaternion_matrix(qr))
                # Unpack the model's absolute world targets
                target_tr = absolute_step[:3]
                pred_quat = absolute_step[3:7] # Expected order: [x, y, z, w]
                graps_width = absolute_step[7] # Your gripper width command index
                rospy.loginfo(graps_width)

                # --- CRITICAL: QUATERNION HEMISPHERE SANITY CHECK ---
                # Check alignment with your current rotation to prevent violent 360-degree spins
                if np.dot(r_ee_rot, pred_quat) < 0.0:
                    pred_quat = -pred_quat

                # Strict normalization to protect your joint actuators from floating point drift
                norm = np.linalg.norm(pred_quat)
                safe_quat = pred_quat / norm if norm > 0 else pred_quat

                # Convert the absolute world targets directly to a 4x4 matrix
                T_world_target = tft.concatenate_matrices(
                    tft.translation_matrix(target_tr), 
                    tft.quaternion_matrix(safe_quat)
                )
                
                # Transform target from WORLD_FRAME to the robot's specific control base ('panda_right_link0')
                T_world_ee_target = T_world_target @ self.T_hand_ee
                T_l0_target_r = np.linalg.inv(self.T_base_l0_r) @ T_world_ee_target
                
                # Helper closure to format PoseStamped message safely
                def to_msg(mat, frame):
                    m = PoseStamped()
                    m.header.stamp = rospy.Time.now() # Update stamp per step
                    m.header.frame_id = frame
                    m.pose.position.x, m.pose.position.y, m.pose.position.z = mat[:3, 3]
                    q = tft.quaternion_from_matrix(mat)
                    m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = q
                    return m

                # Package message and transmit down to your controller
                msg = to_msg(T_l0_target_r, 'panda_right_link0')
                #rospy.loginfo(msg.pose)
                self.pub_right_cartesian_target.publish(msg)
                                
                # Sleep momentarily between steps depending on your controller's execution rates
                # (e.g., if tracking a 10Hz trajectory chunk, sleep 0.1s per loop step)
                #rospy.sleep(0.1)
                
               #self.pub_right_cartesian_target.publish(msg)

                self.rate.sleep()
            
if __name__ == '__main__':
    try:
        node = Pi05RosInterface()
        node.run()
    except rospy.ROSInterruptException:
        pass