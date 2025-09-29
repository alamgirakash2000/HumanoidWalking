import os
import numpy as np
import torch
import pickle
import transforms3d as tf3
import collections

from envs.jvrc_walk_avoidance.jvrc_walk_avoidance_task import JvrcWalkAvoidanceTask
from robots.robot_base import RobotBase
from envs.common import mujoco_env
from envs.common import robot_interface
from envs.common import config_builder

from .gen_xml import *

class JvrcWalkAvoidanceEnv(mujoco_env.MujocoEnv):
    """JVRC walking environment with upper body obstacle avoidance.
    
    - Lower body (12 joints): Continues walking using original walking task
    - Upper body (20 joints): Avoids G1-style moving obstacles  
    """
    
    def __init__(self, path_to_yaml=None):

        ## Load CONFIG from yaml ##
        if path_to_yaml is None:
            path_to_yaml = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'configs/base.yaml')

        self.cfg = config_builder.load_yaml(path_to_yaml)

        sim_dt = self.cfg.sim_dt
        control_dt = self.cfg.control_dt

        self.history_len = self.cfg.obs_history_len

        path_to_xml = '/tmp/mjcf-export/jvrc_walk_avoidance/jvrc_walk_avoidance.xml'
        if not os.path.exists(path_to_xml):
            export_dir = os.path.dirname(path_to_xml)
            builder(export_dir, config={
                'num_obstacles': 3
            })

        mujoco_env.MujocoEnv.__init__(self, path_to_xml, sim_dt, control_dt)

        # Combined actuators: legs + upper body
        all_actuators = LEG_JOINTS + UPPER_BODY_JOINTS
        self.actuators = all_actuators

        # PD gains for all joints (legs + upper body)
        num_joints = len(all_actuators)
        pdgains = np.zeros((2, num_joints))
        
        # Use existing leg gains + add upper body gains
        leg_kp = self.cfg.kp  # 12 leg joints
        leg_kd = self.cfg.kd  # 12 leg joints
        
        # Upper body gains (lighter control)
        upper_body_kp = [30, 30, 30,  # WAIST
                        20, 20, 20,  # HEAD  
                        25, 25, 20, 15, 10, 10, 10,  # Right ARM
                        25, 25, 20, 15, 10, 10, 10]  # Left ARM
        
        upper_body_kd = [3, 3, 3,    # WAIST
                        2, 2, 2,    # HEAD
                        3, 3, 2, 2, 1, 1, 1,  # Right ARM
                        3, 3, 2, 2, 1, 1, 1]  # Left ARM
        
        # Combine gains: legs first, then upper body
        pdgains[0] = np.concatenate([leg_kp, upper_body_kp])
        pdgains[1] = np.concatenate([leg_kd, upper_body_kd])

        # define nominal pose (same as walking + neutral upper body)
        base_position = [0, 0, 0.81]
        base_orientation = [1, 0, 0, 0]
        leg_pose = [-30,  0, 0, 50, 0, -24,  # Right leg
                   -30,  0, 0, 50, 0, -24]  # Left leg
        upper_body_pose = [0, 0, 0,  # WAIST_Y, WAIST_P, WAIST_R
                          0, 0, 0,  # NECK_Y, NECK_R, NECK_P
                          -17, 0, 0, -30, 0, 0, 0,  # Right arm (degrees)
                          -17, 0, 0, -30, 0, 0, 0]  # Left arm (degrees)
        
        self.nominal_pose = (base_position + base_orientation + 
                           np.deg2rad(leg_pose + upper_body_pose).tolist())

        # set up interface
        self.interface = robot_interface.RobotInterface(self.model, self.data, 'R_ANKLE_P_S', 'L_ANKLE_P_S', None)
        
        # Add methods for obstacle management
        def get_body_pos(body_name):
            body_id = mujoco_env.mujoco.mj_name2id(self.model, mujoco_env.mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id >= 0:
                return self.data.xpos[body_id].copy()
            else:
                return np.zeros(3)
        
        def set_body_pos(body_name, position):
            body_id = mujoco_env.mujoco.mj_name2id(self.model, mujoco_env.mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id >= 0:
                self.data.xpos[body_id] = position
        
        self.interface.get_body_pos = get_body_pos
        self.interface.set_body_pos = set_body_pos

        # set up task (walking + avoidance)
        self.task = JvrcWalkAvoidanceTask(client=self.interface,
                                         dt=control_dt,
                                         neutral_foot_orient=np.array([1, 0, 0, 0]),
                                         root_body='PELVIS_S',
                                         lfoot_body='L_ANKLE_P_S',
                                         rfoot_body='R_ANKLE_P_S',
                                         num_obstacles=3)
        
        # set walking parameters (same as original jvrc_walk)
        self.task._goal_height_ref = 0.80
        self.task._total_duration = 1.1
        self.task._swing_duration = 0.75
        self.task._stance_duration = 0.35

        # Load the trained walking policy
        self.walking_policy = self._load_walking_policy()
        self.walk_env = self._create_walking_env()
        
        # set up robot with dual policy controller
        self.robot = DualPolicyRobot(pdgains, control_dt, self.interface, self.task, 
                                   self.walking_policy, self.walk_env)

        # Mirror indices for 76-element observation vector
        # Format: [root(5) + leg_pos(12) + leg_vel(12) + upper_pos(20) + upper_vel(20) + clock(3) + obs(4)]
        base_mir_obs = [-0.1, 1,                     # root orient (0-1)
                        -2, 3, -4,                   # root ang vel (2-4)
                        11, -12, -13, 14, -15, 16,   # leg motor pos [R->L] (5-10)
                         5,  -6,  -7,  8,  -9, 10,   # leg motor pos [L->R] (11-16)
                        23, -24, -25, 26, -27, 28,   # leg motor vel [R->L] (17-22)
                        17, -18, -19, 20, -21, 22,   # leg motor vel [L->R] (23-28)
                        # Upper body positions (29-48): WAIST(3) + NECK(3) + R_ARM(7) + L_ARM(7)
                        29, 30, -31,                  # WAIST_Y, WAIST_P, WAIST_R
                        32, -33, 34,                  # NECK_Y, NECK_R, NECK_P  
                        42, -43, -44, 45, -46, 47, -48,  # L_ARM -> R_ARM position
                        35, -36, -37, 38, -39, 40, -41,  # R_ARM -> L_ARM position
                        # Upper body velocities (49-68): same mirroring pattern
                        49, 50, -51,                  # WAIST velocities
                        52, -53, 54,                  # NECK velocities
                        62, -63, -64, 65, -66, 67, -68,  # L_ARM -> R_ARM velocity
                        55, -56, -57, 58, -59, 60, -61,  # R_ARM -> L_ARM velocity
                        # Clock and goal (69-71): no mirroring
                        69, 70, 71,
                        # Obstacle info (72-75): no mirroring
                        72, 73, 74, 75]
        
        self.robot.clock_inds = [69, 70]  # Clock indices
        self.robot.mirrored_obs = np.array(base_mir_obs, copy=True).tolist()
        
        # Mirror actions: legs + upper body
        leg_mirror_acts = [6, -7, -8, 9, -10, 11,    # Leg mirroring (right<->left)
                          0, -1, -2, 3, -4, 5]       # Leg mirroring (left<->right)
        upper_body_mirror_acts = [12, 13, -14,       # WAIST (mirror R)
                                 15, -16, 17,        # NECK (mirror R) 
                                 25, -26, -27, 28, -29, 30, -31,  # Left arm to right
                                 18, -19, -20, 21, -22, 23, -24]  # Right arm to left
        
        self.robot.mirrored_acts = leg_mirror_acts + upper_body_mirror_acts

        # set action space
        action_space_size = len(self.actuators)
        self.action_space = np.zeros(action_space_size)
        self.prev_prediction = np.zeros(action_space_size)

        # set observation space (extended for upper body)
        self.base_obs_len = self._calculate_obs_length()
        self.observation_history = collections.deque(maxlen=self.history_len)
        self.observation_space = np.zeros(self.base_obs_len*self.history_len)

        # manually define observation mean and std
        obs_mean, obs_std = self._get_obs_normalization()
        self.obs_mean = np.tile(obs_mean, self.history_len)
        self.obs_std = np.tile(obs_std, self.history_len)

    def _calculate_obs_length(self):
        """Calculate observation length: walking obs + obstacle info"""
        # Base walking obs: 5 (root) + 12 (leg pos) + 12 (leg vel) + 3 (clock/goal) = 32
        # Upper body: 20 (pos) + 20 (vel) = 40  
        # Obstacle info: 3 distances + 1 min distance = 4
        return 32 + 40 + 4

    def _get_obs_normalization(self):
        """Get observation normalization parameters"""
        obs_len = self._calculate_obs_length()
        
        # Base walking normalization (from jvrc_walk)
        leg_pose = [-30,  0, 0, 50, 0, -24, -30,  0, 0, 50, 0, -24]
        walking_mean = np.concatenate((
            np.zeros(5),  # Root orientation + angular velocity
            np.deg2rad(leg_pose), np.zeros(12),  # Leg positions + velocities
            [0.5, 0.5, 0.5]  # Clock + goal
        ))
        walking_std = np.concatenate((
            [0.2, 0.2, 1, 1, 1],  # Root
            0.5*np.ones(12), 4*np.ones(12),  # Legs
            [1, 1, 1]  # Clock + goal
        ))
        
        # Upper body normalization
        upper_body_mean = np.zeros(40)  # 20 positions + 20 velocities
        upper_body_std = np.concatenate([
            0.3*np.ones(20),  # Upper body positions
            2.0*np.ones(20)   # Upper body velocities
        ])
        
        # Obstacle normalization
        obstacle_mean = np.ones(4) * 0.5  # Assume obstacles at moderate distance
        obstacle_std = np.ones(4) * 0.5   # Standard deviation for distances
        
        # Combine all
        mean = np.concatenate([walking_mean, upper_body_mean, obstacle_mean])
        std = np.concatenate([walking_std, upper_body_std, obstacle_std])
        
        return mean, std

    def get_obs(self):
        """Get observation: walking state + upper body state + obstacle info"""
        
        # Walking observations (same as jvrc_walk)
        qpos = np.copy(self.interface.get_qpos())
        qvel = np.copy(self.interface.get_qvel())
        root_r, root_p = tf3.euler.quat2euler(qpos[3:7])[0:2]
        root_ang_vel = qvel[3:6]
        
        # Get joint positions and velocities for all actuated joints
        all_motor_pos = self.interface.get_act_joint_positions()
        all_motor_vel = self.interface.get_act_joint_velocities()
        
        # Split into legs and upper body
        leg_motor_pos = all_motor_pos[:12]  # First 12 are legs
        leg_motor_vel = all_motor_vel[:12]
        upper_body_pos = all_motor_pos[12:]  # Rest are upper body
        upper_body_vel = all_motor_vel[12:]
        
        # Walking clock and goal (from base task)
        clock = [np.sin(2 * np.pi * self.task._phase / self.task._period),
                 np.cos(2 * np.pi * self.task._phase / self.task._period)]
        ext_state = np.concatenate((clock, [self.task._goal_speed_ref]))
        
        # Obstacle information
        robot_pos = self.interface.get_body_pos('PELVIS_S')
        obstacle_distances = []
        for obstacle in self.task._obstacle_task:
            obstacle_pos = obstacle.frame[:3, 3]
            dist = np.linalg.norm(robot_pos - obstacle_pos)
            obstacle_distances.append(dist)
        
        # Pad to 3 obstacles if fewer
        while len(obstacle_distances) < 3:
            obstacle_distances.append(2.0)  # Far distance for missing obstacles
        
        min_obstacle_dist = min(obstacle_distances)
        obstacle_info = obstacle_distances + [min_obstacle_dist]

        # Combine all observations
        robot_state = np.concatenate([
            [root_r, root_p], root_ang_vel,  # Root state (5)
            leg_motor_pos, leg_motor_vel,    # Leg state (24)
            upper_body_pos, upper_body_vel,  # Upper body state (40)
        ])
        
        state = np.concatenate([robot_state, ext_state, obstacle_info])
        
        assert state.shape==(self.base_obs_len,), \
            f"State vector length expected to be: {self.base_obs_len} but is {len(state)}"

        if len(self.observation_history)==0:
            for _ in range(self.history_len):
                self.observation_history.appendleft(np.zeros_like(state))
            self.observation_history.appendleft(state)
        else:
            self.observation_history.appendleft(state)
        return np.array(self.observation_history).flatten()

    def step(self, action):
        """Step environment with dual policies: trained walking + PID upper body"""
        
        # NOTE: action is ignored since we use:
        # - Trained policy for walking (lower 12 joints)
        # - PID controller for upper body (upper 20 joints)
        
        # Get nominal offsets for all joints (legs + upper body)
        offsets = []
        joint_start_idx = 7  # Skip base position (3) + base orientation (4)
        for i, jnt in enumerate(self.actuators):
            nominal_idx = joint_start_idx + i
            if nominal_idx < len(self.nominal_pose):
                offsets.append(self.nominal_pose[nominal_idx])
            else:
                offsets.append(0.0)

        # Pass dummy action (not used by DualPolicyRobot)
        dummy_action = np.zeros(len(self.actuators))
        rewards, done = self.robot.step(dummy_action, np.asarray(offsets))
        obs = self.get_obs()

        # Update prediction for consistency (even though not used)
        self.prev_prediction = dummy_action

        return obs, sum(rewards.values()), done, rewards

    def _load_walking_policy(self):
        """Load the trained jvrc_walk policy"""
        try:
            policy_path = "/home/akash/Downloads/HumanoidWalking/trained/jvrc_walk/actor.pt"
            policy = torch.load(policy_path, map_location='cpu', weights_only=False)
            print("✅ Loaded trained jvrc_walk policy for lower body")
            return policy
        except Exception as e:
            print(f"❌ Could not load walking policy: {e}")
            return None
    
    def _create_walking_env(self):
        """Create a jvrc_walk environment for getting walking observations"""
        try:
            from envs.jvrc import JvrcWalkEnv
            return JvrcWalkEnv()
        except Exception as e:
            print(f"❌ Could not create walking environment: {e}")
            return None

    def reset_model(self):
        """Reset model"""
        init_qpos, init_qvel = self.nominal_pose.copy(), [0] * self.interface.nv()

        # set up init state
        self.set_state(
            np.asarray(init_qpos),
            np.asarray(init_qvel)
        )

        self.task.reset(iter_count=self.robot.iteration_count)
        
        # Reset walking environment too
        if self.walk_env is not None:
            self.walk_env.reset_model()

        self.prev_prediction = np.zeros_like(self.prev_prediction)
        self.observation_history = collections.deque(maxlen=self.history_len)
        obs = self.get_obs()
        return obs


class DualPolicyRobot(RobotBase):
    """Robot with dual policies: trained walking policy (lower) + PID avoidance (upper)"""
    
    def __init__(self, pdgains, control_dt, interface, task, walking_policy, walk_env):
        super().__init__(pdgains, control_dt, interface, task)
        
        # Store interface and policies
        self.interface = interface
        self.walking_policy = walking_policy
        self.walk_env = walk_env
        
        # Split gains for legs vs upper body
        self.leg_gains = pdgains[:, :12]      # First 12 joints (legs)
        self.upper_body_gains = pdgains[:, 12:]  # Rest (upper body)
        
        # PID parameters for upper body obstacle avoidance
        self.upper_body_kp = self.upper_body_gains[0]  # P gains
        self.upper_body_kd = self.upper_body_gains[1]  # D gains
        
        # Upper body target positions and previous positions for PID
        self.upper_body_targets = np.zeros(20)  # Neutral pose
        self.prev_upper_body_pos = np.zeros(20)
        
        # Initialize prev_action to avoid None error
        self.prev_action = np.zeros(pdgains.shape[1])
    
    def step(self, dummy_targets, offsets):
        """Execute one step with dual policies: trained walking + PID upper body"""
        
        # Update task (walking + obstacle movement)
        self.task.step()
        
        # === 1. GET WALKING ACTIONS FROM TRAINED POLICY ===
        if self.walking_policy is not None and self.walk_env is not None:
            # Get walking observation
            walk_obs = self.walk_env.get_obs()
            
            # Get leg actions from trained policy
            with torch.no_grad():
                walk_obs_tensor = torch.FloatTensor(walk_obs).unsqueeze(0)
                leg_actions = self.walking_policy(walk_obs_tensor).cpu().numpy().flatten()
            
            # Step the walking environment to keep it synchronized
            self.walk_env.step(leg_actions)
        else:
            # Fallback: use dummy targets for legs
            leg_actions = dummy_targets[:12] if len(dummy_targets) >= 12 else np.zeros(12)
        
        # === 2. COMPUTE UPPER BODY PID ACTIONS FOR OBSTACLE AVOIDANCE ===
        upper_body_actions = self._compute_upper_body_pid_actions()
        
        # === 3. APPLY COMBINED ACTIONS ===
        # Get current state
        current_pos = self.interface.get_act_joint_positions()
        current_vel = self.interface.get_act_joint_velocities()
        
        leg_pos = current_pos[:12]
        leg_vel = current_vel[:12]
        upper_body_pos = current_pos[12:]
        upper_body_vel = current_vel[12:]
        
        # Apply leg actions (from trained policy)
        leg_offsets = offsets[:12] if len(offsets) >= 12 else np.zeros(12)
        leg_position_error = (leg_actions + leg_offsets) - leg_pos
        leg_velocity_error = -leg_vel  # Target velocity is 0
        leg_torques = (self.leg_gains[0] * leg_position_error + 
                      self.leg_gains[1] * leg_velocity_error)
        
        # Apply upper body actions (from PID controller)
        upper_body_offsets = offsets[12:] if len(offsets) > 12 else np.zeros(20)
        upper_body_position_error = (upper_body_actions + upper_body_offsets) - upper_body_pos
        upper_body_velocity_error = -upper_body_vel
        upper_body_torques = (self.upper_body_kp * upper_body_position_error + 
                             self.upper_body_kd * upper_body_velocity_error)
        
        # Combine torques
        combined_torques = np.concatenate([leg_torques, upper_body_torques])
        
        # Apply torque limits
        combined_torques = np.clip(combined_torques, -100, 100)
        
        # Apply control
        self.interface.set_motor_torque(combined_torques)
        
        # Calculate rewards using the combined actions
        combined_actions = np.concatenate([leg_actions, upper_body_actions])
        rewards = self.task.calc_reward(combined_torques, self.prev_action, combined_actions)
        done = self.task.done()
        
        self.prev_action = combined_actions
        self.iteration_count += 1
        
        return rewards, done
    
    def _compute_upper_body_pid_actions(self):
        """Compute PID actions for upper body obstacle avoidance - EXACT G1 BEHAVIOR"""
        
        # Get current upper body positions
        current_pos = self.interface.get_act_joint_positions()
        upper_body_pos = current_pos[12:]
        
        # START WITH GENTLE NEUTRAL POSE (debug mode)
        # WAIST_Y, WAIST_P, WAIST_R, NECK_Y, NECK_R, NECK_P, 
        # R_SHOULDER_P, R_SHOULDER_R, R_SHOULDER_Y, R_ELBOW_P, R_ELBOW_Y, R_WRIST_R, R_WRIST_Y,
        # L_SHOULDER_P, L_SHOULDER_R, L_SHOULDER_Y, L_ELBOW_P, L_ELBOW_Y, L_WRIST_R, L_WRIST_Y
        g1_neutral_pose = np.array([
            0.0, 0.0, 0.0,  # WAIST neutral
            0.0, 0.0, 0.0,  # NECK neutral  
            -0.3, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0,  # Right arm (gentle)
            -0.3, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0   # Left arm (gentle)
        ])
        
        # Start with G1 neutral pose
        target_pose = g1_neutral_pose.copy()
        
        # Get multiple body positions for collision detection (like G1)
        waist_pos = self.interface.get_body_pos('WAIST_Y_S')
        head_pos = self.interface.get_body_pos('NECK_P_S') 
        r_shoulder_pos = self.interface.get_body_pos('R_SHOULDER_P_S')
        l_shoulder_pos = self.interface.get_body_pos('L_SHOULDER_P_S')
        r_elbow_pos = self.interface.get_body_pos('R_ELBOW_P_S')
        l_elbow_pos = self.interface.get_body_pos('L_ELBOW_P_S')
        
        body_positions = [waist_pos, head_pos, r_shoulder_pos, l_shoulder_pos, r_elbow_pos, l_elbow_pos]
        
        # AGGRESSIVE avoidance like G1 - check all obstacles against all body parts
        max_avoidance_strength = 0.0
        for obstacle in self.task._obstacle_task:
            obstacle_pos = obstacle.frame[:3, 3]
            
            # Check distance to all body parts (like G1 collision volumes)
            for body_pos in body_positions:
                if body_pos is not None:
                    obs_to_body = body_pos - obstacle_pos
                    distance = np.linalg.norm(obs_to_body)
                    
                    # DEBUG: Start with gentle movements to prevent falling
                    if distance < 0.5 and distance > 0.01:  # React when close
                        avoidance_dir = obs_to_body / distance
                        avoidance_strength = (0.5 - distance) / 0.5
                        avoidance_strength = np.clip(avoidance_strength, 0, 1)
                        max_avoidance_strength = max(max_avoidance_strength, avoidance_strength)
                        
                        # GENTLE movements first (debug mode)
                        # Waist movements (indices 0-2) - SMALL movements
                        if avoidance_dir[1] > 0:  # Obstacle to left, lean RIGHT
                            target_pose[2] -= 0.2 * avoidance_strength  # WAIST_R - gentle
                        else:  # Obstacle to right, lean LEFT
                            target_pose[2] += 0.2 * avoidance_strength
                        
                        # Arm movements - VISIBLE but stable
                        if avoidance_dir[1] > 0:  # Left side obstacle - move LEFT ARM
                            target_pose[13] -= 0.6 * avoidance_strength  # L_SHOULDER_P - visible
                            target_pose[14] += 0.4 * avoidance_strength  # L_SHOULDER_R - out
                        else:  # Right side obstacle - move RIGHT ARM  
                            target_pose[6] -= 0.6 * avoidance_strength   # R_SHOULDER_P - visible
                            target_pose[7] -= 0.4 * avoidance_strength   # R_SHOULDER_R - out
        
        # DEBUG: Remove oscillation for now to prevent instability
        # import time
        # oscillation = 0.1 * np.sin(time.time() * 10) * max_avoidance_strength
        # target_pose[6:] += oscillation  # Add oscillation to arms
        
        # Ensure movements are within safe joint limits 
        target_pose = np.clip(target_pose, -1.5, 1.5)  # More conservative limits
        
        # Update previous position for next derivative calculation
        self.prev_upper_body_pos = upper_body_pos.copy()
        
        return target_pose
