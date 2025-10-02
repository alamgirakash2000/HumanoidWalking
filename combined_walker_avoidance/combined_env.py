"""
Combined environment that integrates walking policy with upper body obstacle avoidance.

This environment:
1. Loads the trained walking policy for leg control
2. Uses PID+safety controller for upper body obstacle avoidance  
3. Manages both observation spaces and combines actions
4. Handles moving obstacles relative to the walking robot
"""
import os
import sys
import numpy as np
import torch
import collections
import transforms3d as tf3
from pathlib import Path
import mujoco
import mujoco.viewer

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import walking system components
from envs.common import mujoco_env, robot_interface
from envs.common import config_builder
from tasks.walking_task import WalkingTask
from robots.robot_base import RobotBase

# Import upper body system components
from simplified.g1_benchmark_pid.control import (
    PIDPolicy, SafeController, BasicSafeControllerConfig
)
from simplified.g1_benchmark_pid.robot_impl import G1BasicKinematics

# Import our combined configuration
from .robot_config import G1CombinedConfig, create_combined_g1_model
from .task_manager import CombinedBenchmarkTask
from simplified.g1_benchmark_pid.utils import VizColor


class G1CombinedEnv(mujoco_env.MujocoEnv):
    """
    Combined environment that enables the G1 robot to walk while avoiding obstacles
    using its upper body.
    """
    
    def __init__(self, path_to_yaml=None, walking_policy_path=None, enable_viewer=True):
        
        # Set up configuration
        if path_to_yaml is None:
            path_to_yaml = os.path.join(
                PROJECT_ROOT, 'envs/g1/configs/base.yaml'
            )
        
        self.cfg = config_builder.load_yaml(path_to_yaml)
        
        # Create combined robot configuration
        self.combined_config = G1CombinedConfig()

        # Simulation timing
        sim_dt = self.cfg.sim_dt
        control_dt = self.cfg.control_dt

        # Create and setup combined MuJoCo model
        if not os.path.exists(os.path.join(self.combined_config.export_dir, 'g1_combined.xml')):
            create_combined_g1_model(self.combined_config.export_dir)
        
        model_path = os.path.join(self.combined_config.export_dir, 'g1_combined.xml')
        
        # Initialize MuJoCo environment with workaround for MjSpec loading issue
        self._init_mujoco_env_with_workaround(model_path, sim_dt, control_dt)
        
        # Initialize visual elements for spheres  
        self._init_visual_elements()
        
        # Create renderer for visual elements (like original simplified system)
        self.renderer = mujoco.Renderer(self.model)
        
        # Create viewer like original simplified system (this is crucial!)
        self.enable_viewer = enable_viewer
        self.viewer = None  # Will be set by runner when needed
        # Store safety info for visualization
        self._last_safety_info = None
        
        # Setup walking system components
        self._setup_walking_system()
        
        # Setup upper body obstacle avoidance system  
        self._setup_upper_body_system()
        
        # Load trained walking policy
        self._load_walking_policy(walking_policy_path)
        
        # Setup combined task management
        self._setup_task_management()

        # Initialize action and observation spaces
        self._setup_spaces()
        
        # Integrated upper-body position target (convert vel commands to pos)
        self._upper_body_pos_target = np.zeros(17)
        
        print("✅ Combined G1 environment initialized successfully!")
        print(f"   - Walking joints: {len(self.combined_config.leg_joints)}")
        print(f"   - Upper body joints: {len(self.combined_config.upper_body_joints)}")
        print(f"   - Total controlled joints: {len(self.combined_config.all_joints)}")
        
    def _init_mujoco_env_with_workaround(self, model_path, sim_dt, control_dt):
        """
        Initialize MuJoCo environment with workaround for MjSpec loading issues.
        
        The standard MujocoEnv uses MjSpec which doesn't handle complex models
        with mesh dependencies properly. We use direct MjModel loading instead.
        """
        import mujoco
        
        if not model_path.startswith("/"):
            raise Exception("Provide full path to robot description package.")
        if not os.path.exists(model_path):
            raise IOError("File %s does not exist" % model_path)

        # Use direct MjModel loading instead of MjSpec
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None

        # Set frame skip and sim dt (same as MujocoEnv)
        self.frame_skip = (control_dt/sim_dt)
        self.model.opt.timestep = sim_dt

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        
    def _init_visual_elements(self):
        """Initialize visual elements for obstacles and goals."""
        
        # Add visual sites to the model for rendering spheres
        import mujoco
        
        # We'll use MuJoCo's site system to render spheres
        # Add sites for obstacles (red spheres)
        self.obstacle_site_ids = []
        self.goal_site_ids = []
        
        # Create sites programmatically (if model supports it)
        # For now, we'll track positions and render them in the viewer
        self.visual_obstacles = []
        self.visual_goals = []
        self.visual_collision_volumes = []
        
    def _setup_viewer(self):
        """Setup the viewer like original simplified system."""
        if self.viewer:
            # Setup camera
            self.viewer.cam.distance = 4.0
            self.viewer.cam.lookat[0] = 0
            self.viewer.cam.lookat[1] = 0  
            self.viewer.cam.lookat[2] = 1.0
            self.viewer.cam.elevation = -15
            self.viewer.cam.azimuth = 180
            self.viewer.opt.geomgroup = 1  # Show visual geoms
        
    def render_sphere(self, pos, size, color):
        """Render a sphere using MuJoCo's visualization system (like original simplified)."""
        import mujoco
        
        # Add to renderer scene
        if hasattr(self, 'renderer') and self.renderer._scene.ngeom < self.renderer._scene.maxgeom:
            mujoco.mjv_initGeom(
                self.renderer._scene.geoms[self.renderer._scene.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=size * np.ones(3) if np.isscalar(size) else size,
                pos=pos.flatten(),
                mat=np.eye(3).flatten(),
                rgba=color,
            )
            self.renderer._scene.ngeom += 1
            
        # Add to viewer scene if viewer exists
        if hasattr(self, 'viewer') and self.viewer and self.viewer.user_scn.ngeom < self.viewer.user_scn.maxgeom:
            mujoco.mjv_initGeom(
                self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=size * np.ones(3) if np.isscalar(size) else size,
                pos=pos.flatten(),
                mat=np.eye(3).flatten(),
                rgba=color,
            )
            self.viewer.user_scn.ngeom += 1
            
    def render_line_segment(self, pos1, pos2, radius, color):
        """Render a line segment using MuJoCo's visualization (for collision lines).""" 
        import mujoco
        
        pos1 = np.asarray(pos1)
        pos2 = np.asarray(pos2)
        midpoint = (pos1 + pos2) / 2
        length = np.linalg.norm(pos2 - pos1)
        direction = (pos2 - pos1) / (length + 1e-12)
        z_axis = np.array([0, 0, 1])
        axis = np.cross(z_axis, direction)
        axis_len = np.linalg.norm(axis)
        if axis_len > 1e-6:
            axis = axis / axis_len
            angle = np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
            quat = np.zeros(4)
            mujoco.mju_axisAngle2Quat(quat, axis, angle)
            # mjv_initGeom expects a 3x3 matrix flattened; use contiguous array
            rot_matrix = np.zeros((3, 3)).flatten()
            mujoco.mju_quat2Mat(rot_matrix, quat)
        else:
            rot_matrix = np.eye(3).flatten()
            
        # Add to renderer scene
        if hasattr(self, 'renderer') and self.renderer._scene.ngeom < self.renderer._scene.maxgeom:
            mujoco.mjv_initGeom(
                self.renderer._scene.geoms[self.renderer._scene.ngeom],
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                size=[radius, length / 2, 0.0],
                pos=midpoint.flatten(),
                mat=rot_matrix,
                rgba=color,
            )
            self.renderer._scene.ngeom += 1
            
        # Add to viewer scene
        if hasattr(self, 'viewer') and self.viewer and self.viewer.user_scn.ngeom < self.viewer.user_scn.maxgeom:
            mujoco.mjv_initGeom(
                self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                size=[radius, length / 2, 0.0],
                pos=midpoint.flatten(),
                mat=rot_matrix,
                rgba=color,
            )
            self.viewer.user_scn.ngeom += 1
        
    def set_state(self, qpos, qvel):
        """Set the state of the simulation (from MujocoEnv)."""
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.data.qpos.ravel().copy(), self.data.qvel.ravel().copy()
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        import mujoco
        mujoco.mj_forward(self.model, self.data)
        return old_state

    def dt(self):
        """Get the simulation timestep (from MujocoEnv)."""
        return self.model.opt.timestep * self.frame_skip
        
    def _setup_walking_system(self):
        """Setup the walking system components."""
        
        # Joint names for legs (same as original walking env)
        self.leg_names = self.combined_config.leg_joints
        
        # PD gains for all joints (legs + upper body)
        gains_dict = self.cfg.pdgains.to_dict()
        
        # Get gains for leg joints from config
        leg_kp, leg_kd = zip(*[gains_dict[jn] for jn in self.leg_names])
        
        # Set reasonable gains for upper body joints (waist + arms)
        # These will be overridden by the upper body controller anyway
        upper_body_kp = [100.0] * len(self.combined_config.upper_body_joints)  
        upper_body_kd = [10.0] * len(self.combined_config.upper_body_joints)
        
        # Combine gains: legs first, then upper body
        all_kp = list(leg_kp) + upper_body_kp
        all_kd = list(leg_kd) + upper_body_kd
        
        self.combined_pdgains = np.array([all_kp, all_kd])
        self.walking_pdgains = np.array([leg_kp, leg_kd])  # Keep for reference
        
        # Nominal standing pose for combined system
        base_position = [0, 0, 0.80]
        base_orientation = [1, 0, 0, 0]
        leg_pose = [
            0.0, 0.0, -0.2, 0.6, -0.2, 0.0,   # left leg
            0.0, 0.0, -0.2, 0.6, -0.2, 0.0,   # right leg  
        ]
        # Upper body neutral pose (waist + arms)
        upper_body_pose = [0.0] * len(self.combined_config.upper_body_joints)
        
        # Combined pose: base + legs + upper body
        self.nominal_pose = base_position + base_orientation + leg_pose + upper_body_pose
        
        # Robot interface for walking system
        self.interface = robot_interface.RobotInterface(
            self.model, self.data,
            'right_ankle_roll_link', 'left_ankle_roll_link', None
        )
        
        # Walking task setup
        self.walking_task = WalkingTask(
            client=self.interface,
            dt=self.cfg.control_dt,
            neutral_foot_orient=np.array([1, 0, 0, 0]),
            root_body='pelvis',
            lfoot_body='left_ankle_roll_link', 
            rfoot_body='right_ankle_roll_link',
        )
        self.walking_task._goal_height_ref = 0.90
        self.walking_task._total_duration = 1.1
        self.walking_task._swing_duration = 0.75
        self.walking_task._stance_duration = 0.35
        # Set walking speed
        self.walking_task._goal_speed_ref = 0.25
        
        # Walking robot controller (using combined gains for all joints)
        self.walking_robot = RobotBase(
            self.combined_pdgains, self.cfg.control_dt, 
            self.interface, self.walking_task
        )
        
        # Walking observation setup
        self.history_len = self.cfg.obs_history_len
        self.base_obs_len = 5 + 12 + 12 + 3  # Same as G1WalkEnv
        self.walking_observation_history = collections.deque(maxlen=self.history_len)
        
    def _setup_upper_body_system(self):
        """Setup the upper body obstacle avoidance system."""
        
        # Create kinematics for upper body (reusing existing implementation)
        # Note: This uses the reduced model for upper body only
        self.upper_body_kinematics = G1BasicKinematics(
            self.combined_config.upper_body_config
        )
        
        # Upper body control policy  
        self.upper_body_policy = PIDPolicy(
            self.combined_config.upper_body_config,
            self.upper_body_kinematics
        )
        
        # Safety controller for obstacle avoidance
        safety_cfg = BasicSafeControllerConfig()
        safety_cfg.safe_algo.eta = 1.5
        safety_cfg.safe_algo.safety_buffer = 0.05
        
        # Optimize safety controller for better performance
        # Using less strict solver settings to speed up IPOPT calls
        safety_cfg.safe_algo.eta = 2.0  # Less conservative safety margin
        safety_cfg.safe_algo.safety_buffer = 0.03  # Smaller safety buffer
        
        self.safety_controller = SafeController(
            cfg=safety_cfg,
            robot_cfg=self.combined_config.upper_body_config,
            robot_kinematics=self.upper_body_kinematics
        )
        
    def _load_walking_policy(self, policy_path):
        """Load the trained walking policy."""
        
        if policy_path is None:
            policy_path = PROJECT_ROOT / "trained/g1_walk/actor.pt"
            
        if not os.path.exists(policy_path):
            raise FileNotFoundError(f"Walking policy not found at: {policy_path}")
            
        print(f"Loading walking policy from: {policy_path}")
        
        # Load the trained policy
        self.walking_policy = torch.load(policy_path, weights_only=False)
        self.walking_policy.eval()
        
        # Setup observation normalization (same as G1WalkEnv)
        half_sitting_pose = [
            0.0, 0.0, -0.2, 0.6, -0.2, 0.0,   # left
            0.0, 0.0, -0.2, 0.6, -0.2, 0.0,   # right
        ]
        
        self.walking_obs_mean = np.concatenate((
            np.zeros(5),
            half_sitting_pose,
            np.zeros(12),
            [0.5, 0.5, 0.5]
        ))
        self.walking_obs_std = np.concatenate((
            [0.2, 0.2, 1, 1, 1],
            0.5 * np.ones(12),
            4 * np.ones(12),
            [1, 1, 1]
        ))
        self.walking_obs_mean = np.tile(self.walking_obs_mean, self.history_len)
        self.walking_obs_std = np.tile(self.walking_obs_std, self.history_len)
        
    def _setup_task_management(self):
        """Setup the combined task that manages obstacles relative to walking robot."""
        
        self.task_manager = CombinedBenchmarkTask(
            robot_cfg=self.combined_config.upper_body_config,
            robot_kinematics=self.upper_body_kinematics,
            max_episode_length=2000,
            num_obstacles=3
        )
        
    def _setup_spaces(self):
        """Setup action and observation spaces."""
        
        # Total action space: legs (12) + upper body (17) = 29
        self.total_action_space_size = len(self.combined_config.all_joints)
        self.combined_action = np.zeros(self.total_action_space_size)
        
        # Walking observation space (with history)
        self.walking_obs_space_size = self.base_obs_len * self.history_len
        
    def get_walking_observation(self):
        """Get observation for the walking policy (same structure as G1WalkEnv)."""
        
        # Clock and external state
        clock = [np.sin(2 * np.pi * self.walking_task._phase / self.walking_task._period),
                 np.cos(2 * np.pi * self.walking_task._phase / self.walking_task._period)]
        ext_state = np.concatenate((clock, [self.walking_task._goal_speed_ref]))
        
        # Robot state
        qpos = np.copy(self.interface.get_qpos())
        qvel = np.copy(self.interface.get_qvel())
        root_r, root_p = tf3.euler.quat2euler(qpos[3:7])[0:2]
        root_r = np.array([root_r])
        root_p = np.array([root_p])
        root_ang_vel = qvel[3:6]
        
        motor_pos = self.interface.get_act_joint_positions()[:12]  # Only leg joints
        motor_vel = self.interface.get_act_joint_velocities()[:12]  # Only leg joints
        
        robot_state = np.concatenate([root_r, root_p, root_ang_vel, motor_pos, motor_vel])
        state = np.concatenate([robot_state, ext_state])
        
        # History management
        if len(self.walking_observation_history) == 0:
            for _ in range(self.history_len):
                self.walking_observation_history.appendleft(np.zeros_like(state))
                
        self.walking_observation_history.appendleft(state)
        return np.array(self.walking_observation_history).flatten()
        
    def get_upper_body_observation(self):
        """Get observation/feedback for upper body system."""
        
        # Get current joint positions and velocities for upper body
        all_joint_pos = self.interface.get_act_joint_positions()
        all_joint_vel = self.interface.get_act_joint_velocities()
        
        # Extract upper body joints (after the 12 leg joints)
        upper_body_joint_pos = all_joint_pos[12:]  # 17 joints
        upper_body_joint_vel = all_joint_vel[12:]  # 17 joints
        
        # The upper body system expects 20 DOF: 17 joints + 3 base movement
        # Provide actual base XY and yaw from the walking robot so safety aligns with world
        base_pos = self.data.body('pelvis').xpos.copy()
        base_quat = self.data.body('pelvis').xquat.copy()  # [w, x, y, z]
        base_r, base_p, base_yaw = tf3.euler.quat2euler(base_quat)
        base_linear_x = float(base_pos[0])
        base_linear_y = float(base_pos[1])  
        base_rot_yaw = float(base_yaw)
        
        upper_body_pos = np.concatenate([
            upper_body_joint_pos, 
            [base_linear_x, base_linear_y, base_rot_yaw]
        ])  # 20 DOF total
        
        upper_body_vel = np.concatenate([
            upper_body_joint_vel,
            [0.0, 0.0, 0.0]  # Base velocities (handled by walking)
        ])  # 20 DOF total
        
        # Robot base frame (for obstacle avoidance calculations)
        base_rot_mat = tf3.quaternions.quat2mat(base_quat)
        
        robot_base_frame = np.eye(4)
        robot_base_frame[:3, :3] = base_rot_mat
        robot_base_frame[:3, 3] = base_pos
        
        # Create feedback structure expected by upper body system
        agent_feedback = {
            "dof_pos_cmd": upper_body_pos,
            "dof_vel_cmd": upper_body_vel, 
            "dof_pos_fbk": upper_body_pos,
            "robot_base_frame": robot_base_frame,
            "state": upper_body_pos  # 20 DOF state for upper body control
        }
        
        return agent_feedback
        
    def step(self, action=None):
        """
        Combined step that:
        1. Gets walking policy action for legs
        2. Gets upper body policy action for obstacle avoidance  
        3. Combines and applies actions
        4. Updates both task systems
        """
        
        # Get walking policy action
        walking_obs = self.get_walking_observation()
        with torch.no_grad():
            walking_action_tensor = self.walking_policy.forward(
                torch.tensor(walking_obs, dtype=torch.float32), 
                deterministic=True
            )
            walking_action = walking_action_tensor.detach().numpy()
        
        # Get upper body policy action  
        upper_body_feedback = self.get_upper_body_observation()
        task_info = self.task_manager.get_info(upper_body_feedback)

        # Get reference action from PID policy
        u_ref, action_info = self.upper_body_policy.act(
            upper_body_feedback, task_info
        )

        # Apply safety controller (with performance optimization)
        # Skip expensive IPOPT optimization when obstacles are far away
        try:
            robot_pos = upper_body_feedback["robot_base_frame"][:3, 3] 
            min_obstacle_dist = float('inf')
            if "obstacle_task" in task_info and "frames_world" in task_info["obstacle_task"]:
                for obstacle_frame in task_info["obstacle_task"]["frames_world"]:
                    obstacle_pos = obstacle_frame[:3, 3]
                    dist = np.linalg.norm(robot_pos - obstacle_pos)
                    min_obstacle_dist = min(min_obstacle_dist, dist)
                    
            # Skip expensive safety optimization if obstacles are far (>1m)
            if min_obstacle_dist > 1.0:
                u_safe = u_ref  # Use PID control directly - much faster
                combined_info = {"safety_bypassed": True, "min_dist": min_obstacle_dist}
            else:
                # Use full safety controller when obstacles are close
                u_safe, combined_info = self.safety_controller.safe_control(
                    upper_body_feedback["state"], u_ref,
                    upper_body_feedback, task_info, action_info
                )
                combined_info["safety_bypassed"] = False
                combined_info["min_dist"] = min_obstacle_dist
        except Exception:
            # Fallback to full safety on any error
            u_safe, combined_info = self.safety_controller.safe_control(
                upper_body_feedback["state"], u_ref,
                upper_body_feedback, task_info, action_info
            )
        
        # Keep latest safety info for rendering
        self._last_safety_info = combined_info

        # Combine actions: legs (walking) + upper body (obstacle avoidance)
        self.combined_action[:12] = walking_action  # Leg joints  
        
        # ROBUST FIX: Always ensure exactly 17 upper body commands
        u_safe_array = np.asarray(u_safe).flatten()  # Ensure it's a flat numpy array
        
        # Debug counter for potential debugging (disabled by default)
        if not hasattr(self, '_debug_step_count'):
            self._debug_step_count = 0
        self._debug_step_count += 1
        
        if len(u_safe_array) >= 17:
            upper_body_vel_cmd = u_safe_array[:17]
        else:
            # Pad with zeros if somehow less than 17 (shouldn't happen)
            upper_body_vel_cmd = np.pad(u_safe_array, (0, 17 - len(u_safe_array)), 'constant')
            print(f"WARNING: u_safe had {len(u_safe_array)} elements, padded to 17")
        
        # Integrate velocity commands to position targets (match simplified behavior)
        if not hasattr(self, '_upper_body_pos_target') or self._upper_body_pos_target.shape[0] != 17:
            self._upper_body_pos_target = np.zeros(17)
        self._upper_body_pos_target = (
            self._upper_body_pos_target + upper_body_vel_cmd.astype(np.float64) * float(self.cfg.control_dt)
        )
        upper_body_pos_target = self._upper_body_pos_target
        
        # Ensure exactly 17 elements before assignment
        upper_body_pos_target = upper_body_pos_target.astype(np.float64)  # Ensure proper dtype
        assert len(upper_body_pos_target) == 17, f"Expected 17 upper body targets, got {len(upper_body_pos_target)}"
        self.combined_action[12:29] = upper_body_pos_target
        
        # Apply walking system action processing (for legs only)
        leg_targets = self.cfg.action_smoothing * walking_action + \
                     (1 - self.cfg.action_smoothing) * self.prev_walking_prediction if hasattr(self, 'prev_walking_prediction') else walking_action
        
        # Create combined targets: legs + upper body
        combined_targets = np.zeros(29)
        combined_targets[:12] = leg_targets  # Leg targets
        # ROBUST FIX: Always ensure exactly 17 upper body targets
        combined_targets[12:29] = upper_body_pos_target  # Use integrated position targets
        
        # Create combined offsets: legs + upper body  
        leg_offsets = [
            self.nominal_pose[self.interface.get_jnt_qposadr_by_name(jnt)[0]]
            for jnt in self.leg_names
        ]
        upper_body_offsets = [0.0] * 17  # No offsets for upper body
        combined_offsets = np.array(leg_offsets + upper_body_offsets)
        
        # Update walking system (it needs all 29 actions now)
        walking_rewards, walking_done = self.walking_robot.step(combined_targets, combined_offsets)
        
        # Update task manager (obstacles, goals) using the LATEST robot base after stepping
        updated_upper_body_feedback = self.get_upper_body_observation()
        self.task_manager.step(updated_upper_body_feedback)
        
        # Get observations for next step
        walking_obs_new = self.get_walking_observation()
        
        # Store for next step
        self.prev_walking_prediction = walking_action
        
        # Combine rewards and done conditions
        total_reward = sum(walking_rewards.values())
        done = walking_done or task_info.get("done", False)
        
        # Create combined info (use post-step task info for accurate visualization/state)
        task_info_post = self.task_manager.get_info(updated_upper_body_feedback)
        done = walking_done or task_info_post.get("done", False)
        info = {
            "walking_rewards": walking_rewards,
            "task_info": task_info_post,
            "safety_info": combined_info,
            "done": done
        }
        
        return walking_obs_new, total_reward, done, info
        
    def _apply_combined_actions(self, leg_targets, leg_offsets, upper_body_actions):
        """Apply the combined actions to the simulation."""
        
        # Set leg joint targets
        for i, jnt_name in enumerate(self.leg_names):
            actuator_id = self.model.actuator(f"{jnt_name}_motor").id
            self.data.ctrl[actuator_id] = leg_targets[i] + leg_offsets[i]
            
        # Set upper body joint targets
        for i, jnt_name in enumerate(self.combined_config.upper_body_joints):
            actuator_id = self.model.actuator(f"{jnt_name}_motor").id
            self.data.ctrl[actuator_id] = upper_body_actions[i]
            
        # Step simulation
        self.interface.step()
        
    def reset_model(self):
        """Reset the combined environment."""
        
        # Reset MuJoCo state
        init_qpos = self.nominal_pose.copy()  # Already includes all joints
        init_qvel = [0] * self.model.nv  # Use model.nv directly
        
        # Add some initial noise
        c = self.cfg.init_noise * np.pi / 180 if hasattr(self.cfg, 'init_noise') else 0
        root_adr = self.interface.get_jnt_qposadr_by_name('root')[0]
        
        init_qpos[root_adr + 3:root_adr + 7] = tf3.euler.euler2quat(
            np.random.uniform(-c, c), np.random.uniform(-c, c), 0
        )
        
        # Set initial state
        self.set_state(np.asarray(init_qpos), np.asarray(init_qvel))
        
        # Adjust pelvis height (same as G1WalkEnv)
        try:
            lf_z = float(self.interface.get_object_xpos_by_name('lf_force', 'OBJ_SITE')[2])
            rf_z = float(self.interface.get_object_xpos_by_name('rf_force', 'OBJ_SITE')[2])
            min_foot_z = min(lf_z, rf_z)
            clearance = 0.002
            delta = (0.0 + clearance) - min_foot_z
            init_qpos[root_adr + 2] += delta
            self.set_state(np.asarray(init_qpos), np.asarray(init_qvel))
        except Exception:
            pass
            
        # Stabilize
        for _ in range(3):
            self.interface.step()
            
        # Reset task systems
        self.walking_task.reset(iter_count=self.walking_robot.iteration_count)
        self.task_manager.reset()
        
        # Reset observations
        self.prev_walking_prediction = np.zeros(12)
        self.walking_observation_history = collections.deque(maxlen=self.history_len)
        
        obs = self.get_walking_observation()
        return obs
        
    def render(self, mode='human'):
        """Render the environment including obstacles and goals (like original simplified system)."""
        
        # Clear visual geoms each frame (like original simplified system) - optimized
        if hasattr(self, 'renderer') and self.renderer._scene.ngeom > 0:
            self.renderer._scene.ngeom = 0
        if hasattr(self, 'viewer') and self.viewer and self.viewer.user_scn.ngeom > 0:
            self.viewer.user_scn.ngeom = 0
        
        # Get current state for rendering  
        upper_body_feedback = self.get_upper_body_observation()
        task_info = self.task_manager.get_info(upper_body_feedback)
        
        # Calculate robot collision frame positions for visualization
        robot_base_frame = upper_body_feedback["robot_base_frame"]
        # Use full 20 DoF state to compute kinematics like simplified system
        full_state = upper_body_feedback["state"]
        dof_pos = self.upper_body_kinematics.robot_cfg.decompose_state_to_dof(full_state)
        frames = self.upper_body_kinematics.forward_kinematics(dof_pos)
        
        # Transform to world coordinates
        robot_frames_world = np.zeros_like(frames)
        for i in range(len(frames)):
            robot_frames_world[i, :, :] = robot_base_frame @ frames[i, :, :]
        
        # Render moving obstacles (GRAY SPHERES - like original)
        for frame_world, geom in zip(
            task_info["obstacle_task"]["frames_world"], 
            task_info["obstacle_task"]["geom"]
        ):
            if geom.type == "sphere":
                pos = frame_world[:3, 3]
                radius = geom.attributes["radius"]
                # Use original obstacle color: gray [0.5, 0.5, 0.5, 0.7]
                self.render_sphere(pos, radius, geom.color)
        
        # Render goals (LIGHT GREEN SPHERES - like original) 
        goal_left_world = (robot_base_frame @ task_info["goal_teleop"]["left"])[:3, 3]
        goal_right_world = (robot_base_frame @ task_info["goal_teleop"]["right"])[:3, 3]
        # Use original goal color: light green [48/255, 245/255, 93/255, 0.3]
        goal_color = [48/255, 245/255, 93/255, 0.3]
        self.render_sphere(goal_left_world, 0.05, goal_color)
        self.render_sphere(goal_right_world, 0.05, goal_color)
        
        # Render collision volumes around robot with safety coloring (like original)
        env_collision_mask = getattr(self.safety_controller.safety_index, 'env_collision_mask', None)
        self_collision_mask = getattr(self.safety_controller.safety_index, 'self_collision_mask', None)

        def viz_critical_env_pairs(mat, thres, mask, line_width, line_color):
            if mat is None or mask is None or mask.size == 0:
                return []
            masked_mat = mat[mask]
            masked_indices = np.argwhere(mask)
            indices_of_interest = masked_indices[np.argwhere(masked_mat >= thres).reshape(-1)]
            for i, j in indices_of_interest:
                self.render_line_segment(
                    pos1=robot_frames_world[i][:3, 3],
                    pos2=task_info["obstacle"]["frames_world"][j][:3, 3],
                    radius=line_width,
                    color=line_color,
                )
            return indices_of_interest

        def viz_critical_self_pairs(mat, thres, mask, line_width, line_color):
            if mat is None or mask is None or mask.size == 0:
                return []
            masked_mat = mat[mask]
            masked_indices = np.argwhere(mask)
            indices_of_interest = masked_indices[np.argwhere(masked_mat >= thres).reshape(-1)]
            for i, j in indices_of_interest:
                self.render_line_segment(
                    pos1=robot_frames_world[i][:3, 3],
                    pos2=robot_frames_world[j][:3, 3],
                    radius=line_width,
                    color=line_color,
                )
            return indices_of_interest

        action_info = self._last_safety_info or {}
        phi_hold_mat_env = action_info.get("phi_hold_mat_env", None)
        active_pairs_hold_env = viz_critical_env_pairs(
            phi_hold_mat_env, 0.0, env_collision_mask, 0.002, VizColor.hold
        )
        phi_hold_mat_self = action_info.get("phi_hold_mat_self", None)
        active_pairs_hold_self = viz_critical_self_pairs(
            phi_hold_mat_self, 0.0, self_collision_mask, 0.002, VizColor.hold
        )
        phi_safe_mat_env = action_info.get("phi_safe_mat_env", None)
        active_pairs_unsafe_env = viz_critical_env_pairs(
            phi_safe_mat_env, 0.0, env_collision_mask, 0.02, VizColor.unsafe
        )
        phi_safe_mat_self = action_info.get("phi_safe_mat_self", None)
        active_pairs_unsafe_self = viz_critical_self_pairs(
            phi_safe_mat_self, 0.0, self_collision_mask, 0.02, VizColor.unsafe
        )

        # Now draw the robot collision volumes with appropriate colors
        for frame_id, frame_world in enumerate(robot_frames_world):
            try:
                geom = self.combined_config.upper_body_config.CollisionVol[
                    self.combined_config.upper_body_config.Frames(frame_id)
                ]
            except Exception:
                continue

            if any(frame_id in pair for pair in active_pairs_unsafe_self) or any(
                frame_id == _frame_id for _frame_id, _ in active_pairs_unsafe_env
            ):
                color = VizColor.unsafe
            elif any(frame_id in pair for pair in active_pairs_hold_self) or any(
                frame_id == _frame_id for _frame_id, _ in active_pairs_hold_env
            ):
                color = VizColor.hold
            else:
                if (
                    hasattr(self.safety_controller.safety_index, 'env_collision_vol_ignore') and
                    frame_id in self.safety_controller.safety_index.env_collision_vol_ignore
                ):
                    color = VizColor.collision_volume_ignored
                else:
                    color = VizColor.collision_volume

            if geom.type == "sphere":
                pos = frame_world[:3, 3]
                radius = geom.attributes["radius"]
                self.render_sphere(pos, radius, color)
        
        # Collision lines rendered above via viz_critical_* helpers
        
        # Update scene and sync viewer (optimized)
        if hasattr(self, 'renderer'):
            self.renderer.update_scene(self.data)
        if hasattr(self, 'viewer') and self.viewer:
            self.viewer.sync()
            
    def _render_collision_constraints(self, robot_frames_world, task_info):
        """Render lines showing active collision constraints (like original)."""
        
        # Get safety info if available
        try:
            # This would show which collision constraints are active
            # For now, just show lines between robot and close obstacles
            for i, obstacle_frame in enumerate(task_info["obstacle"]["frames_world"]):
                obstacle_pos = obstacle_frame[:3, 3]
                
                # Find closest robot collision volume
                min_dist = float('inf')
                closest_robot_pos = None
                
                for robot_frame in robot_frames_world:
                    robot_pos = robot_frame[:3, 3]
                    dist = np.linalg.norm(robot_pos - obstacle_pos)
                    if dist < min_dist:
                        min_dist = dist
                        closest_robot_pos = robot_pos
                
                # Show line if obstacle is close (potential collision)
                if min_dist < 0.3:  # Within collision detection range
                    if min_dist < 0.15:  # Very close - red line
                        color = [1.0, 0.0, 0.0, 0.8]  # Red
                    else:  # Close - yellow line
                        color = [1.0, 1.0, 0.0, 0.6]  # Yellow
                        
                    self.render_line_segment(closest_robot_pos, obstacle_pos, 0.01, color)
                    
        except Exception:
            # Continue without collision lines if there's an issue
            pass
        
    def close(self):
        """Clean up resources."""
        if hasattr(self, 'viewer') and self.viewer:
            self.viewer.close()
        super().close()


if __name__ == "__main__":
    # Test the combined environment
    env = G1CombinedEnv()
    obs = env.reset()
    
    print(f"Environment created successfully!")
    print(f"Observation shape: {obs.shape}")
    
    # Test a few steps
    for i in range(10):
        obs, reward, done, info = env.step()
        print(f"Step {i}: reward={reward:.4f}, done={done}")
        
        if done:
            obs = env.reset()
            print("Environment reset")
            
    env.close()
    print("Test completed successfully!")
