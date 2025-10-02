"""
Task manager for the combined walking + obstacle avoidance system.

This manager handles:
1. Moving obstacles that stay relative to the walking robot
2. Dynamic goals for the upper body to reach
3. Collision detection and safety constraints
4. Episode management
"""
import numpy as np
from simplified.g1_benchmark_pid.utils import Geometry, VizColor, compute_pairwise_dist


class MovingTaskObject3D:
    """
    Enhanced 3D task object that can move relative to a moving reference frame.
    
    This is an enhanced version of the original TaskObject3D that accounts for
    the robot's movement during walking.
    """
    
    def __init__(self, **kwargs):
        # Original object properties
        self.frame = kwargs.get("frame", np.eye(4))
        self.velocity = kwargs.get("velocity", 1.0)
        self.bound = kwargs.get("bound", np.zeros((3, 2)))
        self.smooth_weight = kwargs.get("smooth_weight", 1.0)
        self.keep_direction_step = kwargs.get("keep_direction_step", 1)
        
        # Enhanced properties for walking robot
        self.follow_robot = kwargs.get("follow_robot", True)
        self.robot_relative_bounds = kwargs.get("robot_relative_bounds", True)
        self.max_distance_from_robot = kwargs.get("max_distance_from_robot", 2.0)
        
        # Internal state
        self.last_direction = np.zeros(3)
        self.step_counter = 0
        self.last_frame = self.frame.copy()
        self.robot_last_position = np.zeros(3)
        
    def move(self, robot_base_frame=None, mode="Brownian"):
        """
        Move the object with consideration for robot movement.
        
        Args:
            robot_base_frame: 4x4 transformation matrix of robot base
            mode: Movement mode ('Brownian', 'Circular', 'Linear')
        """
        
        # Get robot position if provided
        if robot_base_frame is not None:
            robot_pos = robot_base_frame[:3, 3]
        else:
            robot_pos = np.zeros(3)
            
        # Calculate movement relative to robot if enabled
        if self.follow_robot and robot_base_frame is not None:
            # Move bounds relative to robot position
            current_bounds = self.bound.copy()
            for dim in range(3):
                current_bounds[dim, :] += robot_pos[dim]
            # Shift obstacle by the robot's XY displacement to stay attached to robot
            if self.step_counter == 0:
                self.robot_last_position = robot_pos.copy()
            delta_robot = robot_pos - self.robot_last_position
            # Preserve height (Z) exactly as-is; only follow in XY
            delta_robot[2] = 0.0
            self.frame[:3, 3] = self.frame[:3, 3] + delta_robot
            self.robot_last_position = robot_pos.copy()
        else:
            current_bounds = self.bound
            
        self.last_frame = self.frame.copy()
        
        if mode == "Brownian":
            # Generate new direction periodically
            if self.step_counter % self.keep_direction_step == 0:
                direction = np.random.normal(loc=0.0, size=3)
                direction = self.velocity * direction / (np.linalg.norm(direction) + 1e-8)
            else:
                direction = self.last_direction
                
            # Smooth direction change
            update_step = (1 - self.smooth_weight) * self.last_direction + self.smooth_weight * direction
            new_position = self.frame[:3, 3] + update_step
            
            # Enforce bounds
            for dim in range(3):
                if new_position[dim] < current_bounds[dim, 0]:
                    new_position[dim] = self.last_frame[dim, 3] - update_step[dim]
                elif new_position[dim] > current_bounds[dim, 1]:
                    new_position[dim] = self.last_frame[dim, 3] - update_step[dim]
                    
            # Enforce maximum distance from robot if enabled
            if self.follow_robot and robot_base_frame is not None:
                dist_to_robot = np.linalg.norm(new_position - robot_pos)
                if dist_to_robot > self.max_distance_from_robot:
                    # Pull back towards robot
                    direction_to_robot = (robot_pos - new_position) / (dist_to_robot + 1e-8)
                    new_position = robot_pos - direction_to_robot * (self.max_distance_from_robot * 0.9)
                    
            self.frame[:3, 3] = new_position
            self.last_direction = new_position - self.last_frame[:3, 3]
            
        elif mode == "Circular":
            # Circular motion around robot (if following) or origin
            center = robot_pos if self.follow_robot else np.zeros(3)
            angle = self.step_counter * 0.05  # Adjust speed
            radius = np.random.uniform(0.5, 1.5)
            
            self.frame[0, 3] = center[0] + radius * np.cos(angle)
            self.frame[1, 3] = center[1] + radius * np.sin(angle)
            self.frame[2, 3] = center[2] + current_bounds[2, 0] + \
                             (current_bounds[2, 1] - current_bounds[2, 0]) * 0.5
                             
        self.step_counter += 1


class CombinedBenchmarkTask:
    """
    Enhanced benchmark task for combined walking + obstacle avoidance.
    
    This task:
    1. Manages obstacles that move relative to the walking robot
    2. Provides goals for the upper body to reach
    3. Handles episode completion conditions
    4. Tracks performance metrics
    """
    
    def __init__(self, robot_cfg, robot_kinematics, **kwargs):
        self.robot_cfg = robot_cfg
        self.robot_kinematics = robot_kinematics
        
        # Task configuration
        self.task_name = kwargs.get("task_name", "CombinedWalkingAvoidanceTask")
        self.max_episode_length = kwargs.get("max_episode_length", 2000)
        self.num_obstacles = kwargs.get("num_obstacles", 3)
        
        # Performance tracking
        self.episode_length = 0
        self.min_dist_robot_to_env = []
        self.mean_dist_goal = []
        
        self.reset()
        
    def reset(self):
        """Reset the task state."""
        self.episode_length = 0
        
        # Clear performance tracking
        self.min_dist_robot_to_env = []
        self.mean_dist_goal = []
        
        # Create moving obstacles
        self.obstacle_task = []
        self.obstacle_task_geom = []
        
        for i in range(self.num_obstacles):
            # Create obstacles EXACTLY like original simplified system
            obstacle = MovingTaskObject3D(
                velocity=0.01,  # EXACT same as original
                keep_direction_step=500,  # EXACT same as original  
                bound=np.array([[-0.3, 0.5], [-0.3, 0.5], [0.8, 1.0]]),  # EXACT same bounds as original
                follow_robot=True,
                robot_relative_bounds=True,
                max_distance_from_robot=1.2,  # Keep close to robot
                smooth_weight=0.8
            )
            
            # Initialize position exactly like original (relative to robot bounds)
            obstacle.frame[:3, 3] = np.array([
                np.random.uniform(-0.3, 0.5),
                np.random.uniform(-0.3, 0.5), 
                np.random.uniform(0.8, 1.0)
            ])
            
            self.obstacle_task.append(obstacle)
            # EXACT same radius as original: 0.05 (not 0.08!)
            self.obstacle_task_geom.append(Geometry(type="sphere", radius=0.05, color=VizColor.obstacle_task))
            
        # Create goals EXACTLY like original simplified system
        self.goal_left = MovingTaskObject3D(
            velocity=0.001,  # EXACT same as original
            keep_direction_step=10,  # EXACT same as original
            bound=np.array([[0.1, 0.4], [0.1, 0.4], [0.0, 0.2]]),  # EXACT same bounds as original
            follow_robot=True,
            robot_relative_bounds=True,
            max_distance_from_robot=0.6,  # Keep very close to robot
            smooth_weight=0.8  # EXACT same as original
        )
        
        self.goal_right = MovingTaskObject3D(
            velocity=0.001,  # EXACT same as original
            keep_direction_step=10,  # EXACT same as original  
            bound=np.array([[0.1, 0.4], [-0.4, -0.1], [0.0, 0.2]]),  # EXACT same bounds as original
            follow_robot=True,
            robot_relative_bounds=True,
            max_distance_from_robot=0.6,  # Keep very close to robot
            smooth_weight=0.8  # EXACT same as original
        )
        
        # Initialize goal positions EXACTLY like original (relative to robot bounds)
        self.goal_left.frame[:3, 3] = np.array([
            np.random.uniform(0.1, 0.4),
            np.random.uniform(0.1, 0.4), 
            np.random.uniform(0.8, 1.0)  # Higher height for reachability
        ])
        self.goal_right.frame[:3, 3] = np.array([
            np.random.uniform(0.1, 0.4),
            np.random.uniform(-0.4, -0.1), 
            np.random.uniform(0.8, 1.0)  # Higher height for reachability
        ])
        
        # Info structure
        self.info = {
            "goal_teleop": {},
            "obstacle_task": {},
            "obstacle_debug": {},
            "obstacle": {},
            "robot_frames": None,
            "robot_state": {}
        }
        
    def step(self, feedback):
        """Update the task state."""
        self.episode_length += 1
        
        # Get robot base frame for relative movement
        robot_base_frame = feedback.get("robot_base_frame", np.eye(4))
        
        # Update obstacles with robot-relative movement
        for obstacle in self.obstacle_task:
            obstacle.move(robot_base_frame, mode="Brownian")
            
        # Update goals with robot-relative movement  
        self.goal_left.move(robot_base_frame, mode="Brownian")
        self.goal_right.move(robot_base_frame, mode="Brownian")
        
    def get_info(self, feedback):
        """Get current task information."""
        
        # Episode completion check
        self.info["done"] = False
        if self.max_episode_length >= 0 and self.episode_length >= self.max_episode_length:
            self.info["done"] = True
            
        self.info["episode_length"] = self.episode_length
        self.info["robot_base_frame"] = feedback.get("robot_base_frame", np.eye(4))
        
        # Update goal positions (transform to robot base frame for IK)
        robot_base_frame = self.info["robot_base_frame"] 
        base_to_robot = np.linalg.inv(robot_base_frame)
        
        # Transform goal positions to robot base frame coordinates
        goal_left_global = self.goal_left.frame
        goal_right_global = self.goal_right.frame
        
        goal_left_local = base_to_robot @ goal_left_global
        goal_right_local = base_to_robot @ goal_right_global
        
        # Use STATIC goals like the original system (not moving goals!)
        # The original system has fixed relative positions for goals
        self.goal_teleop_static = {}
        self.goal_teleop_static["left"] = np.array(
            [[1.0, 0.0, 0.0, 0.25], [0.0, 1.0, 0.0, 0.25], [0.0, 0.0, 1.0, 0.1], [0.0, 0.0, 0.0, 1.0]]
        )
        self.goal_teleop_static["right"] = np.array(
            [[1.0, 0.0, 0.0, 0.25], [0.0, 1.0, 0.0, -0.25], [0.0, 0.0, 1.0, 0.1], [0.0, 0.0, 0.0, 1.0]]
        )

        self.info["goal_teleop"]["left"] = self.goal_teleop_static["left"]
        self.info["goal_teleop"]["right"] = self.goal_teleop_static["right"]
        
        # Obstacle information (in global frame)
        obstacle_frames = []
        if len(self.obstacle_task) > 0:
            obstacle_frames = [obstacle.frame for obstacle in self.obstacle_task]
        else:
            obstacle_frames = []
            
        self.info["obstacle_task"]["frames_world"] = np.array(obstacle_frames) if obstacle_frames else np.empty((0, 4, 4))
        self.info["obstacle_task"]["geom"] = self.obstacle_task_geom
        
        # Debug obstacles (empty for now)
        self.info["obstacle_debug"]["frames_world"] = np.empty((0, 4, 4))
        self.info["obstacle_debug"]["geom"] = []
        
        # Combined obstacle info
        if len(obstacle_frames) > 0:
            self.info["obstacle"]["frames_world"] = self.info["obstacle_task"]["frames_world"]
        else:
            self.info["obstacle"]["frames_world"] = np.empty((0, 4, 4))
            
        self.info["obstacle"]["geom"] = self.obstacle_task_geom
        self.info["obstacle"]["num"] = len(obstacle_frames)
        
        # Robot state info
        self.info["robot_state"]["dof_pos_cmd"] = feedback.get("dof_pos_cmd", [])
        self.info["robot_state"]["dof_pos_fbk"] = feedback.get("dof_pos_fbk", [])
        self.info["robot_state"]["dof_vel_cmd"] = feedback.get("dof_vel_cmd", [])
        
        return self.info
        
    def calculate_performance_metrics(self, robot_frames_world, task_info):
        """Calculate performance metrics for logging."""
        
        # Distance to obstacles
        if len(task_info["obstacle"]["frames_world"]) > 0:
            dist_env = compute_pairwise_dist(
                frame_list_1=robot_frames_world,
                geom_list_1=self.robot_cfg.CollisionVol.values(),
                frame_list_2=task_info["obstacle"]["frames_world"],
                geom_list_2=task_info["obstacle"]["geom"],
            )
            self.min_dist_robot_to_env.append(np.min(dist_env) if dist_env.size else float('inf'))
        else:
            self.min_dist_robot_to_env.append(float('inf'))
            
        # Distance to goals  
        robot_base_frame = task_info["robot_base_frame"]
        L_ee = self.robot_cfg.Frames.L_ee
        R_ee = self.robot_cfg.Frames.R_ee
        
        goal_left = (robot_base_frame @ task_info["goal_teleop"]["left"])[:3, 3]
        goal_right = (robot_base_frame @ task_info["goal_teleop"]["right"])[:3, 3]
        
        if len(robot_frames_world) > max(L_ee, R_ee):
            dist_goal_left = np.linalg.norm(robot_frames_world[L_ee, :3, 3] - goal_left)
            dist_goal_right = np.linalg.norm(robot_frames_world[R_ee, :3, 3] - goal_right)
            self.mean_dist_goal.append((dist_goal_left + dist_goal_right) / 2)
        else:
            self.mean_dist_goal.append(0.0)
            
    def get_performance_summary(self):
        """Get summary of performance metrics."""
        summary = {}
        
        if self.min_dist_robot_to_env:
            summary["avg_min_dist_to_obstacles"] = np.mean(self.min_dist_robot_to_env)
            summary["min_dist_to_obstacles"] = np.min(self.min_dist_robot_to_env)
        else:
            summary["avg_min_dist_to_obstacles"] = float('inf')
            summary["min_dist_to_obstacles"] = float('inf')
            
        if self.mean_dist_goal:
            summary["avg_dist_to_goals"] = np.mean(self.mean_dist_goal)
            summary["max_dist_to_goals"] = np.max(self.mean_dist_goal)
        else:
            summary["avg_dist_to_goals"] = 0.0
            summary["max_dist_to_goals"] = 0.0
            
        summary["episode_length"] = self.episode_length
        
        return summary


if __name__ == "__main__":
    # Test the task manager
    from simplified.g1_benchmark_pid.robot_impl import G1BasicConfig, G1BasicKinematics
    
    robot_cfg = G1BasicConfig()
    robot_kinematics = G1BasicKinematics(robot_cfg)
    
    task = CombinedBenchmarkTask(robot_cfg, robot_kinematics, num_obstacles=2)
    
    print("Task manager created successfully!")
    print(f"Number of obstacles: {len(task.obstacle_task)}")
    
    # Test a few steps
    dummy_feedback = {
        "robot_base_frame": np.eye(4),
        "dof_pos_cmd": np.zeros(10),
        "dof_pos_fbk": np.zeros(10),
        "dof_vel_cmd": np.zeros(10)
    }
    
    for i in range(5):
        task.step(dummy_feedback)
        info = task.get_info(dummy_feedback)
        print(f"Step {i}: obstacles={info['obstacle']['num']}, done={info['done']}")
        
        # Move robot base to test relative movement
        dummy_feedback["robot_base_frame"][0, 3] += 0.1  # Move forward
        
    print("Task manager test completed successfully!")
