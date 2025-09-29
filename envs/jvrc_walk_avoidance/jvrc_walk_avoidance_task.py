import numpy as np
import transforms3d as tf3
from tasks import rewards
from tasks.walking_task import WalkingTask

class TaskObject3D():
    """EXACT copy of G1 benchmark TaskObject3D class for obstacle movement"""
    def __init__(self, **kwargs):
        self.frame = kwargs.get("frame", np.eye(4))
        self.velocity = kwargs.get("velocity", 1.0)
        self.bound = kwargs.get("bound", np.zeros((3,2)))
        self.smooth_weight = kwargs.get("smooth_weight", 1.0)
        self.last_direction = np.zeros(3)
        self.step_counter = 0
        self.keep_direction_step = kwargs.get("keep_direction_step", 1)
    
    def move(self, mode="Brownian"):
        if mode == "Brownian":
            if self.step_counter % self.keep_direction_step == 0:
                direction = np.random.normal(loc=0.0, size=3)
                direction = self.velocity * direction / np.linalg.norm(direction)
            else:
                direction = self.last_direction
            self.last_frame = self.frame.copy()
            update_step = (1 - self.smooth_weight) * self.last_direction +  self.smooth_weight * direction
            self.frame[:3,3] += update_step
            
            for dim in range(3):
                if self.frame[dim,3] < self.bound[dim][0]:
                    self.frame[dim,3] = self.last_frame[dim,3] - update_step[dim]
                elif self.frame[dim,3] > self.bound[dim][1]:
                    self.frame[dim,3] = self.last_frame[dim,3] - update_step[dim]
            
            self.last_direction = self.frame[:3,3] - self.last_frame[:3,3]

        self.step_counter += 1


class JvrcWalkAvoidanceTask(WalkingTask):
    """Walking task with upper body obstacle avoidance.
    
    Lower body: Continues normal walking using WalkingTask
    Upper body: Avoids G1-style moving obstacles
    """

    def __init__(self,
                 client=None,
                 dt=0.025,
                 neutral_foot_orient=[],
                 root_body='pelvis',
                 lfoot_body='lfoot',
                 rfoot_body='rfoot',
                 waist_r_joint='waist_r',
                 waist_p_joint='waist_p',
                 num_obstacles=3):

        # Initialize base walking task
        super().__init__(client, dt, neutral_foot_orient, root_body, lfoot_body, rfoot_body, waist_r_joint, waist_p_joint)
        
        self._num_obstacles = num_obstacles
        self._obstacle_task = []
        
        # EXACT G1 benchmark obstacle parameters
        self._init_obstacles()
        
        # Upper body avoidance parameters
        self._avoidance_distance = 0.3  # Distance to start avoiding
        self._collision_distance = 0.15  # Collision threshold
        
        # Upper body target positions for avoidance
        self._upper_body_targets = None
        
    def _init_obstacles(self):
        """Initialize obstacles with EXACT G1 benchmark parameters but MORE AGGRESSIVE"""
        self._obstacle_task = []
        for _ in range(self._num_obstacles):
            # Make obstacles MORE aggressive than G1 for better visibility
            obstacle = TaskObject3D(velocity=0.03,  # 3x faster than G1 (was 0.01)
                                  keep_direction_step=200,  # Change direction more often (was 500)
                                  bound=np.array([[-0.4, 0.6],   # Larger bounds around robot
                                                 [-0.4, 0.6], 
                                                 [0.7, 1.3]]))
            # Initialize position closer to robot for immediate interaction
            obstacle.frame[:3,3] = np.array([
                np.random.uniform(-0.2, 0.4),  # Start closer to robot
                np.random.uniform(-0.2, 0.4),
                np.random.uniform(0.8, 1.2)
            ])
            self._obstacle_task.append(obstacle)

    def calc_reward(self, prev_torque, prev_action, action):
        """Calculate reward: walking rewards + obstacle avoidance rewards"""
        
        # Get base walking rewards
        walking_rewards = super().calc_reward(prev_torque, prev_action, action)
        
        # Calculate obstacle avoidance rewards
        avoidance_rewards = self._calc_obstacle_avoidance_reward()
        
        # Combine rewards
        combined_rewards = walking_rewards.copy()
        combined_rewards.update(avoidance_rewards)
        
        return combined_rewards
    
    def _calc_obstacle_avoidance_reward(self):
        """Calculate rewards for upper body obstacle avoidance"""
        
        # Get robot head/torso position
        head_pos = self._client.get_body_pos(self._root_body_name)
        
        # Calculate distances to all obstacles
        obstacle_distances = []
        for obstacle in self._obstacle_task:
            obstacle_pos = obstacle.frame[:3, 3]
            dist = np.linalg.norm(head_pos - obstacle_pos)
            obstacle_distances.append(dist)
        
        min_obstacle_distance = min(obstacle_distances) if obstacle_distances else 1.0
        
        # Obstacle avoidance reward
        if min_obstacle_distance < self._collision_distance:
            obstacle_reward = -10.0  # Large penalty for collision
        elif min_obstacle_distance < self._avoidance_distance:
            # Exponential penalty as we get closer
            obstacle_reward = -3.0 * np.exp(-5.0 * (min_obstacle_distance - self._collision_distance))
        else:
            obstacle_reward = 0.5  # Small reward for maintaining safe distance
        
        return {
            'obstacle_avoidance': 0.3 * obstacle_reward,  # Weight obstacle avoidance
        }

    def _update_obstacle_positions(self):
        """Update obstacle positions using EXACT G1 movement pattern"""
        for i, obstacle in enumerate(self._obstacle_task):
            obstacle.move(mode="Brownian")  # EXACT G1 movement mode
            # Update MuJoCo body position
            new_pos = obstacle.frame[:3, 3]
            self._client.set_body_pos(f'moving_obstacle_{i}', new_pos)

    def _calculate_upper_body_targets(self):
        """Calculate upper body joint targets for obstacle avoidance"""
        
        # Get current robot position
        robot_pos = self._client.get_body_pos(self._root_body_name)
        
        # Default neutral upper body pose (arms at sides, upright torso)
        neutral_targets = {
            'WAIST_Y': 0.0,
            'WAIST_P': 0.0, 
            'WAIST_R': 0.0,
            'NECK_Y': 0.0,
            'NECK_R': 0.0,
            'NECK_P': 0.0,
            'R_SHOULDER_P': -0.3, 'R_SHOULDER_R': 0.0, 'R_SHOULDER_Y': 0.0,
            'R_ELBOW_P': -0.5, 'R_ELBOW_Y': 0.0, 'R_WRIST_R': 0.0, 'R_WRIST_Y': 0.0,
            'L_SHOULDER_P': -0.3, 'L_SHOULDER_R': 0.0, 'L_SHOULDER_Y': 0.0,
            'L_ELBOW_P': -0.5, 'L_ELBOW_Y': 0.0, 'L_WRIST_R': 0.0, 'L_WRIST_Y': 0.0,
        }
        
        # Calculate avoidance adjustments
        avoidance_adjustment = {key: 0.0 for key in neutral_targets.keys()}
        
        for obstacle in self._obstacle_task:
            obstacle_pos = obstacle.frame[:3, 3]
            obs_to_robot = robot_pos - obstacle_pos
            distance = np.linalg.norm(obs_to_robot)
            
            if distance < self._avoidance_distance and distance > 0.01:
                # Calculate avoidance direction
                avoidance_dir = obs_to_robot / distance
                
                # Avoidance strength (stronger when closer)
                avoidance_strength = (self._avoidance_distance - distance) / self._avoidance_distance
                avoidance_strength = np.clip(avoidance_strength, 0, 1)
                
                # Apply avoidance to upper body joints
                # Lean away from obstacle
                if avoidance_dir[1] > 0:  # Obstacle to the left, lean right
                    avoidance_adjustment['WAIST_R'] -= 0.3 * avoidance_strength
                    avoidance_adjustment['L_SHOULDER_R'] += 0.5 * avoidance_strength
                else:  # Obstacle to the right, lean left  
                    avoidance_adjustment['WAIST_R'] += 0.3 * avoidance_strength
                    avoidance_adjustment['R_SHOULDER_R'] -= 0.5 * avoidance_strength
                
                # Bend away in forward/backward direction
                if avoidance_dir[0] > 0:  # Obstacle behind, lean forward
                    avoidance_adjustment['WAIST_P'] += 0.2 * avoidance_strength
                else:  # Obstacle in front, lean back
                    avoidance_adjustment['WAIST_P'] -= 0.2 * avoidance_strength
        
        # Combine neutral pose with avoidance adjustments
        final_targets = {}
        for joint, neutral_val in neutral_targets.items():
            final_targets[joint] = neutral_val + avoidance_adjustment[joint]
        
        return final_targets

    def step(self):
        """Step the task: walking + obstacle avoidance"""
        
        # Step the base walking task
        super().step()
        
        # Update obstacles
        self._update_obstacle_positions()
        
        # Calculate upper body targets for avoidance
        self._upper_body_targets = self._calculate_upper_body_targets()

    def done(self):
        """Check termination: only collision check, NOT walking termination"""
        
        # DON'T check base walking termination - let it continue walking!
        # walking_done = super().done()
        # if walking_done:
        #     return True
        
        # Only check for obstacle collisions
        robot_pos = self._client.get_body_pos(self._root_body_name)
        for obstacle in self._obstacle_task:
            obstacle_pos = obstacle.frame[:3, 3]
            distance = np.linalg.norm(robot_pos - obstacle_pos)
            if distance < self._collision_distance:
                return True  # Collision detected
        
        return False

    def reset(self, iter_count=0):
        """Reset task: walking + obstacles"""
        
        # Reset base walking task
        super().reset(iter_count)
        
        # Reset obstacles to new random positions
        self._init_obstacles()
        
        # Update MuJoCo obstacle positions
        for i, obstacle in enumerate(self._obstacle_task):
            self._client.set_body_pos(f'moving_obstacle_{i}', obstacle.frame[:3, 3])
        
        # Reset upper body targets
        self._upper_body_targets = None
