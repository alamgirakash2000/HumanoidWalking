# Combined Walking + Obstacle Avoidance System

This system combines two previously separate behaviors for the G1 humanoid robot:

1. **Walking Behavior** (Lower body - 12 leg joints)
   - Uses trained PPO policy from `trained/g1_walk/`
   - Enables the robot to walk forward while maintaining balance
   - Controls hip, knee, and ankle joints for both legs

2. **Obstacle Avoidance Behavior** (Upper body - 17 joints: 3 waist + 14 arm)
   - Uses PID control + safety constraints
   - Enables the robot to avoid moving spherical obstacles
   - Reaches for target goals with both hands
   - Maintains safe distances using collision detection

## Key Features

- **Seamless Integration**: Both behaviors run simultaneously without interference
- **Dynamic Obstacles**: Obstacles move relative to the walking robot's position
- **Safety Constraints**: Upper body movements are constrained to avoid collisions
- **Real-time Visualization**: MuJoCo viewer shows the robot, obstacles, and goals
- **Performance Metrics**: Tracks distances to obstacles and goals, safety violations

## Architecture

### Joint Separation
- **Leg Joints (12)**: Controlled by walking policy
  - Left/right hip yaw/roll/pitch, knee, ankle pitch/roll
- **Upper Body Joints (17)**: Controlled by obstacle avoidance system
  - Waist yaw/roll/pitch + left/right arm (7 joints each)

### Policy Integration
- **Walking Policy**: Trained neural network (PPO) loaded from `trained/g1_walk/actor.pt`
- **Upper Body Policy**: PID controller with inverse kinematics for goal reaching
- **Safety Controller**: Real-time collision avoidance using safety set algorithms

### Task Management
- **Moving Obstacles**: 3D spherical obstacles that move with Brownian motion relative to robot
- **Dynamic Goals**: Target positions for left and right hands that move slowly
- **Episode Management**: Tracks performance and resets when needed

## Usage

### Basic Usage
```bash
# Run with default settings (5 episodes, visualization enabled)
python run_combined.py

# Run more episodes  
python run_combined.py --episodes 10

# Run without visualization (faster)
python run_combined.py --no-viz

# Custom policy path
python run_combined.py --policy-path /path/to/actor.pt
```

### Advanced Usage
```python
# Use as a Python module
from combined_walker_avoidance import G1CombinedEnv

# Create environment
env = G1CombinedEnv(walking_policy_path="trained/g1_walk/actor.pt")

# Run simulation loop
obs = env.reset()
for step in range(1000):
    obs, reward, done, info = env.step()
    if done:
        obs = env.reset()
        
env.close()
```

## System Requirements

- Python 3.8+
- MuJoCo 2.3+
- PyTorch (for walking policy)
- NumPy, SciPy (for control algorithms)
- All dependencies from the original HumanoidWalking project

## Performance Metrics

The system tracks several performance metrics:

- **Safety Metrics**:
  - Minimum distance to obstacles
  - Safety controller trigger frequency
  - Collision avoidance success rate

- **Task Metrics**:
  - Distance to target goals
  - Goal reaching success rate
  - Episode completion time

- **Walking Metrics**:
  - Forward walking speed
  - Balance maintenance
  - Step regularity

## Technical Details

### Robot Model
- Uses unified G1 MuJoCo model with all 29 actuated joints
- Maintains collision geometry for safety constraints
- Proper foot contact detection for walking

### Control Loop
1. Get walking policy observation (robot state + task info)
2. Compute walking policy action (12 leg joint targets)
3. Get upper body observation (joint states + obstacle positions)
4. Compute upper body reference action (PID for goal reaching)
5. Apply safety filtering (collision avoidance constraints)
6. Combine and apply all joint commands
7. Step simulation and update task state

### Coordinate Frames
- **World Frame**: Fixed reference for obstacles and goals
- **Robot Base Frame**: Moving with the walking robot
- **Joint Space**: Individual joint angles and velocities

## Troubleshooting

### Common Issues

1. **Walking Policy Not Found**
   - Ensure `trained/g1_walk/actor.pt` exists
   - Check that the walking policy was trained successfully

2. **MuJoCo Model Issues**
   - The system automatically generates the combined model in `/tmp/mjcf-export/g1_combined/`
   - If issues persist, delete this directory and restart

3. **Visualization Problems**
   - Use `--no-viz` flag to run without visualization
   - Ensure proper MuJoCo viewer setup

4. **Performance Issues**
   - The system is computationally intensive due to:
     - Neural network policy inference
     - Real-time collision detection
     - Safety optimization (QP solver)
   - Consider reducing episode length or obstacle count

### Expected Behavior

A successful run should show:
- Robot walking forward with natural gait
- Upper body actively avoiding moving obstacles
- Hands reaching toward target goals
- No collisions between robot and obstacles
- Stable walking despite upper body movements

## Implementation Files

- `robot_config.py`: Combined robot configuration and model generation
- `combined_env.py`: Main environment integrating both systems
- `task_manager.py`: Obstacle and goal management for walking robot
- `runner.py`: Simulation runner with visualization and logging
- `README.md`: This documentation

## Future Enhancements

Possible improvements to the system:
- Adaptive walking speed based on obstacle density
- Learning-based upper body policy (instead of PID)
- More complex obstacle shapes and movements
- Integration with vision-based obstacle detection
- Multi-robot scenarios

