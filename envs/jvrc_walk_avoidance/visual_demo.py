#!/usr/bin/env python3
"""
VISUAL DEMO: Watch JVRC robot walk and avoid obstacles with upper body
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

def visual_demo():
    """Visual demo showing robot walking and avoiding obstacles"""
    
    try:
        import mujoco_viewer
    except ImportError:
        print("❌ mujoco_viewer not available. Install with: pip install mujoco_viewer")
        return False
    
    try:
        print("👀 VISUAL DEMO: JVRC Walking + Upper Body Obstacle Avoidance")
        print("=" * 70)
        print("🦵 LOWER BODY: Trained walking policy")
        print("💪 UPPER BODY: PID obstacle avoidance") 
        print("🔘 GRAY SPHERES: Moving obstacles")
        print("👀 WATCH: Upper body should move when obstacles get close!")
        print("=" * 70)
        
        from envs.jvrc_walk_avoidance import JvrcWalkAvoidanceEnv
        
        print("📦 Creating environment...")
        env = JvrcWalkAvoidanceEnv()
        
        print("🎬 Starting MuJoCo viewer...")
        viewer = mujoco_viewer.MujocoViewer(env.model, env.data)
        
        print("🔄 Resetting environment...")
        obs = env.reset_model()
        
        print("🏃 Starting visual demo...")
        print("   Press ESC to exit")
        print("   Watch for upper body movements when obstacles approach!")
        print()
        
        step_count = 0
        total_steps = 0
        episode = 1
        
        # Make sure obstacles start at reasonable positions
        robot_pos = env.interface.get_body_pos('PELVIS_S')
        print(f"Robot starting position: {robot_pos}")
        
        # Print initial obstacle positions
        for i, obstacle in enumerate(env.task._obstacle_task):
            obs_pos = obstacle.frame[:3, 3]
            dist = np.linalg.norm(robot_pos - obs_pos)
            print(f"Obstacle {i+1} at: {obs_pos}, distance: {dist:.3f}m")
        print()
        
        while viewer.is_alive:
            
            # Dummy action (not used - both policies are automatic)
            dummy_action = np.zeros(len(env.action_space))
            
            # Step the environment
            obs, reward, done, info = env.step(dummy_action)
            
            # Get obstacle info
            obstacle_info = obs[-4:]
            min_obstacle_dist = obstacle_info[-1]
            
            # Update viewer
            viewer.render()
            
            step_count += 1
            total_steps += 1
            
            # Print status every 50 steps
            if step_count % 50 == 0:
                print(f"Episode {episode}, Step {step_count}:")
                print(f"  Reward: {reward:.3f}")
                print(f"  Min obstacle distance: {min_obstacle_dist:.3f}m")
                
                # Highlight when avoiding
                if min_obstacle_dist < 0.4:
                    print(f"  🚨 OBSTACLE NEARBY! Watch upper body avoid!")
                    
                    # Show individual obstacle distances
                    robot_pos = env.interface.get_body_pos('PELVIS_S')
                    for i, obstacle in enumerate(env.task._obstacle_task):
                        obs_pos = obstacle.frame[:3, 3]
                        dist = np.linalg.norm(robot_pos - obs_pos)
                        print(f"     Obstacle {i+1}: {dist:.3f}m")
                
                print()
            
            if done:
                print(f"Episode {episode} ended after {step_count} steps")
                
                if min_obstacle_dist < 0.15:
                    print("  Reason: Collision detected")
                else:
                    print("  Reason: Other condition")
                
                # Reset for new episode
                print("  Resetting for next episode...")
                obs = env.reset_model()
                episode += 1
                step_count = 0
                
                # Print new obstacle positions
                robot_pos = env.interface.get_body_pos('PELVIS_S')
                print(f"  New robot position: {robot_pos}")
                for i, obstacle in enumerate(env.task._obstacle_task):
                    obs_pos = obstacle.frame[:3, 3]
                    dist = np.linalg.norm(robot_pos - obs_pos)
                    print(f"  New obstacle {i+1}: {dist:.3f}m")
                print()
                
                # Stop after 5 episodes for demo
                if episode > 5:
                    print("Demo completed after 5 episodes")
                    break
        
        viewer.close()
        print(f"✅ Visual demo completed! Total steps: {total_steps}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("👀 VISUAL DEMO")
    print("This will show you the robot walking and avoiding obstacles!")
    print("- Lower body uses your trained walking policy")
    print("- Upper body uses PID to avoid gray sphere obstacles")
    print("- Watch for upper body movements when obstacles get close!")
    print()
    
    success = visual_demo()
    
    if success:
        print("\n✅ Visual demo successful!")
        print("Did you see the upper body moving to avoid obstacles?")
    else:
        print("\n❌ Visual demo failed.")
        sys.exit(1)

