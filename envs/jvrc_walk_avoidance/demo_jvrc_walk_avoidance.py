#!/usr/bin/env python3
"""
Demo script for JVRC Walk + Upper Body Obstacle Avoidance

Shows the robot trying to walk while avoiding moving G1-style obstacles with its upper body.
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

def demo_walk_avoidance():
    """Demo the jvrc_walk_avoidance environment with visualization"""
    
    try:
        import mujoco_viewer
    except ImportError:
        print("❌ mujoco_viewer not available. Install with: pip install mujoco_viewer")
        return False
    
    try:
        print("🤖 JVRC Walk + Upper Body Obstacle Avoidance Demo")
        print("=" * 60)
        print("🦵 Lower body: Walking (continues jvrc_walk behavior)")
        print("💪 Upper body: Avoids G1-style moving obstacles")
        print("🔘 Gray spheres: Moving obstacles (exact G1 parameters)")
        print("=" * 60)
        
        from envs.jvrc_walk_avoidance import JvrcWalkAvoidanceEnv
        
        print("📦 Creating environment...")
        env = JvrcWalkAvoidanceEnv()
        
        print("🎬 Starting visualization...")
        viewer = mujoco_viewer.MujocoViewer(env.model, env.data)
        
        print("🔄 Resetting environment...")
        obs = env.reset_model()
        
        print("🏃 Running demo (ESC to exit)...")
        print("Watch the robot:")
        print("  - Trying to walk forward")
        print("  - Moving obstacles around the robot")
        print("  - Upper body reacting to avoid obstacles")
        print()
        
        step_count = 0
        episode = 1
        
        while viewer.is_alive:
            # Use random policy for demo (normally would use trained policy)
            action = np.random.uniform(-0.1, 0.1, size=len(env.action_space))
            
            # Add some structure to the action for better walking
            if step_count % 100 < 50:  # First half of cycle
                action[:6] *= 0.5   # Reduce right leg movement
                action[6:12] *= 1.5 # Increase left leg movement  
            else:  # Second half of cycle
                action[:6] *= 1.5   # Increase right leg movement
                action[6:12] *= 0.5 # Reduce left leg movement
            
            # Step environment
            obs, reward, done, info = env.step(action)
            
            # Extract obstacle information
            obstacle_info = obs[-4:]
            min_obstacle_dist = obstacle_info[-1]
            
            # Update viewer
            viewer.render()
            
            step_count += 1
            
            # Print status every 100 steps
            if step_count % 100 == 0:
                print(f"Episode {episode}, Step {step_count}: "
                     f"Reward={reward:.3f}, Min obstacle dist={min_obstacle_dist:.3f}m")
            
            if done:
                print(f"Episode {episode} ended after {step_count} steps")
                if min_obstacle_dist < 0.15:
                    print("  Reason: Collision with obstacle!")
                else:
                    print("  Reason: Other termination condition")
                
                # Reset for next episode
                obs = env.reset_model()
                episode += 1
                step_count = 0
                
                if episode > 10:  # Limit demo to 10 episodes
                    break
        
        viewer.close()
        print("✅ Demo completed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🎮 JVRC Walk + Upper Body Obstacle Avoidance Demo")
    print()
    print("This demo shows:")
    print("  🚶 Walking: JVRC robot walks forward using lower body")
    print("  🔄 Obstacles: G1-style gray spheres move around randomly")
    print("  💪 Avoidance: Upper body reacts to avoid obstacles")
    print("  🎯 Integration: Both tasks work together")
    print()
    
    success = demo_walk_avoidance()
    
    if success:
        print("\n🎉 Demo completed successfully!")
        print("\n📚 Next steps:")
        print("1. Train a policy:")
        print("   python run_experiment.py train --env jvrc_walk_avoidance --logdir trained/jvrc_walk_avoidance")
        print("\n2. Evaluate trained policy:")
        print("   python run_experiment.py eval --path trained/jvrc_walk_avoidance")
    else:
        print("\n❌ Demo failed. Check error messages above.")
        sys.exit(1)
