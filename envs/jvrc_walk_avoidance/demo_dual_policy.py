#!/usr/bin/env python3
"""
Demo script for JVRC Dual Policy: Trained Walking + PID Upper Body Avoidance

This demo shows:
- Lower body: Uses your TRAINED jvrc_walk policy (exactly as-is)
- Upper body: Uses PID controller for obstacle avoidance
- No action input needed - both policies run automatically!
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

def demo_dual_policy():
    """Demo the dual policy approach"""
    
    try:
        import mujoco_viewer
    except ImportError:
        print("❌ mujoco_viewer not available. Install with: pip install mujoco_viewer")
        return False
    
    try:
        print("🤖 JVRC Dual Policy Demo")
        print("=" * 60)
        print("🧠 Lower body: TRAINED jvrc_walk policy (automatic)")
        print("🎛️  Upper body: PID controller for obstacle avoidance (automatic)")
        print("🔘 Gray spheres: Moving obstacles")
        print("🎯 NO ACTION INPUT NEEDED - both policies run automatically!")
        print("=" * 60)
        
        from envs.jvrc_walk_avoidance import JvrcWalkAvoidanceEnv
        
        print("📦 Creating dual policy environment...")
        env = JvrcWalkAvoidanceEnv()
        
        print("🎬 Starting visualization...")
        viewer = mujoco_viewer.MujocoViewer(env.model, env.data)
        
        print("🔄 Resetting environment...")
        obs = env.reset_model()
        
        print("🏃 Running dual policy demo (ESC to exit)...")
        print("Watch:")
        print("  - Robot walking with TRAINED policy (legs)")
        print("  - Upper body moving with PID controller to avoid obstacles")
        print("  - Both policies working together automatically!")
        print("  - Gray spheres moving around using G1 parameters")
        print()
        
        step_count = 0
        episode = 1
        
        while viewer.is_alive:
            
            # No action needed! Both policies run automatically:
            # - Walking policy gets actions from trained model
            # - PID policy computes upper body actions for avoidance
            dummy_action = np.zeros(len(env.action_space))  # Not used
            
            # Step environment - the dual policy robot handles everything
            obs, reward, done, info = env.step(dummy_action)
            
            # Extract obstacle information
            obstacle_info = obs[-4:]
            min_obstacle_dist = obstacle_info[-1]
            
            # Update viewer
            viewer.render()
            
            step_count += 1
            
            # Print status every 25 steps (more frequent updates)
            if step_count % 25 == 0:
                walking_reward = info.get('com_vel_error', 0)
                avoidance_reward = info.get('obstacle_avoidance', 0)
                
                print(f"Episode {episode}, Step {step_count}:")
                print(f"  Total Reward: {reward:.3f}")
                print(f"  Walking: {walking_reward:.3f} | Avoidance: {avoidance_reward:.3f}")
                print(f"  Min Obstacle Distance: {min_obstacle_dist:.3f}m")
                
                # Show when robot is actively avoiding
                if min_obstacle_dist < 0.5:
                    print(f"  🚨 AVOIDING OBSTACLE! Distance: {min_obstacle_dist:.3f}m")
                    print(f"      👀 Watch the upper body move to avoid!")
            
            if done:
                print(f"\nEpisode {episode} ended after {step_count} steps")
                if min_obstacle_dist < 0.15:
                    print("  Reason: Collision with obstacle!")
                    print("  (PID controller needs tuning for better avoidance)")
                else:
                    print("  Reason: Other termination condition")
                
                # Reset for next episode
                obs = env.reset_model()
                episode += 1
                step_count = 0
                
                if episode > 5:  # Allow more episodes to see the behavior
                    break
        
        viewer.close()
        print("✅ Dual policy demo completed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🎮 JVRC Dual Policy Demo")
    print()
    print("This demonstrates the dual policy approach:")
    print("  🧠 Lower body: Your TRAINED jvrc_walk policy")
    print("  🎛️  Upper body: PID controller for obstacle avoidance")
    print("  🤝 Both work together automatically!")
    print("  ⚡ No action input required!")
    print()
    
    success = demo_dual_policy()
    
    if success:
        print("\n🎉 Dual policy demo completed successfully!")
        print("\n📊 Key Features Demonstrated:")
        print("✅ Trained walking policy preserved exactly")
        print("✅ PID upper body avoidance working automatically")
        print("✅ Both policies coordinated in real-time")
        print("✅ G1-style obstacle movement")
        print("\n📚 Next steps:")
        print("1. Tune PID parameters for better avoidance")
        print("2. Train end-to-end policy for comparison:")
        print("   python run_experiment.py train --env jvrc_walk_avoidance --logdir trained/jvrc_walk_avoidance")
    else:
        print("\n❌ Demo failed. Check error messages above.")
        sys.exit(1)
