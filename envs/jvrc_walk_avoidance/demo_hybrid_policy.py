#!/usr/bin/env python3
"""
Demo script that uses the trained jvrc_walk policy for legs + simple upper body avoidance
"""

import numpy as np
import torch
import pickle
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

def load_jvrc_walk_policy():
    """Load the trained jvrc_walk policy"""
    try:
        policy_path = "/home/akash/Downloads/HumanoidWalking/trained/jvrc_walk/actor.pt"
        metadata_path = "/home/akash/Downloads/HumanoidWalking/trained/jvrc_walk/training_metadata.pkl"
        
        # Load policy (with weights_only=False for compatibility)
        policy = torch.load(policy_path, map_location='cpu', weights_only=False)
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
            
        print(f"✅ Loaded trained jvrc_walk policy")
        print(f"   Policy parameters: {sum(p.numel() for p in policy.parameters())}")
        
        return policy, metadata
        
    except Exception as e:
        print(f"❌ Could not load jvrc_walk policy: {e}")
        return None, None

def demo_hybrid_policy():
    """Demo with hybrid policy: trained walking + simple upper body avoidance"""
    
    try:
        import mujoco_viewer
    except ImportError:
        print("❌ mujoco_viewer not available. Install with: pip install mujoco_viewer")
        return False
    
    try:
        print("🤖 JVRC Hybrid Policy Demo")
        print("=" * 60)
        print("🦵 Lower body: TRAINED jvrc_walk policy (reusing your training!)")
        print("💪 Upper body: Simple reactive avoidance")
        print("🔘 Gray spheres: Moving obstacles")
        print("=" * 60)
        
        # Load trained walking policy
        walking_policy, metadata = load_jvrc_walk_policy()
        if walking_policy is None:
            return False
        
        # Import environments
        from envs.jvrc import JvrcWalkEnv
        from envs.jvrc_walk_avoidance import JvrcWalkAvoidanceEnv
        
        print("📦 Creating environments...")
        # Create original jvrc_walk environment to get walking actions
        walk_env = JvrcWalkEnv()
        
        # Create new walk+avoidance environment for demo
        demo_env = JvrcWalkAvoidanceEnv()
        
        print("🎬 Starting visualization...")
        viewer = mujoco_viewer.MujocoViewer(demo_env.model, demo_env.data)
        
        print("🔄 Resetting environments...")
        walk_obs = walk_env.reset_model()
        demo_obs = demo_env.reset_model()
        
        print("🏃 Running hybrid demo (ESC to exit)...")
        print("Watch:")
        print("  - Robot walking with TRAINED policy (legs)")
        print("  - Upper body avoiding obstacles reactively")
        print("  - Gray spheres moving around")
        print()
        
        step_count = 0
        episode = 1
        
        while viewer.is_alive:
            
            # Get walking action from trained policy (12 leg joints)
            with torch.no_grad():
                walk_obs_tensor = torch.FloatTensor(walk_obs).unsqueeze(0)
                leg_action = walking_policy(walk_obs_tensor).cpu().numpy().flatten()
            
            # Get robot and obstacle positions for upper body avoidance
            robot_pos = demo_env.interface.get_body_pos('PELVIS_S')
            
            # Simple upper body avoidance (20 joints)
            upper_body_action = np.zeros(20)  # Start with neutral pose
            
            # Check obstacles and react
            for obstacle in demo_env.task._obstacle_task:
                obstacle_pos = obstacle.frame[:3, 3]
                obs_to_robot = robot_pos - obstacle_pos
                distance = np.linalg.norm(obs_to_robot)
                
                if distance < 0.4 and distance > 0.01:  # React when close
                    # Calculate avoidance direction
                    avoidance_dir = obs_to_robot / distance
                    avoidance_strength = (0.4 - distance) / 0.4
                    avoidance_strength = np.clip(avoidance_strength, 0, 1)
                    
                    # Waist lean away (indices 0-2: WAIST_Y, WAIST_P, WAIST_R)
                    if avoidance_dir[1] > 0:  # Obstacle to left, lean right
                        upper_body_action[2] -= 0.3 * avoidance_strength  # WAIST_R
                    else:  # Obstacle to right, lean left
                        upper_body_action[2] += 0.3 * avoidance_strength
                    
                    if avoidance_dir[0] > 0:  # Obstacle behind, lean forward
                        upper_body_action[1] += 0.2 * avoidance_strength  # WAIST_P
                    else:  # Obstacle in front, lean back
                        upper_body_action[1] -= 0.2 * avoidance_strength
                    
                    # Arms move away (indices 6-12: R_ARM, 13-19: L_ARM)
                    if avoidance_dir[1] > 0:  # Move left arm up
                        upper_body_action[13] -= 0.4 * avoidance_strength  # L_SHOULDER_P
                        upper_body_action[15] += 0.3 * avoidance_strength  # L_SHOULDER_Y
                    else:  # Move right arm up  
                        upper_body_action[6] -= 0.4 * avoidance_strength   # R_SHOULDER_P
                        upper_body_action[8] -= 0.3 * avoidance_strength   # R_SHOULDER_Y
            
            # Combine leg actions (trained) + upper body actions (reactive)
            combined_action = np.concatenate([leg_action, upper_body_action])
            
            # Step both environments (keep them synchronized)
            walk_obs, _, walk_done, _ = walk_env.step(leg_action)
            demo_obs, reward, demo_done, info = demo_env.step(combined_action)
            
            # Extract obstacle information
            obstacle_info = demo_obs[-4:]
            min_obstacle_dist = obstacle_info[-1]
            
            # Update viewer
            viewer.render()
            
            step_count += 1
            
            # Print status every 50 steps
            if step_count % 50 == 0:
                print(f"Episode {episode}, Step {step_count}: "
                     f"Reward={reward:.3f}, Min obstacle dist={min_obstacle_dist:.3f}m")
            
            # Reset if either environment is done
            if demo_done or walk_done:
                print(f"Episode {episode} ended after {step_count} steps")
                if min_obstacle_dist < 0.15:
                    print("  Reason: Collision with obstacle!")
                elif walk_done:
                    print("  Reason: Walking task terminated")
                else:
                    print("  Reason: Other condition")
                
                # Reset both environments
                walk_obs = walk_env.reset_model()
                demo_obs = demo_env.reset_model()
                episode += 1
                step_count = 0
                
                if episode > 5:  # Limit to 5 episodes for demo
                    break
        
        viewer.close()
        print("✅ Hybrid demo completed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🎮 JVRC Hybrid Policy Demo")
    print()
    print("This demo uses:")
    print("  🧠 TRAINED jvrc_walk policy for legs (reusing your training!)")
    print("  💪 Simple reactive policy for upper body obstacle avoidance")
    print("  🎯 Best of both: stable walking + reactive avoidance")
    print()
    
    success = demo_hybrid_policy()
    
    if success:
        print("\n🎉 Hybrid demo completed successfully!")
        print("\n📚 You can also try:")
        print("1. Pure trained walking:")
        print("   python run_experiment.py eval --path trained/jvrc_walk")
        print("\n2. Train combined policy:")
        print("   python run_experiment.py train --env jvrc_walk_avoidance --logdir trained/jvrc_walk_avoidance")
    else:
        print("\n❌ Demo failed. Check error messages above.")
        sys.exit(1)
