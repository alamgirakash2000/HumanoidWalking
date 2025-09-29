#!/usr/bin/env python3
"""
Step-by-step debug to find what's causing immediate termination
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

def debug_environments():
    """Compare original vs dual policy environments"""
    
    print("🐛 SYSTEMATIC DEBUG")
    print("=" * 60)
    
    try:
        # Test 1: Original walking environment
        print("\n1️⃣ Testing Original JvrcWalkEnv...")
        from envs.jvrc import JvrcWalkEnv
        walk_env = JvrcWalkEnv()
        
        print(f"   Action space: {len(walk_env.action_space)}")
        print(f"   Observation space: {len(walk_env.observation_space)}")
        
        obs = walk_env.reset_model()
        print(f"   Initial obs shape: {obs.shape}")
        
        # Try one step
        action = np.zeros(len(walk_env.action_space))
        obs, reward, done, info = walk_env.step(action)
        print(f"   After 1 step: reward={reward:.3f}, done={done}")
        
        if not done:
            # Try 10 more steps
            for i in range(10):
                action = np.random.uniform(-0.05, 0.05, size=len(walk_env.action_space))
                obs, reward, done, info = walk_env.step(action)
                if done:
                    print(f"   Original env ended at step {i+2}")
                    break
            else:
                print(f"   Original env survived 11 steps ✅")
        
        # Test 2: Dual policy environment
        print("\n2️⃣ Testing JvrcWalkAvoidanceEnv...")
        from envs.jvrc_walk_avoidance import JvrcWalkAvoidanceEnv
        dual_env = JvrcWalkAvoidanceEnv()
        
        print(f"   Action space: {len(dual_env.action_space)}")
        print(f"   Observation space: {len(dual_env.observation_space)}")
        
        obs = dual_env.reset_model()
        print(f"   Initial obs shape: {obs.shape}")
        
        # Try one step
        action = np.zeros(len(dual_env.action_space))
        obs, reward, done, info = dual_env.step(action)
        print(f"   After 1 step: reward={reward:.3f}, done={done}")
        
        if done:
            print("   ❌ Dual env ended immediately!")
            print(f"   Reward breakdown: {info}")
            
            # Check termination reasons
            print("\n🔍 Checking termination reasons...")
            
            # Check task done condition
            task_done = dual_env.task.done()
            print(f"   Task done: {task_done}")
            
            # Check robot state
            qpos = dual_env.interface.get_qpos()
            print(f"   Robot height: {qpos[2]:.3f}")
            print(f"   Robot orientation: {qpos[3:7]}")
            
            # Check obstacles
            robot_pos = dual_env.interface.get_body_pos('PELVIS_S')
            print(f"   Robot position: {robot_pos}")
            
            min_dist = float('inf')
            for i, obs_obj in enumerate(dual_env.task._obstacle_task):
                obs_pos = obs_obj.frame[:3, 3]
                dist = np.linalg.norm(robot_pos - obs_pos)
                min_dist = min(min_dist, dist)
                print(f"   Obstacle {i} distance: {dist:.3f}m")
            
            print(f"   Min obstacle distance: {min_dist:.3f}m")
            print(f"   Collision threshold: {dual_env.task._collision_distance}")
            
        else:
            print("   ✅ Dual env survived first step!")
        
        print("\n📊 SUMMARY:")
        print(f"   Original env: {len(walk_env.action_space)} actions, works ✅")
        print(f"   Dual env: {len(dual_env.action_space)} actions, {'fails ❌' if done else 'works ✅'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during debug: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_environments()
    
    if not success:
        sys.exit(1)

