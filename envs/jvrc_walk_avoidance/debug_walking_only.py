#!/usr/bin/env python3
"""
Debug script to test walking policy only (no upper body movements)
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

def debug_walking_only():
    """Test just the walking part to see if it works"""
    
    try:
        import mujoco_viewer
    except ImportError:
        print("❌ mujoco_viewer not available. Install with: pip install mujoco_viewer")
        return False
    
    try:
        print("🐛 DEBUG: Testing Walking Policy Only")
        print("=" * 50)
        
        from envs.jvrc import JvrcWalkEnv
        
        print("📦 Creating original jvrc_walk environment...")
        env = JvrcWalkEnv()
        
        print("🎬 Starting visualization...")
        viewer = mujoco_viewer.MujocoViewer(env.model, env.data)
        
        print("🔄 Resetting environment...")
        obs = env.reset_model()
        
        print("🏃 Running original walking demo...")
        
        step_count = 0
        
        while viewer.is_alive and step_count < 200:  # Limit steps for debug
            
            # Use random actions like other demos
            action = np.random.uniform(-0.1, 0.1, size=len(env.action_space))
            
            # Step environment
            obs, reward, done, info = env.step(action)
            
            # Update viewer
            viewer.render()
            
            step_count += 1
            
            # Print status every 50 steps
            if step_count % 50 == 0:
                print(f"Step {step_count}: Reward={reward:.3f}, Done={done}")
            
            if done:
                print(f"Episode ended after {step_count} steps")
                break
        
        viewer.close()
        print("✅ Walking debug completed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during debug: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🐛 DEBUG: Testing original walking environment")
    print("This helps us understand if the base walking works")
    print()
    
    success = debug_walking_only()
    
    if success:
        print("\n✅ Walking debug successful!")
    else:
        print("\n❌ Walking debug failed.")
        sys.exit(1)

