#!/usr/bin/env python3
"""
Quick performance test for the combined system
"""
import os
import sys
from pathlib import Path
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def test_quick():
    """Quick test without viewer."""
    print("🚀 Quick Performance Test")
    print("-" * 30)
    
    start_time = time.time()
    
    try:
        print("1. Importing environment...")
        from combined_walker_avoidance.combined_env import G1CombinedEnv
        import_time = time.time()
        print(f"   ✅ Import took {import_time - start_time:.2f}s")
        
        print("2. Creating environment (headless)...")
        env = G1CombinedEnv(enable_viewer=False)
        create_time = time.time()
        print(f"   ✅ Creation took {create_time - import_time:.2f}s")
        
        print("3. Resetting environment...")
        obs = env.reset()
        reset_time = time.time()
        print(f"   ✅ Reset took {reset_time - create_time:.2f}s")
        
        print("4. Running 10 steps...")
        step_start = time.time()
        for i in range(10):
            obs, reward, done, info = env.step()
            if done:
                obs = env.reset()
                
        step_time = time.time()
        print(f"   ✅ 10 steps took {step_time - step_start:.2f}s ({10/(step_time - step_start):.1f} Hz)")
        
        env.close()
        total_time = time.time() - start_time
        print(f"\n✅ TOTAL TEST TIME: {total_time:.2f}s")
        
        if total_time > 10:
            print("⚠️  System is slow - needs optimization")
            return False
        else:
            print("🎉 System performance is good!")
            return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_quick()

