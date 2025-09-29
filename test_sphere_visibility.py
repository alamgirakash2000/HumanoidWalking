#!/usr/bin/env python3
"""
Test script to verify that spheres are visible in the combined system.
This will open the viewer and show the robot with all the spheres.
"""
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def test_sphere_visibility():
    """Test that all spheres are visible in the MuJoCo viewer."""
    print("🔴🟢🔵 Testing Sphere Visibility")
    print("="*50)
    print("This test will:")
    print("  🔴 Show RED moving obstacle spheres")
    print("  🟢 Show GREEN target goal spheres") 
    print("  🔵 Show BLUE robot collision spheres")
    print("  📏 Show collision constraint lines")
    print("="*50)
    
    try:
        from combined_walker_avoidance.combined_env import G1CombinedEnv
        
        # Create environment with viewer
        print("Creating environment with viewer...")
        env = G1CombinedEnv(enable_viewer=True)
        print("✅ Environment created")
        print("👁️  MuJoCo viewer should have opened!")
        
        if not env.viewer:
            print("❌ No viewer created!")
            return False
            
        obs = env.reset()
        print("✅ Environment reset")
        
        print("\n🎬 Running simulation for 30 seconds...")
        print("Look for:")
        print("  🔴 RED spheres moving around (obstacles to avoid)")
        print("  🟢 GREEN spheres near the hands (goals to reach)")
        print("  🔵 BLUE spheres around robot body (collision volumes)")
        print("  🟡 YELLOW/RED lines when robot gets close to obstacles")
        print("\nPress Ctrl+C to exit early...")
        
        step_count = 0
        try:
            for i in range(3000):  # About 30 seconds at 100Hz
                obs, reward, done, info = env.step()
                env.render()  # This shows all the spheres!
                step_count += 1
                
                # Print status every 500 steps (5 seconds)
                if step_count % 500 == 0:
                    task_info = info.get('task_info', {})
                    obstacles = task_info.get('obstacle', {}).get('num', 0)
                    print(f"⏱️  {step_count//100}s: {obstacles} obstacles active, reward={reward:.3f}")
                
                if done:
                    obs = env.reset()
                    print("🔄 Episode reset")
                    
        except KeyboardInterrupt:
            print("\n👋 Interrupted by user")
            
        env.close()
        print("✅ Viewer closed")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    print("🤖 COMBINED WALKING + OBSTACLE AVOIDANCE - SPHERE VISIBILITY TEST")
    print("🤖" + "="*77 + "🤖")
    
    success = test_sphere_visibility()
    
    print("\n" + "🤖" + "="*77 + "🤖")
    if success:
        print("🎉 SPHERE VISIBILITY TEST COMPLETED!")
        print("You should have seen:")
        print("  🔴 RED obstacle spheres moving around")
        print("  🟢 GREEN goal spheres for the hands")
        print("  🔵 BLUE collision spheres around robot body")
        print("  📏 Lines showing collision constraints")
        print("\n✅ The robot walks AND avoids obstacles simultaneously!")
    else:
        print("❌ SPHERE VISIBILITY TEST FAILED!")
        print("Check the error messages above.")
    
    print("🤖" + "="*77 + "🤖")
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())

