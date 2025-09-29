#!/usr/bin/env python3
"""
Entry point for the combined walking + obstacle avoidance system.

This script provides a simple way to run the combined system where the G1 robot:
- Uses its legs to walk (via trained PPO policy)  
- Uses its upper body to avoid obstacles and reach goals (via PID + safety control)

Usage:
    python run_combined.py                          # Run with default settings
    python run_combined.py --episodes 10           # Run 10 episodes
    python run_combined.py --no-viz                # Run without visualization
    python run_combined.py --help                  # Show all options
"""
import os
import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from combined_walker_avoidance.runner import main


def print_banner():
    """Print a nice banner for the combined system."""
    print("🤖" + "="*78 + "🤖")
    print("🚀 G1 COMBINED WALKING + OBSTACLE AVOIDANCE SYSTEM")
    print("="*80)
    print("   This system demonstrates a G1 humanoid robot that can:")
    print("   ✅ Walk using its legs (via trained RL policy)")  
    print("   ✅ Avoid moving obstacles using its upper body (via safety control)")
    print("   ✅ Reach goals with its hands while walking")
    print("   ✅ Maintain balance and safety during combined behaviors")
    print("="*80)
    print("   🎮 Controls: ESC to exit, mouse to control camera")
    print("   📊 Performance metrics will be displayed after each episode")
    print("🤖" + "="*78 + "🤖")
    print()


if __name__ == "__main__":
    # Print banner
    print_banner()
    
    # Check if we're in the safeenv environment
    try:
        import torch
        import mujoco
        print("✅ Required dependencies detected")
    except ImportError as e:
        print(f"❌ Missing required dependency: {e}")
        print("   Make sure you're running this in the safeenv environment!")
        sys.exit(1)
        
    # Check if walking policy exists
    walking_policy_path = PROJECT_ROOT / "trained/g1_walk/actor.pt"
    if not walking_policy_path.exists():
        print(f"❌ Walking policy not found at: {walking_policy_path}")
        print("   Make sure the trained G1 walking policy is available!")
        sys.exit(1)
    else:
        print(f"✅ Walking policy found: {walking_policy_path}")
        
    print("\n🚀 Starting simulation...")
    print("-" * 40)
    
    # Run the main function
    exit_code = main()
    
    if exit_code == 0:
        print("\n🎉 Simulation completed successfully!")
    else:
        print("\n❌ Simulation ended with errors.")
        
    sys.exit(exit_code)

