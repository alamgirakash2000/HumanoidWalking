"""
Main runner for the combined walking + obstacle avoidance system.

This script:
1. Initializes the combined environment  
2. Runs the simulation with both walking and obstacle avoidance
3. Provides visualization and performance logging
4. Handles episode management and statistics
"""
import os
import sys
import time
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# MuJoCo visualization imports
import mujoco
import mujoco.viewer

# Import our combined system
from .combined_env import G1CombinedEnv

# Import visualization utilities
# from simplified.g1_benchmark_pid.utils import Logger  # Disabled - causing hang

# Simple logger replacement
class SimpleLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        
    def log_scalar(self, value, name):
        pass  # No-op logging for now
        
    def flush(self):
        pass  # No-op logging for now


class CombinedWalkingAvoidanceRunner:
    """
    Main runner for the combined walking + obstacle avoidance task.
    """
    
    def __init__(self, config=None, **kwargs):
        """Initialize the runner."""

        self.config = config or self._get_default_config()
        
        # Override config with any kwargs
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        print("🚀 Initializing Combined Walking + Obstacle Avoidance System")
        print("="*60)
        
        # Initialize environment
        self.env = G1CombinedEnv(
            walking_policy_path=self.config.walking_policy_path,
            enable_viewer=self.config.enable_visualization
        )
        
        # Setup logging
        log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)
        self.logger = SimpleLogger(log_dir)
        
        # Performance tracking
        self.episode_count = 0
        self.total_steps = 0
        self.episode_rewards = []
        self.episode_lengths = []
        
        print("✅ System initialized successfully!")
        print(f"   - Walking policy loaded: {self.config.walking_policy_path}")
        print(f"   - Max steps per episode: {self.config.max_steps_per_episode}")
        print(f"   - Total episodes to run: {self.config.num_episodes}")
        print("="*60)
        
    def _get_default_config(self):
        """Get default configuration."""
        class Config:
            walking_policy_path = PROJECT_ROOT / "trained/g1_walk/actor.pt"
            max_steps_per_episode = 2000
            num_episodes = 5
            enable_visualization = True
            save_performance_log = True

        return Config()
        
    def run(self):
        """Run the combined system."""
        
        print("🎯 Starting Combined Walking + Obstacle Avoidance Simulation")
        print("="*60)
        
        if self.config.enable_visualization:
            self._run_with_visualization()
        else:
            self._run_without_visualization()
            
        self._print_final_statistics()
        
    def _run_with_visualization(self):
        """Run with MuJoCo visualization."""
        
        print("📺 Running with visualization (press ESC to exit)")
        
        # Create viewer for visualization
        viewer = mujoco.viewer.launch_passive(self.env.model, self.env.data)
        self.env.viewer = viewer  # Set viewer in environment
        self.env._setup_viewer()  # Setup camera
        
        try:
            for episode in range(self.config.num_episodes):
                self._run_episode(episode, viewer)
                
        except KeyboardInterrupt:
            print("\n⏹️  Simulation interrupted by user")
            
        finally:
            self.env.close()
            
    def _run_without_visualization(self):
        """Run without visualization (headless mode)."""
        
        print("🔧 Running in headless mode")
        
        try:
            for episode in range(self.config.num_episodes):
                self._run_episode(episode, None)
                
        except KeyboardInterrupt:
            print("\n⏹️  Simulation interrupted by user")
            
        finally:
            self.env.close()
            
    def _run_episode(self, episode_num, viewer=None):
        """Run a single episode."""
        
        print(f"\n🎬 Episode {episode_num + 1}/{self.config.num_episodes}")
        
        # Reset environment
        obs = self.env.reset()
        
        episode_reward = 0
        episode_steps = 0
        episode_start_time = time.time()
        
        # Episode loop
        for step in range(self.config.max_steps_per_episode):
            
            step_start_time = time.time()
            
            # Step environment (policies are called internally)
            obs, reward, done, info = self.env.step()
            
            episode_reward += reward
            episode_steps += 1
            self.total_steps += 1
            
            # Update visualization if enabled
            if viewer is not None:
                self._update_visualization(viewer, info)
                viewer.sync()
                
            # Log performance
            self._log_step(info, step)
            
            # Check if episode is done
            if done:
                print(f"   ✅ Episode completed early at step {episode_steps}")
                break
                
            # Control simulation speed for visualization
            if viewer is not None:
                time_until_next_step = max(
                    0, self.env.cfg.control_dt - (time.time() - step_start_time)
                )
                time.sleep(time_until_next_step)
                
        # Episode summary
        episode_time = time.time() - episode_start_time
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_steps)
        
        print(f"   📊 Episode {episode_num + 1} Summary:")
        print(f"      - Steps: {episode_steps}")
        print(f"      - Total reward: {episode_reward:.3f}")
        print(f"      - Average reward/step: {episode_reward/max(episode_steps, 1):.4f}")
        print(f"      - Duration: {episode_time:.2f}s")
        
        # Print task-specific performance
        self._print_episode_performance(info)
        
    def _update_visualization(self, viewer, info):
        """Update visualization with obstacles and goals."""
        
        # Update camera to follow robot
        try:
            robot_pos = self.env.data.body('pelvis').xpos.copy()
            viewer.cam.lookat = robot_pos
        except:
            pass
            
        # Call environment's render method to draw spheres and collision lines
        self.env.render()
        
    def _log_step(self, info, step):
        """Log step information."""
        
        if self.config.save_performance_log:
            # Log walking rewards
            walking_rewards = info.get("walking_rewards", {})
            for reward_name, reward_value in walking_rewards.items():
                self.logger.log_scalar(reward_value, f"walking/{reward_name}")
                
            # Log safety information
            safety_info = info.get("safety_info", {})
            if "trigger_safe" in safety_info:
                self.logger.log_scalar(float(safety_info["trigger_safe"]), "safety/trigger_safe")
                
            # Log task information  
            task_info = info.get("task_info", {})
            if "episode_length" in task_info:
                self.logger.log_scalar(task_info["episode_length"], "task/episode_length")
                
            self.logger.flush()
            
    def _print_episode_performance(self, info):
        """Print episode-specific performance metrics."""
        
        # Get task performance summary
        if hasattr(self.env.task_manager, 'get_performance_summary'):
            perf_summary = self.env.task_manager.get_performance_summary()
            
            print(f"      🎯 Task Performance:")
            print(f"         - Avg distance to obstacles: {perf_summary.get('avg_min_dist_to_obstacles', 0):.3f}m")
            print(f"         - Min distance to obstacles: {perf_summary.get('min_dist_to_obstacles', 0):.3f}m") 
            print(f"         - Avg distance to goals: {perf_summary.get('avg_dist_to_goals', 0):.3f}m")
            
            # Safety warnings
            min_dist = perf_summary.get('min_dist_to_obstacles', float('inf'))
            if min_dist < 0.1:
                print(f"         ⚠️  WARNING: Very close to obstacles! (min: {min_dist:.3f}m)")
            elif min_dist < 0.2:
                print(f"         ⚡ Close to obstacles (min: {min_dist:.3f}m)")
                
    def _print_final_statistics(self):
        """Print final simulation statistics."""
        
        print("\n" + "="*60)
        print("📈 FINAL SIMULATION STATISTICS")
        print("="*60)
        
        if self.episode_rewards:
            print(f"Total Episodes: {len(self.episode_rewards)}")
            print(f"Total Steps: {self.total_steps}")
            print(f"Average Episode Reward: {np.mean(self.episode_rewards):.3f}")
            print(f"Best Episode Reward: {np.max(self.episode_rewards):.3f}")
            print(f"Average Episode Length: {np.mean(self.episode_lengths):.1f} steps")
            print(f"Success Rate: {len([r for r in self.episode_rewards if r > 0]) / len(self.episode_rewards) * 100:.1f}%")
        else:
            print("No episodes completed.")
            
        print("\n🎉 Simulation completed successfully!")
        print("="*60)


def main():
    """Main function to run the combined system."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Combined Walking + Obstacle Avoidance System")
    parser.add_argument("--policy-path", type=str, 
                       default=None,
                       help="Path to trained walking policy")
    parser.add_argument("--episodes", type=int, default=5,
                       help="Number of episodes to run")
    parser.add_argument("--steps-per-episode", type=int, default=2000,
                       help="Maximum steps per episode") 
    parser.add_argument("--no-viz", action="store_true",
                       help="Disable visualization (headless mode)")
    parser.add_argument("--no-log", action="store_true", 
                       help="Disable performance logging")
    
    args = parser.parse_args()
    
    # Configure runner
    class Config:
        walking_policy_path = args.policy_path or (PROJECT_ROOT / "trained/g1_walk/actor.pt")
        num_episodes = args.episodes
        max_steps_per_episode = args.steps_per_episode
        enable_visualization = not args.no_viz
        save_performance_log = not args.no_log
        
    # Initialize and run
    runner = CombinedWalkingAvoidanceRunner(Config())
    
    try:
        runner.run()
    except Exception as e:
        print(f"\n❌ Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
