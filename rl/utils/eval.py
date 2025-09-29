import torch
import time
from pathlib import Path

import mujoco
import mujoco.viewer

import imageio
from datetime import datetime

class EvaluateEnv:
    def __init__(self, env, policy, args):
        self.env = env
        self.policy = policy
        self.ep_len = args.ep_len
        
        # Check if user wants to enable video recording (default: NO VIDEO)
        self.record_video = getattr(args, 'record_video', False)
        
        if self.record_video:
            if args.out_dir is None:
                args.out_dir = Path(args.path.parent, "videos")

            video_outdir = Path(args.out_dir)
            try:
                Path.mkdir(video_outdir, exist_ok=True)
                now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                video_fn = Path(video_outdir, args.path.stem + "-" + now + ".mp4")
                self.writer = imageio.get_writer(video_fn, fps=60)
                print(f"📹 Recording video to: {video_fn}")
            except Exception as e:
                print("Could not create video writer:", e)
                exit(-1)
        else:
            print("✅ Live viewing mode (no video recording)")
            self.writer = None

    @torch.no_grad()
    def run(self):

        height = 480
        width = 640
        renderer = mujoco.Renderer(self.env.model, height, width)
        viewer = mujoco.viewer.launch_passive(self.env.model, self.env.data)
        frames = []

        # Make a camera.
        cam = viewer.cam
        mujoco.mjv_defaultCamera(cam)
        cam.elevation = -20
        cam.distance = 4

        reset_counter = 0
        observation = self.env.reset()
        
        # FORCE FORWARD WALKING: Set goal speed to ensure robot moves forward
        if hasattr(self.env.task, '_goal_speed_ref'):
            self.env.task._goal_speed_ref = 0.35  # Force forward walking speed
            print(f"🚀 Forced goal speed to: {self.env.task._goal_speed_ref}")
        
        # Collect episode rewards for statistics
        ep_rewards = []
        step_count = 0
        
        while self.env.data.time < self.ep_len:

            step_start = time.time()

            # forward pass and step
            raw = self.policy.forward(torch.tensor(observation, dtype=torch.float32), deterministic=True).detach().numpy()
            observation, reward, done, info = self.env.step(raw.copy())
            
            # Collect reward data
            if info:
                ep_rewards.append(info)
            step_count += 1

            # render scene
            cam.lookat = self.env.data.body(1).xpos.copy()
            if self.record_video:
                renderer.update_scene(self.env.data, cam)
                pixels = renderer.render()
                frames.append(pixels)

            viewer.sync()

            if done and reset_counter < 3:
                observation = self.env.reset()
                reset_counter += 1

            time_until_next_step = max(
                0, self.env.frame_skip*self.env.model.opt.timestep - (time.time() - step_start))
            time.sleep(time_until_next_step)

        # Print reward statistics like debug_stepper
        self.print_reward_stats(ep_rewards, step_count)
        
        if self.record_video and self.writer:
            for frame in frames:
                self.writer.append_data(frame)
            self.writer.close()
            print("✅ Video saved successfully!")
        else:
            print("✅ Live evaluation completed (no video saved)")
        self.env.close()
        viewer.close()
    
    def print_reward_stats(self, ep_rewards, step_count):
        """Print detailed reward breakdown like debug_stepper"""
        if not ep_rewards:
            print("⚠️  No reward data collected")
            return
        
        print(f"\n✅ Episode finished after {step_count} timesteps")
        
        mean_rewards = {k: [] for k in ep_rewards[-1].keys()}
        print('\n' + '='*50)
        print(' REWARD BREAKDOWN ')
        print('='*50)
        
        total_reward = 0
        for key in mean_rewards.keys():
            values = [step[key] for step in ep_rewards]
            mean_val = sum(values) / len(values)
            mean_rewards[key] = mean_val
            total_reward += mean_val
            print(f'{key:>20}: {mean_val:>8.4f}')
        
        print('-'*50)
        print(f'{"TOTAL REWARD":>20}: {total_reward:>8.4f}')
        print(f'{"STEPS":>20}: {len(ep_rewards):>8d}')
        print('='*50)
