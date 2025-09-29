import os
import numpy as np

from .utils import Geometry, VizColor, compute_pairwise_dist, Logger
from .robot import G1BasicConfig, G1BasicKinematics
from .sim import G1BasicMujocoAgent
from .control import PIDPolicy, SafeController, BasicSafeControllerConfig


class Config:
    max_num_steps = 2000

    class agent:
        mujoco_model = "g1/scene_29dof.xml"
        dt = 0.01
        obstacle_debug = dict(num_obstacle=0, manual_movement_step_size=0.1)

    class safety:
        cfg = BasicSafeControllerConfig()
        cfg.safe_algo.eta = 1.5
        cfg.safe_algo.safety_buffer = 0.05


class TaskObject3D:
    def __init__(self, **kwargs):
        self.frame = kwargs.get("frame", np.eye(4))
        self.velocity = kwargs.get("velocity", 1.0)
        self.bound = kwargs.get("bound", np.zeros((3, 2)))
        self.smooth_weight = kwargs.get("smooth_weight", 1.0)
        self.last_direction = np.zeros(3)
        self.step_counter = 0
        self.keep_direction_step = kwargs.get("keep_direction_step", 1)

    def move(self, mode="Brownian"):
        if mode == "Brownian":
            if self.step_counter % self.keep_direction_step == 0:
                direction = np.random.normal(loc=0.0, size=3)
                direction = self.velocity * direction / np.linalg.norm(direction)
            else:
                direction = self.last_direction
            self.last_frame = self.frame.copy()
            update_step = (1 - self.smooth_weight) * self.last_direction + self.smooth_weight * direction
            self.frame[:3, 3] += update_step
            for dim in range(3):
                if self.frame[dim, 3] < self.bound[dim][0]:
                    self.frame[dim, 3] = self.last_frame[dim, 3] - update_step[dim]
                elif self.frame[dim, 3] > self.bound[dim][1]:
                    self.frame[dim, 3] = self.last_frame[dim, 3] - update_step[dim]
            self.last_direction = self.frame[:3, 3] - self.last_frame[:3, 3]
        self.step_counter += 1


class BenchmarkTask:
    def __init__(self, robot_cfg, robot_kinematics, agent, **kwargs):
        self.robot_cfg = robot_cfg
        self.robot_kinematics = robot_kinematics
        self.agent = agent
        self.task_name = kwargs.get("task_name", "G1BenchmarkTask")
        self.max_episode_length = kwargs.get("max_episode_length", -1)
        self.num_obstacle_task = kwargs.get("num_obstacle", 3)
        self.reset()

    def reset(self):
        self.episode_length = 0
        self.obstacle_task = []
        self.obstacle_task_geom = []
        for _ in range(self.num_obstacle_task):
            obstacle = TaskObject3D(
                velocity=0.01,
                keep_direction_step=500,
                bound=np.array([[-0.3, 0.5], [-0.3, 0.5], [0.8, 1.0]]),
            )
            obstacle.frame[:3, 3] = np.array([np.random.uniform(low, high) for low, high in obstacle.bound])
            self.obstacle_task.append(obstacle)
            self.obstacle_task_geom.append(Geometry(type="sphere", radius=0.05, color=VizColor.obstacle_task))

        self.goal_teleop = {}
        self.goal_teleop["left"] = np.array(
            [[1.0, 0.0, 0.0, 0.25], [0.0, 1.0, 0.0, 0.25], [0.0, 0.0, 1.0, 0.1], [0.0, 0.0, 0.0, 1.0]]
        )
        self.goal_teleop["right"] = np.array(
            [[1.0, 0.0, 0.0, 0.25], [0.0, 1.0, 0.0, -0.25], [0.0, 0.0, 1.0, 0.1], [0.0, 0.0, 0.0, 1.0]]
        )

        self.robot_goal_left = TaskObject3D(
            velocity=0.001,
            keep_direction_step=10,
            bound=np.array([[0.1, 0.4], [0.1, 0.4], [0.0, 0.2]]),
            smooth_weight=0.8,
        )
        self.robot_goal_left.frame[:3, 3] = np.array([np.random.uniform(low, high) for low, high in self.robot_goal_left.bound])

        self.robot_goal_right = TaskObject3D(
            velocity=0.001,
            keep_direction_step=10,
            bound=np.array([[0.1, 0.4], [-0.4, -0.1], [0.0, 0.2]]),
            smooth_weight=0.8,
        )
        self.robot_goal_right.frame[:3, 3] = np.array([np.random.uniform(low, high) for low, high in self.robot_goal_right.bound])

        self.info = {}
        self.info["goal_teleop"] = {}
        self.info["obstacle_task"] = {}
        self.info["obstacle_debug"] = {}
        self.info["obstacle"] = {}
        self.info["robot_frames"] = None
        self.info["robot_state"] = {}

    def update_robot_goal(self):
        self.robot_goal_left.move()
        self.robot_goal_right.move()

    def update_obstacle_task(self):
        for obstacle in self.obstacle_task:
            obstacle.move()

    def step(self, feedback):
        self.episode_length += 1
        self.update_robot_goal()
        self.update_obstacle_task()

    def get_info(self, feedback) -> dict:
        self.info["done"] = False
        if self.max_episode_length >= 0 and self.episode_length >= self.max_episode_length:
            self.info["done"] = True
        self.info["episode_length"] = self.episode_length
        self.info["robot_base_frame"] = feedback["robot_base_frame"]
        self.info["goal_teleop"]["left"] = self.robot_goal_left.frame
        self.info["goal_teleop"]["right"] = self.robot_goal_right.frame
        self.info["obstacle_task"]["frames_world"] = (
            [obstacle.frame for obstacle in self.obstacle_task] if len(self.obstacle_task) > 0 else np.empty((0, 4, 4))
        )
        self.info["obstacle_task"]["geom"] = self.obstacle_task_geom
        self.info["obstacle_debug"]["frames_world"] = feedback.get("obstacle_debug_frame", np.empty((0, 4, 4)))
        self.info["obstacle_debug"]["geom"] = feedback.get("obstacle_debug_geom", [])
        self.info["obstacle"]["frames_world"] = np.concatenate(
            [self.info["obstacle_task"]["frames_world"], self.info["obstacle_debug"]["frames_world"]], axis=0
        )
        self.info["obstacle"]["geom"] = np.concatenate(
            [self.info["obstacle_task"]["geom"], self.info["obstacle_debug"]["geom"]], axis=0
        )
        self.info["obstacle"]["num"] = len(self.info["obstacle"]["frames_world"])
        self.info["robot_state"]["dof_pos_cmd"] = feedback["dof_pos_cmd"]
        self.info["robot_state"]["dof_pos_fbk"] = feedback["dof_pos_fbk"]
        self.info["robot_state"]["dof_vel_cmd"] = feedback["dof_vel_cmd"]
        return self.info


class SimplifiedBenchmark:
    def __init__(self, cfg: Config):
        self.cfg = cfg

        self.robot_cfg = G1BasicConfig()
        self.robot_kinematics = G1BasicKinematics(self.robot_cfg)

        self.agent = G1BasicMujocoAgent(
            robot_cfg=self.robot_cfg,
            mujoco_model=self.cfg.agent.mujoco_model,
            dt=self.cfg.agent.dt,
            obstacle_debug=self.cfg.agent.obstacle_debug,
        )
        self.task = BenchmarkTask(
            robot_cfg=self.robot_cfg,
            robot_kinematics=self.robot_kinematics,
            agent=self.agent,
            task_name="G1BenchmarkTask",
            max_episode_length=200,
        )

        self.policy = PIDPolicy(self.robot_cfg, self.robot_kinematics)
        self.safe_controller = SafeController(
            cfg=self.cfg.safety.cfg, robot_cfg=self.robot_cfg, robot_kinematics=self.robot_kinematics
        )

        log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../log/debug_g1_benchmark")
        os.makedirs(log_dir, exist_ok=True)
        self.logger = Logger(log_dir)

        self.min_dist_robot_to_env = []
        self.mean_dist_goal = []

    def reset(self):
        self.agent.reset()
        self.task.reset()
        agent_feedback = self.agent.get_feedback()
        task_info = self.task.get_info(agent_feedback)
        return agent_feedback, task_info

    def step(self, u):
        self.agent.step(u)
        agent_feedback = self.agent.get_feedback()
        self.task.step(agent_feedback)
        task_info = self.task.get_info(agent_feedback)
        return agent_feedback, task_info

    def run(self):
        agent_feedback, task_info = self.reset()
        u_ref, action_info = self.policy.act(agent_feedback, task_info)
        u_safe, combined_info = self.safe_controller.safe_control(
            agent_feedback["state"], u_ref, agent_feedback, task_info, action_info
        )

        for _ in range(self.cfg.max_num_steps):
            if task_info.get("done", False):
                agent_feedback, task_info = self.reset()
                u_ref, action_info = self.policy.act(agent_feedback, task_info)
                u_safe, combined_info = self.safe_controller.safe_control(
                    agent_feedback["state"], u_ref, agent_feedback, task_info, action_info
                )

            agent_feedback, task_info = self.step(u_safe)
            u_ref, action_info = self.policy.act(agent_feedback, task_info)
            u_safe, combined_info = self.safe_controller.safe_control(
                agent_feedback["state"], u_ref, agent_feedback, task_info, action_info
            )

            self.log(agent_feedback, task_info, combined_info)
            self.render(agent_feedback, task_info, combined_info)

        print("Simulation ended")
        print("average distance to obstacle: ", np.mean(self.min_dist_robot_to_env))
        print("minimum distance to obstacle: ", np.min(self.min_dist_robot_to_env))
        print("average distance to goal: ", np.mean(self.mean_dist_goal))
        print("maximum distance to goal: ", np.max(self.mean_dist_goal))

    def log(self, agent_feedback, task_info, action_info):
        from collections import OrderedDict

        logs = OrderedDict()
        logs["SSA/phi_safe"] = action_info.get("trigger_safe", False)
        for key, value in logs.items():
            self.logger.log_scalar(value, key)
        self.logger.flush()

        x = agent_feedback["state"]
        dof_pos = self.robot_cfg.decompose_state_to_dof(x)
        robot_frame = self.robot_kinematics.forward_kinematics(dof_pos)
        robot_base_frame = agent_feedback["robot_base_frame"]
        robot_frames_world = np.zeros_like(robot_frame)
        for i in range(len(robot_frame)):
            robot_frames_world[i, :, :] = robot_base_frame @ robot_frame[i, :, :]
        self.robot_frames_world = robot_frames_world

        dist_env = compute_pairwise_dist(
            frame_list_1=robot_frames_world,
            geom_list_1=self.robot_cfg.CollisionVol.values(),
            frame_list_2=task_info["obstacle"]["frames_world"],
            geom_list_2=task_info["obstacle"]["geom"],
        )

        L_ee = self.robot_cfg.Frames.L_ee
        R_ee = self.robot_cfg.Frames.R_ee
        goal_left = (robot_base_frame @ task_info["goal_teleop"]["left"])[:3, 3]
        goal_right = (robot_base_frame @ task_info["goal_teleop"]["right"])[:3, 3]
        dist_goal_left = np.linalg.norm(robot_frames_world[L_ee, :3, 3] - goal_left)
        dist_goal_right = np.linalg.norm(robot_frames_world[R_ee, :3, 3] - goal_right)

        self.min_dist_robot_to_env.append(np.min(dist_env) if dist_env.size else 0.0)
        self.mean_dist_goal.append((dist_goal_left + dist_goal_right) / 2)

    def render(self, agent_feedback, task_info, action_info):
        robot_base_frame = agent_feedback["robot_base_frame"]
        self.agent.render_sphere(
            (robot_base_frame @ task_info["goal_teleop"]["left"])[:3, 3], 0.05 * np.ones(3), VizColor.goal
        )
        self.agent.render_sphere(
            (robot_base_frame @ task_info["goal_teleop"]["right"])[:3, 3], 0.05 * np.ones(3), VizColor.goal
        )

        env_collision_mask = self.safe_controller.safety_index.env_collision_mask
        self_collision_mask = self.safe_controller.safety_index.self_collision_mask

        def viz_critical_env_pairs(mat, thres, mask, line_width, line_color):
            if mat is None:
                return []
            masked_mat = mat[mask]
            masked_indices = np.argwhere(mask)
            indices_of_interest = masked_indices[np.argwhere(masked_mat >= thres).reshape(-1)]
            for i, j in indices_of_interest:
                self.agent.render_line_segment(
                    pos1=self.robot_frames_world[i][:3, 3],
                    pos2=task_info["obstacle"]["frames_world"][j][:3, 3],
                    radius=line_width,
                    color=line_color,
                )
            return indices_of_interest

        def viz_critical_self_pairs(mat, thres, mask, line_width, line_color):
            if mat is None:
                return []
            masked_mat = mat[mask]
            masked_indices = np.argwhere(mask)
            indices_of_interest = masked_indices[np.argwhere(masked_mat >= thres).reshape(-1)]
            for i, j in indices_of_interest:
                self.agent.render_line_segment(
                    pos1=self.robot_frames_world[i][:3, 3],
                    pos2=self.robot_frames_world[j][:3, 3],
                    radius=line_width,
                    color=line_color,
                )
            return indices_of_interest

        phi_hold_mat_env = action_info.get("phi_hold_mat_env", None)
        active_pairs_hold_env = viz_critical_env_pairs(
            phi_hold_mat_env, 0.0, env_collision_mask, 0.002, VizColor.hold
        )
        phi_hold_mat_self = action_info.get("phi_hold_mat_self", None)
        active_pairs_hold_self = viz_critical_self_pairs(
            phi_hold_mat_self, 0.0, self_collision_mask, 0.002, VizColor.hold
        )
        phi_safe_mat_env = action_info.get("phi_safe_mat_env", None)
        active_pairs_unsafe_env = viz_critical_env_pairs(
            phi_safe_mat_env, 0.0, env_collision_mask, 0.02, VizColor.unsafe
        )
        phi_safe_mat_self = action_info.get("phi_safe_mat_self", None)
        active_pairs_unsafe_self = viz_critical_self_pairs(
            phi_safe_mat_self, 0.0, self_collision_mask, 0.02, VizColor.unsafe
        )

        for frame_id, frame_world in enumerate(self.robot_frames_world):
            try:
                geom = self.robot_cfg.CollisionVol[self.robot_cfg.Frames(frame_id)]
            except Exception:
                continue
            if any(frame_id in pair for pair in active_pairs_unsafe_self) or any(
                frame_id == _frame_id for _frame_id, _ in active_pairs_unsafe_env
            ):
                geom.color = VizColor.unsafe
            elif any(frame_id in pair for pair in active_pairs_hold_self) or any(
                frame_id == _frame_id for _frame_id, _ in active_pairs_hold_env
            ):
                geom.color = VizColor.hold
            else:
                if frame_id in self.safe_controller.safety_index.env_collision_vol_ignore:
                    geom.color = VizColor.collision_volume_ignored
                else:
                    geom.color = VizColor.collision_volume
            if geom.type == "sphere":
                self.agent.render_sphere(frame_world[:3, 3], geom.attributes["radius"] * np.ones(3), geom.color)

        for frame_world, geom in zip(
            task_info["obstacle_task"]["frames_world"], task_info["obstacle_task"]["geom"]
        ):
            if geom.type == "sphere":
                self.agent.render_sphere(frame_world[:3, 3], geom.attributes["radius"] * np.ones(3), geom.color)

        self.agent.render()


def main():
    cfg = Config()
    runner = SimplifiedBenchmark(cfg)
    runner.run()


__all__ = [
    "Config",
    "TaskObject3D",
    "BenchmarkTask",
    "SimplifiedBenchmark",
    "main",
]


