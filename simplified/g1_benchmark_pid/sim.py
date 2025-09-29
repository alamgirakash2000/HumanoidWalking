import numpy as np
import mujoco
import mujoco.viewer
from mujoco.glfw import glfw
import time
import os


class MujocoAgent:
    def __init__(self, robot_cfg, **kwargs) -> None:
        self.robot_cfg = robot_cfg
        resource_dir = self._find_resource_dir()
        self.model = mujoco.MjModel.from_xml_path(os.path.join(resource_dir, kwargs["mujoco_model"]))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data, key_callback=self.key_callback)
        self.viewer_setup()
        self.renderer = mujoco.Renderer(self.model)
        self.dt = kwargs["dt"]
        self.model.opt.timestep = self.dt

        self.obstacle_debug = kwargs.get("obstacle_debug", dict(num_obstacle=0, manual_movement_step_size=0.1))
        self.obstacle_debug_geom = []
        self.num_obstacle_debug = self.obstacle_debug["num_obstacle"]
        self.manual_step_size = self.obstacle_debug["manual_movement_step_size"]
        self.obstacle_debug_frame = np.zeros((self.num_obstacle_debug, 4, 4))
        if self.num_obstacle_debug > 0:
            self.obstacle_debug_frame = np.stack([np.eye(4) for _ in range(self.num_obstacle_debug)], axis=0)
        for frame in self.obstacle_debug_frame:
            frame[:3, 3] = np.array([0.6, 0.0, 0.993]) + np.random.uniform(-0.2, 0.2, 3)

        from .utils import Geometry, VizColor
        self.obstacle_debug_geom = [Geometry(type="sphere", radius=0.05, color=VizColor.obstacle_debug) for _ in range(self.num_obstacle_debug)]
        self.obstacle_debug_selected = 0
        self.num_obstacle_debug_change_buf = 0

        self.num_dof = len(self.robot_cfg.DoFs)
        self.dof_pos_cmd = None
        self.dof_vel_cmd = None
        self.dof_acc_cmd = None

    def _find_resource_dir(self):
        env_dir = os.environ.get("SPARK_G1_RESOURCE_DIR")
        if env_dir and os.path.isdir(env_dir):
            return env_dir
        local_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "resources")
        if os.path.isdir(local_dir):
            return local_dir
        repo_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "module", "spark_robot", "resources")
        if os.path.isdir(repo_dir):
            return repo_dir
        raise FileNotFoundError("Could not locate G1 resource directory. Set SPARK_G1_RESOURCE_DIR or place resources under simplified/g1_benchmark_pid/resources.")

    def viewer_setup(self):
        self.viewer.cam.distance = 2
        self.viewer.cam.lookat[0] = 0
        self.viewer.cam.lookat[1] = 0
        self.viewer.cam.lookat[2] = 0.8
        self.viewer.cam.elevation = -10
        self.viewer.cam.azimuth = 180
        self.viewer.opt.geomgroup = 1

    def add_obstacle(self):
        from .utils import Geometry, VizColor
        self.num_obstacle_debug += 1
        self.obstacle_debug_frame = np.concatenate([self.obstacle_debug_frame, np.eye(4)[None, :, :]], axis=0)
        self.obstacle_debug_frame[-1, :3, 3] = np.array([0.6, 0.0, 0.793]) + np.random.uniform(-0.2, 0.2, 3)
        self.obstacle_debug_geom.append(Geometry(type="sphere", radius=0.05, color=VizColor.obstacle_debug))
        self.obstacle_debug_selected = self.num_obstacle_debug - 1

    def remove_obstacle(self):
        if self.num_obstacle_debug > 0:
            self.num_obstacle_debug -= 1
            self.obstacle_debug_frame = np.concatenate(
                [self.obstacle_debug_frame[: self.obstacle_debug_selected, :, :], self.obstacle_debug_frame[self.obstacle_debug_selected + 1 :, :, :]],
                axis=0,
            )
            self.obstacle_debug_geom = self.obstacle_debug_geom[: self.obstacle_debug_selected] + self.obstacle_debug_geom[self.obstacle_debug_selected + 1 :]
            self.obstacle_debug_selected = 0 if self.num_obstacle_debug > 0 else None

    def key_callback(self, key):
        if self.num_obstacle_debug > 0:
            selected = self.obstacle_debug_selected
            step = self.manual_step_size
            if key == glfw.KEY_RIGHT:
                if selected is not None:
                    self.obstacle_debug_frame[selected, 1, 3] += step
            elif key == glfw.KEY_LEFT:
                if selected is not None:
                    self.obstacle_debug_frame[selected, 1, 3] -= step
            elif key == glfw.KEY_UP:
                if selected is not None:
                    self.obstacle_debug_frame[selected, 0, 3] -= step
            elif key == glfw.KEY_DOWN:
                if selected is not None:
                    self.obstacle_debug_frame[selected, 0, 3] += step
            elif key == glfw.KEY_E:
                if selected is not None:
                    self.obstacle_debug_frame[selected, 2, 3] += step
            elif key == glfw.KEY_Q:
                if selected is not None:
                    self.obstacle_debug_frame[selected, 2, 3] -= step
            elif key == glfw.KEY_SPACE:
                self.obstacle_debug_selected = (self.obstacle_debug_selected + 1) % self.num_obstacle_debug
        if key == glfw.KEY_PAGE_UP:
            self.num_obstacle_debug_change_buf += 1
        elif key == glfw.KEY_PAGE_DOWN:
            self.num_obstacle_debug_change_buf -= 1

    def render_sphere(self, pos, size, color):
        pos = np.asarray(pos)
        if pos.shape == (2,):
            pos = np.r_[pos, 0]
        mujoco.mjv_initGeom(
            self.renderer._scene.geoms[self.renderer._scene.ngeom],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=size,
            pos=pos.flatten(),
            mat=np.eye(3).flatten(),
            rgba=color,
        )
        self.renderer._scene.ngeom += 1
        if self.viewer:
            mujoco.mjv_initGeom(
                self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=size * np.ones(3),
                pos=pos.flatten(),
                mat=np.eye(3).flatten(),
                rgba=color,
            )
            self.viewer.user_scn.ngeom += 1

    def render_line_segment(self, pos1, pos2, radius, color):
        pos1 = np.asarray(pos1)
        pos2 = np.asarray(pos2)
        midpoint = (pos1 + pos2) / 2
        length = np.linalg.norm(pos2 - pos1)
        direction = (pos2 - pos1) / (length + 1e-12)
        z_axis = np.array([0, 0, 1])
        axis = np.cross(z_axis, direction)
        axis_len = np.linalg.norm(axis)
        if axis_len > 1e-6:
            axis = axis / axis_len
            angle = np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
            quat = np.zeros(4)
            mujoco.mju_axisAngle2Quat(quat, axis, angle)
            rot_matrix = np.zeros((3, 3)).flatten()
            mujoco.mju_quat2Mat(rot_matrix, quat)
        else:
            rot_matrix = np.eye(3)
        mujoco.mjv_initGeom(
            self.renderer._scene.geoms[self.renderer._scene.ngeom],
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=[radius, length / 2, 0.0],
            pos=midpoint.flatten(),
            mat=rot_matrix.flatten(),
            rgba=color,
        )
        self.renderer._scene.ngeom += 1
        if self.viewer:
            mujoco.mjv_initGeom(
                self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                size=[radius, length / 2, 0.0],
                pos=midpoint.flatten(),
                mat=rot_matrix.flatten(),
                rgba=color,
            )
            self.viewer.user_scn.ngeom += 1

    def render_obstacle_debug(self):
        from .utils import VizColor

        for i, (frame, geom) in enumerate(zip(self.obstacle_debug_frame, self.obstacle_debug_geom)):
            if geom.type == 'sphere':
                self.render_sphere(frame[:3, 3], geom.attributes["radius"] * np.ones(3), geom.color)
                if i == self.obstacle_debug_selected:
                    start = frame[:3, 3] + [0, 0, 0.15]
                    end = frame[:3, 3] + [0, 0, 0.01]
                    self.render_line_segment(start, end, 0.01, VizColor.obstacle_debug)

    def set_dof_pos(self, dof_pos: np.ndarray) -> None:
        qpos = np.zeros(self.model.nq)
        for mj_dof in self.robot_cfg.MujocoDoFs:
            dof = self.robot_cfg.MujocoDoF_to_DoF[mj_dof]
            qpos[mj_dof] = dof_pos[dof]
        self.data.qpos = qpos
        self.model.opt.gravity[:] = [0, 0, 0]
        self.data.qvel[:] = 0
        self.data.qacc[:] = 0
        self.data.qfrc_applied[:] = 0
        self.data.xfrc_applied[:, :] = 0

    def reset(self) -> None:
        pass

    def mujoco_step(self):
        mujoco.mj_step(self.model, self.data)
        if self.viewer:
            self.viewer.user_scn.ngeom = 0
        self.renderer._scene.ngeom = 0
        time.sleep(self.model.opt.timestep)

    def render(self):
        self.render_obstacle_debug()
        self.renderer.update_scene(self.data)
        self.viewer.sync()

    def _send_control_modeled_dynamics(self, command, **kwargs):
        x = self.compose_state()
        x_dot = self.robot_cfg.dynamics_xdot(x, command)
        x += x_dot * self.dt
        self.dof_pos_cmd = self.robot_cfg.decompose_state_to_dof(x)
        self.dof_vel_cmd = self.robot_cfg.decompose_state_to_dof(x_dot)
        self.set_dof_pos(self.dof_pos_cmd)

    def post_control_processing(self, **kwargs):
        num_obstacle_debug_change = self.num_obstacle_debug_change_buf
        self.num_obstacle_debug_change_buf -= num_obstacle_debug_change
        while num_obstacle_debug_change > 0:
            self.add_obstacle(); num_obstacle_debug_change -= 1
        while num_obstacle_debug_change < 0:
            self.remove_obstacle(); num_obstacle_debug_change += 1
        self.mujoco_step()

    def send_control(self, command: np.ndarray, use_sim_dynamics: bool = False, **kwargs) -> None:
        if use_sim_dynamics:
            raise NotImplementedError
        else:
            self._send_control_modeled_dynamics(command, **kwargs)

    def step(self, control: np.ndarray, **kwargs) -> None:
        self.send_control(control, **kwargs)
        self.post_control_processing(**kwargs)

    def compose_state(self) -> np.ndarray:
        x = self.robot_cfg.compose_state_from_dof(self.dof_pos_cmd)
        return x

    def get_feedback(self) -> None:
        ret = {}
        global_position = self.data.body("robot").xpos.copy()
        global_orientation = self.data.body("robot").xmat.copy().reshape(3, 3)
        robot_base_frame = np.eye(4)
        robot_base_frame[:3, :3] = global_orientation
        robot_base_frame[:3, 3] = global_position
        ret["robot_base_frame"] = robot_base_frame
        dof_pos_fbk = np.zeros(self.num_dof)
        for dof in self.robot_cfg.DoFs:
            mj_dof = self.robot_cfg.DoF_to_MujocoDoF[dof]
            dof_pos_fbk[dof] = self.data.qpos[mj_dof]
        ret["dof_pos_fbk"] = dof_pos_fbk
        if self.dof_pos_cmd is None:
            self.dof_pos_cmd = dof_pos_fbk
        if self.dof_vel_cmd is None:
            self.dof_vel_cmd = np.zeros(self.num_dof)
        if self.dof_acc_cmd is None:
            self.dof_acc_cmd = np.zeros(self.num_dof)
        ret["dof_pos_cmd"] = self.dof_pos_cmd
        ret["dof_vel_cmd"] = self.dof_vel_cmd
        ret["dof_acc_cmd"] = self.dof_acc_cmd
        ret["state"] = self.compose_state()
        ret["obstacle_debug_frame"] = self.obstacle_debug_frame
        ret["obstacle_debug_geom"] = self.obstacle_debug_geom
        return ret


class G1BasicMujocoAgent(MujocoAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compose_state(self):
        x = self.robot_cfg.compose_state_from_dof(self.dof_pos_cmd)
        return x

__all__ = ["G1BasicMujocoAgent", "MujocoAgent"]


