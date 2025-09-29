import numpy as np
import osqp
from scipy import sparse

from .utils import compute_pairwise_dist


class PIDPolicy:
    def __init__(self, robot_cfg, robot_kinematics):
        self.robot_cfg = robot_cfg
        self.robot_kinematics = robot_kinematics
        num_dof = len(list(self.robot_cfg.DoFs))
        self.pos_K_p = np.ones(num_dof)
        self.pos_K_d = 0.05 * np.ones(num_dof)
        self.pos_K_p[self.robot_cfg.DoFs.WaistYaw] = 2.0
        self.pos_K_p[self.robot_cfg.DoFs.WaistRoll] = 1.5
        self.pos_K_p[self.robot_cfg.DoFs.WaistPitch] = 1.5
        self.pos_K_p[self.robot_cfg.DoFs.LinearX] = 0.0
        self.pos_K_p[self.robot_cfg.DoFs.LinearY] = 0.0
        self.pos_K_p[self.robot_cfg.DoFs.RotYaw] = 0.0
        self.pos_K_d[self.robot_cfg.DoFs.WaistYaw] = 0.1
        self.pos_K_d[self.robot_cfg.DoFs.WaistRoll] = 0.1
        self.pos_K_d[self.robot_cfg.DoFs.WaistPitch] = 0.1
        self.pos_K_d[self.robot_cfg.DoFs.LinearX] = 0.0
        self.pos_K_d[self.robot_cfg.DoFs.LinearY] = 0.0
        self.pos_K_d[self.robot_cfg.DoFs.RotYaw] = 0.0

    def tracking_pos_with_vel(self, desired_dof_pos, dof_pos, dof_vel):
        nominal_dof_vel = self.pos_K_p * (desired_dof_pos - dof_pos) - self.pos_K_d * dof_vel
        control_limits = np.array([self.robot_cfg.ControlLimit[i] for i in self.robot_cfg.Control])
        return np.clip(nominal_dof_vel, -control_limits, control_limits)

    def act(self, agent_feedback: dict, task_info: dict):
        dof_pos_cmd = agent_feedback["dof_pos_cmd"]
        dof_vel_cmd = agent_feedback["dof_vel_cmd"]
        goal_teleop = task_info["goal_teleop"]
        desired_dof_pos, _ = self.robot_kinematics.inverse_kinematics([goal_teleop["left"], goal_teleop["right"]])
        dof_control = self.tracking_pos_with_vel(desired_dof_pos, dof_pos_cmd, dof_vel_cmd)
        info = {}
        return dof_control, info


class BaseSafeAlgorithm:
    def __init__(self, robot_kinematics, **kwargs):
        self.robot_kinematics = robot_kinematics
        self.robot_cfg = robot_kinematics.robot_cfg
        self.num_dof = len(self.robot_cfg.DoFs)
        self.num_control = len(self.robot_cfg.Control)

    def safe_control(self, x, u_ref, agent_feedback, task_info, action_info):
        raise NotImplementedError


class ValueBasedSafeAlgorithm(BaseSafeAlgorithm):
    def __init__(self, safety_index, **kwargs):
        super().__init__(**kwargs)
        self.safety_index = safety_index
        self.control_max = np.array([self.robot_cfg.ControlLimit[i] for i in self.robot_cfg.Control])
        self.control_min = -self.control_max

    def safe_control(self, x, u_ref, agent_feedback, task_info, action_info):
        raise NotImplementedError


class BaseSafetyIndex:
    def __init__(self, robot_kinematics):
        self.robot_kinematics = robot_kinematics
        self.robot_cfg = robot_kinematics.robot_cfg
        self.num_dof = len(self.robot_cfg.DoFs)
        self.num_state = self.robot_cfg.num_state
        self.num_control = len(self.robot_cfg.Control)
        self.num_constraint = None
        self.phi_mask = None

    def phi(self):
        raise NotImplementedError

    def grad_phi(self):
        raise NotImplementedError

    def decode_constraint_info(self, vec, name):
        raise NotImplementedError


class BasicCollisionSafetyIndex(BaseSafetyIndex):
    def __init__(self, robot_kinematics, **kwargs):
        super().__init__(robot_kinematics)
        self.min_distance = kwargs["min_distance"]
        self.collision_vol_frame_ids = list(self.robot_cfg.CollisionVol.keys())
        self.collision_vol_geom = [self.robot_cfg.CollisionVol[fid] for fid in self.collision_vol_frame_ids]
        self.num_collision_vol = len(self.collision_vol_frame_ids)
        self.enable_self_collision = kwargs.get("enable_self_collision", True)
        self.env_collision_vol_ignore = [
            getattr(self.robot_cfg.Frames, name) for name in kwargs.get("env_collision_vol_ignore", []) if hasattr(self.robot_cfg.Frames, name)
        ]
        self.num_constraint_self = self.num_collision_vol * (self.num_collision_vol - 1) // 2
        self.self_collision_mask = self.enable_self_collision * np.ones((self.num_collision_vol, self.num_collision_vol), dtype=bool)
        self.self_collision_mask[np.tril_indices(self.self_collision_mask.shape[0], k=0)] = False
        for frame_i, frame_j in self.robot_cfg.AdjacentCollisionVolPairs:
            i = self.collision_vol_frame_ids.index(frame_i)
            j = self.collision_vol_frame_ids.index(frame_j)
            self.self_collision_mask[i, j] = False
            self.self_collision_mask[j, i] = False
        for frame_i in self.robot_cfg.SelfCollisionVolIgnored:
            i = self.collision_vol_frame_ids.index(frame_i)
            self.self_collision_mask[i, :] = False
            self.self_collision_mask[:, i] = False
        self.min_dist_self = self.min_distance["self"] * np.ones((self.num_collision_vol, self.num_collision_vol))
        self.update_obstacle_info()

    def update_obstacle_info(self, task_info=None):
        if task_info is not None:
            self.num_obstacles = task_info["obstacle"]["num"]
        else:
            self.num_obstacles = 0
        self.num_constraint_env = self.num_collision_vol * self.num_obstacles
        self.num_constraint = self.num_constraint_env + self.num_constraint_self
        self.env_collision_mask = np.ones((self.num_collision_vol, self.num_obstacles), dtype=bool)
        for fid in self.env_collision_vol_ignore:
            i = self.collision_vol_frame_ids.index(fid)
            self.env_collision_mask[i, :] = False
        self.phi_mask = np.zeros(self.num_constraint, dtype=bool)
        self.phi_mask[: self.num_constraint_env] = self.env_collision_mask.flatten()
        if self.num_constraint_self > 0:
            self.phi_mask[-self.num_constraint_self :] = self.self_collision_mask[
                np.triu_indices(self.self_collision_mask.shape[0], k=1)
            ]
        self.min_dist_env = self.min_distance["environment"] * np.ones((self.num_collision_vol, self.num_obstacles))

    def phi(self, x, task_info):
        self.update_obstacle_info(task_info)
        trans_world2base = task_info["robot_base_frame"]
        obstacle_vol_frames_world = task_info["obstacle"]["frames_world"]
        obstacle_vol_geom = task_info["obstacle"]["geom"]
        frames = self.robot_kinematics.forward_kinematics(self.robot_cfg.decompose_state_to_dof(x))
        trans_world2base = self.robot_kinematics.update_base_frame(
            trans_world2base, self.robot_cfg.decompose_state_to_dof(x)
        )
        collision_vol_frames_base = np.stack([frames[fid] for fid in self.collision_vol_frame_ids], axis=0)
        collision_vol_frames_world = np.zeros_like(collision_vol_frames_base)
        for i in range(self.num_collision_vol):
            collision_vol_frames_world[i, :, :] = trans_world2base @ collision_vol_frames_base[i, :, :]
        dist_env = compute_pairwise_dist(
            collision_vol_frames_world, self.collision_vol_geom, obstacle_vol_frames_world, obstacle_vol_geom
        )
        dist_self = compute_pairwise_dist(
            collision_vol_frames_world, self.collision_vol_geom, collision_vol_frames_world, self.collision_vol_geom
        )
        phi = np.zeros(self.num_constraint)
        phi[: self.num_constraint_env] = (self.min_dist_env - dist_env).flatten()
        if self.num_constraint_self > 0:
            phi[-self.num_constraint_self :] = (self.min_dist_self - dist_self)[
                np.triu_indices(dist_self.shape[0], k=1)
            ]
        return phi

    def grad_phi(self, x, task_info, eps=1e-4):
        grad = np.zeros((self.num_constraint, self.robot_cfg.num_state))
        for i in range(self.robot_cfg.num_state):
            x_forward = x.copy()
            x_forward[i] += eps
            x_backward = x.copy()
            x_backward[i] -= eps
            phi_forward = self.phi(x_forward, task_info)
            phi_backward = self.phi(x_backward, task_info)
            grad[:, i] = (phi_forward - phi_backward) / (2 * eps)
        return grad

    def decode_constraint_info(self, vec, name):
        mat_env = vec[: self.num_constraint_env].reshape(self.num_collision_vol, self.num_obstacles)
        mat_self = np.zeros((self.num_collision_vol, self.num_collision_vol))
        if self.num_constraint_self > 0:
            mat_self[np.triu_indices(self.num_collision_vol, k=1)] = vec[-self.num_constraint_self :]
        return {f"{name}_mat_env": mat_env, f"{name}_mat_self": mat_self}


class SafeSetAlgorithm(ValueBasedSafeAlgorithm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.eta_default = kwargs["eta"]
        self.safety_buffer_default = kwargs["safety_buffer"]
        self.slack_weight_default = kwargs["slack_weight"]
        self.control_weight = np.array(kwargs["control_weight"])
        assert self.control_weight.shape == (self.num_control,)

    def kappa(self, x):
        return np.ones((self.safety_index.num_constraint, 1)) * self.eta_default

    def qp_solver(self, x, u_ref, weight_u, weight_s, grad_phi, phi, phi_mask, k_fn, eps_=1.00e-2, abs_=1.00e-2):
        n = weight_u.shape[0]
        m = weight_s.shape[0]
        Qx = sparse.diags(weight_u)
        Qs = sparse.diags(weight_s)
        Pmat = sparse.block_diag([Qx, Qs]).tocsc()
        q = -np.concatenate([(Qx.T @ u_ref.reshape(-1, 1)).reshape(-1), np.zeros(m)])
        Lf = sparse.csc_matrix(grad_phi @ self.robot_cfg.dynamics_f(x))
        Lg = sparse.csc_matrix(grad_phi @ self.robot_cfg.dynamics_g(x))
        C_upper = sparse.hstack([Lg, -sparse.eye(m)])
        C_lower = sparse.eye(m + n)
        Cmat = sparse.vstack([C_upper, C_lower]).tocsc()
        l = np.concatenate([-np.inf * np.ones(m), self.control_min.flatten(), np.zeros(m)])
        val = np.where((phi_mask > 0) & (phi >= 0), np.asarray(-Lf - k_fn(phi.reshape(-1, 1))).flatten(), np.ones_like(phi_mask) * np.inf)
        u = np.concatenate([val, self.control_max.flatten(), np.inf * np.ones(m)])
        prob = osqp.OSQP()
        prob.setup(Pmat, q, Cmat, l, u, alpha=1.0, eps_abs=abs_, eps_rel=eps_, verbose=False)
        result = prob.solve()
        u_sol = result.x[:n].flatten()
        s_sol = result.x[n:].flatten()
        return u_sol, s_sol

    def safety_buffer_solver(self, x, u_ref, control_weight, grad_phi, phi, phi_mask, k_fn):
        u_ref = np.clip(u_ref, self.control_min, self.control_max)
        Lf = sparse.csc_matrix(grad_phi @ self.robot_cfg.dynamics_f(x))
        Lg = sparse.csc_matrix(grad_phi @ self.robot_cfg.dynamics_g(x))
        active_id = np.where((phi_mask > 0) & (phi >= 0))[0]
        Lg_mask = np.abs(Lg[active_id, :]).sum(axis=0)
        phi_new = phi + Lg * u_ref * 0.02
        if phi_new[self.safety_index.phi_mask > 0].max() - phi[self.safety_index.phi_mask > 0].max() <= 0:
            u_safe = u_ref
        else:
            u_safe = np.where(Lg_mask > 0, 0, u_ref)
        return u_safe

    def trigger_SSA(self, x, task_info, phi_offset=0):
        phi = self.safety_index.phi(x, task_info)
        phi[: self.safety_index.num_constraint_env] += phi_offset
        if phi[self.safety_index.phi_mask > 0].max() < 0:
            return phi, False
        else:
            return phi, True

    def safe_control(self, x, u_ref, agent_feedback, task_info, action_info):
        safe_control_args = {}
        phi_hold, trigger_hold = self.trigger_SSA(x, task_info, phi_offset=self.safety_buffer_default)
        phi_safe, trigger_safe = self.trigger_SSA(x, task_info)
        slack_vars = np.zeros_like(phi_safe)
        if not trigger_hold:
            u_safe = u_ref
        elif not trigger_safe:
            u_safe = self.safety_buffer_solver(
                x=x,
                u_ref=u_ref,
                control_weight=safe_control_args.get("control_weight", self.control_weight),
                grad_phi=self.safety_index.grad_phi(x=x, task_info=task_info),
                phi=phi_hold,
                phi_mask=self.safety_index.phi_mask,
                k_fn=self.kappa,
            )
        else:
            control_weight = safe_control_args.get("control_weight", self.control_weight)
            u_safe, slack_vars = self.qp_solver(
                x=x,
                u_ref=u_ref,
                weight_u=np.where(control_weight > 0, control_weight, np.ones_like(control_weight) * 1e8),
                weight_s=safe_control_args.get("slack_weight", self.slack_weight_default * np.ones(self.safety_index.num_constraint)),
                grad_phi=self.safety_index.grad_phi(x=x, task_info=task_info),
                phi=phi_safe,
                phi_mask=self.safety_index.phi_mask,
                k_fn=self.kappa,
            )
        info = {
            "trigger_hold": trigger_hold,
            "trigger_safe": trigger_safe,
            "phi_hold": phi_hold,
            "phi_safe": phi_safe,
            "slack_vars": slack_vars,
        }
        info.update(self.safety_index.decode_constraint_info(phi_hold, 'phi_hold'))
        info.update(self.safety_index.decode_constraint_info(phi_safe, 'phi_safe'))
        info.update(self.safety_index.decode_constraint_info(slack_vars, 'slack_vars'))
        return u_safe, info


class BasicSafeControllerConfig:
    class safety_index:
        class_name = "BasicCollisionSafetyIndex"
        min_distance = {"environment": 0.05, "self": 0.05}
        enable_self_collision = True
        env_collision_vol_ignore = [
            "pelvis_link_1",
            "pelvis_link_2",
            "pelvis_link_3",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
        ]

    class safe_algo:
        class_name = "SafeSetAlgorithm"
        eta = 1.0
        safety_buffer = 0.1
        slack_weight = 1e3
        control_weight = [
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            0.0, 0.0, 0.0
        ]


class SafeController:
    def __init__(self, cfg, robot_cfg, robot_kinematics):
        self.robot_cfg = robot_cfg
        self.robot_kinematics = robot_kinematics
        self.safety_index = BasicCollisionSafetyIndex(robot_kinematics=self.robot_kinematics, **cfg.safety_index.__dict__)
        self.safe_algo = SafeSetAlgorithm(safety_index=self.safety_index, robot_kinematics=self.robot_kinematics, **cfg.safe_algo.__dict__)

    def safe_control(self, x, u_ref, agent_feedback, task_info, action_info):
        return self.safe_algo.safe_control(x, u_ref, agent_feedback, task_info, action_info)


__all__ = [
    "PIDPolicy",
    "SafeController",
    "BasicSafeControllerConfig",
    "BasicCollisionSafetyIndex",
    "SafeSetAlgorithm",
]


