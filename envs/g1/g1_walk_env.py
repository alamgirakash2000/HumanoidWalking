import os
import numpy as np
import transforms3d as tf3
import collections

from robots.robot_base import RobotBase
from envs.common import mujoco_env
from envs.common import robot_interface
from envs.common import config_builder
from tasks.walking_task import WalkingTask

from .gen_xml import builder, LEG_JOINTS


class G1WalkEnv(mujoco_env.MujocoEnv):
    def __init__(self, path_to_yaml=None):

        if path_to_yaml is None:
            path_to_yaml = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                'configs/base.yaml'
            )

        self.cfg = config_builder.load_yaml(path_to_yaml)

        sim_dt = self.cfg.sim_dt
        control_dt = self.cfg.control_dt

        self.dynrand_interval = self.cfg.dynamics_randomization.interval/control_dt if self.cfg.dynamics_randomization and self.cfg.dynamics_randomization.enable else np.inf
        self.perturb_interval = self.cfg.perturbation.interval/control_dt if self.cfg.perturbation and self.cfg.perturbation.enable else np.inf
        self.history_len = self.cfg.obs_history_len

        path_to_xml = '/tmp/mjcf-export/g1/g1.xml'
        if not os.path.exists(path_to_xml):
            export_dir = os.path.dirname(path_to_xml)
            builder(export_dir, config={})

        mujoco_env.MujocoEnv.__init__(self, path_to_xml, sim_dt, control_dt)

        # Joint names for legs (12 actuators)
        self.leg_names = LEG_JOINTS

        # PD gains from config; map by joint name
        gains_dict = self.cfg.pdgains.to_dict()
        kp, kd = zip(*[gains_dict[jn] for jn in self.leg_names])
        pdgains = np.array([kp, kd])

        # Nominal standing pose: base position, orientation, then joint half-sit
        # Start closer to ground; will be adjusted precisely in reset_model
        base_position = [0, 0, 0.80]
        base_orientation = [1, 0, 0, 0]
        half_sitting_pose = [
            0.0, 0.0, -0.2, 0.6, -0.2, 0.0,   # left: yaw roll pitch knee ankle_pitch ankle_roll
            0.0, 0.0, -0.2, 0.6, -0.2, 0.0,   # right
        ]
        self.nominal_pose = base_position + base_orientation + half_sitting_pose

        # Feet body/link names from G1 XML
        self.interface = robot_interface.RobotInterface(
            self.model, self.data,
            'right_ankle_roll_link', 'left_ankle_roll_link', None
        )

        # Task setup (generic walking task)
        self.task = WalkingTask(
            client=self.interface,
            dt=control_dt,
            neutral_foot_orient=np.array([1, 0, 0, 0]),
            root_body='pelvis',
            lfoot_body='left_ankle_roll_link',
            rfoot_body='right_ankle_roll_link',
        )
        self.task._goal_height_ref = 0.90
        self.task._total_duration = 1.1
        self.task._swing_duration = 0.75
        self.task._stance_duration = 0.35

        # Robot controller wrapper
        self.robot = RobotBase(pdgains, control_dt, self.interface, self.task)

        # Mirror settings (same structure as JVRC/H1; indices depend on obs layout)
        base_mir_obs = [-0.1, 1, -2, 3, -4,
                        10, -11, 12, 13, 14, 15,
                        4, -5, 6, 7, 8, 9,
                        22, -23, 24, 25, 26, 27,
                        16, -17, 18, 19, 20, 21]
        append_obs = [(len(base_mir_obs) + i) for i in range(3)]
        self.robot.clock_inds = append_obs[0:2]
        self.robot.mirrored_obs = np.array(base_mir_obs + append_obs, copy=True).tolist()
        self.robot.mirrored_acts = [6, -7, 8, 9, 10, 11,
                                    0, -1, 2, 3, 4, 5]

        # Action/obs spaces
        action_space_size = len(self.leg_names)
        self.action_space = np.zeros(action_space_size)
        self.prev_prediction = np.zeros(action_space_size)

        self.base_obs_len = 5 + 12 + 12 + 3
        self.observation_history = collections.deque(maxlen=self.history_len)
        self.observation_space = np.zeros(self.base_obs_len * self.history_len)

        self.obs_mean = np.concatenate((
            np.zeros(5),
            half_sitting_pose,
            np.zeros(12),
            [0.5, 0.5, 0.5]
        ))
        self.obs_std = np.concatenate((
            [0.2, 0.2, 1, 1, 1],
            0.5 * np.ones(12),
            4 * np.ones(12),
            [1, 1, 1]
        ))
        self.obs_mean = np.tile(self.obs_mean, self.history_len)
        self.obs_std = np.tile(self.obs_std, self.history_len)

        # Backup model for randomization
        import copy as _copy
        self.default_model = _copy.deepcopy(self.model)

    def get_obs(self):
        clock = [np.sin(2 * np.pi * self.task._phase / self.task._period),
                 np.cos(2 * np.pi * self.task._phase / self.task._period)]
        ext_state = np.concatenate((clock, [self.task._goal_speed_ref]))

        qpos = np.copy(self.interface.get_qpos())
        qvel = np.copy(self.interface.get_qvel())
        root_r, root_p = tf3.euler.quat2euler(qpos[3:7])[0:2]
        root_r = np.array([root_r])
        root_p = np.array([root_p])
        root_ang_vel = qvel[3:6]
        motor_pos = self.interface.get_act_joint_positions()
        motor_vel = self.interface.get_act_joint_velocities()

        if self.cfg.observation_noise and self.cfg.observation_noise.enabled:
            noise_type = self.cfg.observation_noise.type
            scales = self.cfg.observation_noise.scales
            level = self.cfg.observation_noise.multiplier
            if noise_type == "uniform":
                def noise(x, n):
                    return np.random.uniform(-x, x, n)
            elif noise_type == "gaussian":
                def noise(x, n):
                    return np.random.randn(n) * x
            else:
                raise Exception("Observation noise type can only be \"uniform\" or \"gaussian\"")
            root_r += noise(scales.root_orient * level, 1)
            root_p += noise(scales.root_orient * level, 1)
            root_ang_vel += noise(scales.root_ang_vel * level, len(root_ang_vel))
            motor_pos += noise(scales.motor_pos * level, len(motor_pos))
            motor_vel += noise(scales.motor_vel * level, len(motor_vel))

        robot_state = np.concatenate([root_r, root_p, root_ang_vel, motor_pos, motor_vel])
        state = np.concatenate([robot_state, ext_state])
        assert state.shape == (self.base_obs_len,), f"State vector length {len(state)} expected {self.base_obs_len}"

        if len(self.observation_history) == 0:
            for _ in range(self.history_len):
                self.observation_history.appendleft(np.zeros_like(state))
            self.observation_history.appendleft(state)
        else:
            self.observation_history.appendleft(state)
        return np.array(self.observation_history).flatten()

    def step(self, action):
        targets = self.cfg.action_smoothing * action + (1 - self.cfg.action_smoothing) * self.prev_prediction
        offsets = [
            self.nominal_pose[self.interface.get_jnt_qposadr_by_name(jnt)[0]]
            for jnt in self.leg_names
        ]

        rewards, done = self.robot.step(targets, np.asarray(offsets))
        obs = self.get_obs()

        if self.cfg.dynamics_randomization and self.cfg.dynamics_randomization.enable and np.random.randint(self.dynrand_interval) == 0:
            self.randomize_dyn()
        if self.cfg.perturbation and self.cfg.perturbation.enable and np.random.randint(self.perturb_interval) == 0:
            self.randomize_perturb()

        self.prev_prediction = action
        return obs, sum(rewards.values()), done, rewards

    def reset_model(self):
        if self.cfg.dynamics_randomization and self.cfg.dynamics_randomization.enable:
            self.randomize_dyn()

        init_qpos, init_qvel = self.nominal_pose.copy(), [0] * self.interface.nv()
        c = self.cfg.init_noise * np.pi / 180
        root_adr = self.interface.get_jnt_qposadr_by_name('root')[0]
        # Small orientation noise; keep yaw upright to avoid initial topple
        init_qpos[root_adr + 3:root_adr + 7] = tf3.euler.euler2quat(np.random.uniform(-c, c), np.random.uniform(-c, c), 0)
        init_qpos[root_adr + 7:] += np.random.uniform(-c, c, len(self.leg_names))

        # Set an initial state
        self.set_state(np.asarray(init_qpos), np.asarray(init_qvel))

        # Adjust pelvis height so soles are on the ground (z≈0)
        try:
            lf_z = float(self.interface.get_object_xpos_by_name('lf_force', 'OBJ_SITE')[2])
            rf_z = float(self.interface.get_object_xpos_by_name('rf_force', 'OBJ_SITE')[2])
            min_foot_z = min(lf_z, rf_z)
            clearance = 0.002
            delta = (0.0 + clearance) - min_foot_z
            init_qpos[root_adr + 2] += delta
            self.set_state(np.asarray(init_qpos), np.asarray(init_qvel))
        except Exception:
            # Fallback: keep current height
            pass

        for _ in range(3):
            self.interface.step()
        self.task.reset(iter_count=self.robot.iteration_count)

        self.prev_prediction = np.zeros_like(self.prev_prediction)
        self.observation_history = collections.deque(maxlen=self.history_len)
        obs = self.get_obs()
        return obs

    def randomize_perturb(self):
        frc_mag = self.cfg.perturbation.force_magnitude
        tau_mag = self.cfg.perturbation.torque_magnitude
        for body in self.cfg.perturbation.bodies:
            self.data.body(body).xfrc_applied[:3] = np.random.uniform(-frc_mag, frc_mag, 3)
            self.data.body(body).xfrc_applied[3:] = np.random.uniform(-tau_mag, tau_mag, 3)
        if np.random.randint(2) == 0:
            self.data.xfrc_applied = np.zeros_like(self.data.xfrc_applied)

    def randomize_dyn(self):
        dofadr = [self.interface.get_jnt_qveladr_by_name(jn) for jn in self.leg_names]
        for jnt in dofadr:
            self.model.dof_frictionloss[jnt] = np.random.uniform(0, 2)
            self.model.dof_damping[jnt] = np.random.uniform(0.02, 2)

        bodies = ["pelvis"]
        for legjoint in self.leg_names:
            bodyid = self.model.joint(legjoint).bodyid
            bodyname = self.model.body(bodyid).name
            bodies.append(bodyname)
        for body in bodies:
            default_mass = self.default_model.body(body).mass[0]
            default_ipos = self.default_model.body(body).ipos
            self.model.body(body).mass[0] = default_mass * np.random.uniform(0.95, 1.05)
            self.model.body(body).ipos = default_ipos + np.random.uniform(-0.01, 0.01, 3)


