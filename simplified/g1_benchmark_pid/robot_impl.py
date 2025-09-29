from enum import IntEnum
import numpy as np
import casadi
import pinocchio as pin
from pinocchio import casadi as cpin
import os

from .utils import Geometry, VizColor


class G1BasicConfig:
    mjcf_path = 'g1/g1_29dof_upper_body.xml'
    joint_to_lock = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    ]

    class DoFs(IntEnum):
        WaistYaw = 0
        WaistRoll = 1
        WaistPitch = 2
        LeftShoulderPitch = 3
        LeftShoulderRoll = 4
        LeftShoulderYaw = 5
        LeftElbow = 6
        LeftWristRoll = 7
        LeftWristPitch = 8
        LeftWristYaw = 9
        RightShoulderPitch = 10
        RightShoulderRoll = 11
        RightShoulderYaw = 12
        RightElbow = 13
        RightWristRoll = 14
        RightWristPitch = 15
        RightWristYaw = 16
        LinearX = 17
        LinearY = 18
        RotYaw = 19

    class Control(IntEnum):
        vWaistYaw = 0
        vWaistRoll = 1
        vWaistPitch = 2
        vLeftShoulderPitch = 3
        vLeftShoulderRoll = 4
        vLeftShoulderYaw = 5
        vLeftElbow = 6
        vLeftWristRoll = 7
        vLeftWristPitch = 8
        vLeftWristYaw = 9
        vRightShoulderPitch = 10
        vRightShoulderRoll = 11
        vRightShoulderYaw = 12
        vRightElbow = 13
        vRightWristRoll = 14
        vRightWristPitch = 15
        vRightWristYaw = 16
        vLinearX = 17
        vLinearY = 18
        vRotYaw = 19

    ControlLimit = {
        Control.vWaistYaw: 0.0,
        Control.vWaistRoll: 1.0,
        Control.vWaistPitch: 0.0,
        Control.vLeftShoulderPitch: 5.0,
        Control.vLeftShoulderRoll: 5.0,
        Control.vLeftShoulderYaw: 5.0,
        Control.vLeftElbow: 5.0,
        Control.vLeftWristRoll: 5.0,
        Control.vLeftWristPitch: 5.0,
        Control.vLeftWristYaw: 5.0,
        Control.vRightShoulderPitch: 5.0,
        Control.vRightShoulderRoll: 5.0,
        Control.vRightShoulderYaw: 5.0,
        Control.vRightElbow: 5.0,
        Control.vRightWristRoll: 5.0,
        Control.vRightWristPitch: 5.0,
        Control.vRightWristYaw: 5.0,
        Control.vLinearX: 0.0,
        Control.vLinearY: 0.0,
        Control.vRotYaw: 0.0,
    }

    @property
    def num_state(self):
        return len(list(self.DoFs))

    def compose_state_from_dof(self, dof_pos):
        return np.asarray(dof_pos).reshape(-1)

    def decompose_state_to_dof(self, state):
        return np.asarray(state).reshape(-1)

    def dynamics_f(self, state):
        return np.zeros((self.num_state, 1))

    def dynamics_g(self, state):
        return np.eye(self.num_state)

    def dynamics_xdot(self, state, control):
        state_vec = np.asarray(state).reshape(-1, 1)
        control_vec = np.asarray(control).reshape(-1, 1)
        xdot = self.dynamics_f(state_vec) + self.dynamics_g(state_vec) @ control_vec
        return xdot.reshape(-1)

    class MujocoDoFs(IntEnum):
        WaistYaw = 15
        WaistRoll = 16
        WaistPitch = 17
        LeftShoulderPitch = 18
        LeftShoulderRoll = 19
        LeftShoulderYaw = 20
        LeftElbow = 21
        LeftWristRoll = 22
        LeftWristPitch = 23
        LeftWristYaw = 24
        RightShoulderPitch = 25
        RightShoulderRoll = 26
        RightShoulderYaw = 27
        RightElbow = 28
        RightWristRoll = 29
        RightWristPitch = 30
        RightWristYaw = 31
        LinearX = 0
        LinearY = 1
        RotYaw = 2

    class MujocoMotors(IntEnum):
        WaistYaw = 0
        WaistRoll = 1
        WaistPitch = 2
        LeftShoulderPitch = 3
        LeftShoulderRoll = 4
        LeftShoulderYaw = 5
        LeftElbow = 6
        LeftWristRoll = 7
        LeftWristPitch = 8
        LeftWristYaw = 9
        RightShoulderPitch = 10
        RightShoulderRoll = 11
        RightShoulderYaw = 12
        RightElbow = 13
        RightWristRoll = 14
        RightWristPitch = 15
        RightWristYaw = 16
        LinearX = 17
        LinearY = 18
        RotYaw = 19

    MujocoDoF_to_DoF = {
        MujocoDoFs.WaistYaw: DoFs.WaistYaw,
        MujocoDoFs.WaistRoll: DoFs.WaistRoll,
        MujocoDoFs.WaistPitch: DoFs.WaistPitch,
        MujocoDoFs.LeftShoulderPitch: DoFs.LeftShoulderPitch,
        MujocoDoFs.LeftShoulderRoll: DoFs.LeftShoulderRoll,
        MujocoDoFs.LeftShoulderYaw: DoFs.LeftShoulderYaw,
        MujocoDoFs.LeftElbow: DoFs.LeftElbow,
        MujocoDoFs.LeftWristRoll: DoFs.LeftWristRoll,
        MujocoDoFs.LeftWristPitch: DoFs.LeftWristPitch,
        MujocoDoFs.LeftWristYaw: DoFs.LeftWristYaw,
        MujocoDoFs.RightShoulderPitch: DoFs.RightShoulderPitch,
        MujocoDoFs.RightShoulderRoll: DoFs.RightShoulderRoll,
        MujocoDoFs.RightShoulderYaw: DoFs.RightShoulderYaw,
        MujocoDoFs.RightElbow: DoFs.RightElbow,
        MujocoDoFs.RightWristRoll: DoFs.RightWristRoll,
        MujocoDoFs.RightWristPitch: DoFs.RightWristPitch,
        MujocoDoFs.RightWristYaw: DoFs.RightWristYaw,
        MujocoDoFs.LinearX: DoFs.LinearX,
        MujocoDoFs.LinearY: DoFs.LinearY,
        MujocoDoFs.RotYaw: DoFs.RotYaw,
    }

    DoF_to_MujocoDoF = {
        DoFs.WaistYaw: MujocoDoFs.WaistYaw,
        DoFs.WaistRoll: MujocoDoFs.WaistRoll,
        DoFs.WaistPitch: MujocoDoFs.WaistPitch,
        DoFs.LeftShoulderPitch: MujocoDoFs.LeftShoulderPitch,
        DoFs.LeftShoulderRoll: MujocoDoFs.LeftShoulderRoll,
        DoFs.LeftShoulderYaw: MujocoDoFs.LeftShoulderYaw,
        DoFs.LeftElbow: MujocoDoFs.LeftElbow,
        DoFs.LeftWristRoll: MujocoDoFs.LeftWristRoll,
        DoFs.LeftWristPitch: MujocoDoFs.LeftWristPitch,
        DoFs.LeftWristYaw: MujocoDoFs.LeftWristYaw,
        DoFs.RightShoulderPitch: MujocoDoFs.RightShoulderPitch,
        DoFs.RightShoulderRoll: MujocoDoFs.RightShoulderRoll,
        DoFs.RightShoulderYaw: MujocoDoFs.RightShoulderYaw,
        DoFs.RightElbow: MujocoDoFs.RightElbow,
        DoFs.RightWristRoll: MujocoDoFs.RightWristRoll,
        DoFs.RightWristPitch: MujocoDoFs.RightWristPitch,
        DoFs.RightWristYaw: MujocoDoFs.RightWristYaw,
        DoFs.LinearX: MujocoDoFs.LinearX,
        DoFs.LinearY: MujocoDoFs.LinearY,
        DoFs.RotYaw: MujocoDoFs.RotYaw,
    }

    MujocoMotor_to_Control = {
        MujocoMotors.WaistYaw: Control.vWaistYaw,
        MujocoMotors.WaistRoll: Control.vWaistRoll,
        MujocoMotors.WaistPitch: Control.vWaistPitch,
        MujocoMotors.LeftShoulderPitch: Control.vLeftShoulderPitch,
        MujocoMotors.LeftShoulderRoll: Control.vLeftShoulderRoll,
        MujocoMotors.LeftShoulderYaw: Control.vLeftShoulderYaw,
        MujocoMotors.LeftElbow: Control.vLeftElbow,
        MujocoMotors.LeftWristRoll: Control.vLeftWristRoll,
        MujocoMotors.LeftWristPitch: Control.vLeftWristPitch,
        MujocoMotors.LeftWristYaw: Control.vLeftWristYaw,
        MujocoMotors.RightShoulderPitch: Control.vRightShoulderPitch,
        MujocoMotors.RightShoulderRoll: Control.vRightShoulderRoll,
        MujocoMotors.RightShoulderYaw: Control.vRightShoulderYaw,
        MujocoMotors.RightElbow: Control.vRightElbow,
        MujocoMotors.RightWristRoll: Control.vRightWristRoll,
        MujocoMotors.RightWristPitch: Control.vRightWristPitch,
        MujocoMotors.RightWristYaw: Control.vRightWristYaw,
        MujocoMotors.LinearX: Control.vLinearX,
        MujocoMotors.LinearY: Control.vLinearY,
        MujocoMotors.RotYaw: Control.vRotYaw,
    }

    class Frames(IntEnum):
        waist_yaw_joint = 0
        waist_roll_joint = 1
        waist_pitch_joint = 2
        left_shoulder_pitch_joint = 3
        left_shoulder_roll_joint = 4
        left_shoulder_yaw_joint = 5
        left_elbow_joint = 6
        left_wrist_roll_joint = 7
        left_wrist_pitch_joint = 8
        left_wrist_yaw_joint = 9
        right_shoulder_pitch_joint = 10
        right_shoulder_roll_joint = 11
        right_shoulder_yaw_joint = 12
        right_elbow_joint = 13
        right_wrist_roll_joint = 14
        right_wrist_pitch_joint = 15
        right_wrist_yaw_joint = 16
        L_ee = 17
        R_ee = 18
        torso_link_1 = 19
        torso_link_2 = 20
        torso_link_3 = 21
        pelvis_link_1 = 22
        pelvis_link_2 = 23
        pelvis_link_3 = 24

    CollisionVol = {
        Frames.waist_yaw_joint: Geometry(type='sphere', radius=0.05, color=VizColor.collision_volume),
        Frames.waist_roll_joint: Geometry(type='sphere', radius=0.05, color=VizColor.collision_volume),
        Frames.waist_pitch_joint: Geometry(type='sphere', radius=0.05, color=VizColor.collision_volume),
        Frames.left_shoulder_pitch_joint: Geometry(type='sphere', radius=0.05, color=VizColor.collision_volume),
        Frames.left_shoulder_roll_joint: Geometry(type='sphere', radius=0.06, color=VizColor.collision_volume),
        Frames.left_shoulder_yaw_joint: Geometry(type='sphere', radius=0.05, color=VizColor.collision_volume),
        Frames.left_elbow_joint: Geometry(type='sphere', radius=0.05, color=VizColor.collision_volume),
        Frames.left_wrist_roll_joint: Geometry(type='sphere', radius=0.05, color=VizColor.collision_volume),
        Frames.left_wrist_pitch_joint: Geometry(type='sphere', radius=0.05, color=VizColor.collision_volume),
        Frames.left_wrist_yaw_joint: Geometry(type='sphere', radius=0.05, color=VizColor.collision_volume),
        Frames.right_shoulder_pitch_joint: Geometry(type='sphere', radius=0.05, color=VizColor.collision_volume),
        Frames.right_shoulder_roll_joint: Geometry(type='sphere', radius=0.06, color=VizColor.collision_volume),
        Frames.right_shoulder_yaw_joint: Geometry(type='sphere', radius=0.05, color=VizColor.collision_volume),
        Frames.right_elbow_joint: Geometry(type='sphere', radius=0.05, color=VizColor.collision_volume),
        Frames.right_wrist_roll_joint: Geometry(type='sphere', radius=0.05, color=VizColor.collision_volume),
        Frames.right_wrist_pitch_joint: Geometry(type='sphere', radius=0.05, color=VizColor.collision_volume),
        Frames.right_wrist_yaw_joint: Geometry(type='sphere', radius=0.05, color=VizColor.collision_volume),
        Frames.L_ee: Geometry(type='sphere', radius=0.05, color=VizColor.collision_volume),
        Frames.R_ee: Geometry(type='sphere', radius=0.05, color=VizColor.collision_volume),
        Frames.torso_link_1: Geometry(type='sphere', radius=0.10, color=VizColor.collision_volume),
        Frames.torso_link_2: Geometry(type='sphere', radius=0.10, color=VizColor.collision_volume),
        Frames.torso_link_3: Geometry(type='sphere', radius=0.08, color=VizColor.collision_volume),
        Frames.pelvis_link_1: Geometry(type='sphere', radius=0.05, color=VizColor.collision_volume),
        Frames.pelvis_link_2: Geometry(type='sphere', radius=0.05, color=VizColor.collision_volume),
        Frames.pelvis_link_3: Geometry(type='sphere', radius=0.05, color=VizColor.collision_volume),
    }

    AdjacentCollisionVolPairs = [
        [Frames.waist_yaw_joint, Frames.waist_roll_joint],
        [Frames.waist_yaw_joint, Frames.waist_pitch_joint],
        [Frames.waist_yaw_joint, Frames.torso_link_1],
        [Frames.waist_roll_joint, Frames.waist_pitch_joint],
        [Frames.waist_roll_joint, Frames.torso_link_1],
        [Frames.waist_pitch_joint, Frames.torso_link_1],
        [Frames.torso_link_1, Frames.torso_link_2],
        [Frames.torso_link_1, Frames.torso_link_3],
        [Frames.torso_link_2, Frames.torso_link_3],
        [Frames.left_shoulder_pitch_joint, Frames.torso_link_1],
        [Frames.left_shoulder_pitch_joint, Frames.torso_link_2],
        [Frames.left_shoulder_roll_joint, Frames.torso_link_1],
        [Frames.left_shoulder_roll_joint, Frames.torso_link_2],
        [Frames.left_shoulder_pitch_joint, Frames.left_shoulder_roll_joint],
        [Frames.left_shoulder_pitch_joint, Frames.left_shoulder_yaw_joint],
        [Frames.left_shoulder_roll_joint, Frames.left_shoulder_yaw_joint],
        [Frames.left_shoulder_yaw_joint, Frames.left_elbow_joint],
        [Frames.left_elbow_joint, Frames.left_wrist_roll_joint],
        [Frames.left_wrist_roll_joint, Frames.left_wrist_pitch_joint],
        [Frames.left_wrist_roll_joint, Frames.left_wrist_yaw_joint],
        [Frames.left_wrist_roll_joint, Frames.L_ee],
        [Frames.left_wrist_pitch_joint, Frames.left_wrist_yaw_joint],
        [Frames.left_wrist_pitch_joint, Frames.L_ee],
        [Frames.left_wrist_yaw_joint, Frames.L_ee],
        [Frames.right_shoulder_pitch_joint, Frames.torso_link_1],
        [Frames.right_shoulder_pitch_joint, Frames.torso_link_2],
        [Frames.right_shoulder_roll_joint, Frames.torso_link_1],
        [Frames.right_shoulder_roll_joint, Frames.torso_link_2],
        [Frames.right_shoulder_pitch_joint, Frames.right_shoulder_roll_joint],
        [Frames.right_shoulder_pitch_joint, Frames.right_shoulder_yaw_joint],
        [Frames.right_shoulder_roll_joint, Frames.right_shoulder_yaw_joint],
        [Frames.right_shoulder_yaw_joint, Frames.right_elbow_joint],
        [Frames.right_elbow_joint, Frames.right_wrist_roll_joint],
        [Frames.right_wrist_roll_joint, Frames.right_wrist_pitch_joint],
        [Frames.right_wrist_roll_joint, Frames.right_wrist_yaw_joint],
        [Frames.right_wrist_roll_joint, Frames.R_ee],
        [Frames.right_wrist_pitch_joint, Frames.right_wrist_yaw_joint],
        [Frames.right_wrist_pitch_joint, Frames.R_ee],
        [Frames.right_wrist_yaw_joint, Frames.R_ee],
    ]

    SelfCollisionVolIgnored = [
        Frames.waist_yaw_joint,
        Frames.waist_roll_joint,
        Frames.waist_pitch_joint,
        Frames.left_shoulder_pitch_joint,
        Frames.left_shoulder_yaw_joint,
        Frames.left_wrist_roll_joint,
        Frames.left_wrist_pitch_joint,
        Frames.left_wrist_yaw_joint,
        Frames.right_shoulder_pitch_joint,
        Frames.right_shoulder_yaw_joint,
        Frames.right_wrist_roll_joint,
        Frames.right_wrist_pitch_joint,
        Frames.right_wrist_yaw_joint,
        Frames.pelvis_link_1,
        Frames.pelvis_link_2,
        Frames.pelvis_link_3,
    ]


class G1BasicKinematics:
    def __init__(self, robot_cfg: G1BasicConfig, **kwargs) -> None:
        self.robot_cfg = robot_cfg
        np.set_printoptions(precision=5, suppress=True, linewidth=200)
        resource_dir = self._find_resource_dir()
        self.robot = pin.RobotWrapper.BuildFromMJCF(os.path.join(resource_dir, self.robot_cfg.mjcf_path))

        self.mixed_jointsToLockIDs = self.robot_cfg.joint_to_lock
        self.add_extra_frames()
        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0.0] * self.robot.model.nq),
        )

        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
        self.cTf_l = casadi.SX.sym("tf_l", 4, 4)
        self.cTf_r = casadi.SX.sym("tf_r", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        self.L_hand_id = self.reduced_robot.model.getFrameId("L_ee")
        self.R_hand_id = self.reduced_robot.model.getFrameId("R_ee")

        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [casadi.vertcat(
                self.cdata.oMf[self.L_hand_id].translation - self.cTf_l[:3, 3],
                self.cdata.oMf[self.R_hand_id].translation - self.cTf_r[:3, 3],
            )],
        )
        self.rotational_error = casadi.Function(
            "rotational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [casadi.vertcat(
                cpin.log3(self.cdata.oMf[self.L_hand_id].rotation @ self.cTf_l[:3, :3].T),
                cpin.log3(self.cdata.oMf[self.R_hand_id].rotation @ self.cTf_r[:3, :3].T),
            )],
        )
        self._FK = self.create_fk_function()

        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)
        self.param_tf_l = self.opti.parameter(4, 4)
        self.param_tf_r = self.opti.parameter(4, 4)
        self.translational_cost = casadi.sumsqr(
            self.translational_error(self.var_q, self.param_tf_l, self.param_tf_r)
        )
        self.rotation_cost = casadi.sumsqr(self.rotational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        self.regularization_cost = casadi.sumsqr(self.var_q)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)
        self.opti.subject_to(
            self.opti.bounded(
                self.reduced_robot.model.lowerPositionLimit, self.var_q, self.reduced_robot.model.upperPositionLimit
            )
        )
        self.opti.minimize(
            50 * self.translational_cost + self.rotation_cost + 0.02 * self.regularization_cost + 0.1 * self.smooth_cost
        )
        opts = {'ipopt': {'print_level': 0, 'max_iter': 20, 'tol': 1e-4}, 'print_time': False, 'calc_lam_p': False}
        self.opti.solver("ipopt", opts)
        self.init_data = np.zeros(self.reduced_robot.model.nq)

    def _find_resource_dir(self):
        env_dir = os.environ.get("SPARK_G1_RESOURCE_DIR")
        if env_dir and os.path.isdir(env_dir):
            return env_dir
        local_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "resources")
        if os.path.isdir(local_dir):
            return local_dir
        repo_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "module", "spark_robot", "resources"
        )
        if os.path.isdir(repo_dir):
            return repo_dir
        raise FileNotFoundError(
            "Could not locate G1 resource directory. Set SPARK_G1_RESOURCE_DIR or place resources under simplified/g1_benchmark_pid/resources."
        )

    def add_extra_frames(self):
        self.robot.model.addFrame(
            pin.Frame(
                'L_ee',
                self.robot.model.getJointId('left_wrist_yaw_joint'),
                pin.SE3(np.eye(3), np.array([0.05, 0, 0]).T),
                pin.FrameType.OP_FRAME,
            )
        )
        self.robot.model.addFrame(
            pin.Frame(
                'R_ee',
                self.robot.model.getJointId('right_wrist_yaw_joint'),
                pin.SE3(np.eye(3), np.array([0.05, 0, 0]).T),
                pin.FrameType.OP_FRAME,
            )
        )
        self.robot.model.addFrame(
            pin.Frame(
                'torso_link_1',
                self.robot.model.getJointId('waist_pitch_joint'),
                pin.SE3(np.eye(3), np.array([0.0, 0.0, 0.1]).T),
                pin.FrameType.OP_FRAME,
            )
        )
        self.robot.model.addFrame(
            pin.Frame(
                'torso_link_2',
                self.robot.model.getJointId('waist_pitch_joint'),
                pin.SE3(np.eye(3), np.array([0.0, 0.0, 0.2]).T),
                pin.FrameType.OP_FRAME,
            )
        )
        self.robot.model.addFrame(
            pin.Frame(
                'torso_link_3',
                self.robot.model.getJointId('waist_pitch_joint'),
                pin.SE3(np.eye(3), np.array([0.0, 0.0, 0.4]).T),
                pin.FrameType.OP_FRAME,
            )
        )
        self.robot.model.addFrame(
            pin.Frame(
                'pelvis_link_1',
                self.robot.model.getJointId('waist_yaw_joint'),
                pin.SE3(np.eye(3), np.array([0.0, 0.0, 0.0]).T),
                pin.FrameType.OP_FRAME,
            )
        )
        self.robot.model.addFrame(
            pin.Frame(
                'pelvis_link_2',
                self.robot.model.getJointId('waist_yaw_joint'),
                pin.SE3(np.eye(3), np.array([0.0, 0.15, 0.0]).T),
                pin.FrameType.OP_FRAME,
            )
        )
        self.robot.model.addFrame(
            pin.Frame(
                'pelvis_link_3',
                self.robot.model.getJointId('waist_yaw_joint'),
                pin.SE3(np.eye(3), np.array([0.0, -0.15, 0.0]).T),
                pin.FrameType.OP_FRAME,
            )
        )

    def create_fk_function(self):
        transforms = []
        for frame in self.robot_cfg.Frames:
            id_ = self.reduced_robot.model.getFrameId(frame.name)
            rotation = self.cdata.oMf[id_].rotation
            translation = self.cdata.oMf[id_].translation
            transform = casadi.vertcat(
                casadi.horzcat(rotation, translation),
                np.array([[0.0, 0.0, 0.0, 1.0]]),
            )
            transforms.append(transform)
        full_transform = casadi.vertcat(*transforms)
        return casadi.Function("FK", [self.cq], [full_transform])

    def update_base_frame(self, trans_world2base, dof):
        try:
            x = dof[self.robot_cfg.DoFs.LinearX]
            y = dof[self.robot_cfg.DoFs.LinearY]
            yaw = dof[self.robot_cfg.DoFs.RotYaw]
        except Exception:
            return trans_world2base

        cz = np.cos(yaw)
        sz = np.sin(yaw)
        Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
        T = np.eye(4)
        T[:3, :3] = Rz
        T[2, 3] = trans_world2base[2, 3]
        T[0, 3] = x
        T[1, 3] = y
        return T

    def forward_kinematics(self, dof):
        frames = self._FK(dof[: self.reduced_robot.model.nq])
        frames_full = frames.full().reshape(-1, 4, 4)
        return frames_full

    def inverse_kinematics(self, T, current_lr_arm_motor_q=None, current_lr_arm_motor_dq=None):
        left_wrist, right_wrist = T[0], T[1]
        if current_lr_arm_motor_q is not None:
            self.init_data = current_lr_arm_motor_q
        self.opti.set_initial(self.var_q, self.init_data)
        self.opti.set_value(self.param_tf_l, left_wrist)
        self.opti.set_value(self.param_tf_r, right_wrist)
        self.opti.set_value(self.var_q_last, self.init_data)
        try:
            sol = self.opti.solve()
            sol_q = self.opti.value(self.var_q)
            v = current_lr_arm_motor_dq * 0.0 if current_lr_arm_motor_dq is not None else (sol_q - self.init_data) * 0.0
            self.init_data = sol_q
            sol_tauff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v, np.zeros(self.reduced_robot.model.nv))
            sol_tauff = np.concatenate([sol_tauff, np.zeros(len(self.robot_cfg.DoFs) - sol_tauff.shape[0])], axis=0)
            info = {"sol_tauff": sol_tauff}
            dof = np.zeros(len(self.robot_cfg.DoFs))
            dof[: len(sol_q)] = sol_q
            return dof, info
        except Exception:
            sol_q = self.opti.debug.value(self.var_q)
            v = current_lr_arm_motor_dq * 0.0 if current_lr_arm_motor_dq is not None else (sol_q - self.init_data) * 0.0
            self.init_data = sol_q
            sol_tauff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v, np.zeros(self.reduced_robot.model.nv))
            sol_tauff = np.concatenate([sol_tauff, np.zeros(len(self.robot_cfg.DoFs) - sol_tauff.shape[0])], axis=0)
            info = {"sol_tauff": sol_tauff * 0.0}
            dof = np.zeros(len(self.robot_cfg.DoFs))
            dof[: len(current_lr_arm_motor_q)] = current_lr_arm_motor_q
            return dof, info


__all__ = ["G1BasicConfig", "G1BasicKinematics"]


