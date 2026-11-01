# 本ファイルは構成を説明するための骨組みである。報酬の重み・接触条件・終了条件の値・参照生成のパラメータは非公開(None のプレースホルダ)で、このままでは学習結果を再現できない。© 2026 okakei. All rights reserved.
# This file is a skeleton that describes the structure only. Reward weights, contact conditions, termination values and reference-generation parameters are withheld (None placeholders), so the learned results cannot be reproduced as is. © 2026 okakei. All rights reserved.
"""Mimic task skeleton: wheelchair sit/stand (Isaac Lab + unitree_rl_lab).

This is a structural skeleton. Reward weights, stds, thresholds and randomization ranges are
placeholders (W / P at the top, values used in training are not published).
Scene layout, observations, events and terminations are kept as-is.
"""
from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg

##
# Pre-defined configs
##
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import unitree_rl_lab.tasks.mimic.mdp as mdp
from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_MIMIC_ACTION_SCALE
from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_MIMIC_CFG as ROBOT_CFG

##
# Scene definition
##

# 学習に用いた値は非公開。W は報酬の重み、P は係数・しきい値・外乱などの調整値。
# None のままでは動かない。使う場合は各自の設定で値を入れる。
W = {
    "joint_acc": None,
    "joint_torque": None,
    "action_rate": None,
    "joint_limit": None,
    "anchor_pos": None,
    "anchor_ori": None,
    "body_pos": None,
    "body_ori": None,
    "body_lin_vel": None,
    "body_ang_vel": None,
    "undesired_contacts": None,
}
P = {
    "obs_noise_anchor_ori": None,
    "obs_noise_ang_vel": None,
    "obs_noise_joint_pos": None,
    "obs_noise_joint_vel": None,
    "velocity_range": None,
    "push_velocity_range": None,
    "ground_static_friction": None,
    "ground_dynamic_friction": None,
    "contact_sensor_force_threshold": None,
    "reset_pose_range": None,
    "reset_joint_position_range": None,
    "randomize_static_friction_range": None,
    "randomize_dynamic_friction_range": None,
    "randomize_restitution_range": None,
    "randomize_joint_default_pos": None,
    "randomize_torso_com_range": None,
    "push_interval_s": None,
    "std_anchor_pos": None,
    "std_anchor_ori": None,
    "std_body_pos": None,
    "std_body_ori": None,
    "std_body_lin_vel": None,
    "std_body_ang_vel": None,
    "undesired_contact_force_threshold": None,
    "term_anchor_pos_z": None,
    "term_anchor_ori": None,
    "term_ee_pos_z": None,
}

VELOCITY_RANGE = P["velocity_range"]

# 立ち上がりを壊さない弱い外乱(学習を安定させる)
PUSH_VELOCITY_RANGE = P["push_velocity_range"]


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=P["ground_static_friction"],
            dynamic_friction=P["ground_dynamic_friction"],
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )
    # robots
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )

    # 車いす(市販の跳ね上げ式手動車いす): メッシュ実体=見た目+厳密三角メッシュ衝突(全部材すり抜け不可)。USD側にCollisionAPI焼き込み済
    chair = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Chair",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{os.path.dirname(__file__)}/assets/wheelchair_up.usd",
            scale=(0.0086, 0.0086, 0.0086),
        ),
        # 参照モーションの正準座標系(着座向き=+X, 着座骨盤=原点)に合わせて配置する
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.03, 0.0, 0.0), rot=(0.70711, 0.70711, 0.0, 0.0)),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, force_threshold=P["contact_sensor_force_threshold"], debug_vis=True
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    motion = mdp.MotionCommandCfg(
        asset_name="robot",
        # generate npz file before training
        # python python scripts/mimic/csv_to_npz.py -f path/to/G1_Take_102.bvh_60hz.csv --input_fps 60
        motion_file=f"{os.path.dirname(__file__)}/motion.npz",
        anchor_body_name="torso_link",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        pose_range=P["reset_pose_range"],
        velocity_range=VELOCITY_RANGE,
        joint_position_range=P["reset_joint_position_range"],
        body_names=[
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ],
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=UNITREE_G1_29DOF_MIMIC_ACTION_SCALE, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        motion_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-P["obs_noise_anchor_ori"], n_max=P["obs_noise_anchor_ori"])
        )
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-P["obs_noise_ang_vel"], n_max=P["obs_noise_ang_vel"]))
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-P["obs_noise_joint_pos"], n_max=P["obs_noise_joint_pos"]))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-P["obs_noise_joint_vel"], n_max=P["obs_noise_joint_vel"]))
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": P["randomize_static_friction_range"],
            "dynamic_friction_range": P["randomize_dynamic_friction_range"],
            "restitution_range": P["randomize_restitution_range"],
            "num_buckets": 64,
        },
    )

    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": P["randomize_joint_default_pos"],
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": P["randomize_torso_com_range"],
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=P["push_interval_s"],
        params={"velocity_range": PUSH_VELOCITY_RANGE},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- base
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=W["joint_acc"])
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=W["joint_torque"])
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=W["action_rate"])
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=W["joint_limit"],
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )

    # -- tracking
    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=W["anchor_pos"],
        params={"command_name": "motion", "std": P["std_anchor_pos"]},
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=W["anchor_ori"],
        params={"command_name": "motion", "std": P["std_anchor_ori"]},
    )
    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=W["body_pos"],
        params={"command_name": "motion", "std": P["std_body_pos"]},
    )
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=W["body_ori"],
        params={"command_name": "motion", "std": P["std_body_ori"]},
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=W["body_lin_vel"],
        params={"command_name": "motion", "std": P["std_body_lin_vel"]},
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=W["body_ang_vel"],
        params={"command_name": "motion", "std": P["std_body_ang_vel"]},
    )

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=W["undesired_contacts"],
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!.*ankle)(?!.*wrist)(?!.*pelvis)(?!.*hip).+$"
                ],
            ),
            "threshold": P["undesired_contact_force_threshold"],
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": P["term_anchor_pos_z"]},
    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": P["term_anchor_ori"]},
    )
    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": P["term_ee_pos_z"],
            "body_names": [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ],
        },
    )


##
# Environment configuration
##


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum = None

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 30.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.sim.physx.gpu_collision_stack_size = 2 ** 28  # フルメッシュ衝突x4096envの接触量対策(既定64MB→256MB)


class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1e9
