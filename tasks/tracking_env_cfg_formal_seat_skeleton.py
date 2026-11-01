# 本ファイルは構成を説明するための骨組みである。報酬の重み・接触条件・終了条件の値・参照生成のパラメータは非公開(None のプレースホルダ)で、このままでは学習結果を再現できない。© 2026 okakei. All rights reserved.
# This file is a skeleton that describes the structure only. Reward weights, contact conditions, termination values and reference-generation parameters are withheld (None placeholders), so the learned results cannot be reproduced as is. © 2026 okakei. All rights reserved.
"""座位目標の導入段階の task skeleton: 式典座位に終わる着座 / 式典座位から始まる起立.

Stage 1 の骨組み(`tracking_env_cfg_skeleton.py`)を継承し、前段階までの追加項と、この段階で足す項を書く。
報酬の重み・しきい値は非公開(W / P はプレースホルダ)。
`.chair_mdp`(足・上体と車いすの距離罰、骨盤 xy の罰、固定開始コマンド)は自作で本リポジトリに含めない。
座位目標の項は `formal_seat_mdp.py`、参照は `motions/formal_seat.py` の出力。

gym id の形:
  Unitree-G1-29dof-Mimic-Wheelchair-Sit-Formal-Seat     (着座。参照 540 frame)
  Unitree-G1-29dof-Mimic-Wheelchair-Stand-Formal-Seat   (起立。参照 490 frame)

補助制御: 環境変数 ARM_PD_BLEND > 0 のとき、起立の座位窓 [0, 100) で腕 14 関節の PD 目標を
0.3 × 方策 + 0.7 × 参照角にする(使用値 0.7、窓末尾 15 frame で解除)。着座は補助なし。
"""
from __future__ import annotations

import os

import numpy as np

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

# 自作(非同梱): 固定開始コマンド、足・上体と車いすの距離罰、骨盤 xy の罰と終了
from .chair_mdp import (
    FixedStartMotionCommand,
    foot_chair_penalty,
    foot_chair_termination,
    root_xy_error_squared,
    root_xy_termination,
    upper_chair_penalty,
)
from . import formal_seat_mdp as fs
from .tracking_env_cfg_skeleton import RobotEnvCfg as Stage1RobotEnvCfg

# 学習に用いた値は非公開。W は報酬の重み、P は係数・しきい値・外乱などの調整値。
# None のままでは動かない。使う場合は各自の設定で値を入れる。
W = {
    "foot_chair_margin": None,
    "upper_chair_margin": None,
    "root_xy_world": None,
    "formal_pelvis": None,
    "formal_torso": None,
    "formal_shoulder": None,
    "formal_arm_sq": None,
    "formal_wrist": None,
}
P = {
    "foot_chair_margin_m": None,
    "upper_chair_margin_m": None,
    "foot_chair_term_m": None,
    "root_xy_term_m": None,
    "formal_pose_term_m": None,
    "arm_motion_scale": None,
    "reset_pose_range": None,
    "reset_joint_position_range": None,
}

HERE = os.path.dirname(__file__)
HOLD_START = 50   # 先頭の保持 step
HOLD_END = 90     # 末尾の保持 step


@configclass
class FormalSeatEnvCfg(Stage1RobotEnvCfg):
    """着座・起立で共通。TASK だけが違う。"""

    TASK = "sit"

    def __post_init__(self):
        super().__post_init__()
        parts = os.path.join(HERE, "chair_parts.json")

        # --- エピソード長: 先頭の保持 + 参照 + 末尾の保持
        with np.load(os.path.join(HERE, "motion.npz")) as motion:
            frames = len(motion["joint_pos"])
        self.episode_length_s = (HOLD_START + frames + HOLD_END) * 0.02

        # --- 前段階まで: 固定開始・小さい初期化ノイズ・押し外乱なし
        self.commands.motion.class_type = FixedStartMotionCommand
        self.commands.motion.pose_range = P["reset_pose_range"]
        self.commands.motion.velocity_range = {}
        self.commands.motion.joint_position_range = P["reset_joint_position_range"]
        self.events.add_joint_default_pos = None
        self.events.push_robot = None
        self.events.base_com = None
        self.scene.robot.soft_joint_pos_limit_factor = 0.98

        # --- 前段階まで: 足と上体の近接罰(余裕の外で罰、足は極小距離で終了)、骨盤 xy
        self.rewards.foot_chair_margin = RewTerm(
            func=foot_chair_penalty, weight=W["foot_chair_margin"], params={"geometry_path": parts, "margin": P["foot_chair_margin_m"]})
        self.terminations.foot_chair = DoneTerm(
            func=foot_chair_termination, params={"geometry_path": parts, "threshold": P["foot_chair_term_m"]})
        self.rewards.upper_chair_margin = RewTerm(
            func=upper_chair_penalty, weight=W["upper_chair_margin"],
            params={"geometry_path": os.path.join(HERE, "upper_geometry.json"), "margin": P["upper_chair_margin_m"]})
        self.terminations.ee_body_pos.params["body_names"] = ["left_ankle_roll_link", "right_ankle_roll_link"]
        self.rewards.root_xy_world = RewTerm(
            func=root_xy_error_squared, weight=W["root_xy_world"], params={"command_name": "motion"})
        self.terminations.root_xy_world = DoneTerm(
            func=root_xy_termination, params={"command_name": "motion", "threshold": P["root_xy_term_m"]})

        # --- この段階: 座位窓の姿勢項(着座は [440, 540)、起立は [0, 50))
        task = {"task": self.TASK}
        self.rewards.formal_pelvis = RewTerm(func=fs.pelvis_penalty, weight=W["formal_pelvis"], params=task)
        self.rewards.formal_torso = RewTerm(func=fs.torso_penalty, weight=W["formal_torso"], params=task)
        self.rewards.formal_shoulder = RewTerm(func=fs.shoulder_penalty, weight=W["formal_shoulder"], params=task)
        # 腕 14 関節の追従(全相。座位窓は係数 1、動作中は係数 arm_motion_scale)と、手首の腿相対距離(座位窓)
        self.rewards.formal_arm_sq = RewTerm(
            func=fs.arm_sq_penalty, weight=W["formal_arm_sq"], params={**task, "motion_scale": P["arm_motion_scale"]})
        self.rewards.formal_wrist = RewTerm(func=fs.wrist_thigh_penalty, weight=W["formal_wrist"], params=task)
        # 座位窓で骨盤が式典座位から大きく離れたら終了
        self.terminations.formal_pose = DoneTerm(
            func=fs.pose_termination, params={**task, "threshold": P["formal_pose_term_m"]})

        # --- 補助制御(起立のみ): 環境変数 ARM_PD_BLEND > 0 で有効
        blend = float(os.environ.get("ARM_PD_BLEND", "0"))
        if blend > 0:
            fs.ArmBlendJointPositionAction.alpha = blend
            self.actions.JointPositionAction.class_type = fs.ArmBlendJointPositionAction


@configclass
class SitFormalSeatEnvCfg(FormalSeatEnvCfg):
    TASK = "sit"


@configclass
class StandFormalSeatEnvCfg(FormalSeatEnvCfg):
    TASK = "stand"


class PlayMixin:
    """再生用: 1 env、初期化ノイズなし、エピソードは終わらない。"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1e9
        self.observations.policy.enable_corruption = False
        self.commands.motion.pose_range = {}
        self.commands.motion.velocity_range = {}
        self.commands.motion.joint_position_range = (0.0, 0.0)


class SitFormalSeatPlayEnvCfg(PlayMixin, SitFormalSeatEnvCfg):
    pass


class StandFormalSeatPlayEnvCfg(PlayMixin, StandFormalSeatEnvCfg):
    pass
