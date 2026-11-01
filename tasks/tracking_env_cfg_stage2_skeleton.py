# 本ファイルは構成を説明するための骨組みである。報酬の重み・接触条件・終了条件の値・参照生成のパラメータは非公開(None のプレースホルダ)で、このままでは学習結果を再現できない。© 2026 okakei. All rights reserved.
# This file is a skeleton that describes the structure only. Reward weights, contact conditions, termination values and reference-generation parameters are withheld (None placeholders), so the learned results cannot be reproduced as is. © 2026 okakei. All rights reserved.
"""Stage 2 task skeleton: walk -> turn -> sit  /  stand -> forward -> depart (early trials turned).

Stage 1 の骨組み(`tracking_env_cfg_skeleton.py`)を継承し、Stage 2 で変える点だけを書く。
報酬の重みとしきい値は非公開(W / P はプレースホルダ)。
`.stage2_mdp` 等の自作 MDP 関数・コマンド実装は本リポジトリに含めない(仕様は tasks/README.md)。

gym id の形:
  Unitree-G1-29dof-Mimic-Wheelchair-Approach-Sit-Skeleton     (着座: 歩いて接近 -> 転回 -> 着座)
  Unitree-G1-29dof-Mimic-Wheelchair-Stand-Depart-Skeleton     (起立: 起立 -> 前進 -> 退去。初期の試行は転回あり)
"""
from __future__ import annotations

import json
import os

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

# 自作(非同梱): 静止開始率つきコマンド、世界座標 xy 罰、着座/退去の相別の罰、足・上体と車いすの距離罰
from .stage2_mdp import (
    Stage2MotionCommandCfg,
    clip_complete,
    foot_chair_penalty,
    foot_chair_termination,
    root_xy_error_squared,
    root_xy_termination,
    seated_xy_error_squared,
    terminal_height_error_squared,
    upper_chair_penalty,
)
from .tracking_env_cfg_skeleton import RobotEnvCfg as Stage1RobotEnvCfg

# 学習に用いた値は非公開。W は報酬の重み、P は係数・しきい値・外乱などの調整値。
# None のままでは動かない。使う場合は各自の設定で値を入れる。
W = {
    "root_xy_world": None,
    "foot_chair_margin": None,
    "upper_chair_margin": None,
    "seated_xy": None,
    "terminal_height": None,
}
P = {
    "foot_chair_margin_m": None,
    "upper_chair_margin_m": None,
    "root_xy_term_m": None,
    "foot_chair_term_m": None,
    "reset_pose_range": None,
    "reset_joint_position_range": None,
}

HERE = os.path.dirname(__file__)
# motions/stage2_motion.py が出すレポート(--json)を validation_motion.json として task 直下に置く
with open(os.path.join(HERE, "validation_motion.json"), encoding="utf-8") as _f:
    VALIDATION = json.load(_f)
SEATED_START_FRAME = VALIDATION["alignment"]["seated_suffix_start"]  # 着座相の開始フレーム
SEATED_XY_TARGET = None  # 着座相の骨盤 xy 目標(車いす座標系、座面付近。値は非公開)


@configclass
class Stage2EnvCfg(Stage1RobotEnvCfg):
    """共通部分(着座・起立で同じ)。"""

    def __post_init__(self):
        super().__post_init__()
        parts = os.path.join(HERE, "chair_parts.json")

        # --- コマンド: 参照は可変長、先頭の一定割合は必ずフレーム 0 から開始(値は非公開)、末尾は 50 step 保持して終了
        self.commands.motion = Stage2MotionCommandCfg(
            asset_name="robot",
            motion_file=os.path.join(HERE, "motion.npz"),
            anchor_body_name="torso_link",
            resampling_time_range=(1.0e9, 1.0e9),
            minimum_start_fraction=None,  # 静止開始率(値は非公開)
            hold_frames=50,
            pose_range=P["reset_pose_range"],
            velocity_range={},
            joint_position_range=P["reset_joint_position_range"],
            body_names=self.commands.motion.body_names,
        )

        # --- シーン/エピソード
        self.scene.env_spacing = 4.0  # 歩行分の広さ
        self.scene.robot.soft_joint_pos_limit_factor = 0.98  # 参照はこの範囲に収まる(既定 0.90 だと転回姿勢が切れる)
        self.episode_length_s = VALIDATION["frames"] / 50 + 1.04  # 参照の長さ + 保持 50 step + 余裕
        self.events.add_joint_default_pos = None
        self.events.base_com = None
        self.events.push_robot = None  # 押し外乱は無効(まず公称軌道の完走を優先)

        # --- 世界座標(車いす固定)の骨盤 xy 誤差: 罰 + 大きく外れたら終了
        self.rewards.root_xy_world = RewTerm(
            func=root_xy_error_squared, weight=W["root_xy_world"], params={"command_name": "motion"}
        )
        self.terminations.root_xy_world = DoneTerm(
            func=root_xy_termination, params={"command_name": "motion", "threshold": P["root_xy_term_m"]}
        )

        # --- 全相での車いすとの距離: 足の外形ボックス(余裕の外で罰、極小距離で終了)、上体(余裕の外で罰)
        self.rewards.foot_chair_margin = RewTerm(
            func=foot_chair_penalty, weight=W["foot_chair_margin"], params={"geometry_path": parts, "margin": P["foot_chair_margin_m"]}
        )
        self.terminations.foot_chair = DoneTerm(
            func=foot_chair_termination, params={"geometry_path": parts, "threshold": P["foot_chair_term_m"]}
        )
        self.rewards.upper_chair_margin = RewTerm(
            func=upper_chair_penalty,
            weight=W["upper_chair_margin"],
            params={"geometry_path": os.path.join(HERE, "upper_geometry.json"), "margin": P["upper_chair_margin_m"]},
        )

        # --- 終了: 手首の高さ偏差は外し(腕の接触で決まるため)、足首のみ。クリップ完了でタイムアウト
        self.terminations.ee_body_pos.params["body_names"] = ["left_ankle_roll_link", "right_ankle_roll_link"]
        self.terminations.clip_complete = DoneTerm(func=clip_complete, time_out=True)


@configclass
class ApproachSitEnvCfg(Stage2EnvCfg):
    """着座: 着座相(座位区間の手前 25 frame かけて立ち上げ)の骨盤 xy を座面付近の目標へ寄せる。"""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.seated_xy = RewTerm(
            func=seated_xy_error_squared,
            weight=W["seated_xy"],
            params={"start_frame": SEATED_START_FRAME, "ramp_frames": 25, "target_xy": SEATED_XY_TARGET},
        )


@configclass
class StandDepartEnvCfg(Stage2EnvCfg):
    """起立+退去: 退去終端の骨盤高さ(参照との z 誤差の二乗)を保持相の 10 step かけて罰する。"""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.terminal_height = RewTerm(
            func=terminal_height_error_squared, weight=W["terminal_height"], params={"ramp_steps": 10}
        )


class PlayMixin:
    """再生用: 1 env、常にフレーム 0 から、初期化ノイズなし、クリップ完了で終わらない。"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1e9
        self.commands.motion.start_at_zero = True
        self.terminations.clip_complete = None
        self.observations.policy.enable_corruption = False
        self.commands.motion.pose_range = {}
        self.commands.motion.joint_position_range = (0.0, 0.0)


class ApproachSitPlayEnvCfg(PlayMixin, ApproachSitEnvCfg):
    pass


class StandDepartPlayEnvCfg(PlayMixin, StandDepartEnvCfg):
    pass
