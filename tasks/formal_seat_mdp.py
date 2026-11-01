# 本ファイルは構成を説明するための骨組みである。報酬の重み・接触条件・終了条件の値・参照生成のパラメータは非公開(None のプレースホルダ)で、このままでは学習結果を再現できない。© 2026 okakei. All rights reserved.
# This file is a skeleton that describes the structure only. Reward weights, contact conditions, termination values and reference-generation parameters are withheld (None placeholders), so the learned results cannot be reproduced as is. © 2026 okakei. All rights reserved.
"""座位目標段階の報酬項・終了条件・補助制御(Isaac Lab / PyTorch)。

姿勢の罰は座位窓だけで働く(起立・着座の動作中は上体の前傾を妨げない)。
  着座: 参照 frame [440, 540)(末尾の保持を含む)
  起立: 参照 frame [0, 50)(開始の保持を含む)
腕関節の追従と手首の罰の「座位窓」は 着座 [390, 540)、起立 [0, 100)。
目標は `motions/formal_seat_pose.json` の式典座位(env 原点を足して使う)。

補助制御(ARM_PD_BLEND): 起立の座位窓 [0, 100) で腕 14 関節の PD 目標を
  目標 = (1 - a) * 方策の目標 + a * 参照の関節角
とする。a = ARM_PD_BLEND(使用値 0.7)、窓の末尾 15 frame で smoothstep により 0 へ戻し、
窓の外では方策だけが腕を動かす。学習・再生の両方に適用する。
"""
import torch

from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction

# 座位窓(参照 frame)
SEGMENTS = {
    'sit': dict(relaxed_window=(440, 540), seated_window=(390, 540)),
    'stand': dict(relaxed_window=(0, 50), seated_window=(0, 100)),
}
# 学習に用いた値は非公開。None のままでは動かない。
# 式典座位の骨盤位置(車いす座標系)と、罰の尺度に使う許容幅。評価用の基準は scripts/eval_formal_seat.py にある。
P = dict(
    pelvis_target=None,   # (x, y, z) [m]
    pelvis_xy_tol_m=None, pelvis_z_tol_m=None,
    torso_pitch_abs_max_deg=None, torso_yaw_abs_max_deg=None,
    shoulder_dx_max_m=None,
)

_CACHE = {}

# 学習に用いた値は非公開。None のままでは動かない。
CLAMP_MAX = None   # 座位目標の罰の上限


def _data(env, task):
    key = (task, str(env.device))
    if key not in _CACHE:
        robot = env.scene['robot']
        names = list(robot.body_names)
        seg = SEGMENTS[task]
        _CACHE[key] = dict(
            window=seg['relaxed_window'], seated=seg['seated_window'], crit=P,
            pelvis=torch.tensor(P['pelvis_target'], device=env.device, dtype=torch.float32),
            torso=names.index('torso_link'),
            shoulders=[names.index('left_shoulder_roll_link'), names.index('right_shoulder_roll_link')],
            arm_joints=[i for i, n in enumerate(robot.joint_names)
                        if any(k in n for k in ('shoulder', 'elbow', 'wrist'))])
    return _CACHE[key]


def _active(env, d, command_name):
    t = env.command_manager.get_term(command_name).time_steps
    return ((t >= d['window'][0]) & (t < d['window'][1])).float()


def _seated(env, d, command_name):
    t = env.command_manager.get_term(command_name).time_steps
    return ((t >= d['seated'][0]) & (t < d['seated'][1])).float()


def _rot(q):
    q = q/torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w), 2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], -1).reshape(*q.shape[:-1], 3, 3)


def pelvis_error(env, d):
    e = env.scene['robot'].data.root_pos_w-env.scene.env_origins-d['pelvis']
    return torch.linalg.vector_norm(e[:, :2], dim=-1), e[:, 2].abs()


def torso_angles_deg(env, d):
    r = _rot(env.scene['robot'].data.body_quat_w[:, d['torso']])
    pitch = torch.rad2deg(torch.atan2(r[:, 0, 2], r[:, 2, 2]))
    yaw = torch.rad2deg(torch.atan2(r[:, 1, 0], r[:, 0, 0]))
    return pitch, yaw


def shoulder_dx(env, d):
    p = env.scene['robot'].data.body_pos_w[:, d['shoulders']]
    return (p[:, 0, 0]-p[:, 1, 0]).abs()


def pelvis_penalty(env, task, command_name='motion'):
    """骨盤: min((xy/許容)^2 + (z/許容)^2, 上限)。座位窓。"""
    d = _data(env, task); c = d['crit']
    xy, z = pelvis_error(env, d)
    value = ((xy/c['pelvis_xy_tol_m'])**2+(z/c['pelvis_z_tol_m'])**2).clamp(max=CLAMP_MAX)
    return value*_active(env, d, command_name)


def torso_penalty(env, task, command_name='motion'):
    """上体: min((pitch/許容)^2, 上限) + min((腰ヨー/許容)^2, 上限)。座位窓。"""
    d = _data(env, task); c = d['crit']
    pitch, yaw = torso_angles_deg(env, d)
    value = (((pitch/c['torso_pitch_abs_max_deg'])**2).clamp(max=CLAMP_MAX)
             + ((yaw/c['torso_yaw_abs_max_deg'])**2).clamp(max=CLAMP_MAX))
    return value*_active(env, d, command_name)


def shoulder_penalty(env, task, command_name='motion'):
    """肩の左右差: min((Δx/許容)^2, 上限)。座位窓。"""
    d = _data(env, task); c = d['crit']
    return ((shoulder_dx(env, d)/c['shoulder_dx_max_m'])**2).clamp(max=CLAMP_MAX)*_active(env, d, command_name)


def arm_sq_penalty(env, task, motion_scale, command_name='motion'):
    """腕 14 関節の追従: mean_j (q_j - q*_j)^2 [rad^2]。上限なし。
    座位窓で係数 1、それ以外の全相で係数 motion_scale。
    exp 型の追従項は誤差が大きいと勾配が 0 になるため、二乗の平均にした。"""
    d = _data(env, task)
    cmd = env.command_manager.get_term(command_name)
    j = d['arm_joints']
    err = ((env.scene['robot'].data.joint_pos[:, j]-cmd.joint_pos[:, j])**2).mean(-1)
    seated = _seated(env, d, command_name)
    return err*(seated+(1-seated)*motion_scale)


_THIGH = {}


def _thigh_ids(env, cmd):
    key = id(cmd)
    if key not in _THIGH:
        rb = list(env.scene['robot'].body_names); cb = list(cmd.cfg.body_names)
        pairs = [('left_wrist_yaw_link', 'left_hip_roll_link'), ('right_wrist_yaw_link', 'right_hip_roll_link')]
        _THIGH[key] = ([rb.index(w) for w, _ in pairs], [rb.index(h) for _, h in pairs],
                       [cb.index(w) for w, _ in pairs], [cb.index(h) for _, h in pairs])
    return _THIGH[key]


def _local(pw, pt, qt):
    return torch.einsum('...ji,...j->...i', _rot(qt), pw-pt)


def wrist_thigh_distance(env, command_name='motion'):
    """手首の位置を腿(hip_roll_link)座標系で見たときの、参照との距離 [m]。左右の平均。"""
    cmd = env.command_manager.get_term(command_name)
    rw, rt, cw, ct = _thigh_ids(env, cmd)
    data = env.scene['robot'].data
    cur = _local(data.body_pos_w[:, rw], data.body_pos_w[:, rt], data.body_quat_w[:, rt])
    ref = _local(cmd.body_pos_w[:, cw], cmd.body_pos_w[:, ct], cmd.body_quat_w[:, ct])
    return torch.linalg.vector_norm(cur-ref, dim=-1).mean(-1)


def wrist_thigh_penalty(env, task, command_name='motion'):
    """手首の腿相対距離 [m]。上限なし、座位窓のみ。"""
    d = _data(env, task)
    return wrist_thigh_distance(env, command_name)*_seated(env, d, command_name)


def pose_termination(env, task, threshold, command_name='motion'):
    """座位窓で骨盤が式典座位から threshold [m] を超えて離れたら終了。"""
    d = _data(env, task)
    xy, z = pelvis_error(env, d)
    return (torch.sqrt(xy**2+z**2) > threshold) & (_active(env, d, command_name) > 0)


# --- 補助制御: 腕の PD 目標を参照角へ寄せる(起立の座位窓のみ) ---
BLEND_WINDOW = (0, 100)
BLEND_RAMP = 15


def blend_alpha(t, alpha, window=BLEND_WINDOW, ramp=BLEND_RAMP):
    """窓の中で alpha、窓の末尾 ramp frame は smoothstep で 0 へ、窓の外は 0。"""
    lo, hi = window
    s = ((hi-t.float())/ramp).clamp(0, 1)
    s = s*s*(3-2*s)
    return alpha*s*((t >= lo) & (t < hi)).float()


class ArmBlendJointPositionAction(JointPositionAction):
    """腕関節の PD 目標を (1 - a) * 方策 + a * 参照 にする関節位置アクション。"""
    alpha = 0.0
    command_name = 'motion'

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        names = list(self._joint_names)
        self._arm_cols = [i for i, n in enumerate(names) if any(k in n for k in ('shoulder', 'elbow', 'wrist'))]
        robot_names = list(self._asset.joint_names)
        self._arm_robot = [robot_names.index(names[i]) for i in self._arm_cols]

    def process_actions(self, actions):
        super().process_actions(actions)
        if self.alpha <= 0:
            return
        cmd = self._env.command_manager.get_term(self.command_name)
        a = blend_alpha(cmd.time_steps, self.alpha)[:, None]
        ref = cmd.joint_pos[:, self._arm_robot]
        cur = self._processed_actions[:, self._arm_cols]
        self._processed_actions[:, self._arm_cols] = (1-a)*cur+a*ref
