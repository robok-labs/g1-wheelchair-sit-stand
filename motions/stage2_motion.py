#!/usr/bin/env python3
# 本ファイルは構成を説明するための骨組みである。報酬の重み・接触条件・終了条件の値・参照生成のパラメータは非公開(None のプレースホルダ)で、このままでは学習結果を再現できない。© 2026 okakei. All rights reserved.
# This file is a skeleton that describes the structure only. Reward weights, contact conditions, termination values and reference-generation parameters are withheld (None placeholders), so the learned results cannot be reproduced as is. © 2026 okakei. All rights reserved.
"""Stage 2 参照モーション補正(初期の試行): 歩いて接近 -> 転回 -> 着座 / 起立 -> 転回 -> 退去.
採用した退去は 起立 -> 前進 -> 退去(転回なし)で、その参照生成コードは同梱していない。

NumPy のみ。前向き(着座)クリップ全体に対して次を行う。

1. 着座相を Stage 1 の座位姿勢へ剛体整列(yaw 回転 + xy 並進)
2. 接地検出で足の接地区間を求め、接地中は床に固定、遊脚は区間の両端へ滑らかに橋渡し
3. 車いすの部材にぶつかる遊脚だけ、低い摺り足 -> 持ち上げの順で回避
4. 転回の終わりに向けて、Stage 1 と同じ座位姿勢(骨盤位置・上体傾斜・腕)へブレンド
5. 脚だけ IK で解き直す(フレーム数・時間は変えない)
6. 起立->退去は、補正済みの前向きクリップを逆再生して作る
7. 運動学的な検証(FK 一致・関節可動域・速度・接地・車いすとの距離)

使い方:
  python stage2_motion.py --input approach_sit_raw.npz --mode approach_sit \\
      --urdf g1_29dof_rev_1_0.urdf --anchor anchor.json --parts chair_parts.json \\
      --output approach_sit.npz --json approach_sit.json

入力 npz のキー: fps, joint_pos, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w
(50 Hz、29 関節、30 ボディ)。PhysHSI 由来のデータは同梱しない。
anchor.json は Stage 1 の座位基準(pelvis_xyz, pelvis_quat_wxyz, standing_pelvis_y)。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import footclear as g  # noqa: E402  足の FK/IK と車いすとの距離(本リポジトリの自作部分)

VELOCITIES = ('joint_vel', 'body_lin_vel_w', 'body_ang_vel_w')
CHAIR_FEET = None  # 座位の足首目標(2x3)
CHAIR_CENTER = np.array([-.03, 0.])                               # 座面中心 xy
FOOT_VISUAL_MIN = None   # 足の外形ボックス(足首座標系)
FOOT_VISUAL_MAX = None
SOFT_LIMIT = .98                                                  # Isaac Lab の関節ソフト制限率
ARM_IDS = [11, 12, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
ARM_SEATED = None  # 座位の腕の関節角(14 関節)
# 学習・生成に用いた値は非公開。None のままでは動かない。
SEATED_Z = None   # 座位とみなす骨盤高さ [m]
CONTACT_SOLE_M = None; CONTACT_SPEED = None; CONTACT_VZ = None   # 接地判定: 床包絡からの高さ、水平・鉛直速度
READY_DISTANCE = None; READY_YAW_DEG = None   # 転回完了の判定: 座面中心までの距離、向きの誤差
PHASE_DISTANCE = None; PHASE_Z = None   # 着座相の開始の判定
STATIONARY_ROOT_M = None; STATIONARY_FOOT_M = None   # 歩いていない参照の判定
SOLE_OFFSET = None; GROUND_EPS = None   # 足裏球の取り付けオフセット、接地高さの余裕 [m]
FOOT_BOX_GRID = None; FOOT_BOX_CLEAR = None   # 足の外形ボックスの格子間隔、既定の距離 [m]
SHUFFLE_OFFSETS = None; LIFT_RANGE = None   # 遊脚の回避候補: 摺り足の前後量、持ち上げ高さ (start, stop, step)
PULSE_EDGE = None; SHUFFLE_Z = None   # 回避の山の立ち上がり比、摺り足の足首の持ち上げ [m]
DROP_STEP = None   # 骨盤を下げる刻み [m]
VALID_FOOT_CLEARANCE = None; VALID_TERMINATION = None; VALID_FOOT_BOX = None; VALID_SLIP = None   # 検証のしきい値 [m]
GROUND_TOL = None   # 接地の許容 [m]
SEAT_DROP = None; LATERAL_MAX = None   # 座位の骨盤低下、横補正の上限 [m]
SEAT_X = None   # 骨盤 x の補正 (後退量, 山なりの前方量, 山の振幅) [m]
PITCH_DEG = None   # 上体傾斜 (座位, 移乗中の上乗せ, 立位への上乗せ) [deg]
WAIST_CLIP = None   # 腰関節の角度制限 [rad]
FRONT_X = None; TURN_MIN = None   # 正面開始の判定、転回の最小値
ROLL_WIDTH = None; ROLL_QUANTILE = None   # 床包絡の窓幅 [frame] と分位
DROP_ATTEMPTS = None   # 骨盤を下げる試行回数の上限
SMOOTH_RADIUS = None   # 下げ量の包絡の膨張・平滑の半径 [frame]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    """npz を読み、形状・有限性・50 Hz・クォータニオンのノルムを検査する。"""
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k].copy() for k in archive.files}
    data['fps'] = np.atleast_1d(data['fps'])
    n = len(data['joint_pos'])
    expected = {'joint_pos': (n, 29), 'body_pos_w': (n, 30, 3), 'body_quat_w': (n, 30, 4),
                'body_lin_vel_w': (n, 30, 3), 'body_ang_vel_w': (n, 30, 3)}
    if n < 50 or data['fps'].shape != (1,) or abs(float(data['fps'][0]) - 50) > 1e-6:
        raise ValueError('50 フレーム以上かつ 50 Hz ちょうどが必要')
    for key, shape in expected.items():
        if data[key].shape != shape or not np.isfinite(data[key]).all():
            raise ValueError('不正な配列: ' + key)
    if np.max(np.abs(np.linalg.norm(data['body_quat_w'], axis=-1) - 1)) > .001:
        raise ValueError('クォータニオンのノルムが不正')
    return data


def reverse(data):
    """時間反転。速度の符号も反転する。"""
    return {k: (v.copy() if k == 'fps' else v[::-1].copy() * (-1 if k in VELOCITIES else 1))
            for k, v in data.items()}


def yaw(rotation):
    return np.arctan2(rotation[..., 1, 0], rotation[..., 0, 0])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def smooth(a):
    a = np.clip(a, 0, 1)
    return a * a * (3 - 2 * a)


def rotation_log(r):
    """回転行列 -> 回転ベクトル。180 度付近はクォータニオン経由で安定化する。"""
    result = g.rotvec(r)
    near_pi = np.trace(r, axis1=-2, axis2=-1) < -1 + 1e-6
    if np.any(near_pi):
        q = g.matq(r[near_pi])
        q[q[:, 0] < 0] *= -1
        norm = np.linalg.norm(q[:, 1:], axis=1)
        result[near_pi] = q[:, 1:] * (2 * np.arctan2(norm, q[:, 0]) / np.maximum(norm, 1e-12))[:, None]
    return result


def runs(mask):
    """True が連続する区間 [start, stop) の一覧。"""
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def seated_suffix(pos):
    """末尾の座位区間の開始フレーム(骨盤 z が座位の高さ未満で続く最後の区間)。"""
    z = pos[:, 0, 2]
    if z[-1] >= SEATED_Z:
        raise ValueError('前向きクリップは座位の高さ未満の着座で終わる必要がある')
    above = np.flatnonzero(z >= SEATED_Z)
    start = int(above[-1] + 1) if len(above) else 0
    if len(z) - start < 10:
        raise ValueError('座位区間が 0.2 秒未満')
    return start


def align(data, anchor):
    """着座相を Stage 1 の座位に剛体整列する(yaw 回転 + xy 並進)。"""
    pos = data['body_pos_w'].astype(float)
    rot = g.qmat(data['body_quat_w'])
    seated = seated_suffix(pos)
    sample = slice(max(seated, len(pos) - 50), None)
    center = np.median(pos[sample, 0, :2], axis=0)
    headings = yaw(rot[sample, 0])
    heading = np.arctan2(np.sin(headings).mean(), np.cos(headings).mean())
    target_yaw = float(yaw(g.qmat(np.array(anchor['pelvis_quat_wxyz']))))
    angle = float(wrap(target_yaw - heading))
    r = g.axis_rotation([0, 0, 1], angle)
    translation = np.array(anchor['pelvis_xyz'])[:2] - r[:2, :2] @ center
    pos = pos @ r.T
    pos[:, :, :2] += translation
    rot = r @ rot
    result = {k: v.copy() for k, v in data.items()}
    result['body_pos_w'] = pos
    result['body_quat_w'] = g.matq(rot)
    for key in ['body_lin_vel_w', 'body_ang_vel_w']:
        result[key] = data[key] @ r.T
    info = dict(yaw_rotation_rad=angle, translation_xy_m=translation.tolist(),
                representative_source_xy=center.tolist(), representative_source_yaw_rad=float(heading),
                target_xy=anchor['pelvis_xyz'][:2], target_yaw_rad=target_yaw,
                seated_suffix_start=seated, representative_window=[sample.start, len(pos)])
    return result, info


def rolling_quantile(x, width=ROLL_WIDTH, quantile=ROLL_QUANTILE):
    h = width // 2
    return np.array([np.quantile(x[max(0, i - h):min(len(x), i + h + 1)], quantile) for i in range(len(x))])


def alternating_contacts(contact, stop):
    """左右が交互に片足支持になる歩行らしい接地列(各 3 フレーム以上)があるか。"""
    sequence = []
    code = np.where(contact[:, 0] & ~contact[:, 1], 1, np.where(contact[:, 1] & ~contact[:, 0], 2, 0))[:stop]
    for side in [1, 2]:
        for a, b in runs(code == side):
            if b - a >= 3:
                sequence.append((a, side))
    sequence.sort()
    sides = [s for _, s in sequence]
    return len(sides) >= 2 and any(a != b for a, b in zip(sides, sides[1:]))


def contact_targets(data, alignment):
    """接地検出と足首の目標位置・姿勢。

    足裏球の高さが床包絡(局所の下位分位)に近く、足の水平・鉛直速度が小さいフレームを接地とする。
    接地区間は xy を中央値に固定し足裏を平らに置く。遊脚は前後の接地区間のずれを滑らかに橋渡しする。
    最終の接地区間は座位の足位置(CHAIR_FEET)に合わせる。
    """
    pos = data['body_pos_w']
    rot = g.qmat(data['body_quat_w'])
    n = len(pos)
    fps = 50.
    feet = pos[:, g.ANKLES].copy()
    foot_r = rot[:, g.ANKLES].copy()
    sole = g.sphere_centers(pos, rot)[..., 2].min(2) - g.RADIUS
    envelope = rolling_quantile(sole.min(1))
    speed = np.linalg.norm(np.gradient(feet, 1 / fps, axis=0)[:, :, :2], axis=-1)
    vz = np.abs(np.gradient(feet[:, :, 2], 1 / fps, axis=0))
    contact = (sole <= envelope[:, None] + CONTACT_SOLE_M) & (speed <= CONTACT_SPEED) & (vz <= CONTACT_VZ)
    for side in range(2):
        for a, b in runs(contact[:, side]):
            if b - a < 3:
                contact[a:b, side] = False

    root = pos[:, 0]
    distance = np.linalg.norm(root[:, :2] - CHAIR_CENTER, axis=1)
    facing = np.abs(wrap(yaw(rot[:, 0]) - alignment['target_yaw_rad']))
    ready = (distance < READY_DISTANCE) & (facing < np.deg2rad(READY_YAW_DEG))
    bad = np.flatnonzero(~ready)
    settle = int(bad[-1] + 1) if len(bad) else 0   # 転回が終わり座面に正対したフレーム
    seated = alignment['seated_suffix_start']
    if settle > seated:
        raise ValueError('着座前に転回が終わらないため、最終の足位置を固定できない')
    phase_start = int(np.flatnonzero((distance < PHASE_DISTANCE) | (root[:, 2] < PHASE_Z))[0])

    # 着座遷移に届く「最後の」接地区間を選ぶ(それ以前の接地が後の遊脚を飲み込まないように)
    final_starts = []
    for side in range(2):
        options = [(a, b) for a, b in runs(contact[:, side]) if a <= seated and b >= seated - 10]
        if not options:
            raise ValueError(f'着座遷移で信頼できる接地がない: foot={side}')
        start = int(options[-1][0])
        final_starts.append(start)
        contact[start:, side] = True

    stationary = bool(settle == 0
                      and np.linalg.norm(root[:, :2] - root[0, :2], axis=1).max() < STATIONARY_ROOT_M
                      and np.linalg.norm(feet[:, :, :2] - feet[:1, :, :2], axis=-1).max() < STATIONARY_FOOT_M
                      and not alternating_contacts(contact, seated))
    if stationary:  # 歩いていない参照(Stage 1 相当)は常時両足接地とみなす
        contact[:] = True
        final_starts = [0, 0]

    target = feet.copy()
    target_r = foot_r.copy()
    episodes = []
    for side in range(2):
        anchors = []
        for a, b in runs(contact[:, side]):
            terminal = bool(b == n)
            xy = CHAIR_FEET[side, :2] if terminal else np.median(feet[a:b, side, :2], axis=0)
            h = 0. if terminal else float(np.median(np.unwrap(yaw(foot_r[a:b, side]))))
            r = g.axis_rotation([0, 0, 1], h)
            height = SOLE_OFFSET - float((g.SPHERES @ r.T)[:, 2].min())
            p = np.r_[xy, height]
            target[a:b, side] = p
            target_r[a:b, side] = r
            anchors.append((a, b, p, h))
            episodes.append(dict(foot=side, start=int(a), stop=int(b), terminal=terminal,
                                 target_xyz=p.tolist(), yaw_rad=h,
                                 source_xy_correction_max_m=float(np.linalg.norm(feet[a:b, side, :2] - xy, axis=1).max())))
        # 接地中は固定し、ずれは間の遊脚区間だけで解消する
        for a, b in runs(~contact[:, side]):
            prev = next((x for x in reversed(anchors) if x[1] <= a), None)
            nxt = next((x for x in anchors if x[0] >= b), None)
            d0 = prev[2][:2] - feet[prev[1] - 1, side, :2] if prev else np.zeros(2)
            d1 = nxt[2][:2] - feet[nxt[0], side, :2] if nxt else d0
            for i in range(a, b):
                u = (i - a + 1) / (b - a + 1)
                w = float(smooth(u))
                target[i, side, :2] = feet[i, side, :2] + (1 - w) * d0 + w * d1
                raw_yaw = float(yaw(foot_r[i, side]))
                h0 = prev[3] if prev else raw_yaw
                h1 = nxt[3] if nxt else raw_yaw
                h = (raw_yaw
                     + (1 - w) * float(wrap(h0 - float(yaw(foot_r[max(a - 1, 0), side]))))
                     + w * float(wrap(h1 - float(yaw(foot_r[min(b, n - 1), side])))))
                # 遊脚の途中は元のピッチ/ロールを残し、接地の直前直後は平らにする
                flat = g.axis_rotation([0, 0, 1], h)
                tilted = g.axis_rotation([0, 0, 1], h - raw_yaw) @ foot_r[i, side]
                q0, q1 = g.matq(flat), g.matq(tilted)
                if q0 @ q1 < 0:
                    q1 = -q1
                retain = min(1., float(smooth((i - a + 1) / 5)), float(smooth((b - i) / 5)))
                target_r[i, side] = g.qmat((1 - retain) * q0 + retain * q1)
                target[i, side, 2] = feet[i, side, 2] - envelope[i] + GROUND_EPS
                if prev:
                    target[i, side, 2] = prev[2][2] + float(smooth((i - a + 1) / 5)) * (target[i, side, 2] - prev[2][2])
                if nxt:
                    target[i, side, 2] = nxt[2][2] + float(smooth((b - i) / 5)) * (target[i, side, 2] - nxt[2][2])
                bottom = float((g.SPHERES @ target_r[i, side].T)[:, 2].min()) - g.RADIUS
                target[i, side, 2] = max(target[i, side, 2], GROUND_EPS - bottom)
    info = dict(phase_start=phase_start, turn_settled_frame=settle, final_stance_starts=final_starts,
                stationary_reference=stationary, episodes=episodes,
                ground_offset_per_frame_m=envelope.tolist(),
                thresholds=dict(relative_sole_height_m=CONTACT_SOLE_M, xy_speed_m_s=CONTACT_SPEED,
                                vertical_speed_m_s=CONTACT_VZ, min_contact_frames=3))
    return target, target_r, contact, info


def foot_box_clearance(feet, rotations, parts, threshold=None):  # threshold 省略時は FOOT_BOX_CLEAR
    """足の外形ボックス全体と車いす部材との距離の下界(格子 + 1-Lipschitz 評価)。"""
    threshold = FOOT_BOX_CLEAR if threshold is None else threshold
    axes = [np.linspace(a, b, int(np.ceil((b - a) / FOOT_BOX_GRID)) + 1) for a, b in zip(FOOT_VISUAL_MIN, FOOT_VISUAL_MAX)]
    grid = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
    radius = .5 * np.linalg.norm([a[1] - a[0] for a in axes])
    result = np.full((len(feet), 2, len(parts)), np.inf)
    for i in range(len(feet)):
        for side in range(2):
            points = grid @ rotations[i, side].T + feet[i, side]
            lo, hi = points.min(0), points.max(0)
            for j, part in enumerate(parts):
                if part['kind'] == 'box':
                    half = np.abs(part['rotation']) @ np.array(part['half'])
                    a, b = np.array(part['center']) - half, np.array(part['center']) + half
                elif part['kind'] == 'capsule':
                    a = np.minimum(part['a'], part['b']) - part['radius']
                    b = np.maximum(part['a'], part['b']) + part['radius']
                else:  # 車輪(円柱)
                    half = np.array([part['radius'], part['half'], part['radius']])
                    a, b = np.array(part['center']) - half, np.array(part['center']) + half
                bound = np.linalg.norm(np.maximum(np.maximum(a - hi, lo - b), 0))
                if bound >= threshold:
                    result[i, side, j] = bound
                elif bound > 0:
                    result[i, side, j] = max(bound, float(g.point_sdf(points, part).min() - radius))
                else:
                    result[i, side, j] = float(g.point_sdf(points, part).min() - radius)
    return result


def avoid_swing_obstacles(target, target_r, contact, parts):
    """車いすにぶつかる遊脚だけ修正する。接地区間と時間は変えない。

    まず低い摺り足(x を少しずらし、高さを床すれすれに)、だめなら持ち上げを順に試す。
    """
    result = target.copy()
    rot = target_r.copy()
    adjustments = []
    initial = foot_box_clearance(target, target_r, parts)
    for side in range(2):
        for a, b in runs(~contact[:, side]):
            if initial[a:b, side].min() >= GROUND_EPS:
                continue
            u = (np.arange(a, b) - a + 1) / (b - a + 1)
            pulse = smooth(u / PULSE_EDGE) * smooth((1 - u) / PULSE_EDGE)
            candidates = [('low_shuffle', dx) for dx in SHUFFLE_OFFSETS]
            candidates += [('lift', h) for h in np.arange(*LIFT_RANGE)]
            accepted = False
            for method, value in candidates:
                trial = result[a:b].copy()
                rr = rot[a:b].copy()
                if method == 'low_shuffle':
                    trial[:, side, 0] = target[a:b, side, 0] + value * pulse
                    trial[:, side, 2] = CHAIR_FEET[side, 2] + SHUFFLE_Z * pulse
                    rr[:, side] = g.axis_rotation([0, 0, 1], yaw(target_r[a:b, side]))
                else:
                    trial[:, side, 2] = target[a:b, side, 2] + value * pulse
                bound = foot_box_clearance(trial, rr, parts)[:, side]
                if bound.min() >= GROUND_EPS:
                    result[a:b, side] = trial[:, side]
                    rot[a:b, side] = rr[:, side]
                    adjustments.append(dict(
                        foot=side, start=int(a), stop=int(b), method=method,
                        extra_lift_m=float(value) if method == 'lift' else 0.,
                        delta_x_m=float(value) if method == 'low_shuffle' else 0.,
                        before_lower_bound_m=float(initial[a:b, side].min()),
                        after_lower_bound_m=float(bound.min()),
                        max_xyz_change_m=float(np.linalg.norm(trial[:, side] - target[a:b, side], axis=1).max())))
                    accepted = True
                    break
            if not accepted:
                raise ValueError(f'遊脚の回避が解けない: foot={side} frames={a}:{b}')
    final = foot_box_clearance(result, rot, parts)
    if final.min() < GROUND_EPS:
        raise ValueError(f'足ボックスと車いすの距離が {final.min():.6f} m のまま。足位置の再設計が必要')
    return result, rot, adjustments


def solve_legs(model, root_p, root_r, q0, target, target_r, strict=True):
    """両脚(各 6 関節)の減衰最小二乗 IK。全フレームを同時に反復する。"""
    q = q0.copy()
    lower = np.array([j['lower'] for j in model['joints']])
    upper = np.array([j['upper'] for j in model['joints']])
    mid = (lower + upper) / 2
    lower = mid + SOFT_LIMIT * (lower - mid)
    upper = mid + SOFT_LIMIT * (upper - mid)
    chains = [[0, 3, 6, 9, 13, 17], [1, 4, 7, 10, 14, 18]]
    for iteration in range(g.IK_ITERS):
        p, r, axes = g.fk(model, root_p, root_r, q, True)
        maximum = 0.
        for side, chain in enumerate(chains):
            bi = g.ANKLES[side]
            ep = target[:, side] - p[:, bi]
            er = rotation_log(target_r[:, side] @ r[:, bi].transpose(0, 2, 1))
            maximum = max(maximum, float(np.abs(ep).max()), float(np.abs(er).max()))
            jp = np.cross(axes[:, chain], p[:, bi, None, :] - p[:, np.array(chain) + 1])
            j = np.concatenate([jp.transpose(0, 2, 1), g.IK_ORI_W * axes[:, chain].transpose(0, 2, 1)], axis=1)
            jt = j.transpose(0, 2, 1)
            err = np.concatenate([ep, g.IK_ORI_W * er], axis=1)
            step = np.linalg.solve(jt @ j + g.IK_DAMP * np.eye(6), (jt @ err[..., None]))[..., 0]
            step *= np.minimum(1., g.IK_STEP_MAX / np.maximum(np.abs(step).max(1), 1e-12))[:, None]
            q[:, chain] = np.clip(q[:, chain] + step, lower[chain] + g.IK_LIMIT_MARGIN, upper[chain] - g.IK_LIMIT_MARGIN)
        if maximum < g.IK_CONV:
            break
    p, r = g.fk(model, root_p, root_r, q)
    pe = np.linalg.norm(p[:, g.ANKLES] - target, axis=-1)
    re = np.linalg.norm(rotation_log(target_r @ r[:, g.ANKLES].swapaxes(-1, -2)), axis=-1)
    if strict and (pe.max() > g.POS_TOL or re.max() > g.ANG_TOL):
        frame, side = np.unravel_index(pe.argmax(), pe.shape)
        raise ValueError(f'IK が解けない: frame={frame} foot={side} position_m={pe.max():.6g} angle_rad={re.max():.6g}')
    info = dict(iterations=iteration + 1, position_residual_m=float(pe.max()),
                angle_residual_rad=float(re.max()),
                residual_by_frame_m=pe.max(1).tolist(), angle_by_frame_rad=re.max(1).tolist())
    return q, p, r, info


def reachable_root(model, root, root_r, q0, target, target_r):
    """脚が届かないフレームだけ骨盤を一定の刻みで下げ(上限あり)、滑らかな包絡にして解き直す。

    xy と向きは変えない。上限で届かなければエラーにして、無理な姿勢を隠さない。
    """
    required = np.zeros(len(root))
    trial = root.copy()
    for attempt in range(DROP_ATTEMPTS):
        q, p, r, ik = solve_legs(model, trial, root_r, q0, target, target_r, False)
        bad = (np.array(ik['residual_by_frame_m']) > g.POS_TOL) | (np.array(ik['angle_by_frame_rad']) > g.ANG_TOL)
        if not bad.any():
            break
        if attempt == DROP_ATTEMPTS - 1:
            raise ValueError('骨盤の下げ量の上限でも届かない: frames ' + str(np.flatnonzero(bad).tolist()))
        required[bad] += DROP_STEP
        trial = root.copy()
        trial[:, 2] -= required
    # 半径 SMOOTH_RADIUS の膨張 -> 同じ半径の正規化平滑で、必要な下げ量を下回らず階段状にもならない
    envelope = np.array([required[max(0, i - SMOOTH_RADIUS):min(len(root), i + SMOOTH_RADIUS + 1)].max() for i in range(len(root))])
    kernel = np.hanning(2 * SMOOTH_RADIUS + 3)[1:-1]
    kernel /= kernel.sum()
    drop = np.convolve(np.pad(envelope, (SMOOTH_RADIUS, SMOOTH_RADIUS), mode='edge'), kernel, mode='valid')
    final = root.copy()
    final[:, 2] -= drop
    q, p, r, ik = solve_legs(model, final, root_r, q0, target, target_r)
    ik['reachability_extra_drop_m'] = drop.tolist()
    ik['max_extra_drop_m'] = float(drop.max())
    return final, q, p, r, ik


def validate(data, model, parts, contact, chair_mask):
    """運動学的な検証。checks の全項目が True なら kinematic_pass。"""
    q = data['joint_pos'].astype(float)
    p = data['body_pos_w'].astype(float)
    r = g.qmat(data['body_quat_w'])
    fp, fr = g.fk(model, p[:, 0], r[:, 0], q)
    pe = float(np.linalg.norm(fp - p, axis=-1).max())
    re = float(np.linalg.norm(rotation_log(fr @ r.swapaxes(-1, -2)), axis=-1).max())
    d = g.clearance(p, r, parts)
    ground = g.sphere_centers(p, r)[..., 2] - g.RADIUS
    limit = np.array([[j['lower'], j['upper']] for j in model['joints']])
    limits_ok = bool(np.all(q >= limit[:, 0] - 1e-6) and np.all(q <= limit[:, 1] + 1e-6))
    mid = limit.mean(1)
    soft = mid[:, None] + SOFT_LIMIT * (limit - mid[:, None])
    soft_ok = bool(np.all(q >= soft[:, 0] - 1e-6) and np.all(q <= soft[:, 1] + 1e-6))
    expected = g.derivatives(q, p, r, 50)
    velocity_ok = all(np.allclose(data[k], v, atol=g.ANG_TOL, rtol=1e-4) for k, v in zip(VELOCITIES, expected))
    vmax = np.array([j['velocity'] for j in model['joints']])
    ratio = float((np.abs(data['joint_vel']) / vmax).max())
    adjacent_ratio = float((np.abs(np.diff(q, axis=0)) * 50 / vmax).max())

    slips = []
    for side in range(2):
        for a, b in runs(contact[:, side]):
            xy = p[a:b, g.ANKLES[side], :2]
            slips.append(dict(foot=side, start=int(a), stop=int(b),
                              drift_m=float(np.linalg.norm(xy - xy[0], axis=1).max()),
                              speed_max_m_s=float(np.linalg.norm(np.diff(xy, axis=0), axis=1).max() * 50) if b - a > 1 else 0.))
    worst = np.unravel_index(d.argmin(), d.shape)
    chair_min = float(d[chair_mask].min())
    box = foot_box_clearance(p[:, g.ANKLES], r[:, g.ANKLES], parts)
    checks = dict(
        fk_position=pe <= g.POS_TOL, fk_angle=re <= g.ANG_TOL,
        joint_limits=limits_ok, isaac_soft_joint_limits=soft_ok,
        velocity_consistency=velocity_ok,
        joint_velocity_limits=ratio <= 1 + 1e-5, adjacent_joint_velocity_limits=adjacent_ratio <= 1 + 1e-5,
        all_feet_above_floor=bool(ground.min() >= -GROUND_TOL),
        contact_floor=bool(np.abs(ground.min(2)[contact] - GROUND_EPS).max() < GROUND_TOL),
        chair_clearance=chair_min >= VALID_FOOT_CLEARANCE,   # 着座相の部材との距離
        walking_termination_margin=bool(d.min() >= VALID_TERMINATION),  # 全相で学習時の終了条件に掛からない
        foot_visual_box=bool(box.min() >= VALID_FOOT_BOX),
        contact_slip=max(s['drift_m'] for s in slips) <= VALID_SLIP)
    return dict(
        checks=checks, kinematic_pass=all(checks.values()),
        fk_position_error_m=pe, fk_angle_error_rad=re,
        joint_velocity_limit_ratio_max=ratio, adjacent_joint_velocity_limit_ratio_max=adjacent_ratio,
        max_joint_step_rad=float(np.abs(np.diff(q, axis=0)).max()),
        hard_limit_margin_rad=float(np.minimum(q - limit[:, 0], limit[:, 1] - q).min()),
        ground_min_m=float(ground.min()),
        chair_phase_min_clearance_m=chair_min, all_frames_min_clearance_m=float(d.min()),
        walking_min_clearance_m=float(d[~chair_mask].min()) if (~chair_mask).any() else None,
        foot_visual_box_min_clearance_m=float(box.min()),
        worst_frame=int(worst[0]), worst_foot=int(worst[1]), worst_part=parts[worst[3]]['name'],
        stance_episodes=slips, max_stance_drift_m=max(s['drift_m'] for s in slips),
        pelvis_start=p[0, 0].tolist(), pelvis_end=p[-1, 0].tolist())


def process(source, mode, model, parts, anchor):
    """前向きクリップ全体を補正して (出力 dict, レポート dict) を返す。"""
    aligned, alignment = align(source, anchor)
    p = aligned['body_pos_w']
    r = g.qmat(aligned['body_quat_w'])
    fp, fr = g.fk(model, p[:, 0], r[:, 0], aligned['joint_pos'])
    if np.linalg.norm(fp - p, axis=-1).max() > g.POS_TOL or np.linalg.norm(rotation_log(fr @ r.swapaxes(-1, -2)), axis=-1).max() > g.ANG_TOL:
        raise ValueError('入力の FK・関節順・モデルが一致しない')
    target, target_r, contact, phases = contact_targets(aligned, alignment)
    target, target_r, avoidance = avoid_swing_obstacles(target, target_r, contact, parts)

    n = len(p)
    seated = alignment['seated_suffix_start']
    settle = phases['turn_settled_frame']
    z0 = float(np.median(p[max(seated, n - 50):, 0, 2]))  # 座位の骨盤高さ(元)
    z1 = float(np.median(p[:min(25, seated), 0, 2]))      # 歩行時の骨盤高さ(元)
    if z1 - z0 <= SEAT_DROP:
        raise ValueError('Stage 1 互換の高さ補正に必要な高さ差が足りない')
    height_phase = np.maximum((p[:, 0, 2] - z0) / (z1 - z0), 0)
    n_idx = np.arange(n)

    # 骨盤: 歩行中は床の包絡分だけ下げ、転回の終わりへ向けて座位用の低下へブレンド
    blend = smooth((n_idx - (settle - 25)) / 25)
    root = p[:, 0].copy()
    bias = np.maximum(np.array(phases['ground_offset_per_frame_m']) - GROUND_EPS, 0)
    root[:, 2] -= (1 - blend) * bias + blend * SEAT_DROP * height_phase

    # 骨盤 y: 最終の足位置が決まる前後で Stage 1 の立位 y へ寄せる
    final_stance = max(phases['final_stance_starts'])
    lateral_blend = smooth((n_idx - (final_stance - 25)) / 25)
    standing_y = float(np.median(p[final_stance:final_stance + 10, 0, 1]))
    lateral_delta = float(anchor['standing_pelvis_y'] - standing_y)
    if abs(lateral_delta) > LATERAL_MAX:
        raise ValueError('車いす座標系の横補正が上限を超える。整列を見直す')
    root[:, 1] += lateral_blend * lateral_delta * np.minimum(height_phase, 1)

    # 骨盤 x(座位姿勢): 座面への後退を一度だけ適用し、移乗中は前へ山なりに動かす。
    # 後退量は末尾 50 フレームで減衰させ、元クリップの後ろへの動きを残す。
    h = np.clip(height_phase, 0, 1)
    chair_blend = smooth((n_idx - (settle - 25)) / 25)
    age = n_idx[::-1]
    seat_blend = 1 - smooth((age - 50) / 40)
    root[:, 0] += chair_blend * (-SEAT_X[0] * seat_blend - SEAT_X[1] * h + SEAT_X[2] * np.sin(np.pi * h) ** 2)

    # 腰: 上体傾斜(座位、移乗中、立位の 3 値)を腰 3 関節へ配分
    q0 = aligned['joint_pos'].astype(float).copy()
    torso_yaw = yaw(r[:, 9])
    pitch = np.deg2rad(PITCH_DEG[0] + PITCH_DEG[1] * np.sin(np.pi * h) + PITCH_DEG[2] * h)
    desired = g.axis_rotation([0, 0, 1], torso_yaw) @ g.axis_rotation([0, 1, 0], pitch)
    relative = r[:, 0].transpose(0, 2, 1) @ desired
    waist = np.stack([np.arctan2(-relative[:, 0, 1], relative[:, 1, 1]),
                      np.clip(np.arcsin(np.clip(relative[:, 2, 1], -1, 1)), -WAIST_CLIP, WAIST_CLIP),
                      np.clip(np.arctan2(-relative[:, 2, 0], relative[:, 2, 2]), -WAIST_CLIP, WAIST_CLIP)], axis=1)
    q0[:, [2, 5, 8]] += (waist - q0[:, [2, 5, 8]]) * chair_blend[:, None]
    # 腕: 座位では肘掛けに当たらない姿勢へ(立位に近いほど元の腕振りを残す)
    q0[:, ARM_IDS] += (ARM_SEATED - q0[:, ARM_IDS]) * (chair_blend * (1 - smooth(h)))[:, None]

    root, q, pos, rot, ik = reachable_root(model, root, r[:, 0], q0, target, target_r)
    output = dict(fps=np.array([50], dtype=source['fps'].dtype), joint_pos=q.astype('float32'),
                  body_pos_w=pos.astype('float32'), body_quat_w=g.matq(rot).astype('float32'))
    derived = g.derivatives(output['joint_pos'].astype(float), output['body_pos_w'].astype(float),
                            g.qmat(output['body_quat_w']), 50)
    for key, value in zip(VELOCITIES, derived):
        output[key] = value.astype('float32')

    chair_mask = n_idx >= phases['phase_start']
    raw_root = p[:, 0]
    distance = float(np.linalg.norm(raw_root[0, :2] - raw_root[seated, :2]))
    heading = np.unwrap(yaw(r[:, 0]))
    turn = float(np.rad2deg(heading.max() - heading.min()))
    front = bool(raw_root[0, 0] > FRONT_X and raw_root[0, 0] - CHAIR_CENTER[0] > abs(raw_root[0, 1]))
    incoming = bool(np.dot(raw_root[min(25, n - 1), :2] - raw_root[0, :2], CHAIR_CENTER - raw_root[0, :2]) > 0)

    if mode == 'stand_depart':  # 補正済みの前向きクリップを逆再生
        output = reverse(output)
        contact = contact[::-1].copy()
        chair_mask = chair_mask[::-1].copy()

    report = validate(output, model, parts, contact, chair_mask)
    report['kinematic_pass'] = all(report['checks'].values())
    report['swing_avoidance'] = avoidance

    end = output['body_pos_w'][-1, 0]
    if mode == 'approach_sit':
        terminal_pass = bool(.55 <= end[2] <= .68 and np.linalg.norm(end[:2] - CHAIR_CENTER) <= .10)
    else:
        terminal_pass = bool(end[0] - CHAIR_CENTER[0] >= .8 and end[2] >= .75)
    walk_contact = contact[::-1] if mode == 'stand_depart' else contact
    coverage = dict(front_approach=front, incoming_motion=incoming, approach_displacement_m=distance,
                    turn_range_deg=turn, distance_pass=distance >= .8, turn_pass=turn >= TURN_MIN,
                    alternating_walk_contacts=bool(alternating_contacts(walk_contact, min(phases['final_stance_starts']))),
                    terminal_pose_pass=terminal_pass, terminal_hold_target_xyz=end.tolist())
    report.update(mode=mode, frames=n, fps=50, alignment=alignment, contact_detection=phases,
                  coverage=coverage, ik=ik, prefix_added_frames=0,
                  reverse_of_corrected_forward=mode == 'stand_depart',
                  pelvis_lateral_correction_m=lateral_delta,
                  final_velocity_norm_m_s=float(np.linalg.norm(output['body_lin_vel_w'][-1, 0])))
    report['pass_all'] = bool(report['kinematic_pass'] and front and incoming and distance >= .8
                              and turn >= TURN_MIN and coverage['alternating_walk_contacts'] and terminal_pass)
    return output, report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, required=True, help='前向き(着座)クリップの npz')
    parser.add_argument('--mode', choices=['approach_sit', 'stand_depart'], required=True)
    parser.add_argument('--urdf', type=Path, required=True)
    parser.add_argument('--anchor', type=Path, required=True)
    parser.add_argument('--parts', type=Path, default=HERE / 'chair_parts.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--json', type=Path, required=True)
    args = parser.parse_args()
    args.json.parent.mkdir(parents=True, exist_ok=True)
    try:
        source = load(args.input)
        model = g.make_model(args.urdf)
        parts = json.loads(args.parts.read_text())['parts']
        anchor = json.loads(args.anchor.read_text())
        output, report = process(source, args.mode, model, parts, anchor)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.output, **output)
        report.update(input_sha256=sha(args.input), output_sha256=sha(args.output))
        args.json.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        keys = ['mode', 'frames', 'kinematic_pass', 'pass_all', 'coverage',
                'chair_phase_min_clearance_m', 'max_stance_drift_m', 'fk_position_error_m']
        print(json.dumps({k: report[k] for k in keys}, indent=2))
        print('STAGE2_MOTION_PASS' if report['pass_all'] else 'STAGE2_MOTION_NOT_READY')
        return 0 if report['pass_all'] else 2
    except (ValueError, KeyError, FileNotFoundError) as error:
        args.json.write_text(json.dumps(dict(pass_all=False, mode=args.mode, error=str(error)), indent=2) + '\n',
                             encoding='utf-8')
        print('STAGE2_MOTION_FAIL', error)
        return 2


if __name__ == '__main__':
    sys.exit(main())
