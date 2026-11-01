#!/usr/bin/env python3
# 本ファイルは構成を説明するための骨組みである。報酬の重み・接触条件・終了条件の値・参照生成のパラメータは非公開(None のプレースホルダ)で、このままでは学習結果を再現できない。© 2026 okakei. All rights reserved.
# This file is a skeleton that describes the structure only. Reward weights, contact conditions, termination values and reference-generation parameters are withheld (None placeholders), so the learned results cannot be reproduced as is. © 2026 okakei. All rights reserved.
"""座位目標段階の参照生成: 式典座位(formal seated pose)に終わる着座 / 始まる起立の参照.

NumPy と SciPy のみ。入力は `footclear.py` で足を補正した起立・着座クリップ(npz)。
PhysHSI 由来のデータは同梱しない。式典座位は `motions/formal_seat_pose.json`。

式典座位(目標姿勢): 足裏接地・上体鉛直・腰 0・手は腿の上(学習結果では手は腿の上方に留まった)。関節の目標角を使い、骨盤は
`pelvis_initial_pos`・姿勢は単位行列に置く。

全フレームを左右対称にする。
  骨盤 y = 0、骨盤 yaw/roll = 0(x, z, pitch は人の動作のまま)
  腰 yaw/roll = 0、腕の関節は左右の平均
  脚は IK で解き直し、両足を平置きに保つ
着座 = 直立静止 50f -> 直立から人の着座へ 50f -> 人の着座 290f -> 式典座位へ 50f -> 式典座位の静止 100f (540f)
起立 = 式典座位の静止 50f -> 人の起立へ 50f -> 人の起立 290f -> 直立へ 50f -> 直立の静止 50f (490f)
ブレンドは関節・骨盤位置が線形、骨盤姿勢が slerp、いずれも smoothstep。
足は人の x から式典座位の x へ滑らせる。腕は肘掛けの手前で内側へ寄せてから下ろす。

最後に `naturalize_arms` で動作区間の腕を直立の体側 <-> 式典座位の間で補間し、
人の肩ピッチ・肘の動きを一部だけ重ねる(前へ伸ばしたままにしない)。

使い方:
  python motions/formal_seat.py --sit sit_human.npz --stand stand_human.npz \\
      --urdf g1_29dof_rev_1_0.urdf --pose motions/formal_seat_pose.json --out-dir out/
入力 npz のキー: fps, joint_pos, body_pos_w, body_quat_w(50 Hz、29 関節、30 ボディ)。
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import footclear as g

BLEND = 50
STILL_SIT = 100
STILL_STAND = 50
STAND_BLEND = 50
STAND_STILL = 50
# 学習・生成に用いた値は非公開。None のままでは動かない。
HALF_STANCE = None        # 足の半歩隔 [m]
ANKLE_Z = None            # 平置きの足首高さ [m]
HUMAN_FOOT_X = None       # 人の足首 x [m]
FORMAL_FOOT_X = None      # 式典座位の足首 x(実測)[m]
UPRIGHT_Z = None          # 直立の骨盤高さ [m]
SIT_UPRIGHT_X = None      # 着座クリップ先頭の直立姿勢の骨盤 x [m]
POSE_KEY = 'pose_feet_on_floor'
WRIST_ABS_Y = None        # 手首 |y| の最適化目標 [m]

# (左, 右, 符号): 左右対称化する関節の対
PAIRS = [(0, 1, 1), (3, 4, -1), (6, 7, -1), (9, 10, 1), (13, 14, 1), (17, 18, -1),
         (11, 12, 1), (15, 16, -1), (19, 20, -1), (21, 22, 1), (23, 24, -1), (25, 26, 1), (27, 28, -1)]
ARM_INWARD = [15, 16, 19, 20]      # 肩 roll / yaw
ARM_LOWER = [11, 12, 21, 22]       # 肩 pitch / 肘

# naturalize_arms の設定
ARM_IDS = [11, 12, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
ARM_AMP = None                # 残す人の腕の動きの振幅(自身の線形トレンドからの差の比)
SH_PITCH_MIN = None           # 肩の前方挙上の上限 [rad、前方が負]
ELBOW_MOTION = None           # 動作中の肘関節角の範囲 (下限, 上限) [rad]
ARM_VMAX = None               # 腕の関節速度の上限 [rad/s]
ARM_TAPER = None              # 人の腕の動きを重ねる両端のテーパ [frame]
SIGMA_MAX = None              # 腕の平滑の打ち切り幅 [frame]
IK_POS_W = None; IK_REG_W = None; IK_PALM_W = None   # 腕 IK の最適化係数(位置、正則化、手のひらの向き)
NEUTRAL_REG_W = None          # 直立の腕 IK の正則化係数
JOINT_MARGIN = None           # 腕 IK で関節限界から空ける余裕 [rad]
COM_ITERS = None; COM_TOL = None   # 直立の骨盤 x を重心から決める反復数と許容 [m]
SIT_OUTWARD_WARP = None       # 着座で腕を開く区間(ブレンド比)
NEUTRAL_WRIST_OFFSET = None   # 直立の手首を肩からどこへ置くか (x, y, z) [m]
INWARD_WARP = None            # 腕を内側へ寄せる区間 (開始, 終了)(ブレンド比)
LOWER_WARP = None             # 腕を下ろす区間 (開始, 終了)(ブレンド比)
OUTWARD_LEAD = None           # 起立で腕を外へ開き始める、人の動作終端からの前倒し [frame]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def smooth(n):
    s = np.arange(1, n+1, dtype=float)/(n+1)
    return s*s*(3-2*s)


def warp(t, start, end):
    u = np.clip((t-start)/(end-start), 0, 1)
    return u*u*(3-2*u)


def slerp(r0, r1, s):
    v = g.rotvec((r0.T@r1)[None])[0]
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.repeat(r0[None], len(s), 0)
    return np.stack([r0@g.axis_rotation(v/n, n*t) for t in s])


def feet_targets(n, foot_x):
    x = np.broadcast_to(np.asarray(foot_x, float), (n,))
    t = np.zeros((n, 2, 3)); t[:, :, 0] = x[:, None]
    t[:, 0, 1], t[:, 1, 1], t[:, :, 2] = HALF_STANCE, -HALF_STANCE, ANKLE_Z
    return t


def mirror_average(q):
    """左右の関節を平均して対称にし、腰 yaw/roll を 0 にする。"""
    q = np.array(q, float, copy=True)
    for l, r, sign in PAIRS:
        m = (q[..., l]+sign*q[..., r])/2
        q[..., l], q[..., r] = m, sign*m
    q[..., 2] = q[..., 5] = 0.
    return q


def pitch_only(rot):
    pitch = np.arctan2(rot[..., 0, 2], rot[..., 2, 2])
    return g.axis_rotation([0, 1, 0], pitch)


def leg_ik(model, root_p, root_r, q0, foot_x):
    q, _, _ = g.solve_legs(model, root_p, root_r, q0, feet_targets(len(q0), foot_x))
    return q


def joint_vector(model, joints):
    return np.array([float(joints[j['name']]) for j in model['joints']])


def load_formal(path, model):
    """式典座位: 関節目標 + pelvis_initial_pos + 単位姿勢を左右対称化し、脚と腕を解き直す。"""
    pose = json.loads(Path(path).read_text(encoding='utf-8'))[POSE_KEY]
    q_target = joint_vector(model, pose['joint_pos_target_rad'])
    root_p = np.asarray(pose['pelvis_initial_pos'], float).copy(); root_p[1] = 0.
    root_r = np.eye(3)
    q = mirror_average(q_target)
    q = leg_ik(model, root_p[None], root_r[None], q[None], FORMAL_FOOT_X)[0]
    q = arms_inward(model, root_p, root_r, q)
    return dict(q=q, root_p=root_p, root_r=root_r, foot_x=FORMAL_FOOT_X, source_sha256=sha(path))


def arms_inward(model, root_p, root_r, q, abs_y=WRIST_ABS_Y):
    """手首の x, z は式典座位のまま |y| だけ内側へ寄せる。左腕を解き、右腕は鏡像。"""
    from scipy.optimize import minimize
    ids = [11, 15, 19, 21, 23, 25, 27]
    p0, r0 = g.fk(model, root_p[None], root_r[None], q[None])
    target = p0[0, 28].copy(); target[1] = abs_y
    palm = r0[0, 28][:, 2]   # 手のひらの向きを保つ

    def cost(x):
        qq = q.copy(); qq[ids] = x
        pp, rr = g.fk(model, root_p[None], root_r[None], qq[None])
        return (np.sum((pp[0, 28]-target)**2)*IK_POS_W+np.sum((x-q[ids])**2)*IK_REG_W
                + IK_PALM_W*np.sum((rr[0, 28][:, 2]-palm)**2))
    lo = np.array([model['joints'][i]['lower'] for i in ids])+JOINT_MARGIN
    hi = np.array([model['joints'][i]['upper'] for i in ids])-JOINT_MARGIN
    res = minimize(cost, q[ids], method='L-BFGS-B', bounds=list(zip(lo, hi)))
    out = q.copy(); out[ids] = res.x
    return mirror_average(out)


def human(model, d):
    """人の参照を左右対称にし、足を平置きの位置に固定する。"""
    q = mirror_average(d['joint_pos'].astype(float))
    root_p = d['body_pos_w'][:, 0].astype(float).copy(); root_p[:, 1] = 0.
    root_r = pitch_only(g.qmat(d['body_quat_w'][:, 0]))
    q = leg_ik(model, root_p, root_r, q, HUMAN_FOOT_X)
    return q, root_p, root_r


def blend(model, a, b, foot_a, foot_b, to_formal=True):
    """a, b = (q, root_p, root_r)。間の BLEND frame を作る。
    式典座位へ向かうときは腕を内側(roll/yaw)へ寄せてから下ろし(pitch/肘)、離れるときは逆順。"""
    s = smooth(BLEND)
    t = np.arange(1, BLEND+1, dtype=float)/(BLEND+1)
    w = np.repeat(s[:, None], 29, 1)
    first, second = (ARM_INWARD, ARM_LOWER) if to_formal else (ARM_LOWER, ARM_INWARD)
    w[:, first] = warp(t, *INWARD_WARP)[:, None]
    w[:, second] = warp(t, *LOWER_WARP)[:, None]
    q = a[0][None]*(1-w)+b[0][None]*w
    root_p = a[1][None]*(1-s[:, None])+b[1][None]*s[:, None]
    root_r = slerp(a[2], b[2], s)
    return leg_ik(model, root_p, root_r, q, foot_a*(1-s)+foot_b*s), root_p, root_r


def assemble(model, frames, fps):
    q = np.concatenate([f[0] for f in frames]).astype(float)
    root_p = np.concatenate([f[1] for f in frames]).astype(float)
    root_r = np.concatenate([f[2] for f in frames]).astype(float)
    pos, rot = g.fk(model, root_p, root_r, q)
    d = dict(fps=np.asarray(fps).copy(), joint_pos=q.astype('float32'), body_pos_w=pos.astype('float32'),
             body_quat_w=g.matq(rot).astype('float32'))
    for key, value in zip(['joint_vel', 'body_lin_vel_w', 'body_ang_vel_w'],
                          g.derivatives(d['joint_pos'].astype(float), d['body_pos_w'].astype(float),
                                        g.qmat(d['body_quat_w']), float(d['fps'][0]))):
        d[key] = value.astype('float32')
    return d


def still(frame, n):
    return tuple(np.repeat(np.asarray(x)[None], n, 0) for x in frame)


def concat(frames):
    return tuple(np.concatenate([f[i] for f in frames]).astype(float) for i in range(3))


def neutral_arms(model, root_p, root_r, q):
    """腕を体側に下ろす。左腕を最適化(手首を肩の下、わずかに外側)し、右腕は鏡像。"""
    from scipy.optimize import minimize
    ids = [11, 15, 19, 21]
    wrist = [23, 25, 27]
    q = q.copy(); q[wrist] = 0.
    p0, _ = g.fk(model, root_p[None], root_r[None], q[None])
    target = p0[0, 16]+np.asarray(NEUTRAL_WRIST_OFFSET)

    def cost(x):
        qq = q.copy(); qq[ids] = x
        pp, _ = g.fk(model, root_p[None], root_r[None], qq[None])
        return np.sum((pp[0, 28]-target)**2)*IK_POS_W+np.sum(x**2)*NEUTRAL_REG_W
    lo = np.array([model['joints'][i]['lower'] for i in ids])+JOINT_MARGIN
    hi = np.array([model['joints'][i]['upper'] for i in ids])-JOINT_MARGIN
    res = minimize(cost, np.clip(q[ids], lo, hi), method='L-BFGS-B', bounds=list(zip(lo, hi)))
    q[ids] = res.x
    return mirror_average(q)


def upright_pose(model, end_q, fixed_x=None):
    """直立の静止姿勢: 骨盤水平、腰 0、足は人の足 x。
    骨盤 x は fixed_x、なければ重心が足裏接触球の中心に来る位置。"""
    if fixed_x is not None:
        root_p = np.array([fixed_x, 0., UPRIGHT_Z]); root_r = np.eye(3)
        q = mirror_average(end_q); q[8] = 0.
        q = leg_ik(model, root_p[None], root_r[None], q[None], HUMAN_FOOT_X)[0]
        return neutral_arms(model, root_p, root_r, q), root_p, root_r
    q = mirror_average(end_q); q[8] = 0.
    root_r = np.eye(3)
    centre = HUMAN_FOOT_X+float(g.SPHERES[:, 0].mean())
    x = HUMAN_FOOT_X
    for _ in range(COM_ITERS):
        root_p = np.array([x, 0., UPRIGHT_Z])
        q = leg_ik(model, root_p[None], root_r[None], q[None], HUMAN_FOOT_X)[0]
        q = neutral_arms(model, root_p, root_r, q)
        pos, rot = g.fk(model, root_p[None], root_r[None], q[None])
        err = centre-float(g.center_of_mass(model, pos, rot)[0, 0])
        if abs(err) < COM_TOL:
            break
        x += err
    return q, root_p, root_r


def taper(n, k=ARM_TAPER):
    w = np.ones(n)
    r = .5-.5*np.cos(np.pi*np.arange(1, k+1)/(k+1))
    w[:k] = r; w[-k:] = np.minimum(w[-k:], r[::-1])
    return w


def gauss_smooth(x, sigma, pad_a, pad_b):
    """axis 0 のガウス平滑(両端は静止窓の値で埋める)。"""
    if sigma <= 0:
        return x
    r = int(3*sigma)+1
    k = np.exp(-.5*(np.arange(-r, r+1)/sigma)**2); k /= k.sum()
    xp = np.concatenate([np.repeat(pad_a[None], r, 0), x, np.repeat(pad_b[None], r, 0)])
    return np.stack([np.convolve(xp[:, j], k, mode='valid') for j in range(x.shape[1])], 1)


def naturalize_arms(q, root_p, region, human_seg, arm_a, arm_b, rising, fps):
    """region [a, b) の腕関節を arm_a -> arm_b へ骨盤高さの進捗で補間する。
    人の腕の動き(線形トレンドからの差)を ARM_AMP 倍で重ね、肩 pitch と肘を制限し、
    左右平均したうえで、腕の関節速度が ARM_VMAX 以下になるまでガウス平滑する。"""
    q = q.copy(); a, b = region; h0, h1 = human_seg
    z = root_p[a:b, 2]
    p = (z-z[0])/(z[-1]-z[0]) if rising else (z[0]-z)/(z[0]-z[-1])
    p = np.maximum.accumulate(np.clip(p, 0, 1)); p = p*p*(3-2*p)
    # 内外へ動く関節(肩 roll/yaw・手首)は手が肘掛けから離れている間に動かす。
    inward = [ARM_IDS.index(i) for i in (15, 16, 19, 20, 23, 24, 25, 26, 27, 28)]
    pj = np.repeat(p[:, None], len(ARM_IDS), 1)
    idx = np.arange(b-a, dtype=float)
    pj[:, inward] = (warp(idx, h1-a-OUTWARD_LEAD, b-a-1.) if rising else warp(p, *SIT_OUTWARD_WARP))[:, None]
    base = arm_a[None]+pj*(arm_b-arm_a)[None]
    qh = q[h0:h1][:, ARM_IDS]
    lin = qh[0][None]+np.linspace(0, 1, h1-h0)[:, None]*(qh[-1]-qh[0])[None]
    w = taper(h1-h0)[:, None]
    swing = [ARM_IDS.index(i) for i in (11, 12, 21, 22)]   # 前後の振りだけ(肩 pitch, 肘)
    seg = base[h0-a:h1-a].copy(); seg[:, swing] += ARM_AMP*((qh-lin)*w)[:, swing]
    el = [ARM_IDS.index(21), ARM_IDS.index(22)]
    cw = w*np.sin(np.pi*p[h0-a:h1-a])[:, None]**2   # 骨盤が動いている間だけ肘の範囲を制限
    seg[:, el] = (1-cw)*seg[:, el]+cw*np.clip(seg[:, el], *ELBOW_MOTION)
    base[h0-a:h1-a] = seg
    sp = [ARM_IDS.index(11), ARM_IDS.index(12)]
    base[:, sp] = np.maximum(base[:, sp], SH_PITCH_MIN)
    q[a:b, ARM_IDS] = base
    q = mirror_average(q)
    raw = q[a:b, ARM_IDS].copy(); sigma = 0.
    while True:
        sm = gauss_smooth(raw, sigma, arm_a, arm_b)
        full = np.concatenate([arm_a[None], sm, arm_b[None]])
        vmax = float(np.abs(np.diff(full, axis=0)).max()*fps)
        if vmax <= ARM_VMAX or sigma > SIGMA_MAX:
            break
        sigma += 1.
    q[a:b, ARM_IDS] = sm
    print('naturalize_arms region', region, 'human', human_seg, 'sigma', sigma, 'arm_vmax_rad_s', round(vmax, 3))
    return mirror_average(q), dict(sigma=sigma, arm_vmax_rad_s=vmax)


def build_sit(model, sit, formal):
    """直立静止 -> 直立から人の着座へ -> 人の着座 -> 式典座位へ -> 式典座位の静止。"""
    hq, hp, hr = human(model, sit)
    f = (formal['q'], formal['root_p'], formal['root_r'])
    up = upright_pose(model, hq[0], SIT_UPRIGHT_X)
    s = smooth(STAND_BLEND)
    uq = up[0][None]*(1-s[:, None])+hq[0][None]*s[:, None]
    upp = up[1][None]*(1-s[:, None])+hp[0][None]*s[:, None]
    ur = slerp(up[2], hr[0], s)
    uq = leg_ik(model, upp, ur, uq, HUMAN_FOOT_X)
    bq, bp, br = blend(model, (hq[-1], hp[-1], hr[-1]), f, HUMAN_FOOT_X, formal['foot_x'])
    Q, P, R = concat([still(up, STAND_STILL), (uq, upp, ur), (hq, hp, hr), (bq, bp, br), still(f, STILL_SIT)])
    n = len(Q); a = STAND_STILL+STAND_BLEND; b = a+len(hq)
    Q, _ = naturalize_arms(Q, P, (STAND_STILL, n-STILL_SIT), (a, b), up[0][ARM_IDS], f[0][ARM_IDS],
                           False, float(sit['fps'][0]))
    d = assemble(model, [(Q, P, R)], sit['fps'])
    return d, dict(upright_still=[0, STAND_STILL], upright_blend=[STAND_STILL, a], human=[a, b],
                   blend=[b, b+BLEND], still=[n-STILL_SIT, n], relaxed_window=[n-STILL_SIT, n],
                   seated_window=[b, n], sit_start_window=[0, STAND_STILL])


def build_stand(model, stand, formal):
    """式典座位の静止 -> 人の起立へ -> 人の起立 -> 直立へ -> 直立の静止。"""
    hq, hp, hr = human(model, stand)
    f = (formal['q'], formal['root_p'], formal['root_r'])
    bq, bp, br = blend(model, f, (hq[0], hp[0], hr[0]), formal['foot_x'], HUMAN_FOOT_X, to_formal=False)
    up = upright_pose(model, hq[-1])
    s = smooth(STAND_BLEND)
    uq = hq[-1][None]*(1-s[:, None])+up[0][None]*s[:, None]
    upp = hp[-1][None]*(1-s[:, None])+up[1][None]*s[:, None]
    ur = slerp(hr[-1], up[2], s)
    uq = leg_ik(model, upp, ur, uq, HUMAN_FOOT_X)
    Q, P, R = concat([still(f, STILL_STAND), (bq, bp, br), (hq, hp, hr), (uq, upp, ur), still(up, STAND_STILL)])
    a = STILL_STAND+BLEND; b = a+len(hq); n = len(Q)
    Q, _ = naturalize_arms(Q, P, (STILL_STAND, n-STAND_STILL), (a, b), f[0][ARM_IDS], up[0][ARM_IDS],
                           True, float(stand['fps'][0]))
    d = assemble(model, [(Q, P, R)], stand['fps'])
    return d, dict(still=[0, STILL_STAND], blend=[STILL_STAND, a], human=[a, b], upright_blend=[b, b+STAND_BLEND],
                   upright_still=[n-STAND_STILL, n], relaxed_window=[0, STILL_STAND], seated_window=[0, a],
                   stand_end_window=[n-STAND_STILL, n])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sit', required=True, help='補正済みの人の着座クリップ(npz)')
    ap.add_argument('--stand', required=True, help='補正済みの人の起立クリップ(npz)')
    ap.add_argument('--urdf', required=True)
    ap.add_argument('--pose', default=str(Path(__file__).with_name('formal_seat_pose.json')))
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    model = g.make_model(args.urdf)
    formal = load_formal(args.pose, model)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    segments = {}
    for name, clip, build in [('sit', args.sit, build_sit), ('stand', args.stand, build_stand)]:
        with np.load(clip) as z:
            d, seg = build(model, dict(z), formal)
        np.savez(out/f'{name}_formal_seat.npz', **d)
        segments[name] = dict(frames=len(d['joint_pos']), **seg)
    (out/'formal_seat_segments.json').write_text(json.dumps(segments, indent=1), encoding='utf-8')
    print(json.dumps({k: v['frames'] for k, v in segments.items()}))


if __name__ == '__main__':
    main()
