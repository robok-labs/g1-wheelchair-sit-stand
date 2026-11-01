#!/usr/bin/env python3
# 本ファイルは構成を説明するための骨組みである。報酬の重み・接触条件・終了条件の値・参照生成のパラメータは非公開(None のプレースホルダ)で、このままでは学習結果を再現できない。© 2026 okakei. All rights reserved.
# This file is a skeleton that describes the structure only. Reward weights, contact conditions, termination values and reference-generation parameters are withheld (None placeholders), so the learned results cannot be reproduced as is. © 2026 okakei. All rights reserved.
"""座位目標段階の腕込み判定: 式典座位の窓で姿勢と腕を見る.

入力 CSV(1 行 = 1 制御ステップ、50 Hz)の列:
  motion_step                     参照の frame 番号
  root_{x,y,z}                    骨盤(env 原点を引いたもの)
  torso_{qw,qx,qy,qz}             torso_link の姿勢
  lshoulder_x, rshoulder_x        左右の肩 roll リンクの世界 x
  lwrist_{x,y,z}, rwrist_{x,y,z}  左右の手首 yaw リンク
  arms_armrest_clearance_m        腕と肘掛けの箱近似の距離(正なら離れている)。
                                  上体形状は本リポジトリに含まないため、別途計算した値を列で渡す

判定する窓: 着座は motion_step [440, 540) の最後の 50 step、起立は [0, 50)。
判定項目(すべて満たして合格。終了コードもこれ):
  - 骨盤: 窓の平均が式典座位から xy 3 cm 以内・z 2 cm 以内
  - 上体の傾き(pitch)の窓平均の絶対値 < 5 度
  - 腰ヨー(torso yaw)の窓平均の絶対値 < 5 度
  - 肩の左右差(x)の窓平均 < 3 cm
  - 腕と肘掛けの箱近似の距離 > 0(窓の最小値)。列がなければ未評価として不合格
  - 手首: 全 step・左右とも z < 0.62 m かつ(肘掛けの x 範囲の外、または |y| < 0.135 m)

使い方:
  python scripts/eval_formal_seat.py trace.csv sit
  python scripts/eval_formal_seat.py trace.csv stand --json result.json
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "motions"))
import footclear as g  # noqa: E402

PELVIS_TARGET = np.array([-0.005, 0.0, 0.582])   # 式典座位の骨盤(motions/formal_seat_pose.json の pelvis_initial_pos)
WINDOW = {"sit": (440, 540), "stand": (0, 50)}
LAST_N = {"sit": 50, "stand": None}              # 着座は窓の最後の 50 step、起立は窓全体
MIN_STEPS = 20
PELVIS_XY_TOL, PELVIS_Z_TOL = 0.03, 0.02
TORSO_PITCH_MAX, TORSO_YAW_MAX = 5.0, 5.0
SHOULDER_DX_MAX = 0.03
WRIST_Z_MAX = 0.62
ARMREST_X_RANGE = (-0.1418, 0.099)
FRONT_OF_BODY_ABS_Y_MAX = 0.135


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def column(rows, idx, key):
    return np.array([float(rows[i][key]) for i in idx])


def evaluate(path, task):
    rows = load_rows(path)
    lo, hi = WINDOW[task]
    steps = np.array([int(float(r["motion_step"])) for r in rows])
    idx = np.flatnonzero((steps >= lo) & (steps < hi))
    if LAST_N[task]:
        idx = idx[-LAST_N[task]:]
    if len(idx) < MIN_STEPS:
        raise ValueError(f"window too short: {len(idx)} < {MIN_STEPS}")

    pelvis = np.stack([column(rows, idx, f"root_{c}") for c in "xyz"], 1).mean(0)
    xy_err = float(np.linalg.norm(pelvis[:2] - PELVIS_TARGET[:2]))
    z_err = float(pelvis[2] - PELVIS_TARGET[2])

    quat = np.stack([column(rows, idx, f"torso_{c}") for c in ("qw", "qx", "qy", "qz")], 1)
    rot = g.qmat(quat)
    pitch = float(np.degrees(np.arctan2(rot[:, 0, 2], rot[:, 2, 2])).mean())
    yaw = float(np.degrees(np.arctan2(rot[:, 1, 0], rot[:, 0, 0])).mean())

    shoulder_dx = float(np.abs(column(rows, idx, "lshoulder_x") - column(rows, idx, "rshoulder_x")).mean())

    wrists = np.stack([np.stack([column(rows, idx, f"{s}wrist_{c}") for c in "xyz"], 1) for s in "lr"], 1)
    outside = (wrists[..., 0] < ARMREST_X_RANGE[0]) | (wrists[..., 0] > ARMREST_X_RANGE[1])
    front = np.abs(wrists[..., 1]) < FRONT_OF_BODY_ABS_Y_MAX
    wrist_ok = (wrists[..., 2] < WRIST_Z_MAX) & (outside | front)

    has_clearance = "arms_armrest_clearance_m" in rows[0]
    clearance = float(column(rows, idx, "arms_armrest_clearance_m").min()) if has_clearance else None

    checks = dict(
        pelvis_xy=xy_err <= PELVIS_XY_TOL,
        pelvis_z=abs(z_err) <= PELVIS_Z_TOL,
        torso_pitch=abs(pitch) < TORSO_PITCH_MAX,
        torso_yaw=abs(yaw) < TORSO_YAW_MAX,
        shoulder_dx=shoulder_dx < SHOULDER_DX_MAX,
        armrest_box=bool(has_clearance and clearance > 0),
        wrists=bool(wrist_ok.all()),
    )
    return dict(
        task=task,
        window_steps=int(len(idx)),
        pelvis_xyz=pelvis.tolist(),
        pelvis_xy_error_m=xy_err,
        pelvis_z_error_m=z_err,
        torso_pitch_deg=pitch,
        torso_yaw_deg=yaw,
        shoulder_dx_m=shoulder_dx,
        arms_armrest_min_m=clearance,
        wrist_z_max_m=float(wrists[..., 2].max()),
        checks={k: bool(v) for k, v in checks.items()},
        pass_all=bool(all(checks.values())),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("task", choices=["sit", "stand"])
    ap.add_argument("--json", help="結果を書き出す JSON のパス")
    args = ap.parse_args()
    result = evaluate(args.trace, args.task)
    text = json.dumps(result, indent=1, ensure_ascii=False)
    print(text)
    if args.json:
        Path(args.json).write_text(text, encoding="utf-8")
    sys.exit(0 if result["pass_all"] else 1)


if __name__ == "__main__":
    main()
