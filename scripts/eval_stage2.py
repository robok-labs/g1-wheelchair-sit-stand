#!/usr/bin/env python3
# 本ファイルは構成を説明するための骨組みである。報酬の重み・接触条件・終了条件の値・参照生成のパラメータは非公開(None のプレースホルダ)で、このままでは学習結果を再現できない。© 2026 okakei. All rights reserved.
# This file is a skeleton that describes the structure only. Reward weights, contact conditions, termination values and reference-generation parameters are withheld (None placeholders), so the learned results cannot be reproduced as is. © 2026 okakei. All rights reserved.
"""Stage 2(歩いて接近->転回->着座 / 起立->前進->退去。初期の試行は転回あり)の rollout 合否判定.

入力 CSV(1 行 = 1 制御ステップ、50 Hz)の列:
  root_{x,y,z,qw,qx,qy,qz}      骨盤(root)の世界座標(env 原点を引いたもの)
  left_{x,y,z,qw,qx,qy,qz}      左足首リンク
  right_{x,y,z,qw,qx,qy,qz}     右足首リンク
  reset(任意)                   0/1。1 が 1 回でもあれば無効

判定は 2 系統を出す。

1. Stage 2 固有(合格はこちらで決める。終了コードもこれ)
   - 接近/退去の距離: 着座は開始位置から座面中心の方向へ 0.8 m 以上、退去は座面中心から前方 0.8 m 以上
   - 転回: 骨盤 yaw の振れ幅が 100 度以上
   - 終端 z: 着座は最終 50 step 平均 0.55〜0.68 m、退去は 0.70 m 以上
   - 座面 xy: 着座は最終 50 step 平均が座面中心から x, y とも ±0.12 m 以内
   - 最小距離: 足と車いす部材の最小距離が 0 より大きい
   - 転倒なし: 骨盤傾斜 60 度未満、骨盤 z が 0.4 m 以上

2. Stage 1 互換(同じ軌跡を Stage 1 の基準で見たもの。参考値)
   - 着座: z 0.55〜0.68 m、座面 xy は x, y とも ±0.10 m 以内
   - 退去: z 0.75 m 以上、参照末尾との差 3 cm 以内(--ref-end-z 指定時のみ)
   - 共通: 最小距離 > 0、転倒なし

使い方:
  python scripts/eval_stage2.py trace.csv motions/chair_parts.json approach_sit
  python scripts/eval_stage2.py trace.csv motions/chair_parts.json stand_depart --ref-end-z 0.766
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "motions"))
import footclear as g  # noqa: E402

LAST_N = 50
CHAIR_CENTER = np.array([-0.03, 0.0])
FIELDS = ["x", "y", "z", "qw", "qx", "qy", "qz"]


def load_trace(path):
    """CSV から root と両足首の位置・クォータニオン(と reset)を読む。"""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < LAST_N:
        raise ValueError(f"trace too short: {len(rows)} < {LAST_N}")
    n = len(rows)
    pos = np.zeros((n, 30, 3))
    quat = np.zeros((n, 30, 4))
    quat[:, :, 0] = 1.0
    for name, body in [("root", 0), ("left", g.ANKLES[0]), ("right", g.ANKLES[1])]:
        pos[:, body] = [[float(r[f"{name}_{c}"]) for c in FIELDS[:3]] for r in rows]
        quat[:, body] = [[float(r[f"{name}_{c}"]) for c in FIELDS[3:]] for r in rows]
    if not (np.isfinite(pos).all() and np.isfinite(quat).all()):
        raise ValueError("non-finite values in trace")
    if np.max(np.abs(np.linalg.norm(quat[:, [0, *g.ANKLES]], axis=-1) - 1)) > 1e-3:
        raise ValueError("invalid quaternion norm")
    resets = sum(int(float(r.get("reset", 0))) for r in rows)
    return pos, quat, resets


def evaluate(path, parts, task, ref_end_z=None):
    pos, quat, resets = load_trace(path)
    rot = g.qmat(quat)
    root = pos[:, 0]
    end = root[-LAST_N:].mean(0)

    # 足と車いす部材の最小距離(足裏球ベース、正なら非接触)
    min_distance = float(g.clearance(pos, rot, parts).min())
    # 骨盤傾斜: 骨盤 z 軸と鉛直のなす角
    tilt = np.degrees(np.arctan2(np.linalg.norm(rot[:, 0, :2, 2], axis=-1), rot[:, 0, 2, 2]))
    # 骨盤 yaw の振れ幅
    heading = np.unwrap(np.arctan2(rot[:, 0, 1, 0], rot[:, 0, 0, 0]))
    turn = float(np.degrees(heading.max() - heading.min()))

    common = dict(
        no_reset=resets == 0,
        min_distance_positive=min_distance > 0,
        nonfall=bool(tilt.max() < 60 and root[:, 2].min() >= 0.4),
    )
    seat_dx, seat_dy = float(end[0] - CHAIR_CENTER[0]), float(end[1] - CHAIR_CENTER[1])

    if task == "approach_sit":
        # 開始位置から座面中心へ向かう距離(開始時の座面中心までの距離 - 終端の距離)
        travel = float(np.linalg.norm(root[0, :2] - CHAIR_CENTER) - np.linalg.norm(end[:2] - CHAIR_CENTER))
        specific = dict(
            approach_distance=travel >= 0.8,
            turn=turn >= 100.0,
            terminal_z=bool(0.55 <= end[2] <= 0.68),
            seat_xy=bool(abs(seat_dx) <= 0.12 and abs(seat_dy) <= 0.12),
        )
        compat = dict(
            terminal_z=bool(0.55 <= end[2] <= 0.68),
            seat_xy=bool(abs(seat_dx) <= 0.10 and abs(seat_dy) <= 0.10),
        )
        distance = travel
    else:
        distance = seat_dx  # 座面中心から前方(+x)への距離
        specific = dict(
            depart_distance=distance >= 0.8,
            turn=turn >= 100.0,
            terminal_z=bool(end[2] >= 0.70),
            reference_z=True if ref_end_z is None else bool(abs(end[2] - ref_end_z) <= 0.05),
        )
        compat = dict(terminal_z=bool(end[2] >= 0.75))
        if ref_end_z is not None:
            compat["reference_z"] = bool(abs(end[2] - ref_end_z) <= 0.03)
    specific.update(common)
    compat.update(common)

    return dict(
        task=task,
        rows=len(pos),
        distance_m=distance,
        turn_deg=turn,
        last50_pelvis_xyz=end.tolist(),
        seat_center_xy_error_m=[seat_dx, seat_dy],
        min_foot_chair_distance_m=min_distance,
        max_pelvis_tilt_deg=float(tilt.max()),
        resets=resets,
        stage2_checks=specific,
        stage2_pass=all(specific.values()),
        stage1_compat_checks=compat,
        stage1_compat_pass=all(compat.values()),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv")
    p.add_argument("parts", help="車いす衝突形状 JSON(motions/chair_parts.json)")
    p.add_argument("task", choices=["approach_sit", "stand_depart"])
    p.add_argument("--ref-end-z", type=float, default=None, help="参照モーション末尾の骨盤 z(退去の参照差判定用)")
    a = p.parse_args()
    parts = json.loads(Path(a.parts).read_text(encoding="utf-8"))["parts"]
    result = evaluate(a.csv, parts, a.task, a.ref_end_z)
    print(json.dumps(result, indent=2))
    print(f"stage2={'PASS' if result['stage2_pass'] else 'FAIL'} "
          f"stage1_compat={'PASS' if result['stage1_compat_pass'] else 'FAIL'}")
    sys.exit(0 if result["stage2_pass"] else 1)


if __name__ == "__main__":
    main()
