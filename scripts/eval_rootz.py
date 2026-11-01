#!/usr/bin/env python3
# 本ファイルは構成を説明するための骨組みである。報酬の重み・接触条件・終了条件の値・参照生成のパラメータは非公開(None のプレースホルダ)で、このままでは学習結果を再現できない。© 2026 okakei. All rights reserved.
# This file is a skeleton that describes the structure only. Reward weights, contact conditions, termination values and reference-generation parameters are withheld (None placeholders), so the learned results cannot be reproduced as is. © 2026 okakei. All rights reserved.
"""Pass/fail check of the pelvis (root) height trace of a rollout.

Input CSV needs a column `root_z` (one row per control step).
Criterion: mean of the last 50 steps
  stand: z >= 0.75 m
  sit  : 0.55 m <= z <= 0.68 m
Exit code 0 on pass, 1 on fail.

Usage:
  python scripts/eval_rootz.py trace.csv stand
  python scripts/eval_rootz.py trace.csv sit
"""
import argparse
import csv
import sys

import numpy as np

LAST_N = 50
STAND_MIN = 0.75
SIT_RANGE = (0.55, 0.68)


def evaluate(path, task):
    with open(path, newline="", encoding="utf-8") as f:
        z = np.array([float(r["root_z"]) for r in csv.DictReader(f)])
    if len(z) < LAST_N:
        raise ValueError(f"trace too short: {len(z)} < {LAST_N}")
    if not np.isfinite(z).all():
        raise ValueError("non-finite values in trace")
    z_end = float(z[-LAST_N:].mean())
    if task == "stand":
        ok = z_end >= STAND_MIN
    else:
        ok = SIT_RANGE[0] <= z_end <= SIT_RANGE[1]
    return z_end, ok


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv")
    p.add_argument("task", choices=["stand", "sit"])
    a = p.parse_args()
    z_end, ok = evaluate(a.csv, a.task)
    print(f"task={a.task} last{LAST_N}_mean_root_z={z_end:.4f} -> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
