> 本ファイルは構成を説明するための骨組みである。報酬の重み・接触条件・終了条件の値・参照生成のパラメータは非公開(None のプレースホルダ)で、このままでは学習結果を再現できない。© 2026 okakei. All rights reserved.
> This file is a skeleton that describes the structure only. Reward weights, contact conditions, termination values and reference-generation parameters are withheld (None placeholders), so the learned results cannot be reproduced as is. © 2026 okakei. All rights reserved.

# motions

参照モーションのデータ本体は同梱しません。

## 出自

PhysHSI(人-物体インタラクションのヒューマノイド向けデータセット)に含まれる SAMP 由来の椅子クリップ `chair_mo006` の standUp を使います。各自、PhysHSI の公式リポジトリの手順でデータを取得してください。

## 変換手順(pt → csv → npz)

1. PhysHSI の `.pt` から、G1 29dof の関節角・ルート姿勢を取り出して csv に書き出す(フレームレートは 60Hz 想定)。
2. unitree_rl_lab の `scripts/mimic/csv_to_npz.py` で npz に変換する。

```
python scripts/mimic/csv_to_npz.py -f <motion>.csv --input_fps 60 --headless
```

注意: `csv_to_npz.py` は npz を書き終えても Isaac Sim の終了処理で止まることがある。npz が出力されていればプロセスを終了してよい。

3. 座位側は、着座骨盤を原点・着座向きを +X にそろえる(`tasks/README.md` の車いす配置と一致させる)。
4. `footclear.py` で足の接地・歩隔を補正する。

## footclear.py

G1 の URDF から順運動学を NumPy だけで実装し、足首を平らに接地させる脚の逆運動学と、車いす(`chair_parts.json`)との解析的な最小距離を計算します。

- 入力: 起立クリップと着座クリップの npz、G1 29dof の URDF
- 処理: 足を床に平置きにして歩隔を固定、骨盤高さを滑らかに下げ、脚だけを IK で解き直す(フレーム数は変えない)
- 検証: 車いすとの距離の下限、関節可動域、FK 一致、足の接地、速度の整合、関節速度上限
- 既定値: 足 x、半歩隔、骨盤の低下量(値は非公開)

```
python motions/footclear.py build --urdf g1_29dof_rev_1_0.urdf --input <npz dir> --output <out dir>
python motions/footclear.py verify <out dir>
```

`chair_parts.json` は `footclear.py` が内部で生成する車いす衝突形状と同じ内容です。

## Stage 2: 歩いて接近 -> 転回 -> 着座 / 起立 -> 転回 -> 退去(`stage2_motion.py`。初期の試行。採用した退去は 起立 -> 前進 -> 退去(初期の試行は転回あり))

Stage 1 は車いすの前に立った状態から始まる。Stage 2 は、数歩歩いて近づき、車いすに背を向ける向きへ転回してから座る動作(着座)と、その逆(起立 -> 転回 -> 退去)を扱う。

### 参照クリップ

- PhysHSI が配布する SAMP 由来の椅子クリップ `chair_mo006` の「座る」を使う(データは同梱しない)。
- 元クリップの接近距離は約 1.0 m、転回は約 132 度。
- 起立->退去は専用のクリップを使わず、補正済みの着座クリップを逆再生して作る(速度の符号も反転)。

### 補正の流れ

1. **整列**: 着座相(骨盤 z が座位の高さを最後に下回ってから末尾まで)の位置と向きの代表値を求め、Stage 1 の座位姿勢に合うよう、クリップ全体を yaw 回転と xy 並進で剛体整列する。
2. **歩行相の足の補正**: 足裏球の高さが床の包絡(局所の下位分位)に近く、足の水平・鉛直速度が小さいフレームを接地とみなす。接地中の足は xy を中央値に固定して床に平置きにし、遊脚は前後の接地とのずれを間の区間だけで滑らかに解消する。最後の接地は座位の足位置に合わせる。
3. **遊脚の回避**: 車いすの部材にぶつかる遊脚だけ、低い摺り足、だめなら持ち上げの順に直す。接地区間と時間は変えない。
4. **着座相の接続**: 転回が終わる 25 フレーム前から、Stage 1 と同じ座位姿勢へブレンドする。足は Stage 1 と同じ位置に置き、骨盤の高さと x(座面への後退)、上体傾斜(座位用の値)、腕(肘掛けに当たらない姿勢)を座位用に入れ替える。
5. **脚の IK**: 補正した足の目標に対し、両脚だけを IK で解き直す。届かないフレームは骨盤を一定の刻みで上限まで下げ、滑らかな包絡にしてから再度解く。フレーム数は変えない。
6. **逆再生**(起立->退去のみ)。

### 検証指標

`validate()` が運動学的な項目を機械判定し、`--json` のレポートに残す。項目は次のとおり。基準と実測の値は非公開。

| 指標 | 内容 |
|---|---|
| 足と車いす部材の最小距離(全相) | 下限あり(学習時の終了条件に掛けない) |
| 腕と肘掛けの最小距離 | 診断値(判定は上体全体で下限あり) |
| 上体傾斜(最大) | 上限あり |
| 接地中の足の滑り | 上限あり |

そのほか FK 一致、関節可動域(ソフト制限 98% を含む)、速度の整合、関節速度上限、足が床を貫通しないことを判定する。腕と肘掛け、上体傾斜は上体形状の別ファイルで測る値のため、本リポジトリの `stage2_motion.py` には含まない(足・部材の距離と接地は含む)。

### 使い方

```
python motions/stage2_motion.py --input <前向き着座クリップ>.npz --mode approach_sit \
    --urdf g1_29dof_rev_1_0.urdf --anchor anchor.json --parts motions/chair_parts.json \
    --output approach_sit.npz --json approach_sit.json
```

`--mode stand_depart` にすると、同じ補正をしたうえで逆再生した起立->退去クリップを出す。`anchor.json` は Stage 1 の座位基準(`pelvis_xyz`, `pelvis_quat_wxyz`, `standing_pelvis_y`)で、Stage 1 のクリップから作る。
