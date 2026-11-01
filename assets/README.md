> 本ファイルは構成を説明するための骨組みである。報酬の重み・接触条件・終了条件の値・参照生成のパラメータは非公開(None のプレースホルダ)で、このままでは学習結果を再現できない。© 2026 okakei. All rights reserved.
> This file is a skeleton that describes the structure only. Reward weights, contact conditions, termination values and reference-generation parameters are withheld (None placeholders), so the learned results cannot be reproduced as is. © 2026 okakei. All rights reserved.

# assets

## 車いすメッシュ

`gen_wheelchair.py` は trimesh で市販の跳ね上げ式手動車いすのメッシュを、公開カタログ寸法から生成します。フットレストを下げた `down` と跳ね上げた `up` の2変種を、`out/` に obj / stl / glb で出力します。

```
python assets/gen_wheelchair.py
```

### 寸法の根拠(公式カタログ値)

| 部位 | 値 |
|---|---|
| 座面高 | 50 cm |
| 座幅 | 40 cm |
| 座奥行 | 37 cm |
| 背もたれ高 | 40 cm(全高 90 cm) |
| 後輪 | 20 inch (50.8 cm) |
| 前輪 | 6 inch (15.24 cm) |
| 全幅 | 61 cm |
| 全長 | 95 cm |

アームレスト高 22 cm はカタログ値からの目安、細部の形状(スポーク数、押手など)は見た目に合わせた近似です。

## USD 化

Isaac Sim の kit `asset_converter`(`omni.kit.asset_converter`)で `wheelchair_up.glb` を USD に変換します。

注意点:

- 出力 glb は Y-up で出力されている(スクリプト内で Rx(-90) を適用)。変換後の USD は Y-up かつ単位が cm になる。そのままシーンに置くと横倒しかつ 100 倍の大きさになる。
- そのためシーン側で `scale = 0.0086`(cm→m の 0.01 に、車いすの実寸補正 0.86 を掛けた値)と、`Rx(90)`(クォータニオン `(0.70711, 0.70711, 0, 0)`)を与えて Z-up の m 単位に戻す。
- 配置は `pos = (-0.03, 0, 0)`。参照モーションの正準座標系(着座向き +X、着座時の骨盤が原点)に合わせてある。
- 衝突は厳密三角メッシュ。USD 側に CollisionAPI を焼き込んでおく。
- `motions/chair_parts.json` の衝突形状は、同じ寸法に 0.86 倍と位置オフセットを掛けたワールド座標の近似(箱・カプセル・円柱)。

変換は上の手順を Isaac Sim の GUI で行った。kit を起動する変換スクリプトは本リポジトリに含めていない。
