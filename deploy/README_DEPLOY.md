# deploy/ — 実機 G1(Linux)展開バンドル

シミュレーションで E2E 検証済みのミッションコントローラを、実機の Linux PC に
持ち込むための一式。**コントローラ本体はシムと実機で同一ファイル**
(`g1_mission_controller.py`、依存は numpy + onnxruntime のみ)。

## 構成

| ファイル | 役割 |
|---|---|
| `g1_mission_controller.py` | ミッションFSM+方策実行(**シムでE2E検証されたデプロイ本体**) |
| `config.json` | 全校正値(椅子座標系のドック点・ゲート・ゲイン・着座レシピ) |
| `g1_walk_29dof.onnx` | 歩行方策(公式 playground、ビルド時に models/ からコピー) |
| `g1_climb_backstep.onnx` | 後ろ向き段差登り方策(本リポジトリでRL学習→ONNX化) |
| `export_climb_onnx.py` | SB3→ONNX 変換(正規化統計を焼き込み、torch不要化) |
| `run_on_robot.py` | **Linux アダプタ(テンプレート)** — unitree_sdk2py 接続部 |

バンドル作成:

```sh
cp ../models/policies/g1_joystick_29dof.onnx g1_walk_29dof.onnx
../.venv-rl/bin/python export_climb_onnx.py        # runs_climb/ の学習済みから生成
```

ロボット側の依存: `pip install numpy onnxruntime unitree_sdk2py`(torch 不要)。

## 実機までの距離(正直な現状)

✅ 済: コントローラはシムの全ミッション(障害物回避→回頭→後退→段差登り→着座)を
1本の連続物理で完走する同一コードとして検証
⬜ 未: `run_on_robot.py` の TODO(HW) 6箇所(SDK配線・関節インデックス対応・状態推定・
椅子座標系の自己位置・LiDAR整形・E-stop)。**ハードウェアでの実行実績はゼロ**です。

## sim2real チェックリスト(この順で)

1. **ゲイン対応の検証**: config の `walk_kp` はシム学習時の値(75/20/2)。シムは
   XML の関節ダンピングが kd を兼ねるため `walk_kd` は Unitree 慣行値を仮置きしている
   — 吊り下げ状態で1関節ずつステップ応答を比較し、実機 kd を校正すること。
   SIT/CLIMB モードの kp300/kd8 も同様(まず空中で)。
2. **関節インデックスマップ**: `config.actuator_names` の順序と G1 lowcmd のモータ
   番号の対応表を作る(unitree_rl_gym の deploy_real を参照)。
3. **椅子座標系の自己位置**: 椅子の位置・向きを一度測量(AprilTag/LiDAR/メジャー)し、
   オドメトリと合成して base_xy/yaw を椅子フレームで供給。誤差バジェットは
   ドッキングゲート ±4cm/±6°(config.gates)。
4. **段階投入**(各段階で吊りハーネス+E-stop):
   a. 立位保持のみ(コントローラ出力を damping と切替できることを確認)
   b. SIT 単体: 台上に手で立たせ、SIT フェーズだけ実行(着座は最も低リスク)
   c. CLIMB 単体: 柔らかいモック段差(ウレタン0.22m)で後ろ向き登り
   d. 歩行・回頭・後退(障害物なし)
   e. フルチェーン(実椅子)
5. **既知の sim2real ギャップ**(docs/FULL_MISSION_DEPLOY.md 詳細):
   クッション剛性(シムは剛体=保守側)/ センサ遅延・ノイズ(シムは理想)/
   脚×障害物の衝突はシムでは骨盤ボックスのみ物理(回避マージンは VFH 任せ)/
   CLIMB 方策の学習分布外の外乱。
