# MuJoCo on Mac スキル ― 実現戦略(レビュー用ドキュメント)

| 項目 | 内容 |
|---|---|
| 文書バージョン | **v2.2**(v2.0 = Part II 行動スキル層追加 / v2.1 = 整合性修正 / **v2.2 = 旧方針の完全アーカイブ化**:Part I のクラウド記述・旧ロードマップを「旧方針/スコープ外」に統一し、実装者が Part II だけを正準として読めるようにした) |
| 作成日 | 2026-06-05(v1.0)/ 2026-06-05 追補(v2.0) |
| ステータス | **戦略策定段階 / 実装前**。本書を複数AI・人間で精査し、議論を深めてから開発着手する |
| 作成プロセス | Claude による並列Webリサーチ → 統合 → 敵対的レビュー(+ **M5 Max 実機ベンチ**)のワークフロー成果を編集。Codex のレビューを反映 |
| 最終ゴール(v2.0で明確化) | 「**歩く・座る/立つ・走る※・障害物回避**を **Unitree G1(ヒューマノイド)・H1(ヒューマノイド)・GO2(四足)** ほか各種ロボットで実現するスキルズセット」を、NVIDIA環境なしの Mac(M5 Max 128GB をローカル旗艦と想定)で提供する。**※走る(run)** は NVIDIA-free のローカルでは現状**学習不可**のため、決定2(100%ローカル, II-9)の下では**当面スコープ外**。将来、走行の事前学習方策が公開されれば**再生**で追加する保留枠とする(II-9)。 |
| 目的 | 「GTC Taipei 2026-06-01 で公開された NVIDIA/skills を範に、**macOS(Apple Silicon)用 MuJoCo スキルを作り、NVIDIA環境を持たない全Macユーザーに提供する**」ための実現戦略を、検証可能な形で記録する |

> **📌 実装の正準ソース(最初に読む)**
> 本書は2部構成。**実装の正式な指針は Part II**(特に **II-4 スキルセット / II-5 ロボット別マトリクス / II-8 ロードマップ / II-9 確定決定**)。**Part I(§1〜§15)は基盤の検証記録**だが、その **§7(クラウド)と §10(旧ロードマップ)は「旧方針」でアーカイブ済**。決定2により **クラウド層は不採用**、**run(走行)はスコープ外/保留枠**。Part I と Part II で食い違って見える箇所は **常に Part II / II-9 が正**。
>
> **要点**:
> - **Part II = ロボット行動スキル層(本丸)**。中心命題=**「モデルベース制御(MuJoCo MPC / PD / CPG / convex-MPC)を行動生成の背骨に据え、ゼロ学習RL と run はスコープ外(100%ローカル堅持)」**。
> - **M5 Max 実機ベンチを実測済**(Part II-1)。GO2 のモデルベース制御は余裕でリアルタイム、G1 はギリギリ。
> - **訂正**: §11-A4 の「jax-metal 死亡 / 全issueクローズ」は**過剰主張**だった。正しくは「**メンテ停止・0.1.1/jaxlib0.4.34 で凍結・Apple は今も experimental 表記**」。実務結論(ベースに据えない)は不変。Part II-3 参照。

---

## 0. この文書の使い方(レビュアー / 他AI向け)

- 本書は**自己完結**を意図している。前提知識なしで精査できる。
- すべての主要主張に **【確度】** と **出典(§14)** を付した。**検証済み事実**と**推論・設計判断**を区別している。鵜呑みにせず出典を当たってほしい。
- **争点・未検証事項は §12・§13 に集約**した。議論はまずそこから始めると効率的。
- 数値(性能・バージョン等)には「**実測ではなく出典記載値**」「**要実機検証**」のラベルを付けた箇所がある。開発判断の根拠にする前に再検証すること。
- この戦略は一度、辛口の敵対的レビュー(§11)を通している。元案にあった楽観的な記述は**既に修正済み**で、本文は修正後の見解を反映している。

---

## 目次

1. [TL;DR(結論)](#1-tldr結論)
2. [背景と問題設定](#2-背景と問題設定)
3. [検証済み事実(出典・確度付き)](#3-検証済み事実出典確度付き)
4. [成果物の定義(何を作るか)](#4-成果物の定義何を作るか)
5. [アーキテクチャ(スキル形式・配布)](#5-アーキテクチャスキル形式配布)
6. [ローカルコア(NVIDIA不要で動く範囲)](#6-ローカルコアnvidia不要で動く範囲)
7. [クラウド・フォールバック層 ※旧方針・不採用(II-9)](#7-クラウドフォールバック層)
8. [実現可能性マップ](#8-実現可能性マップ)
9. [Claude × Codex 協業モデル](#9-claude--codex-協業モデル)
10. [フェーズ別ロードマップ ※旧版 → II-8 が正式](#10-フェーズ別ロードマップ)
11. [敵対的レビュー:成否を分ける地雷](#11-敵対的レビュー成否を分ける地雷)
12. [リスク・未検証事項一覧(確度付き)](#12-リスク未検証事項一覧確度付き)
13. [他AIに議論してほしい論点](#13-他aiに議論してほしい論点)
14. [出典一覧](#14-出典一覧)
15. [用語集](#15-用語集)

---

## 1. TL;DR(結論)

- **作れる。しかも「空いている穴」である。** NVIDIA/skills カタログには現状 **MuJoCo スキルが1本も存在しない**。【確度: 高】
- **スキルの形式・配布は NVIDIA 専用ではなくオープン規格**。`SKILL.md`(agentskills.io 仕様)+ Vercel製 `skills` CLI。`github.com/<あなた>/mujoco-skills` という**独立リポジトリを公開するだけ**で、Claude Code にも Codex にも `npx skills add` で導入できる。**NVIDIAへのPR・署名・承認は一切不要**。【確度: 高】
- **MuJoCo は Apple Silicon ネイティブで、NVIDIA完全不要で動く**(CPU物理・描画・Menagerieモデル)。【確度: 高】
- **真の難所は「NVIDIA境界の説明」ではなく「ローカルMacの最初の5分の摩擦」**。敵対的レビューが、初手で事故る地雷を実証付きで3つ発見した(§11)。これを設計段階で回避できるかが成否を分ける。【確度: 高】
- **手の届かない領域は正直に切る**:数千env並列RL、フォトリアル合成データ、大規模VLA推論、そして **run(走行)とゼロ学習ヒューマノイドRL** は CUDA専用 or 学習不可で Macローカルでは不可能。**決定2(100%ローカル堅持, II-9)によりこれらはスコープ外=クラウド層は作らない**。**ベース体験は決してNVIDIAを要求しない**。【確度: 高】

> **一言でいうと**: NVIDIAのスキルは「Macユーザーが触れないGPUクラスタを操作する」もの。我々が作るのは「同じエージェント(Claude/Codex)に、ローカルMacだけで **制御・物理・小規模学習・評価** をやらせる」スキル。

---

## 2. 背景と問題設定

### 2.1 何が起きたか

2026-06-01、GTC Taipei で NVIDIA は **Physical AI 向けのエージェントスキル/ツール群(約110〜134本)** を OSS 公開した(`github.com/NVIDIA/skills`)。これらは「どのツールを呼び、何を出力し、どう検証するか」をエージェントに教える指示セットである。【出典: NVIDIA Newsroom】【確度: 高】

### 2.2 ギャップ(誰が排除されているか)

NVIDIA の Physical AI 系スキルは、その**大半が OSMO/Kubernetes クラスタや NIM エンドポイントにジョブを投げる「ルーター/オーケストレーター」**であり、以下を**ハード要求**する:

- NVIDIA GPU(しばしばマルチGPU)、CUDA 12.x
- OptiX/RTX レイトレーシング(例:defect-image 生成は `/usr/share/nvidia/nvoptix.bin` と GPU≥2 を要求)
- ゲートされた NGC / HuggingFace アセット、APIキー

計算境界は固い:**NVIDIA Warp / MuJoCo Warp は CUDA 専用で、Apple Silicon GPU では加速しない(CPU動作のみ)**。NVIDIA自身が「MuJoCo Playground を補完的学習フロントエンド」と位置づける Newton ですら、GPU経路はMacで加速しない。【確度: 高】

→ 結果として、**NVIDIA環境を持たない全Mac開発者・学生・研究者・愛好家が、ハードウェア・ゲート資産・リモートクラスタ前提によって締め出されている**。

### 2.3 なぜ MuJoCo が橋渡しになるか

- MuJoCo は **Apple Silicon ネイティブ・NVIDIA依存ゼロ**で動く(§3.3)。
- **NVIDIAカタログに MuJoCo スキルは存在しない** → 競合ではなく未占有の穴。
- NVIDIA の `omniverse-cad-to-simready` スキルは**入力に MJCF を既に受け付けている** → MuJoCo は NVIDIA エコシステムの「外」ではなく「中」に座れる(対立ではなく補完というフレーミングが可能)。【確度: 高】

---

## 3. 検証済み事実(出典・確度付き)

リサーチを5領域に分けて実施した。以下は各領域の確定事項。

### 3.1 スキル形式(skill-format)【総合確度: 高】

- NVIDIA/skills は**オープンな Agent Skills 仕様(agentskills.io/specification)に完全準拠**したカタログ。NVIDIA独自スキーマは存在しない。Claude Code / Cursor / Codex / Windsurf が使うのと**同一の `SKILL.md` 規格**。【高】
- **必須フロントマター**: `name`(最大64字、`[a-z0-9-]` のみ、先頭末尾連続ハイフン禁止、**親ディレクトリ名と一致必須**)と `description`(最大1024字、「何をするか」+「いつ使うか」を書く)。【高】
- **任意フロントマター**: `license` / `compatibility`(最大500字、ランタイム要件を書く場所)/ `metadata`(任意の文字列マップ、version/author/tags はここに入れ子)/ `allowed-tools`(**実験的**、エージェントごとに対応差)。【高】
- **NVIDIA実運用の頻度**(133ファイル調査): name 133, license 133, description 133, metadata 96, version 41, when_to_use 34, compatibility 21, allowed-tools 19。→ `license` は仕様上任意だがNVIDIA実務では事実上必須。version/author/tags は `metadata` 配下が安全。【高】
- **進行的開示(progressive disclosure)3段階**: ①name+description(~100トークン、常時ロード) ②本文(発火時ロード、**<5000トークン/<500行推奨**) ③`references/scripts/assets`(オンデマンド)。ファイル参照は相対パス・1階層まで。【高】
- 1スキルのディレクトリ実例(`accelerated-computing-cudf`): `SKILL.md` / `skill-card.md` / `BENCHMARK.md` / `evals/`(evals.json + fixtures) / `references/*.md` / `skill.oms.sig`。【高】
- **署名・skill-card・evals は「NVIDIA-Verified カタログ掲載」のゲートでのみ必須**。スキルの動作には `SKILL.md` 単体で十分。同期パイプラインがこれら成果物を欠くスキルを公開前に弾く。【高】
- **第三者は NVIDIA-Verified バッジを取得できない**:SkillSpector スキャン + NVSkills-Eval(Tier 1/2/3)+ `nv-agent-root-cert.pem` に対するOMS署名は **NVIDIA内部の非公開パイプライン**。self-sign(自己署名)はできるが NVIDIA 検証とは別物。【高】
- **署名検証はMacで動く**: `pip install model-signing`(pure-Python `py3-none-any`、GPU依存ゼロ、Apple Silicon可)。ただし PyPI ステータスは "3-Alpha"(v1.1.1, 2025-10)でAPI変更リスクあり。【高/中】

### 3.2 配布・CLI(skills-cli-distribution)【総合確度: 高】

- `skills` CLI は **Vercel Labs 製**(`github.com/vercel-labs/skills`、npm `skills`、サイト skills.sh)。**NVIDIAはインストーラを所有していない**。【高】
- **任意のGitソースからレジストリ不要で導入可能**:`owner/repo` 短縮形 / フルURL / リポジトリ内サブパス / GitLab / git URL / ローカルパス。【高】
- インストール構文: `npx skills add <source> [-s <name|'*'>] [-a <agent>] [-g] [-y] [--copy]`。`-a` 複数指定で一括導入可:`npx skills add owner/repo -a claude-code -a codex -a cursor`。【高】
- エージェント別配置先:Claude Code → `.claude/skills/`(project)/ `~/.claude/skills/`(global);Codex → `.agents/skills/`(project)/ `~/.codex/skills/`(global、**ただし§12で要検証**);Cursor → `.agents/skills/`;Kiro → `.kiro/skills/`。既定は **symlink**、`--copy` でコピー。【高、Codex globalは中】
- **NVIDIA/skills はカタログのミラー**(製品リポジトリから日次同期)であり、コミュニティスキルをPRする場所ではない。**独立リポジトリで出すべき**。【高】
- **Codex のスキルはネイティブ機能**(developers.openai.com/codex/skills):`.agents/skills` を project スコープで上方探索、`$HOME/.agents/skills` と `/etc/codex/skills` を personal/system スコープで探索。自動ロード、`/skills`・`$`・暗黙マッチで発火。【高】
- **Write-once / run-everywhere**:同一 `SKILL.md` が Claude Code・Codex・Cursor・Kiro 等70+エージェントで無改変動作。配置先ディレクトリだけが異なる。→ Claude+Codex 協業を1リポジトリで実現可能。【高】

### 3.3 MuJoCo on Mac(mujoco-on-mac)【総合確度: 高(Apple-GPU MJXのみ低)】

- **arm64 wheel `mujoco` 3.9.0** がネイティブ動作。CPU物理は高速(単一humanoidで概ね 650K〜900K steps/s、**出典記載値・要実機再検証**)。【高/数値は中】
- **オフスクリーン描画は CGL 経由でRGB/深度/セグメンテーション取得可**。macOSでは EGL/OSMesa 不要。【高】
- **対話ビューア(`mujoco.viewer` passive)は macOS で `mjpython` が必要**(GUIをメインスレッドで回すため)。【高】
- **MJX(MuJoCo XLA)は JAX 上で動く**。Apple GPU 利用は実験的な `jax-metal` プラグイン依存で、**jax-metal はメンテ停止・凍結**(§11-A4 / Part II-3、最新 0.1.1=2024-10、jaxlib≥0.4.34、Apple は今も experimental 表記。baseline にはしない)。MJX-on-Mac は CPU-via-XLA(単一シーンでネイティブMuJoCoの約10倍遅)。【高、Apple-GPU部分は低】
- **MuJoCo Warp / Newton の GPU 経路は CUDA 専用**。Apple Silicon では CPU のみ。MuJoCo Warp は RTX 4090 で MJX 比 最大152倍(locomotion)/313倍(manipulation)と NVIDIA は主張するが、これは**Macには来ない加速**。【高】
- **mujoco_menagerie 70+ モデル**がそのまま動く。【高】
- Mac で信頼できる加速は **PyTorch MPS バックエンド**と **Apple MLX**(jax-metal ではない)。【高】

### 3.4 NVIDIA Physical AI スキルの実態(nvidia-physical-ai-skills)【総合確度: 高(Isaac系内部構造は中)】

代表的スキルと、その下回りツール・ハード要件:

| スキル | 役割 | 下回り / ハード要件 |
|---|---|---|
| `physical-ai-neural-reconstruction` | AVセンサ記録→編集可能3Dシーン | Omniverse NuRec/NRE、3D Gaussian Splatting。**GPU+NVIDIA Container Toolkit+Docker必須**、Isaac Sim 5.1連携、NGCキー+ゲートHF |
| `physical-ai-defect-image-generation` | 製造AOI向け合成欠陥画像 | OSMOクラスタ投入、**OptiX nvoptix.bin・GPU≥2**、~80GB+ストレージ |
| `physical-ai-video-data-augmentation` | 動画拡張+自動ラベル | Cosmos Transfer、NIM(Qwen系)、CUDA 12.x、コールドスタート45–80分 |
| `omniverse-cad-to-simready` | CAD/URDF/**MJCF**/USD→SimReady USD | オーケストレーションはLinux/**macOS**+Python 3.12可、ただしプロパティ割当はGPU+Container Toolkit+API_KEY必須 |
| `physicsnemo-discover` | PhysicsNeMo ナビ | **RLは明示的にスコープ外**。計算は実行せずルーティングのみ |

- **MuJoCo-viable な部分** = 制御+物理+中規模RL(MJCF/URDF読込、接触剛体シミュ、単機RL、基本描画)。**手が届かない部分** = フォトリアル合成データ、ニューラル再構成、数千GPU並列RL、VLA推論。【高】
- NVIDIA は MuJoCo(Playground)を Newton の補完的学習フロントエンドと**自ら位置づけている**。【高】

### 3.5 NVIDIA非依存ブリッジ戦略(no-nvidia-bridge-strategy)【総合確度: 中〜高】

- 単一humanoidで M3 Max が約 650K SPS、A100(batch 8192)が約 950K total SPS という**出典記載の数値**。→ **単環境RLでは単一Macが対抗可能だが、これは並列RLの実時間とは別問題**(§11-A2 の注意必読)。【数値は中・解釈は高】
- MJX は Apple Silicon 対応だが単一シーンで native の約10倍遅。【中】
- **jax-metal はメンテ停止・凍結**(最新 0.1.1=2024-10、jaxlib≥0.4.34、Apple は今も experimental 表記。「死亡/終息」は誤りで Part II-3 で訂正)。【高】
- **MuJoCo Warp は NVIDIA-CUDA 専用、Apple Silicon 非対応**。【高】
- Mac の加速方針 = **小RL網は CPU 既定**(MPS は小MLPで遅いことが多い)/ 大網は **Apple MLX**。`PYTORCH_ENABLE_MPS_FALLBACK=1` は性能が読めないため**既定にせず診断時のみ**(§11-A3)。128GB統合メモリは大モデルに有利。【中〜高】
- NVIDIA非依存のロボット学習参照実装:**LeRobot**(HuggingFace, il_sim)と **UniLab**。※UniLabの3–11倍主張は**ベンダ自己申告**であり独立再現なし、事実として引用しない。【中/UniLabは低】
- クラウド・フォールバック候補(**旧検討候補、決定2により不採用 / II-9**):Colab T4(無料・12h上限・不安定)、Modal/Replicate/Baseten、build.nvidia.com の NIM/GR00T。【中〜高】

---

## 4. 成果物の定義(何を作るか)

**独立OSSリポジトリ1本** = `SKILL.md` 準拠の MuJoCo スキル(基盤3〜4本 + 行動層、完全な構成は II-4)。`skills` CLI で Claude Code / Codex 両方に1コマンド導入。NVIDIA同様の「オーケストレーター(SKILL.md がツールを駆動)」だが、**計算は全部ローカルで完結**(**決定2: クラウドGPUは不採用**, II-9。Part II が正準)。

> **設計原則**: スキルの価値は「MuJoCo公式docの書き写し」ではなく **Mac固有の落とし穴の知識を埋め込むこと**。薄いラッパーはエージェントが自分でdocを読めば済むので無価値。各スキルは「**非自明なMac固有ハマりを数時間ぶん節約できるか?**」で選別する。

### スキル構成

| # | スキル名 | スコープ | 価値の核 |
|---|---|---|---|
| 1 | `mujoco-env-setup` | arm64/macOS・CUDA無しを検出 → `mujoco` 導入、スモークテスト、**導入後の可視性チェック**(§11-C1 の #851 回避)、MPS vs CPU 実機ベンチで推奨提示 | 環境プローブ + 可視性検証 |
| 2 | `mujoco-viewer` | **mjpython 専用・対話ビューアのみ** | mjpython メインスレッド作法 |
| 3 | `mujoco-offscreen-render` | **通常 python・ヘッドレスのみ**(RGB/深度/セグ・動画)。**mjpython では絶対に呼ばない** | CGL オフスクリーンの罠回避 |
| 4 | `mujoco-rl-train`(+評価) | 単機RL(PPO/SAC)+ BC + ロールアウト評価 + ドメインランダム化。**収束する実タスク**を同梱 | CPU/MLX で実際に学習が回る設定 |

> ②③の分離が最重要(§11-A1)。~~クラウド連携が必要になれば後から `mujoco-cloud-offload` を追加~~ → **決定2(II-9)でクラウドは不採用**。ベース3〜4本(+ Part II 行動層)は**クラウド無しで完結**し、それが製品の全て。

---

## 5. アーキテクチャ(スキル形式・配布)

### 5.1 1スキルのディレクトリ(NVIDIAモデルのオープン部分集合)

```
mujoco-rl-train/
  SKILL.md                 # 必須(name=ディレクトリ名, description)
  references/              # *.md: PPO/SAC レシピ、MPS vs CPU、トラブルシュート
  scripts/                 # train.py, rollout.py, randomize.py(実ローカル計算)
  assets/                  # 例 MJCF, config
  evals/
    evals.json             # 正例タスク + 負例(デコイ)タスク ≥1
    files/<task>/code/*.py # fixtures
  skill-card.md            # 任意(ガバナンス/出自カード)
  BENCHMARK.md             # 任意(評価レポート)
  skill.oms.sig            # 任意(自己署名)※費用対効果は低い、§11-C3
```

### 5.2 リポジトリ全体(独立構成)

```
mujoco-skills/                      # github.com/<user>/mujoco-skills
  README.md                         # no-NVIDIA 宣言・スコープ限定を明記
  skills.sh.json                    # $schema https://skills.sh/schemas/skills.sh.schema.json
  .agents/skills/                   # エージェント非依存パス(推奨)
    mujoco-env-setup/
    mujoco-viewer/
    mujoco-offscreen-render/
    mujoco-rl-train/
  CONTRIBUTING.md  LICENSE  DCO
```

`.agents/skills/`(エージェント非依存)を推奨。`.claude/skills` 等エージェント固有パスは避ける。

### 5.3 フロントマター方針

- `name`(=ディレクトリ名)・`description`(何を+いつ)は必須。
- `license: Apache-2.0` を付ける(NVIDIA実務でも事実上必須)。
- `compatibility` に **no-NVIDIA の約束を明記**(仕様公認の場所、最大500字):
  > 例:`Requires Python 3.11+, mujoco (pip) >=3.9; runs CPU-only on macOS Apple Silicon, no NVIDIA GPU required. Viewer needs mjpython. Optional MPS/MLX acceleration.`
- `version`/`author`/`tags` は `metadata` 配下に入れ子。
- 非仕様キー(`when_to_use`/`tools`/`permissions`/`triggers`/`user_invocable`)はトップレベルに置かず `metadata` 配下へ。`allowed-tools` は実験的扱い、サンドボックスとして依存しない。

### 5.4 クロスエージェント導入

```
npx skills add <user>/mujoco-skills --skill '*' --agent claude-code --agent codex
```

§11-C1(#851)と §12 の Codex パス問題を Phase 1 で実機検証し、`--copy`/project スコープ + 可視性チェックで回避する。

### 5.5 「独立公開 vs NVIDIA/skills へPR」── 結論: **独立公開**

NVIDIA/skills は製品リポジトリの日次ミラーで、コミュニティスキルをPRできない。NVIDIA-Verified は第三者には構造的に取得不可能。独立リポジトリでも導入体験は同一。discoverability 目的で skills.sh 掲載は検討可(コミュニティ掲載経路は未文書化=低確度、導入自体は依存しない)。

### 5.6 ローカル検証・自己署名(全てMacネイティブ)

- `skills-ref validate ./<skill>` でフロントマター・命名を検証。
- `evals/evals.json` は手書き可、Claude Code でローカル実行。NVSkills-Eval 本体は非公開。
- `model-signing` で自己署名・自己検証は可能だが、第三者には信頼根が無く**費用対効果は低い**(§11-C3)。ロードマップでは降格。

---

## 6. ローカルコア(NVIDIA不要で動く範囲)

stock な Apple Silicon Mac + `pip install mujoco` + Node/npx + PyTorch-MPS or MLX。**CUDA・NVIDIA GPU・クラウド一切不要。**

| 能力 | ローカル動作 | 正直な性能 |
|---|---|---|
| CPU剛体物理・接触シミュ | ネイティブ arm64 `mujoco` 3.9.0 | 単環境は高速。**並列RLの実時間は別問題(§11-A2)** |
| 対話ビューア | `mjpython` 必須 | 通常シーンはリアルタイム |
| オフスクリーン描画 | CGL(EGL不要)→ RGB+深度+セグ | RL/評価に十分。**フォトリアルではない**、Mac深度精度は限定 |
| Menagerie 70+モデル | 箱出しで動く | 全カバレッジ確認済 |
| 単機RL(PPO/SAC) | 物理=CPU MuJoCo、網=**CPU既定**(§11-A3)、必要時のみ MLX | 小タスク(cartpole/reacher/簡単な四足歩容)向き。サンプル効率律速タスクは遅い |
| Behavior Cloning | MLX/PyTorch-MPS | 実用的。128GB統合メモリが有利 |
| ロールアウト+小規模評価 | 種固定の決定論的ロールアウト+指標+動画 | 研究・教育に十分 |
| ドメインランダム化 | 質量/摩擦/照明/テクスチャ | 安価。Cosmos合成データの**明確に弱い**代替 |

**エンジン選択(修正済み・重要)**:
- 学習の既定を **CPU(小RL網)** とする。**MPS は小MLPでCPUより遅いことが多い**(§11-A3)。GPUを使うなら **MLX** を優先しベンチで判断。MJX+jax-metal はベースに据えない(jax-metal はメンテ停止・凍結 → Part II-3)。
- `mujoco-env-setup` がユーザー実機で **MPS vs CPU を計測して勝者を推奨**する。

参照すべきNVIDIA非依存実装:**LeRobot**(il_sim)。UniLab は要注意(自己申告ベンチ)。

---

## 7. クラウド・フォールバック層(※v2.1:決定2により【旧方針 / 不採用】)

> **【v2.1 注記】本節は決定2(100%ローカル堅持・クラウド不採用, II-9)により製品には実装しない「旧方針」。NVIDIA-free 純度を最優先する判断のため `mujoco-cloud-offload` は作らない。以下は当時の検討記録として参考保存する。**

CUDA必須の作業だけ、**「GPU非検出 → ローカルで実行 → ローカル結果を先に提示 → それを超えたい時だけクラウドを提案」**。

| 領域 | 手段 | 注意 |
|---|---|---|
| 大規模並列RL(MJWarp/AlpaGym級) | Colab T4 / Modal / Replicate / Baseten | Colabは12h上限・アイドル切断・本番不向き |
| フォトリアル合成データ(Cosmos/NuRec/OmniDreams) | NIM / build.nvidia.com | 多くはゲート資産。ローカルは安価DRのみ=**実質手が届かない** |
| 大規模VLA推論(GR00T, Alpamayo 32B) | NIM/GR00Tエンドポイント | データセンタGPU前提、Mac実測なし |

**ベースをNVIDIA-freeに保つ規律**:
1. クラウドは**別スキル**/オプトイン分岐。スキル1〜4の依存にしない。
2. ローカル能力にクラウド資格情報・NGCキー・HFゲートtokenを要求しない。
3. 各クラウド選択肢にコスト/レイテンシ/信頼性の注意を併記。
4. クラウド経路に入る瞬間「**ここから先はNVIDIA-free保証を外れます**」と1回バナー表示(ゲート資産の壁に必ずぶつかるため)。

---

## 8. 実現可能性マップ

> **【v2.2 注記】下表「クラウドGPU」列は v1.0 当時の整理。決定2(100%ローカル, II-9)により**クラウドは不採用**=この列の項目は**スコープ外**(参考表示)。製品が約束するのは「ローカルMac ✅」列のみ。最新の行動別判定は II-5 を見ること。**

| 能力 | ローカルMac | クラウドGPU(=決定2でスコープ外) | 手が届かない |
|---|:--:|:--:|:--:|
| CPU物理・接触シミュ | ✅ | | |
| 対話ビューア / オフスクリーン描画 | ✅ | | |
| MJCF/URDF/Menagerie 読込・検証 | ✅ | | |
| 単機〜少数並列RL(PPO/SAC) | ✅ | | |
| Behavior Cloning | ✅ | | |
| ロールアウト+小規模評価 | ✅ | | |
| ドメインランダム化 | ✅ | | |
| MJX GPU加速 on Mac | ⚠️脆い(jax-metal メンテ停止・凍結) | ✅(CUDAクラウド) | |
| 数千env並列RL(MJWarp級) | | ✅ | |
| フォトリアル合成データ | | ⚠️NIM経由 | ✅ ローカルは安価DRのみ |
| Madrona バッチ視覚RL | | ✅(CUDA+Linux) | |
| 大規模VLA推論(GR00T/Alpamayo) | | ✅ | |
| NVIDIA-Verified バッジ | | | ✅ 第三者には不可能(self-signのみ) |

**結論**: 制御+物理+中規模RL+評価のスライスは**完全かつ正直にMacローカル**。フォトリアル合成データと大規模GPU-RL/VLA は不可能で、そこは偽らない。

---

## 9. Claude × Codex 協業モデル

両者が**同一の `SKILL.md`** を読む(write-once/run-everywhere)。フォーマット分岐ではなく**役割分担**。1リポジトリを `--agent` 複数指定で両方に導入。

- **Claude Code** = 設計・執筆・評価・レビュー。`SKILL.md` 本文と `references/` の長文技術記述、`evals.json`(デコイ含む)設計、`BENCHMARK.md`、`skills-ref validate`、no-NVIDIA の一貫性管理。
- **Codex** = コード生成・実行・環境配線。`scripts/`(train/rollout/randomize)生成、mujoco導入・mjpythonスモーク・MPS/MLX配線、実行時 `/skills`・`$` で起動される側。
- **共通基盤 / 受け渡し**:リポジトリが単一の真実源。Claudeが書いたスキルを Codex が即実行できる(形式が同一だから)。
- **クロスエージェント評価をゲートに**:NVIDIA の NVSkills-Eval が claude-code と codex 両方で採点するのに倣い、各スキルの `evals.json` を**両エージェントでスポットチェック**(エージェント差を検出)。※ただし完全自動の二重評価ハーネスは工数大(§11-e)、Phase 1 は手動スモークに留める。
- **DCO サインオフ**(`git commit -s`)で出自を明確化(NVIDIAの `dco.yml` に倣う)。

---

## 10. フェーズ別ロードマップ

> **【v2.2 注記 ― 旧版アーカイブ】これは v1.0 当時の「基盤のみ」ロードマップ(Phase 2 が RL 中心、Phase 3 にクラウド/Plugin)。現在は**旧版**。**正式なロードマップは II-8**(モデルベース背骨・Phase 1.5 で Plugin・Phase 2 で GO2 灯台デモ・クラウド不採用)。**実装は II-8 に従うこと。** 以下は経緯の記録として残す。**

### Phase 1 — MVP(NVIDIA無しで入って動く)
- スキル①②③を出荷。各々:有効な `SKILL.md`、`## Instructions` + `## Examples`、最小 `scripts/`+`references/`。
- 1コマンドで Claude Code / Codex 両方に導入できることを実機確認。**Codex書込先と #851 を必ず潰す**(§11-C1, §12)。
- スモークテスト:導入 → Menagerieモデル読込 → step → オフスクリーン描画 →(別工程で)mjpythonビューア起動。
- リポジトリ公開、`skills.sh.json`、READMEに no-NVIDIA 宣言とスコープ限定。
- **マイルストーン**: NVIDIA環境ゼロのMacユーザーが `npx skills add` → 描画済みロールアウトまで10分以内。

### Phase 2 — 実用(本当に学習・評価できる)
- スキル④を追加。学習既定は **CPU(小RL網)**、必要時 MLX。MJX/Playground は注意付きフラグの裏に。
- 各スキルに `evals/evals.json`(正例+負例)を用意、**実機で**ベンチして `BENCHMARK.md`。
- 任意で `skill-card.md`(*Deployment Geography: Global*、*Known Risks/Mitigations* に CPU性能注意)。
- **マイルストーン**: MacBook 上で四足/簡単タスクのRLポリシーを end-to-end 学習し評価。完全オフライン。

### Phase 3 — エコシステム
- ~~任意 `mujoco-cloud-offload`(厳格オプトイン)~~ → **決定2で不採用(II-9)。クラウド層は作らない。**
- LeRobot 相互運用、Menagerie タスクテンプレ拡充、Playground env カタログ。
- Codex **plugin.json** 対応(§11-C2)。skills.sh 掲載検討。
- CI:ローカル `skills-ref validate` + 二重エージェント評価、DCO 強制。
- ~~**マイルストーン**: ローカル⇄クラウドへのバースト~~ → **決定2で不採用**。正式なマイルストーンは II-8 を参照。

---

## 11. 敵対的レビュー:成否を分ける地雷

> 本節は辛口レビューエージェントが**実証付きで**発見した問題。元案の楽観記述はここで修正済み。**NVIDIA境界は綺麗に扱える。死ぬのは地味なローカルUX。**

### (a) 「Apple Siliconで快適に動く」が崩れる箇所

**A1. ビューアとオフスクリーン描画は同一プロセスで両立できない【確度: 高・実証済】**
`launch_passive` は mjpython(メインスレッド描画)が必要。だが mjpython 配下で `mujoco.Renderer`(オフスクリーン)を回すと `NSInternalInconsistencyException: NSWindow drag regions should only be invalidated on the Main Thread!`(MuJoCo issue #798)。→ **スキルを②ビューア(mjpython)と③オフスクリーン(通常python)に分離**。オフスクリーンは mjpython 不要(CGL)。最初に試されるのが描画なので、ここでスレッド例外が出たら即死。

**A2. 「単一A100に匹敵」はカテゴリ誤り【確度: 高(数値は中)】**
Mac単環境スループットとA100単環境スループットの比較は無意味。RL実時間は**並列env数**で決まり、Macに並列envのGPU経路は無い(Warp/MJX-GPUはCUDA専用)。4090で約1時間のhumanoid学習がMacでは数日〜非現実的になり得る。→「A100匹敵」は削除。スコープは数百万ステップで収束する小タスクに限定。**数字は実機 `testspeed` で実測してから出す**。

**A3. 「ポリシー網はMPS既定」はRLで逆効果【確度: 高】**
RLは小MLP×小バッチ×微小op連鎖で、MPSはディスパッチoverheadでCPUより遅いことが多い。`PYTORCH_ENABLE_MPS_FALLBACK=1` は無対応opの暗黙CPU退避=性能予測不能の地雷でもある。→ **小RL網はCPU既定**。GPUを使うなら **MLX**。`env-setup` が実機で MPS vs CPU を計測して推奨。

**A4. jax-metal は「メンテ停止・凍結」【確度: 高 / ※v2.0で表現を訂正】**
~~JAX保守側が2025-12に jax-metal の全issueをクローズ。~~ ← **この記述は過剰主張だったため訂正**(v2.0)。検証で確認できた事実:**jax-metal の PyPI 最新は 0.1.1(2024-10-08)で約20か月更新なし、jaxlib≥0.4.34 要求、float64/complex 非対応。Apple 公式ページ・JAX 公式ドキュメントは今も「experimental」表記のまま(正式な deprecation はされていない)**。JAX issue #34648 では保守側が "seems unmaintained" と述べる。現行 JAX(0.8/0.9系 jaxlib)は jax-metal を壊す。生きている fork は **jax-mps 0.10.3(Alpha, jaxlib 0.10.x, ResNet18 で CPU比約3倍)** と **applejax(jaxlib 0.9.x)**。→ **正しい言い方は「死亡」ではなく「メンテ停止・version-fragile・baseline には使わない」**。fork は**将来の watch 項目**(未検証、jaxlib系統が異なるためバージョン固定必須)。**実務結論(ベース学習に据えない、CPU/MLX を既定)は不変。**

### (b) NVIDIA/クラウド依存がこっそり戻る箇所

**B1. ドメインランダム化はフォトリアル合成データの代替にならない【確度: 中】** ローカルは安価な視覚DRのみ。sim-to-real視覚には往々に不足。→ READMEでスコープを「制御・物理・中規模RL・評価」に明示限定し、DRが同等であるかのように見せない。

**B2. クラウド経路はNGC/HFゲートに必ずぶつかる【確度: 中】** → ~~`mujoco-cloud-offload` 起動時にバナー~~。**※決定2(II-9)でクラウド不採用のため本項は不適用**(そもそもクラウドを使わないので、この壁に当たらない)。

### (c) スキル形式・配布の誤った前提

**C1. `npx skills add -g -a claude-code` は現在バグで不可視【確度: 高・実証済】**
vercel-labs/skills **Issue #851(オープン)**:Claude Code へのグローバル導入が `~/.agents/skills/` に書き、`~/.claude/skills/` シンボリックリンクを作らず、**スキルが見えない**。Phase 1 マイルストーンを直撃。→ `--copy`/project スコープ導入 + `env-setup` に導入後可視性チェック(無ければシンボリックリンク作成)。CLIバージョン固定。PR #1089 マージ後に再検証。

**C2. Codex は plugin システムが配布の第一級になりつつある【確度: 中】**
Codex CLI v0.117.0(2026-03)で **plugin(plugin.json)** が配布primitive、skill は authoring primitive に。生のSKILL.md配布も動くが非idiomaticに。→ Phase 2 で `plugin.json` を追加。

**C3. 自己署名は本用途では儀式に過ぎない【確度: 中】** 第三者に信頼根が無く、3-Alphaで動く依存。→ ロードマップから降格、1行の「将来」注記に。工数は C1 の可視性チェックに回す。

### (d) 6スキルは価値があるか薄いラッパーか
- **薄い/境界**: `env-setup`(価値は可視性チェック+no-NVIDIAプローブに集約せよ)、`model-load`(価値はエラー修復/Menagerie取得の実質に依存)。
- **本当に価値**: viewer/offscreen(mjpython/CGLの地雷を正しく回避するなら)、`rl-train`(**実際に収束する**CPU/MLX PPOを同梱するなら)、`rollout-eval`(種固定+指標は本物の糊)。
- → **「数時間のデバッグを節約する非自明なMac固有知識を埋め込むか?」**で判定。**6本より3〜4本の強いスキル**。

### (e) スコープ・工数の現実
- **二重エージェント評価ハーネスが隠れ工数**:NVSkills-Eval非公開のため手作りになり、スキル本体並みの工数。→ Phase 1 は手動スモークのみ、自動化はPhase 3 か「両方でスポットチェック」に緩める。
- **Macベンチの穴**:引用SPSは未実測。正直な `BENCHMARK.md` には**実機(M系)が必須**。→ 実機の有無が前提条件。
- **保守のトレッドミル**:`skills` CLI(未解決バグ)、Codex plugin(2026-03に変化)、MuJoCo/JAX の3つの動く標的。一発作りでなく継続保守を見込む。

### 成功に必須な3条件
1. **描画が初回で"ただ動く"**:ビューア(mjpython)とオフスクリーン(通常python)の分離。初手で `NSWindow` 例外が出たら即死。(A1)
2. **導入してスキルが実際に見える**:#851 を `--copy`/project スコープ+可視性チェックで回避。(C1)
3. **少なくとも1つのRLスキルが実機CPUで許容時間内に収束**:実測したタスクで。(A2/A3+ベンチ穴)

### 最も起きやすい失敗
**「CUDA境界」ではなく「最初の5分の摩擦」による死**。導入してスキルが見えない or 描画デモがスレッド例外で落ち、ユーザーが「またMacで動かないやつか」と結論=このプロジェクトが否定したかったまさにその評決。難しいのはNVIDIAについて正直であることではなく、**ローカルMac経路を最初のコマンドに耐えさせること**。

---

## 12. リスク・未検証事項一覧(確度付き)

| 項目 | 内容 | 確度 | 対応 |
|---|---|---|---|
| jax-metal メンテ停止・凍結(※「死亡」は訂正、Part II-3) | ベース学習に使えない | 高 | CPU/MLX既定、forkはwatchのみ |
| Warp/Newton GPUはCUDA専用 | 数千env並列RLはMac不可 | 高 | スコープ外(決定2でクラウド不採用) |
| フォトリアル/大VLAにMac代替なし | DRは弱い代替 | 高 | 正直に明記 |
| NVIDIA-Verified不可 | 第三者は取得不能 | 高 | self-signのみ、endorsement匂わせ禁止 |
| `allowed-tools` 実験的 | 構文/対応に差 | 高 | サンドボックスに依存しない |
| ~~**Codex導入先**~~ **✅検証済(II-A)** | global は **`~/.agents/skills/`**(`~/.codex/skills` は `.system` のみ) | — | 解決:Codex は `~/.agents/skills/` を読む |
| ~~**#851 グローバル導入バグ**~~ **✅検証済(II-A)** | skills CLI **1.5.10 では再現せず**=`~/.claude/skills/` に正しく着地・可視 | — | 解決(ただしCLIバージョン固定推奨)。既定は copy で symlink 脆弱性も回避 |
| Kiro エージェントID | `--agent kiro-cli` 未一次確認 | 中 | `npx skills add --help` で確認 |
| CLI symlink既定 | キャッシュ削除で壊れ得る | 中 | `--copy` 推奨 |
| skills.sh 掲載経路 | コミュニティ掲載未文書化 | 低 | 導入は非依存、discoverabilityのみ |
| model-signing 3-Alpha | API変更リスク | 中 | 自己署名は nice-to-have |
| **Macベンチ未実測** | 公開のM系MuJoCoベンチ無し | 高 | 実機で測定(前提条件) |
| UniLab 3–11倍 | 自己申告・未再現 | 低 | 事実として引用しない |

---

## 13. 他AIに議論してほしい論点

開発着手前に、別AI・人間で詰めたい問い:

1. **スキル粒度**: 4本に絞るか、umbrella 1本にするか、逆に分割するか。進行的開示の発火精度・per-skill eval のしやすさ vs 保守コストのトレードオフ。
2. **学習エンジンの既定**: CPU既定は妥当か。MLX を第一GPU候補にすべきか、PyTorch-MPS とどちらをデフォルト推奨にするか。タスク種別ごとの分岐ルールをどう書くか。
3. **「収束する実タスク」の選定**: Phase 2 マイルストーンの「許容時間内に収束」を満たす具体タスク(cartpole/reacher/quadruped gait 等)はどれか。許容時間の定義は。
4. **オフスクリーン描画の堅牢性**: macOS の CGL オフスクリーンが CI / ヘッドレス(SSH/Docker for Mac)でどこまで安定か。深度・セグメンテーションの精度限界は実用に足るか。
5. **配布バグ #851 / Codex パス**: 現時点の `skills` CLI 実挙動の確定(実機ログ)。PR #1089 の状況。plugin.json を Phase いつで入れるか。
6. ~~**クラウド・フォールバックの設計**~~ → **決定2(II-9)でクラウド不採用のため本論点はクローズ**。
7. **評価(evals)設計**: 負例(デコイ)を「MacでIsaac Labをセットアップ」→ `expected_skill: null` のように境界教育に使うべきか。二重エージェント評価をどこまで自動化するか。
8. **ライセンス/出自**: Apache-2.0 で良いか。MuJoCo/Menagerie/LeRobot 由来コードのライセンス整合。
9. **NVIDIAとの関係性のフレーミング**: 「補完」を公式にどう打ち出すか。商標・endorsement の線引き。
10. **実機前提**: M系Mac実機の確保(ベンチ・地雷再現に必須)。対象とする最小スペック(メモリ/チップ世代)。

---

## 14. 出典一覧

**スキル形式・配布**
- NVIDIA/skills リポジトリ — https://github.com/NVIDIA/skills
- NVIDIA/skills スキル一覧 — https://github.com/NVIDIA/skills/tree/main/skills
- Agent Skills 仕様 — https://agentskills.io/specification
- NVIDIA-Verified Agent Skills(ガバナンス) — https://developer.nvidia.com/blog/nvidia-verified-agent-skills-provide-capability-governance-for-ai-agents/
- スキル署名docs — https://docs.nvidia.com/skills/signing-agent-skills
- Skill Card テンプレ — https://github.com/NVIDIA/Trustworthy-AI/blob/main/Skill%20Card.md
- skills-ref(検証) — https://github.com/agentskills/agentskills/tree/main/skills-ref
- model-signing — https://pypi.org/project/model-signing/
- Vercel Labs `skills` CLI — https://github.com/vercel-labs/skills
- `skills` CLI README — https://raw.githubusercontent.com/vercel-labs/skills/main/README.md
- npm `skills` — https://www.npmjs.com/package/skills
- skills.sh(Claude Code) — https://www.skills.sh/agent/claude-code
- OpenAI Codex Skills docs — https://developers.openai.com/codex/skills
- OpenAI skills カタログ — https://github.com/openai/skills

**GTC Taipei 発表・Physical AI**
- NVIDIA Newsroom(Physical AIスキル公開) — https://nvidianews.nvidia.com/news/nvidia-releases-major-collection-of-open-source-agent-tools-and-skills-for-physical-ai
- CVPR Physical AI research agent skills — https://blogs.nvidia.com/blog/cvpr-physical-ai-research-agent-skills/
- The Robot Report(GTC Taipei) — https://www.therobotreport.com/nvidia-releases-new-updated-tools-physical-ai-gtc-taipei-computex/
- Newton physics engine — https://blogs.nvidia.com/blog/newton-physics-engine-openusd/
- Newton 発表(NVIDIA Developer) — https://developer.nvidia.com/blog/announcing-newton-an-open-source-physics-engine-for-robotics-simulation/

**MuJoCo / Mac / 加速**
- mujoco(PyPI) — https://pypi.org/project/mujoco/
- MuJoCo Python docs(mjpython/CGL) — https://mujoco.readthedocs.io/en/stable/python.html
- MJX docs — https://mujoco.readthedocs.io/en/stable/mjx.html
- MJWarp docs — https://mujoco.readthedocs.io/en/stable/mjwarp/index.html
- mujoco_warp — https://github.com/google-deepmind/mujoco_warp
- Warp FAQ(CUDA要件) — https://nvidia.github.io/warp/faq.html
- Apple Metal JAX — https://developer.apple.com/metal/jax/
- Apple Metal PyTorch — https://developer.apple.com/metal/pytorch/
- LeRobot il_sim — https://huggingface.co/docs/lerobot/il_sim
- UniLab(自己申告ベンチ・要注意) — https://unilabsim.github.io/
- build.nvidia.com モデル — https://build.nvidia.com/models

**敵対的レビューの実証**
- jax-metal status(unmaintained / コミュニティfork議論) — https://github.com/jax-ml/jax/discussions/34648
- applejax / jax-mps fork — https://github.com/danielpcox/applejax
- MuJoCo issue #798(mjpython+オフスクリーン衝突) — https://github.com/google-deepmind/mujoco/issues/798
- vercel-labs/skills issue #851(グローバル導入バグ) — https://github.com/vercel-labs/skills/issues/851
- Codex CLI plugin system v0.117.0 — https://codex.danielvaughan.com/2026/03/30/codex-cli-plugin-system/
- PyTorch MPS が小workloadでCPUより遅い — https://www.codegenes.net/blog/pytorch-mps-slower-than-cpu/
- 大規模並列humanoid locomotion(並列env学習時間) — https://arxiv.org/pdf/2407.05148

> ⚠️ **出典の鮮度に関する注意**: 本書の一部はWebFetchによる要約に基づく。GTC Taipei 2026 や Codex plugin(2026-03)、#851 など2026年の動きは流動的。開発判断の根拠にする前に、特に §12 の「中/低確度」項目は一次情報で再確認すること。

---

## 15. 用語集

| 用語 | 説明 |
|---|---|
| Agent Skill / SKILL.md | エージェントに「何をいつどう行うか」を教えるオープン規格の指示セット。`SKILL.md`(YAMLフロントマター+Markdown本文)が本体 |
| 進行的開示 | name+description→本文→参照ファイル の3段階でコンテキストを段階的にロードする仕組み |
| `skills` CLI | Vercel Labs製の導入ツール。`npx skills add` で任意GitリポジトリからエージェントへSKILL.mdを配置 |
| MuJoCo | Google DeepMind の物理シミュレータ。Apple Silicon ネイティブ動作 |
| MJX | MuJoCo の JAX/XLA 実装。GPU並列向けだがMacのApple GPU加速は jax-metal 依存=メンテ停止・凍結のため当面不可(Part II-3) |
| MuJoCo Warp / Newton | NVIDIA Warp ベースのGPU高速MuJoCo / 物理エンジン。**CUDA専用** |
| mjpython | macOSで対話ビューアを動かすための専用ランチャ(GUIをメインスレッドで回す) |
| CGL | macOSのOpenGLコンテキスト。オフスクリーン描画に使う(EGL不要) |
| MPS / MLX | Apple GPU向けPyTorchバックエンド / Apple製ML框架 |
| NIM / OSMO / NGC | NVIDIAの推論マイクロサービス / ジョブ基盤 / コンテナ・モデルレジストリ(いずれもNVIDIA環境前提) |
| VLA | Vision-Language-Action モデル(例:GR00T, Alpamayo) |
| Menagerie | MuJoCo公式のロボットモデル集(70+) |
| NVIDIA-Verified | NVIDIA非公開パイプラインによるスキル検証バッジ。第三者取得不可 |
| OMS署名 | OpenSSF Model Signing形式の検証署名。`skill.oms.sig` |

---

*本書 Part I は基盤の検証記録である(§13 は当時の論点記録)。**意思決定は II-9 で確定済**。**実装は Part II / II-8(ロードマップ)/ II-9(決定)に従う。***

---
---

# Part II ― ロボット行動スキル層(v2.0 ブラッシュアップ)

> **このパートの位置づけ**: Part I(§1〜§15)は「Mac で MuJoCo を NVIDIA なしに動かす**基盤**」戦略だった。Codex のレビューで「**基盤だけで、ロボット**行動**を生成・訓練・評価する層が無い**」と指摘され、それが正しいので追加した層。ユーザーの最終ゴール=**Unitree G1/H1/GO2 ほかで 歩く・座る/立つ・走る※・障害物回避** を実現するスキルズセット(**※run は決定2でスコープ外=将来の事前学習方策待ちの保留枠, II-9**)。本パートは Part I §11-A4 の「jax-metal 死亡」表現を訂正する(Part II-3)。

## II-1. M5 Max 実機ベンチマーク(実測)

Part I が繰り返し「Mac実機性能が未測定」と空白にしていた点を、**実機(M5 Max, 18コア/12性能コア, 128GB, mujoco 3.9.0)で実測**して埋めた。Menagerie の実モデル(G1=35DOF/29act, GO2=18DOF/12act)を使用。

| モデル | forward step | リアルタイム比 | `mjd_transitionFD`(FDヤコビアン, 単一スレッド) |
|---|---|---|---|
| **GO2** | 60,342 sps | **121×** | **0.99 ms/call** |
| **G1**(29act) | 31,310 sps | **63×** | **3.50 ms/call** |
| G1+両手(43act) | 22,322 sps | 45× | 7.20 ms/call |

**この FD コストはモデルベース歩行の実現性を左右する最重要指標の一つ**(ただし歩行の成否は FD 単独では決まらず、接触スケジュール・コスト設計・摩擦・アクチュエータ制約・遅延・姿勢安定化にも依存する。下の判定はこれら前提込み)。
- **GO2**: FD 0.99ms → 予測制御は余裕でリアルタイム(121×)。**モデルベース歩行は高確度で実現可能**。
- **G1**: FD 3.50ms。iLQR の前向き線形化スイープ(地平線H=30, 単一スレッド)で約105ms(~9.5Hz)。性能コアで約8倍並列化を楽観すると ~13ms/sweep(~76Hz)。MPCステップは複数反復+ロールアウト+ラインサーチを 50–100Hz で要するため、**ギリギリ(margin-free)= 不可能ではないが約束できない「研究賭け」**。
- 結論:**ヒューマノイド歩行で約束してよいのは「事前学習方策の再生」。モデルベース歩行は実機クローズドループ計測が取れるまで売り文句にしない。**

## II-2. 中心的な再設計 ― モデルベース制御を「背骨」に

Part I は暗黙に **RL** を行動エンジンに想定していた。**これは Mac では逆**。検証の結論:

- **Mac が原理的にできない唯一のこと = ゼロからの移動RL**(数千env並列=CUDA前提。Playground は Go1 約5分/G1 30分未満だが**dual RTX 4090・チェックポイント非同梱**。MJX-on-Mac は CPU-via-XLA で約10倍遅、jax-metal 凍結)。
- **Mac が確実にできること = モデルベース制御**(PD / CPG / convex-MPC / iLQG。全依存に**arm64 ネイティブwheel**: osqp 1.1.x, `pin`(Pinocchio), CasADi, qpsolvers)。

→ **行動生成の背骨をモデルベース制御に置き、RL は四足の小規模ローカル実験に限定する**(run / ゼロ学習ヒューマノイドRL / クラウドバースト は決定2でスコープ外・不採用, II-9)。

**裏付け(検証済)**:
- **MuJoCo MPC(mjpc)** が `humanoid/{stand, walk, tracking, interact}` と quadruped タスク(Unitree A1)を同梱。`quadruped.cc` は完全なモデルベース歩容スケジューラ(Stand/Trot/Walk/Canter/Gallop + Flip + Biped + **Scramble=目標指向の障害物ナビ**)を Predictive Sampling / iLQG / Gradient で駆動、CPUマルチスレッド・リアルタイム設計。
- **存在証明**: `go2-convex-mpc`(GO2, CasADi+OSQP+Pinocchio, **~2.70ms MPCサイクル / 30–50Hz / GPUなし・RLなし**=README記載);whole-body iLQR 論文(arXiv:2503.04613, `mujoco_mpc_deploy`)が **Go1/Go2 移動 + H1 ヒューマノイド歩行**を iLQR で実演(※ただし **20コア Intel i9** 上、~50Hz)。
- **事前学習方策**: `unitree_rl_gym` が **G1/H1/H1_2 の `motion.pt` を同梱**、`deploy_mujoco.py` が **`torch.jit.load` で CPU 実行(CUDA不要)**。→ ヒューマノイド歩行の最速・確実な NVIDIA-free 勝ち筋。**GO2 は同梱方策が無い**(=GO2はRL/制御ロボット、G1/H1は再生ロボット、と役割が綺麗に反転)。

**mjpc の注意**: macOS CI(macOS-15)はあるが、**Apple Silicon ネイティブビルドの未修正バグ #378**(`-msse4.1` の x86 フラグが arm64 ビルドに漏れる)があり、制御は **C++ からビルドした gRPC `agent_server` 経由(pip 一発ではない)**。→ mjpc は**ビルド検証ゲート付きの「上級」スキル**であり Phase 1 依存にはしない。

## II-3. Codex の3点への賛否

| # | Codex の指摘 | 判定 | 内容 |
|---|---|---|---|
| 1 | jax-metal「死亡」は強すぎ→「experimental/version-fragile」に | **調整して同意** | 検証で Part I §11-A4 の「2025-12 全issueクローズ/死亡」は**裏取れず=過剰主張**と判明。正しくは **0.1.1(2024-10)で凍結・jaxlib≥0.4.34・Apple は今も experimental 表記・"seems unmaintained"**。fork は jax-mps 0.10.3 / applejax。**実務結論(baseline に使わない)は不変。**→ §11-A4 訂正済。 |
| 2 | plugin.json を前倒し(Phase 1.5/2) | **強く同意・さらに前へ** | 検証確認:**Codex の `.codex-plugin/plugin.json`(name/version/description)が配布単位**で、2つ以上のskill+MCP+apps+hooks を束ねる。`.agents/skills` はローカル発見のみ。行動層スキルセット(6〜9本)はまさに plugin が束ねる対象。→ **Phase 1.5 で plugin.json 作成**。ただし #851(Claude Code グローバル導入でスキル不可視)と Codex 導入パスの実機検証は別問題で Phase 1 で先に潰す。 |
| 3 | 行動スキル層が無い | **最重要・全面同意(ただし1段深く)** | 検証は「穴」どころか**これが製品本体**と裏付け。ただし Codex の7スキルは Part I「薄いラッパーを作らない」原則に一部反する。→ **少数の強いスキル + アセットpack** に再編(II-4)。`mujoco-controller-baselines` は背骨として第一級で全面同意。 |

加えて Codex の **「M5 Max 128GB をローカル旗艦に」も同意**(実測でも 614GB/s 帯域・GO2 121×・G1 63×)。ただし**CUDA数千env の代替ではない**ことは明記する。

## II-4. 統合スキルセット(3層・削ぎ落とし版)

Codex の7スキルを「少数の強いスキル + pack」に再編。**敵対的レビューで `robot-layer`/`task-design` を独立スキルにすると薄くなると指摘されたため、行動スキルへ畳み込む**。

### 基盤層(Part I の4本を踏襲・微改名)
| スキル | スコープ |
|---|---|
| `mujoco-env-setup` | arm64/CUDA無し検出、導入、スモークテスト、**導入後可視性チェック(#851)**、MPS vs CPU 実機ベンチ |
| `mujoco-viewer` | **mjpython 専用**・対話ビューア |
| `mujoco-offscreen-render` | **通常python専用**・RGB/深度/セグ/動画(mjpython下では呼ばない) |
| `mujoco-rl-train-eval` | **小規模CPU/MLX RL(四足寄り)+ BC + ロールアウト評価 + ドメインランダム化**。※「行動エンジン」から**診断/四足RL**へ降格 |

### ロボット層 + 行動層(新規・強いスキルのみ)
| スキル | 種別 | スコープと**実現手段** |
|---|---|---|
| `mujoco-controller-baselines` | **第一級(背骨)** | PD / 姿勢 / 足先軌道 / COM / **CPG歩容** / **convex-MPC** / 簡易MPC。**GO2 trot/walk と G1/H1 stand・sit-to-stand を直接実現**。RL失敗時の診断オラクル。 |
| `mujoco-pretrained-deploy` | **第一級** | Unitree同梱の G1/H1/H1_2 方策(`motion.pt`, CPU `torch.jit`)を実行・軽量fine-tune。**ヒューマノイド歩行の確実な勝ち筋**。※**観測構築/制御周波数/MJCF束縛の契約を所有**(II-6 の訂正必須)。 |
| `mujoco-obstacle-navigation` | **第一級** | rangefinder/幾何/深度の**ローカルプランナが (vx,vy,wz) を出力**し、歩容/方策の上に乗る階層分離。**sim内なら完全 Mac-native**。 |
| `mujoco-mpc-advanced` | **延期/任意(ビルドゲート)** | mjpc(Predictive Sampling/iLQG)。**#378 ビルド検証 + gRPC サーバ管理ゲートの後にのみ出荷**。 |
| `locomotion-pack` / `posture-pack` | **アセットpack(スキルではない)** | trot/walk と stand/sit/get-up の MJCF+コスト仕様+ゲイン(**run は含めない=保留枠, II-9**。将来は別途 `future-run-pack` として方策公開後に追加)。上のスキルが消費。 |

> **削ぎ落とし方針(敵対的レビュー反映)**: 行動スキルは **`controller-baselines` / `pretrained-deploy` / `obstacle-navigation` の3本が本体**。`robot-layer`(MJCF読込/センサ付与/共通(vx,vy,wz)・(q,dq,kp,kd,tau)インタフェース)と `task-design`(コスト/終了/観測。**mjpcヒューマノイドは疎報酬で破綻=コスト正則化必須** という HumanoidBench 知見を内蔵)は**この3本に畳み込む**。mjpc は #378 がarm64でビルド可能と実証できるまで単一の延期スキル。

## II-5. ロボット別 実現可能性マトリクス(実測反映)

凡例:✅ ローカルMac(NVIDIA-free) / ❌ NVIDIA-free では手が届かない / ⚠️ ローカル可だが**実機未検証 or 要重チューニング** /「スコープ外」= 決定2(クラウド不採用)で対象外(II-9)

| 能力 | **GO2**(四足) | **G1**(ヒューマノイド29DOF) | **H1**(ヒューマノイド) |
|---|:---:|:---:|:---:|
| stand / sit-stand(モデルベース) | ✅(PD/CPG, mjpc A1 stand) | ✅(PD。sit-stand用 MJCF+コスト要作成) | ✅(同) |
| walk(モデルベース) | ✅ **高確度**(go2-convex-mpc + CPG の2経路, **FD 0.99ms 実測**) | ⚠️ mjpc/iLQR は存在も**20コアIntelで~50Hz**。**G1 FD 3.5ms→ ~76Hz単反復でmargin-free。要実機クローズドループ検証** | ⚠️ 同(H1歩行はiLQR実演もIntel i9上) |
| walk(ローカルRL) | ⚠️ 境界(CPUで「数時間」=**推論・実測なし**) | ❌ 数千CUDA env要 | ❌ 同 |
| run | ❌ スコープ外/保留枠(II-9) | ❌ 同 | ❌ 同 |
| 障害物回避(sim) | ✅(rangefinderプランナ→歩容に速度指令) | ✅(事前学習方策上の速度指令層) | ✅ 同 |
| 事前学習のローカルfine-tune | ⚠️ GO2同梱方策無し→対象少 | ⚠️ 軽量ローカルfine-tuneのみ(本格fine-tuneはスコープ外, 決定2) | ⚠️ 同 |
| ゼロからのローカルRL | ⚠️ 境界(四足は最良候補だが時間未実測) | ❌ スコープ外(決定2) | ❌ スコープ外(決定2) |

**正直な読み方**: **GO2 = Mac のスイートスポット**(モデルベース歩行=高確度・最良のローカルRL候補)。**G1/H1 = stand は堅実・walk が本当の不確実性**。歩行周波数の引用は全て20コアIntelデスクトップ由来で、実機計測は本書の II-1 が初。mjpc のヒューマノイドは**汎用MuJoCoヒューマノイドで G1/H1 ではない**(MJCF差替+コスト再調整が要る)。→ **ヒューマノイド歩行は「事前学習再生」が確実な勝ち筋、モデルベース歩行は実機計測が取れるまで研究賭け。**

## II-6. 事前学習方策の正しい使い方(重要な訂正)

敵対的レビューが指摘した**最も過大評価されやすい「確実」主張**:
- ✅ **真**: 推論は自明に動く(G1規模MLPで ~0.02ms/call ≈ 5万Hz, CPU)。
- ❌ **罠**: 「**Menagerie の MJCF を読み込んで `motion.pt` を再生**」は**静かに破綻**する。方策は**学習時のMJCF(関節順序・dqスケール・projected gravity・前回action・位相クロック・コマンドベクトル)に束縛**されており、Menagerie の関節順/センサ定義/フレーム規約と一致保証がない。`deploy_mujoco.py` が動くのは**自前の一致したMJCF+観測ビルダを同梱しているから**。Menagerie に向けると**無言で異常歩容**になる。さらに**decimation/PDゲイン**(action スケールに焼き込み)や **`torch.jit` バージョン差**も無言破綻源。
- → **正しい成果物 = 「ベンダの `deploy_mujoco.py` を無改変で Mac CPU 実行」**(これは本当に動く・NVIDIA-free)。**「Menagerie差替+方策再生」は約束しない。** `pretrained-deploy` は方策↔MJCFの束縛を所有し、不一致を**大声で拒否**する責務を負う。

## II-7. sim-to-real のスコープ(Linux/DDS 境界)

**「実機 Unitree で動かす」が何を要求するか**(検証=高確度):
- **Mac-native(sim側)**: 物理・制御ループ・ドメインランダム化・センサ作成・方策export/検査(.pt/.onnx)・**sim内障害物回避**は全て Mac ネイティブ。鍵は**階層分離**(高位プランナが (vx,vy,wz) を出し、低位移動制御が追従)→ 回避層は GPU移動RLを要しない。
- **非Mac-native(deploy側)**: Unitree スタックは **Linux 一択**。`unitree_sdk2_python` は **cyclonedds==0.10.2**・**Linux専用導入手順**、リンクは **DDS/CycloneDDS over Ethernet(192.168.123.x)**。`unitree_mujoco`(公式sim-to-sim検証器)も同じDDSで **Ubuntu専用**。**macOS の Docker Desktop は DDS マルチキャストを実機へ確実に橋渡しできない**(`--net=host` でも)= コードサンドボックス止まり。

→ **推奨: sim-first + 分離した deploy 拡張**。スキルセットは「MuJoCo で検証済の方策/回避」まで**完全 Mac-native**(=ユーザーの行動ゴールを**全ロボット・シミュレーション上で**カバー)。実機 deploy は **`mujoco-deploy-handoff`(Mac側export + 正確な deploy コマンド生成だが、実行はロボットのオンボードLinux/Jetsonクラスや有線Linux機で)** という**文書化されたハンドオフ**にする。**正直性ガード**: 文献は DR を効かせても sim-to-real 成功率 **~15–30%低下**を報告。「MuJoCoで動く」を転移保証として提示しない。実世界の視覚ベース回避(GR00Tクラス)は GPU フォトリアル描画=クラウド、Macコア対象外。

**沈黙の破綻源(敵対的レビュー)**: ①アクチュエータモデル差(Menagerie は理想化PDサーボ、実機はトルク上限/電流飽和/ギア摩擦/遅延=sim が上限を持たないので**sim内で無言**)②接触/摩擦・タイムステップ感度(soft contact dt=2ms 対 実機剛接触)③観測遅延・IMUノイズ。④**ハンドオフスキルは Mac から実行も検証も不可**(Linux/DDS無し)=作者が自機で検証不能な構造。→ 転移を語る前に**アクチュエータ上限+遅延+IMUノイズを sim のコスト/DR に入れる**。

## II-8. 改訂ロードマップ

**Phase 1 — 基盤(NVIDIA-free 導入+実行)** 【+ plugin.json を 1.5 で】
- `env-setup`/`viewer`/`offscreen-render` 出荷。**導入の地雷を実機で潰す**(#851 は `--copy`/projectスコープ+可視性チェック、Codex導入パス、mjpython/CGL スレッド分離=issue #798)。**Phase 1.5: `.codex-plugin/plugin.json` で束ねる**、CLIバージョン固定。
- **マイルストーン**: NVIDIA ゼロの Mac で `npx skills add … -a claude-code -a codex` → モデル読込 → step → オフスクリーン描画 → mjpythonビューア が **10分以内・スレッド例外なし・スキル可視**。

**Phase 2 — ロボット層 + controller-baselines + 最初の実働行動**
- `controller-baselines`/`rl-train-eval`(再スコープ)出荷(+robot層/task-designを畳み込み)。
- **灯台デモ: GO2 trot をモデルベース制御で。** 根拠: **2つの独立存在証明**(go2-convex-mpc + mjpc CPGスケジューラ)、**全依存arm64 wheel**、**マトリクス最高確度**、**mjpcビルドゲート非依存**(純Python convex-MPC/CPG)、**FD 0.99ms 実測**で余裕のリアルタイム。中心命題(モデルベース・NVIDIA-free・実ロボット)を最小リスクで端から端まで証明。
- **マイルストーン**: M5 Max 上で GO2 が MuJoCo 内で trot、**QP解時間/計画レートを実測して `BENCHMARK.md` 公開**。

**Phase 3 — ヒューマノイド stand/walk + 姿勢**
- `pretrained-deploy`(G1/H1/H1_2 `motion.pt` を**ベンダハーネス無改変**で CPU 再生=確実な歩行)、`posture-pack` 出荷。
- **G1/H1 stand + sit-to-stand を `controller-baselines` で**(高確度)。mjpc/iLQR ヒューマノイド歩行は `mujoco-mpc-advanced`(#378ゲート)で**試行し正直に報告**。
- **マイルストーン**: G1 が PD/軌道最適で stand・sit-to-stand;G1/H1 walk は事前学習再生で;モデルベース歩行が Mac で実用かの**実測付き判定**を公開。

**Phase 4 — ナビ + sim-to-real ハンドオフ**(※**決定2によりクラウド層は不採用**)
- `obstacle-navigation`(速度指令プランナ, Scramble型)、`mujoco-deploy-handoff`(Mac export → Linux/ロボット実行、II-7ガード付き)。
- ~~`mujoco-cloud-offload`~~ は **決定2(100%ローカル堅持)により不採用**。**run / ゼロ学習ヒューマノイドRL / 本格fine-tune はスコープ外**(II-9 参照)。
- **マイルストーン**: GO2+ヒューマノイドの sim内回避;`unitree_mujoco`(Linux)を通したdeployハンドオフ1本検証。

## II-9. 意思決定(2026-06-05 **確定**)

> 開発前の3つの分岐をユーザーが確定。以降の設計はこれに従う。

| # | 決定 | 選択 | 帰結 |
|---|---|---|---|
| 1 | 成果物スコープ | **sim-first + 文書化ハンドオフ** | MuJoCo(Mac)で行動を生成・評価する完全Mac-nativeな製品。実機 deploy は `mujoco-deploy-handoff`(Mac export → ロボットのオンボードLinuxで実行)。Macからは実機を直接駆動しない。 |
| 2 | 手の届かない行動のクラウド姿勢 | **100%ローカル堅持(クラウド不採用)** | `mujoco-cloud-offload` を**作らない**。**`run`(走行)と「ゼロからのヒューマノイド歩行RL」と「本格fine-tune」はスコープ外**。行動の実現経路は **{モデルベース制御} ∪ {既存の事前学習方策の再生(+軽量ローカルfine-tune)}** に限定される。 |
| 3 | mjpc 投資 | **延期・純Python背骨で先行** | 行動層の背骨は `controller-baselines`(PD/CPG/convex-MPC, 全依存arm64)+ `pretrained-deploy`。`mujoco-mpc-advanced` は **#378 が arm64 でビルド可能と実証できた後にのみ**上級/任意スキルとして追加。 |

**決定2の重要な含意(明示)**:
- ✅ **達成可能(100%ローカル)**: GO2 歩行/トロット(モデルベース)、G1/H1 立つ・座る(PD/軌道最適)、**G1/H1 歩行(Unitree 同梱の事前学習方策を `deploy_mujoco.py` 無改変で再生)**、sim内の幾何/レンジファインダ障害物回避。→ **「歩く・座る・障害物回避」は全ロボットで成立。**
- ❌ **スコープ外(現時点)**: **走る(run)**、ゼロからのヒューマノイド歩行RL、フォトリアル視覚回避。
- ⚠️ **「走る」の将来の唯一のローカル経路**: 100%ローカルでも、**誰か(Unitree/コミュニティ)が走行の事前学習方策を公開すれば、それを再生する形でローカルに `run` を後から追加できる**(再生はローカルで完結するため)。ただし**ローカルでの走行学習は不可**。→ run は「方策が出てきたら再生で入れる」保留枠とする(**Codexレビューで確認済 = 将来の事前学習方策待ち**。クラウド例外は設けない)。
- この選択は「**全Macユーザーが NVIDIA 一切なしで使える**」という当初の強い動機を**最も純粋に守る**判断。スコープから外れるのは元々 NVIDIA-free では不可能だった部分のみ。

## II-A. Phase 1 実機検証ログ(2026-06-05, M5 Max 実機)

> Codex 優先順(Codex導入パス → skills CLI可視性 → viewer/offscreen)で実機検証。**3件すべて GREEN**。当初「最も死にやすい」とされた導入・描画の地雷は de-risk 済み。

**環境(実測)**: M5 Max / 18コア(perflevel0=6) / 128GB / arm64 / macOS 26.5。Python 3.10.13(pyenv)に **mujoco 3.9.0 + mjpython** 導入済。node v23.11 / npm 10.9.2。**codex-cli 0.137.0** と **claude 2.1.162** 両方導入済。

| # | 地雷 | 検証結果 | 判定 |
|---|---|---|---|
| 1 | **#798**(viewer+offscreen 同居クラッシュ) | 使い捨てMJCFで `mujoco.Renderer` のオフスクリーン RGB+深度を **python3 でも mjpython でも成功**(クラッシュ無し)。crash は `launch_passive`+`Renderer` 併用時のみ=**②viewer/③offscreen 分離で回避済**。深度は `ARB_clip_control unavailable…depth accuracy limited` 警告=**Mac の深度精度は限定的**(研究どおり)。 | ✅ 設計で回避・むしろ制約が緩い |
| 2 | **#851**(Claude Code global 導入でスキル不可視) | `skills` CLI **1.5.10** で使い捨てスキルを `add -a claude-code -g` → **`~/.claude/skills/mjm-probe` に正しく着地・可視**。**バグ再現せず=現行版で解消**。既定挙動は **copy**(symlink 脆弱性も回避)。 | ✅ 解消(CLIバージョン固定は推奨) |
| 3 | **Codex 導入パス**(`~/.codex/skills` か `~/.agents/skills` か) | `add -a codex -g` → **`~/.agents/skills/mjm-probe` に着地**。`~/.codex/skills` は `.system`(Codex同梱)のみ。**Codex の user global は `~/.agents/skills/`** と確定(OpenAI docs と整合)。 | ✅ 確定:`~/.agents/skills/` |

**CLI 事実**: npm `skills` = vercel-labs/skills、最新 **1.5.10**。フラグは戦略想定どおり(`add <source>` はローカルパス可 / `-g` global / `-a` agent / `-s` skill / `--copy` / `--all` / `init`)。検証後は全テスト物を削除し原状復帰済。

**含意(設計反映)**:
- `mujoco-env-setup` の「導入後可視性チェック」は**保険として残すが、現行CLIでは必須ブロッカーではない**。CLIバージョンは固定/明記する。
- **配布パスは確定**: Claude Code=`~/.claude/skills/`、Codex=`~/.agents/skills/`。リポジトリ側は `.agents/skills/` レイアウトで両対応(II-4 のまま)。
- `mujoco-offscreen-render` は **python3 ベースで RGB+深度+セグ取得可**。深度精度の限界は skill の `compatibility`/ドキュメントに明記。
- 次の実機検証候補:(a) `launch_passive`+`Renderer` 併用クラッシュの対話確認(窓が開くため要同席)、(b) ~~GO2 Menagerie モデルのロード+モデルベース trot~~ **✅ 達成(II-B)**。

## II-B. Phase 2 灯台デモ達成(2026-06-05, M5 Max 実機)

> **中心命題を実証**:NVIDIA なし・クラウドなし・RL なし、Mac ローカルのモデルベース制御だけで **Unitree GO2 が前進トロット**(~0.23 m/s、8秒で +1.86m、姿勢安定 z≈0.23)。

- **実体スキルを作成**: `.agents/skills/mujoco-controller-baselines/`(SKILL.md + `scripts/`{inspect_go2, go2_stand, go2_trot, render_go2_trot} + `references/go2-trot-recipe.md` + `assets/go2_trot.gif`)。
- **手法**: CPG 足先軌道 → 解析的 2リンク IK(L1=L2=0.213、home角を完全再現)→ ソフトPD(`τ=kp(q_des−q)−kd·q̇`、ctrlrange でクリップ)。対角トロット(FL-RR / FR-RL)。全依存 arm64 ネイティブ。
- **検証で得た3つの要点**(recipe に記録):①GO2 は `<motor>`=トルク制御、PDはソフト実装 ②`forcerange=(0,0)` は「無効」で 0N·m ではない=**トルク上限は `ctrlrange`**(これを誤ると常時トルク0で崩れる)③前進/後退は x 符号でなく **duty+lift** で決まる(duty0.6/lift0.06=接地99%で引きずり→後退、**duty0.5/lift0.10**=踏み出し→前進)。
- **現状の限界**: 開ループのため緩やかな yaw/横ドリフト有り(8秒で横0.19m)。次段は heading/速度フィードバック、その先は convex-MPC(CasADi+OSQP+Pinocchio, 全arm64)。
- **判定**: マトリクス II-5 の「GO2 walk(モデルベース)= ✅ 高確度」を**実機で確認**。


## II-10. v2.0 追加出典

**ロボットモデル / モデルベース制御**
- MuJoCo Menagerie — https://github.com/google-deepmind/mujoco_menagerie (unitree_g1 / unitree_h1 / unitree_go2 ほか)
- MuJoCo MPC(mjpc) — https://github.com/google-deepmind/mujoco_mpc(tasks/humanoid, tasks/quadruped/quadruped.cc)
- mjpc Apple Silicon ビルドバグ #378 — https://github.com/google-deepmind/mujoco_mpc/issues/378
- Predictive Sampling — https://arxiv.org/abs/2212.00541
- Whole-Body MPC of Legged Robots with MuJoCo(iLQR, H1歩行) — https://arxiv.org/html/2503.04613v1 / https://github.com/johnzhang3/mujoco_mpc_deploy
- HumanoidBench MJPC(コスト正則化必須) — https://arxiv.org/abs/2408.00342
- GO2 convex-MPC(CPU, arm64) — https://github.com/elijah-waichong-chan/go2-convex-mpc
- arm64 wheel: osqp https://pypi.org/project/osqp/ , Pinocchio https://pypi.org/project/pin/

**RL / 事前学習方策**
- unitree_rl_gym(G1/H1 motion.pt, deploy_mujoco.py) — https://github.com/unitreerobotics/unitree_rl_gym
- MuJoCo Playground — https://github.com/google-deepmind/mujoco_playground / 技報 https://playground.mujoco.org/assets/playground_technical_report.pdf
- unitree_rl_mjlab(mujoco_warp=CUDA) — https://github.com/unitreerobotics/unitree_rl_mjlab
- 四足RL(MuJoCo) — https://github.com/nimazareian/quadruped-rl-locomotion

**sim-to-real / ナビ**
- unitree_sdk2_python(Linux/CycloneDDS) — https://github.com/unitreerobotics/unitree_sdk2_python
- unitree_mujoco(公式sim-to-sim) — https://github.com/unitreerobotics/unitree_mujoco
- G1 DDS services — https://support.unitree.com/home/en/G1_developer/dds_services_interface
- unitree-g1-docker-sdk(Macはdev専用) — https://github.com/Uvesh-patel/unitree-g1-docker-sdk
- MuJoCo rangefinder/LiDAR — https://github.com/google-deepmind/mujoco/issues/1654 / https://mujoco.readthedocs.io/en/latest/XMLreference.html
- 階層ナビ(velocity-command分離) — Agile But Safe https://arxiv.org/html/2401.17583v2 / Omni-Perception https://arxiv.org/pdf/2505.19214

**Codex訂正点**
- jax-metal(0.1.1, experimental) — https://developer.apple.com/metal/jax/ / https://pypi.org/project/jax-metal/ / https://docs.jax.dev/en/latest/installation.html
- JAX #34648(unmaintained, forks) — https://github.com/jax-ml/jax/discussions/34648
- Codex plugins(plugin.json) — https://developers.openai.com/codex/plugins

---

*v2.2 結論: Codex の「行動層を足せ」は正しく、検証もそれを製品本体と裏付けた。**背骨はモデルベース制御**(GO2は実測で余裕、G1 stand は堅実)。**ヒューマノイド歩行の約束は事前学習再生に置く。** 3決定確定(II-9, 2026-06-05):**sim-first + ハンドオフ / 100%ローカル堅持(クラウド不採用・run はスコープ外/保留枠)/ mjpc 延期・純Python背骨で先行**。**実装の正準ソースは Part II(II-4 / II-5 / II-8 / II-9)。Part I の §7・§10 は旧方針アーカイブ。** 次は Phase 1 の地雷潰し(#851・#798・Codex導入パス)と GO2 灯台デモから着手する。*
