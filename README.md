# llm-math

LLM（GPT-4o-mini）を使用して数学問題を解き、精度を評価するプロジェクトです。

## 主な機能

- **Leave-One-Out バリデーション**: 訓練データを用いた交差検証でモデル性能を評価
- **テスト推論**: テストデータに対して訓練データから few-shot を選択して予測
- **Type-Matching**: 問題タイプ（代数、幾何など）に基づいた few-shot 選択
- **非同期 API 実行**: 並列リクエストによる高速推論
- **SymPy 同値判定**: LaTeX 数式の数学的同値判定
- **タイムスタンプ管理**: 実行ごとにタイムスタンプ付きディレクトリで結果を管理
- **ジャンル別結果保存**: 問題タイプ別にファイル分割、正解/不正解でソート

## ディレクトリ構成

```
llm-math/
├── src/                    # ソースコード
│   ├── utils.py            # 共通ユーティリティ
│   ├── validate.py         # Leave-One-Out バリデーション
│   ├── predict.py          # テストデータ推論
│   └── score.py            # 採点スクリプト
├── data/                   # 入力データ
│   ├── train.jsonl         # 訓練データ
│   └── test.jsonl          # テストデータ
├── output/                 # 出力ファイル
│   ├── validation_YYYYMMDD_HHMMSS/  # バリデーション結果
│   │   ├── summary.txt              # 評価サマリー
│   │   ├── all_results.jsonl        # 全結果
│   │   ├── Algebra.jsonl            # 代数問題（正解→不正解順）
│   │   ├── Geometry.jsonl           # 幾何問題（正解→不正解順）
│   │   └── ...
│   └── prediction_YYYYMMDD_HHMMSS/  # 予測結果
│       └── predictions.jsonl
├── config.yaml             # 設定ファイル
├── legacy/                 # 旧バージョンのファイル
└── MY/                     # 分析・ドキュメント
```

## セットアップ

```bash
# 依存関係インストール
uv sync

# 環境変数設定
cp .env.example .env
# .env に OPENAI_API_KEY を設定
```

## 設定（config.yaml）

プロジェクトの設定は `config.yaml` で管理されます。

```yaml
# 実験メモ
experiment_memo: "Baseline experiment with Type-Matching few-shot selection"

# データ・出力ディレクトリ
data_dir: data
output_dir: output

# データファイル
train_file: train.jsonl
test_file: test.jsonl

# モデル設定
model: gpt-4o-mini
temperature: 0.0

# Few-shot設定
max_fewshot: 5

# 並行実行設定
concurrent_requests: 20
validation_concurrent_requests: 20

# プロンプト設定
prompt:
  # デフォルト（Geometry以外）のプロンプト
  default:
    instructions: "構造化されたアプローチ (UNDERSTAND → PLAN → EXECUTE → VERIFY)"
    input_template: "..."
  
  # Geometry専用のプロンプト
  geometry:
    instructions: "幾何専用の体系的アプローチ (IDENTIFY → VISUALIZE → THEOREM → CALCULATE → VERIFY)"
    input_template: "..."
```

**プロンプト設定について**:
- `default`: Geometry以外の問題タイプに使用される標準プロンプト
  - UNDERSTAND → PLAN → EXECUTE → VERIFY の4ステップアプローチ
- `geometry`: Geometry問題専用の特化プロンプト
  - IDENTIFY → VISUALIZE → THEOREM → CALCULATE → VERIFY の5ステップ
  - 角度の検証、相似三角形の対応関係の明示など、幾何特有のチェック項目を含む
- 問題の`type`フィールドが`"Geometry"`の場合、自動的にgeometryプロンプトが使用される

## 使い方

### Leave-One-Out バリデーション

訓練データの各問題に対して、自分自身を除いた同タイプの問題から few-shot を選択し、推論・採点を行う。

```bash
cd src && uv run validate.py
```

**出力**:
- `output/validation_YYYYMMDD_HHMMSS/`
  - `summary.txt`: 評価結果サマリー
  - `all_results.jsonl`: 全結果（ID順）
  - `<Type>.jsonl`: タイプ別結果（正解が上部、不正解が下部）

### テストデータ推論

テストデータに対して、訓練データから Type-Matching で few-shot を選択して予測。

```bash
cd src && uv run predict.py
```

**出力**: `output/prediction_YYYYMMDD_HHMMSS/predictions.jsonl`

### 採点（正解ファイルがある場合）

```bash
# デフォルト（予測ファイルと同じディレクトリにsummary.txtを保存）
cd src && uv run score.py --gold ../data/test_teacher.jsonl --pred ../output/prediction_YYYYMMDD_HHMMSS/predictions.jsonl

# 別のディレクトリに保存する場合
cd src && uv run score.py --gold ../data/test_teacher.jsonl --pred ../output/prediction_YYYYMMDD_HHMMSS/predictions.jsonl --output-dir ../output/evaluation
```

## 出力形式

### summary.txt（評価サマリー）

### タイプ別JSONLファイル

各タイプのファイルは、正解（`is_correct: true`）が上部、不正解が下部に配置されます。

## プロジェクト構造

### legacy/ディレクトリ

旧バージョンのスクリプトや出力ファイルを保管しています。現在の実装では使用されません。

- `main.py`, `validation.py`, `validation_2.py`: 旧バージョンのスクリプト
- `*.jsonl`: 過去の実行結果

## 技術スタック

- **Language**: Python 3.12+
- **LLM**: OpenAI GPT-4o-mini
- **Libraries**: `openai`, `sympy`, `python-dotenv`, `pyyaml`
- **Async**: `asyncio` による非同期 API リクエスト
- **Tooling**: `uv`

## 詳細ドキュメント

- 実装計画・詳細: [MY/rq.md](MY/rq.md)
- エラー分析: [MY/error_analysis.md](MY/error_analysis.md)