# llm-math

LLM（Large Language Models）を使用して数学の問題を解き、その精度を評価するためのプロジェクトです。
主に OpenAI の GPT-4o-mini を活用し、few-shot プロンプティングを用いて数学の問題に回答します。

## 主な機能

- **自動解答生成**: `main.py` を実行することで、指定されたテストデータの数学問題を LLM が解き、結果を JSONL 形式で出力します。
- **精度評価**: `score_math_test.py` を使用して、生成された回答の正誤を判定し、精度（Accuracy）を算出します。
- **数式判定**: Sympy を用いて、LaTeX 形式の数式が数学的に同値であるかを判定します。

## セットアップ

### 1. 環境構築

このプロジェクトはパッケージ管理に [uv](https://github.com/astral-sh/uv) を使用しています。以下のコマンドで依存関係をインストールできます。

```bash
uv sync
```

### 2. 環境変数の設定

`.env.example` を参考に `.env` ファイルを作成し、OpenAI の API キーを設定してください。

```bash
cp .env.example .env
# .env ファイルを編集して OPENAI_API_KEY を入力
```

## 使い方

### 解答の生成

テストデータ (`math_level12_easy_test100_student.jsonl`) に対して LLM で解答を生成します。

```bash
uv run main.py
```

実行後、解答結果が `my_preds.jsonl` に保存されます。

### 精度評価

生成された解答の正確さを評価します。

> **注意**: 評価には正解データ（teacher ファイル）が必要です。`score_math_test.py` 内の `gold_path` や `pred_path` を必要に応じて調整してください。

```bash
uv run score_math_test.py
```

## プロジェクト構成

- `main.py`: 解答生成のメインスクリプト。
- `score_math_test.py`: 解答の評価スクリプト。
- `math_level12_easy_...`: 数学問題のデータセット（訓練用・テスト用）。
- `pyproject.toml`: プロジェクトの依存関係定義。
- `.env`: APIキーなどの秘匿情報を管理。

## 技術スタック

- **Language**: Python 3.12+
- **LLM**: OpenAI GPT-4o-mini
- **Libraries**: `openai`, `sympy`, `python-dotenv`
- **Tooling**: `uv`
