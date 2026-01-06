"""共通ユーティリティモジュール。

データ読み込み、few-shotプロンプト生成、LLM API呼び出しなどの
共通機能を提供する。
"""

import json
import logging
import re
from pathlib import Path
from datetime import datetime
import yaml
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

# =========================
# 設定読み込み
# =========================
def load_config() -> dict:
    """config.yamlから設定を読み込む。

    Returns:
        設定情報を含む辞書。
    """
    # プロジェクトルートのconfig.yamlを探す
    current_dir = Path(__file__).parent
    config_path = current_dir.parent / "config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found at {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


config = load_config()

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# プロジェクトルート
PROJECT_ROOT = Path(__file__).parent.parent

# パス設定
DATA_DIR = PROJECT_ROOT / config["data_dir"]
OUTPUT_DIR = PROJECT_ROOT / config["output_dir"]
TRAIN_PATH = DATA_DIR / config["train_file"]
TEST_PATH = DATA_DIR / config["test_file"]

# モデル設定
MODEL = config["model"]
TEMPERATURE = config["temperature"]

# Few-shot設定
MAX_FEWSHOT = config["max_fewshot"]

# 並行実行設定
CONCURRENT_REQUESTS = config["concurrent_requests"]

client = AsyncOpenAI()


# =========================
# データ I/O
# =========================
def load_jsonl(path: Path) -> list[dict]:
    """JSONLファイルを読み込む。

    Args:
        path: 読み込むJSONLファイルのパス。

    Returns:
        読み込んだデータのリスト。各要素は辞書型。
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(data: list[dict], path: Path) -> None:
    """JSONLファイルに保存する。

    Args:
        data: 保存するデータのリスト。各要素は辞書型。
        path: 保存先のパス。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def create_timestamped_dir(base_dir: Path, prefix: str = "") -> Path:
    """タイムスタンプ付きディレクトリを作成する。

    Args:
        base_dir: ベースディレクトリ。
        prefix: ディレクトリ名のプレフィックス。

    Returns:
        作成されたディレクトリのパス。
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{prefix}{timestamp}" if prefix else timestamp
    output_dir = base_dir / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_results_by_type(results: list[dict], output_dir: Path) -> None:
    """結果をタイプ別にファイル分割して保存する。

    各タイプ内で、正解を上部、不正解を下部に配置してソートする。

    Args:
        results: 結果データのリスト。各要素は'type'と'is_correct'を含む辞書。
        output_dir: 保存先ディレクトリ。
    """
    from collections import defaultdict
    
    # タイプ別にグループ化
    by_type = defaultdict(list)
    for result in results:
        by_type[result["type"]].append(result)
    
    # タイプ別にファイル保存
    for type_name, items in by_type.items():
        # 正解を上部、不正解を下部にソート
        sorted_items = sorted(items, key=lambda x: (not x["is_correct"], x["id"]))
        
        # ファイル名を安全にする
        safe_type_name = type_name.replace(" ", "_").replace("/", "_")
        file_path = output_dir / f"{safe_type_name}.jsonl"
        save_jsonl(sorted_items, file_path)
        logger.info(f"Saved {len(sorted_items)} items to {file_path.name}")


def save_evaluation_summary(stats: dict, output_dir: Path, filename: str = "summary.txt") -> None:
    """評価結果のサマリーをテキストファイルに保存する。

    Args:
        stats: 評価統計情報を含む辞書。
        output_dir: 保存先ディレクトリ。
        filename: 保存するファイル名。
    """
    summary_path = output_dir / filename
    
    with open(summary_path, "w", encoding="utf-8") as f:
        # 実験メモがあれば冒頭に記載
        if "experiment_memo" in config and config["experiment_memo"]:
            f.write("=" * 60 + "\n")
            f.write(" Experiment Memo\n")
            f.write("=" * 60 + "\n")
            f.write(f"{config['experiment_memo']}\n")
            f.write("=" * 60 + "\n\n")
        
        f.write("=" * 60 + "\n")
        f.write(f" Evaluation Results\n")
        f.write("=" * 60 + "\n\n")
        
        # 全体の精度
        total_correct = stats["total_correct"]
        total_count = stats["total_count"]
        acc = total_correct / total_count if total_count > 0 else 0
        f.write(f"Total Accuracy: {acc:.2%} ({total_correct}/{total_count})\n\n")
        
        # タイプ別の精度
        f.write("-" * 60 + "\n")
        f.write("By Type:\n")
        f.write("-" * 60 + "\n")
        
        type_stats = stats["type_stats"]
        for t_type in sorted(type_stats.keys()):
            type_data = type_stats[t_type]
            c, t = type_data["correct"], type_data["total"]
            type_acc = c / t if t > 0 else 0
            f.write(f"  {t_type.ljust(25)}: {type_acc:6.2%} ({c:3d}/{t:3d})\n")
        
        f.write("=" * 60 + "\n")
    
    logger.info(f"Saved evaluation summary to {summary_path.name}")


# =========================
# Few-shot プロンプト生成
# =========================
def build_fewshot_prompt(examples: list[dict]) -> str:
    """選択された例のリストからfew-shotプロンプトを生成する。

    Args:
        examples: few-shotとして使用する例のリスト。
                  各例は'problem', 'solution', 'answer'キーを持つ辞書。

    Returns:
        生成されたfew-shotプロンプト文字列。
    """
    parts = []
    for ex in examples:
        parts.append(
            f"""problem:
{ex["problem"]}

answer:
{ex["solution"]}
FINAL: {ex["answer"]}
"""
        )
    return "\n---\n".join(parts)


# =========================
# LLM API 呼び出し
# =========================
async def solve_item(item: dict, fewshot_text: str, semaphore) -> dict:
    """単一問題をChatGPTに解かせる（非同期）。

    Args:
        item: 問題データ。'id'と'problem'キーを持つ辞書。
        fewshot_text: few-shotプロンプト文字列。
        semaphore: 並行実行数を制御するセマフォ。

    Returns:
        問題IDと予測結果を含む辞書。
    """
    async with semaphore:
        problem_id = item["id"]
        problem_text = item["problem"]

        logger.info(f"Solving id={problem_id}")

        try:
            response = await client.responses.create(
                model=MODEL,
                instructions="You are a math assistant. Be sure to write a final answer after 'FINAL:'. Think step by step.",
                input=f"The following is a sample answer.\n\n{fewshot_text}\n\n ---\nPlease solve the following problem.\n\nproblem:\n{problem_text}",
                temperature=TEMPERATURE,
            )
            answer_text = response.output_text
        except Exception as e:
            logger.error(f"Error solving id={problem_id}: {e}")
            answer_text = f"Error: {e}"

        return {
            "id": problem_id,
            "prediction": answer_text,
        }


# =========================
# 採点用ユーティリティ
# =========================
def extract_final_answer(text: str) -> str | None:
    """LLM出力から最終回答を抽出する。

    Args:
        text: LLMの生成した出力テキスト。

    Returns:
        抽出された最終回答文字列。抽出できない場合はNone。
    """
    if text is None:
        return None
    s = str(text)

    # 1) FINAL: を優先
    m = re.search(r"FINAL:\s*(.+)", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # 2) \boxed{...}
    m = re.search(r"\\boxed\{([^}]*)\}", s)
    if m:
        return m.group(1).strip()

    # 3) 最後の非空行（最後の手段）
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    return lines[-1] if lines else None


def normalize_latex(expr: str) -> str | None:
    """LaTeX数式を正規化する。

    Args:
        expr: 正規化対象のLaTeX数式文字列。

    Returns:
        正規化された数式文字列。正規化できない場合はNone。
    """
    if expr is None:
        return None
    x = expr.strip()

    x = x.replace("$", "")
    x = x.replace("\\left", "").replace("\\right", "")
    x = re.sub(r"\\[,\s;!]+", "", x)
    x = x.replace("\\pi", "pi")

    # \frac{a}{b} -> (a)/(b)
    while True:
        m = re.search(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", x)
        if not m:
            break
        a, b = m.group(1), m.group(2)
        x = x[:m.start()] + f"(({a})/({b}))" + x[m.end():]

    # \sqrt{a} -> sqrt(a)
    while True:
        m = re.search(r"\\sqrt\{([^{}]+)\}", x)
        if not m:
            break
        a = m.group(1)
        x = x[:m.start()] + f"sqrt({a})" + x[m.end():]

    x = x.replace("^", "**")
    x = re.sub(r"\s+", "", x)
    return x


def equivalent(gold: str, pred: str) -> bool:
    """2つの数式が同値かどうか判定する。

    Args:
        gold: 正解の数式文字列。
        pred: 予測された数式文字列。

    Returns:
        同値の場合True、それ以外False。
    """
    g = normalize_latex(gold)
    p = normalize_latex(pred)
    if g is None or p is None:
        return False
    if g == p:
        return True

    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import (
            parse_expr, standard_transformations, implicit_multiplication_application
        )
        trans = standard_transformations + (implicit_multiplication_application,)
        g_expr = parse_expr(g, transformations=trans, evaluate=True)
        p_expr = parse_expr(p, transformations=trans, evaluate=True)
        return sp.simplify(g_expr - p_expr) == 0
    except Exception:
        return False
