import asyncio
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

# =========================
# 設定
# =========================
here = Path(__file__).parent
TRAIN_PATH = here / "math_level12_easy_train100_student_with_answer_solution.jsonl"
TEST_PATH = here / "math_level12_easy_test100_student.jsonl"
OUTPUT_PATH = here / "my_preds.jsonl"

MODEL = "gpt-4o-mini"
MAX_FEWSHOT = 5     # トークン超過を防ぐため
CONCURRENT_REQUESTS = 20  # 同時に送るリクエスト数

client = AsyncOpenAI()

# =========================
# ユーティリティ
# =========================
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def build_fewshot_prompt(train_data, type, max_examples=5):
    """
    few-shot 用のテキストを生成
    """
    examples = []
    for d in train_data:
        if d["type"] == type:
            examples.append(d)
            if len(examples) >= max_examples:
                break
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


async def solve_item(item, fewshot_text, semaphore):
    """
    単一問題を ChatGPT に解かせる（非同期）
    """
    async with semaphore:
        problem_id = item["id"]
        problem_text = item["problem"]

        print(f"Solving id={problem_id} ...")

        try:
            response = await client.responses.create(
                model=MODEL,
                instructions="You are a math assistant. Be sure to write a final answer after 'FINAL:'. Think step by step.",
                input=f"The following is a sample answer.\n\n{fewshot_text}\n\n ---\nPlease solve the following problem.\n\nproblem:\n{problem_text}",
                temperature=0.0,
            )
            answer_text = response.output_text
        except Exception as e:
            print(f"Error solving id={problem_id}: {e}")
            answer_text = f"Error: {e}"

        return {
            "id": problem_id,
            "prediction": answer_text,
        }


# =========================
# メイン処理
# =========================
async def main():
    # データ読み込み
    train_data = load_jsonl(TRAIN_PATH)
    test_data = load_jsonl(TEST_PATH)

    # 同時実行リクエストを制限
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

    # タスクの作成
    tasks = [solve_item(item, build_fewshot_prompt(train_data, item["type"], MAX_FEWSHOT), semaphore) for item in test_data]

    # 並列実行
    predictions = await asyncio.gather(*tasks)

    # id順にソート（非同期実行だと順番が入れ替わることがあるため）
    predictions.sort(key=lambda x: x["id"])

    # jsonl 形式で出力
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")

    print(f"Finished. Output written to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
