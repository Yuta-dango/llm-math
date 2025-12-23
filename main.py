from openai import OpenAI
import json
from pathlib import Path

# =========================
# 設定
# =========================
here = Path(__file__).parent
TRAIN_PATH = here / "math_level12_easy_train100_student_with_answer_solution.jsonl"
TEST_PATH = here / "math_level12_easy_test100_student.jsonl"
OUTPUT_PATH = here / "my_preds.jsonl"

MODEL = "gpt-4o-mini"
MAX_FEWSHOT = 5     # トークン超過を防ぐため

client = OpenAI()

# =========================
# ユーティリティ
# =========================
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def build_fewshot_prompt(train_data, max_examples=5):
    """
    few-shot 用のテキストを生成
    """
    examples = train_data[:max_examples]
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


def solve_problem(problem_text, fewshot_text):
    """
    単一問題を ChatGPT に解かせる（Responses API 使用）
    """
    response = client.responses.create(
        model=MODEL,
        instructions="You are a math assistant. Be sure to write a final answer after 'FINAL:'.",
        input=f"The following is a sample answer.\n\n{fewshot_text}\n\n ---\nPlease solve the following problem.\n\nproblem:\n{problem_text}",
        temperature=0.0,
    )

    # Responses API では output_text にまとめて入る
    return response.output_text


# =========================
# メイン処理
# =========================
def main():
    # データ読み込み
    train_data = load_jsonl(TRAIN_PATH)
    test_data = load_jsonl(TEST_PATH)

    # few-shot プロンプト生成
    fewshot_text = build_fewshot_prompt(train_data, MAX_FEWSHOT)

    predictions = []

    for item in test_data:
        problem_id = item["id"]
        problem_text = item["problem"]

        print(f"Solving id={problem_id} ...")

        answer_text = solve_problem(problem_text, fewshot_text)

        predictions.append(
            {
                "id": problem_id,
                "prediction": answer_text,
            }
        )

    # jsonl 形式で出力
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")

    print(f"Finished. Output written to {OUTPUT_PATH}")


# def main():
#     client = OpenAI()
#     response = client.responses.create(
#         model="gpt-4o-mini",
#         instructions="Answer the question.",
#         input="Where is the capital of Japan?"
#     )
#     print(response.output_text)


if __name__ == "__main__":
    main()
