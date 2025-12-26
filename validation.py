import json
import random
import re
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# .envの読み込み
load_dotenv()

# =========================
# 設定
# =========================
here = Path(__file__).parent
# NOTE: 評価(Validation)を行うためには正解ラベルが必要なため、
# "train"と名付けられた正解付きデータセットを使用し、これを内部で分割します。
DATA_PATH = here / "math_level12_easy_train100_student_with_answer_solution.jsonl"
OUTPUT_PATH = here / "validation_results.jsonl"

MODEL = "gpt-4o-mini"
MAX_FEWSHOT = 5       # Few-shotに使用する例示の数
TRAIN_RATIO = 0.8     # 学習データの割合 (80:20分割)
SEED = 42             # 再現性のための乱数シード

client = OpenAI()

# =========================
# ユーティリティ関数
# =========================
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def build_fewshot_prompt(train_examples):
    """
    Trainデータセットの中からFew-shot用のテキストを生成
    """
    parts = []
    for ex in train_examples:
        # prompt構成: 問題 -> 解法 -> FINAL: 答え
        parts.append(
            f"""problem:
{ex["problem"]}

answer:
{ex["solution"]}
FINAL: {ex["answer"]}
"""
        )
    return "\n---\n".join(parts)

def extract_final_answer(text):
    """
    LLMの出力から 'FINAL: ' 以降の答え部分を抽出する
    """
    # "FINAL:" または "FINAL Answer:" などのパターンに対応
    match = re.search(r"FINAL:\s*(.*)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()

def solve_problem(problem_text, fewshot_text):
    """
    OpenAI APIを使って問題を解く
    """
    prompt = f"The following are sample answers.\n\n{fewshot_text}\n\n ---\nPlease solve the following problem. Be sure to write the final answer after 'FINAL:'.\n\nproblem:\n{problem_text}"
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a math assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    return response.choices[0].message.content

def is_correct(pred, true_val):
    """
    正解判定ロジック
    単純な文字列一致だけでなく、数値的な等価性などを考慮する場合ここにロジックを追加
    """
    # 簡易的な正規化（空白削除、小文字化）
    p = str(pred).strip().lower()
    t = str(true_val).strip().lower()
    
    # 数値としての比較を試みる (例: "12.0" == "12")
    try:
        return float(p) == float(t)
    except ValueError:
        pass
        
    return p == t

def score_math_test(y_true, y_pred, raw_preds=None):
    """
    精度評価を行い、結果を表示する関数
    """
    total = len(y_true)
    correct_count = 0
    
    for i in range(total):
        # 抽出した答え(y_pred)と正解(y_true)を比較
        if is_correct(y_pred[i], y_true[i]):
            correct_count += 1
    
    accuracy = correct_count / total if total > 0 else 0
    
    print("\n" + "="*40)
    print("      SCORE MATH TEST REPORT      ")
    print("="*40)
    print(f"Total Validation Samples : {total}")
    print(f"Correct Predictions      : {correct_count}")
    print(f"Accuracy                 : {accuracy:.2%}")
    print("="*40 + "\n")
    
    return accuracy

# =========================
# メイン処理
# =========================
def main():
    print(f"Loading data from {DATA_PATH} ...")
    full_data = load_jsonl(DATA_PATH)
    
    # --- 1. データの分割 (80:20) ---
    random.seed(SEED)
    random.shuffle(full_data)
    
    split_idx = int(len(full_data) * TRAIN_RATIO)
    train_set = full_data[:split_idx]
    val_set = full_data[split_idx:]
    
    print(f"Data Split -> Train: {len(train_set)}, Validation: {len(val_set)}")

    # --- 2. Few-shotプロンプトの構築 ---
    # Trainセットの中から最大MAX_FEWSHOT個だけ例示として使う
    fewshot_examples = train_set[:MAX_FEWSHOT]
    fewshot_text = build_fewshot_prompt(fewshot_examples)
    
    print(f"Few-shot prompt prepared with {len(fewshot_examples)} examples.")

    # --- 3. 検証実行 (Validation Loop) ---
    y_true = []       # 正解リスト
    y_pred_raw = []   # LLMの生出力
    y_pred_extracted = [] # 抽出した答え
    
    results = []

    print("Starting validation loop...")
    for i, item in enumerate(val_set):
        problem_text = item["problem"]
        true_answer = item["answer"] # 正解データには "answer" がある前提
        
        # LLM推論
        raw_output = solve_problem(problem_text, fewshot_text)
        
        # 答えの抽出
        extracted_answer = extract_final_answer(raw_output)
        
        # リストに保存
        y_true.append(true_answer)
        y_pred_raw.append(raw_output)
        y_pred_extracted.append(extracted_answer)
        
        # ログ保存用データ作成
        is_acc = is_correct(extracted_answer, true_answer)
        results.append({
            "id": item.get("id", i),
            "problem": problem_text,
            "true_answer": true_answer,
            "prediction_raw": raw_output,
            "prediction_extracted": extracted_answer,
            "is_correct": is_acc
        })
        
        # 進捗表示 (例: 10件ごと)
        if (i + 1) % 5 == 0:
            print(f"Processed {i + 1}/{len(val_set)} samples...")

    # --- 4. 結果保存 ---
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
    print(f"Detailed results saved to {OUTPUT_PATH}")

    # --- 5. 精度評価 ---
    score_math_test(y_true, y_pred_extracted)

if __name__ == "__main__":
    main()