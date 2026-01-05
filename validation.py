import asyncio
import json
import random
from pathlib import Path
from collections import defaultdict

# main.py から必要な関数・変数をインポート
from main import (
    load_jsonl, 
    build_fewshot_prompt, 
    solve_item, 
    TRAIN_PATH, 
    MAX_FEWSHOT
)

# 採点用ロジックをインポート
from score_math_test import extract_final_answer, equivalent

# 出力ファイル設定
VALIDATION_OUTPUT = Path(__file__).parent / "validation_leave_one_out.jsonl"
CONCURRENT_REQUESTS = 5  # 検証用なので安全のため少し絞る

async def run_validation():
    print("--- Leave-One-Out Validation Start ---")

    # 1. データ準備 (TRAINデータを全件読み込み)
    full_data = load_jsonl(TRAIN_PATH)
    print(f"Data Loaded: {len(full_data)} items")

    # Typeごとにデータをグループ化（高速化用）
    train_by_type = defaultdict(list)
    data_map = {}  # IDから正解データなどを引くための辞書
    for item in full_data:
        train_by_type[item["type"]].append(item)
        data_map[item["id"]] = item

    # 2. タスク生成 (自分以外 & Type一致 のFew-shotを選択)
    print("Generating tasks with Type-Matching & Leave-One-Out strategy...")
    
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    tasks = []

    for target in full_data:
        # A. 同じTypeの候補を取得
        candidates = train_by_type.get(target["type"], full_data)
        
        # B. 自分自身(ID)を除外
        valid_candidates = [c for c in candidates if c["id"] != target["id"]]
        
        # (万が一、同じTypeの他データが0件なら、全体から自分以外を選ぶフォールバック)
        if not valid_candidates:
            valid_candidates = [c for c in full_data if c["id"] != target["id"]]

        # C. ランダムにサンプリング
        k = min(len(valid_candidates), MAX_FEWSHOT)
        selected_shots = random.sample(valid_candidates, k)

        # D. プロンプト作成
        fewshot_text = build_fewshot_prompt(selected_shots)

        # E. タスク追加
        tasks.append(solve_item(target, fewshot_text, semaphore))

    # 3. 推論実行 (API)
    print(f"Running Inference on {len(tasks)} items...")
    raw_results = await asyncio.gather(*tasks)
    print("\nInference Finished! (API通信完了)")

    # 4. 採点 & 集計
    print("Evaluating (SymPy)...")
    
    final_results = []
    
    # 全体スコア
    total_correct = 0
    total_count = 0
    
    # Type別スコア集計用
    type_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    # 結果はID順とは限らないためループで処理
    # (raw_resultsはAPIが返した順序のリスト)
    # ただし今回はasyncio.gatherの引数順(tasks順)に返ってくる仕様だが、
    # 安全のため data_map から正解情報を引く
    
    for i, res in enumerate(raw_results):
        item_id = res["id"]
        original_item = data_map[item_id]
        
        gold = original_item["answer"]
        pred_raw = res["prediction"]
        item_type = original_item.get("type", "Unknown")

        # 進捗表示
        print(f"[{i+1}/{len(full_data)}] ID:{item_id} ({item_type}) ...", end="", flush=True)

        # 回答抽出
        pred_clean = extract_final_answer(pred_raw)

        # 正誤判定
        is_ok = equivalent(gold, pred_clean)

        if is_ok:
            total_correct += 1
            type_stats[item_type]["correct"] += 1
            print(" OK")
        else:
            print(f" NG (Gold:{gold} vs Pred:{pred_clean})")

        type_stats[item_type]["total"] += 1
        total_count += 1

        # 結果行を作成
        result_row = {
            "id": item_id,
            "type": item_type,
            "is_correct": is_ok,
            "gold_answer": gold,
            "pred_answer": pred_clean,
            "raw_output": pred_raw,
            "used_prompt_strategy": "Leave-One-Out TypeMatch"
        }
        final_results.append(result_row)

    # 5. ファイル保存
    # ID順にソートして保存
    final_results.sort(key=lambda x: x["id"])
    with open(VALIDATION_OUTPUT, "w", encoding="utf-8") as f:
        for row in final_results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 6. 最終レポート表示
    print("\n" + "="*40)
    print(f" VALIDATION SUMMARY (N={total_count})")
    print("="*40)
    
    # 全体精度
    acc = total_correct / total_count if total_count > 0 else 0
    print(f"Total Accuracy: {acc:.2%} ({total_correct}/{total_count})\n")
    
    # Type別精度
    print("--- By Type ---")
    for t_type, stats in sorted(type_stats.items()):
        c = stats["correct"]
        t = stats["total"]
        type_acc = c / t if t > 0 else 0
        print(f"{t_type.ljust(15)}: {type_acc:.2%} ({c}/{t})")
        
    print("="*40)
    print(f"Saved details to: {VALIDATION_OUTPUT.name}")

if __name__ == "__main__":
    asyncio.run(run_validation())