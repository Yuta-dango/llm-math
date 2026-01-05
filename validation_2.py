import asyncio
import json
import random
from collections import defaultdict
from main import (
    load_jsonl, 
    build_fewshot_prompt, 
    solve_item, 
    TRAIN_PATH, TEST_PATH, OUTPUT_PATH, CONCURRENT_REQUESTS
)

async def main():
    # 1. データの読み込み (main.pyの関数を再利用)
    train_data = load_jsonl(TRAIN_PATH)
    test_data = load_jsonl(TEST_PATH)

    # 2. 戦略: Typeごとにデータをグループ化
    train_by_type = defaultdict(list)
    for t in train_data:
        train_by_type[t["type"]].append(t)

    # 3. タスクの作成
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    tasks = []

    print("Generating tasks with Type-Matching strategy...")
    for target in test_data:
        # 戦略ロジック: 同じTypeから候補を選び、ランダムにサンプリング
        candidates = train_by_type.get(target["type"], train_data) # typeが無い場合は全体から
        selected_shots = random.sample(candidates, min(len(candidates), 5))

        # main.pyの関数を再利用してプロンプト作成 & リクエスト作成
        # build_fewshot_prompt はリストを渡せばそのまま整形してくれます
        fewshot_text = build_fewshot_prompt(selected_shots)
        
        # solve_item もそのまま使えます (個別のプロンプトを渡すだけ)
        tasks.append(solve_item(target, fewshot_text, semaphore))

    # 4. 並列実行 & 保存
    predictions = await asyncio.gather(*tasks)
    predictions.sort(key=lambda x: x["id"])

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    
    print(f"Finished. Output written to {OUTPUT_PATH}")

if __name__ == "__main__":
    asyncio.run(main())