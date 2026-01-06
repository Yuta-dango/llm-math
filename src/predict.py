"""テストデータに対する推論モジュール。

訓練データからType-Matchingでfew-shotを選択し、
テストデータの各問題に対して予測を生成する。
"""

import asyncio
import logging
import random
from collections import defaultdict

from utils import (
    load_jsonl,
    save_jsonl,
    build_fewshot_prompt,
    solve_item,
    create_timestamped_dir,
    TRAIN_PATH,
    TEST_PATH,
    OUTPUT_DIR,
    MAX_FEWSHOT,
    CONCURRENT_REQUESTS,
    logger,
)


async def run_prediction():
    """テストデータ推論のメイン処理を実行する。

    訓練データを読み込み、Type-Matchingによるfew-shot選択を行い、
    テストデータに対して並列推論を実行する。
    """
    logger.info("=" * 50)
    logger.info(" Test Prediction")
    logger.info("=" * 50)

    # 1. データ読み込み
    train_data = load_jsonl(TRAIN_PATH)
    test_data = load_jsonl(TEST_PATH)
    logger.info(f"Train: {len(train_data)} items from {TRAIN_PATH.name}")
    logger.info(f"Test:  {len(test_data)} items from {TEST_PATH.name}")

    # 2. Type ごとにグループ化
    train_by_type: dict[str, list] = defaultdict(list)
    for item in train_data:
        train_by_type[item["type"]].append(item)

    # 3. タスク生成
    logger.info("Generating tasks (Type-Matching)")
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    tasks = []

    for target in test_data:
        # 同じ Type から候補を取得（なければ全体から）
        candidates = train_by_type.get(target["type"], train_data)
        
        # ランダムにサンプリング
        k = min(len(candidates), MAX_FEWSHOT)
        selected_shots = random.sample(candidates, k)

        # プロンプト作成 & タスク追加
        fewshot_text = build_fewshot_prompt(selected_shots)
        tasks.append(solve_item(target, fewshot_text, semaphore))

    # 4. 推論実行
    logger.info(f"Running inference on {len(tasks)} items")
    predictions = await asyncio.gather(*tasks)
    logger.info("Inference finished")

    # 5. ファイル保存
    # タイムスタンプ付きディレクトリを作成
    run_dir = create_timestamped_dir(OUTPUT_DIR, prefix="prediction_")
    logger.info(f"Saving results to: {run_dir}")
    
    predictions.sort(key=lambda x: x["id"])
    predictions_path = run_dir / "predictions.jsonl"
    save_jsonl(predictions, predictions_path)

    logger.info("=" * 50)
    logger.info(f"Saved {len(predictions)} predictions to: {run_dir}")


if __name__ == "__main__":
    asyncio.run(run_prediction())
