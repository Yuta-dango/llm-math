"""Leave-One-Outバリデーションモジュール。

訓練データを使って、各問題に対して自分自身を除いた
同じタイプの問題からfew-shotを選択し、推論・採点を行う。
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
    extract_final_answer,
    equivalent,
    create_timestamped_dir,
    backup_prompts_to_output,
    save_results_by_type,
    save_evaluation_summary,
    TRAIN_PATH,
    OUTPUT_DIR,
    MAX_FEWSHOT,
    config,
    logger,
)

CONCURRENT_REQUESTS = config["validation_concurrent_requests"]


async def run_validation():
    """Leave-One-Out検証のメイン処理を実行する。

    訓練データを使用し、各問題について自分自身を除いた
    同じタイプの問題からfew-shotを選択して推論・採点を行う。
    """
    logger.info("=" * 50)
    logger.info(" Leave-One-Out Validation")
    logger.info("=" * 50)

    # 1. データ準備
    full_data = load_jsonl(TRAIN_PATH)
    logger.info(f"Data Loaded: {len(full_data)} items from {TRAIN_PATH.name}")

    # Type ごとにグループ化
    train_by_type: dict[str, list] = defaultdict(list)
    data_map: dict[int, dict] = {}
    for item in full_data:
        train_by_type[item["type"]].append(item)
        data_map[item["id"]] = item

    # 2. タスク生成
    logger.info("Generating tasks (Type-Matching + Leave-One-Out)")
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    tasks = []

    for target in full_data:
        # 同じ Type から候補を取得
        candidates = train_by_type.get(target["type"], full_data)
        # 自分自身を除外
        valid_candidates = [c for c in candidates if c["id"] != target["id"]]
        
        # フォールバック: 同じ Type の他データが 0 件なら全体から選ぶ
        if not valid_candidates:
            valid_candidates = [c for c in full_data if c["id"] != target["id"]]

        # ランダムにサンプリング
        k = min(len(valid_candidates), MAX_FEWSHOT)
        selected_shots = random.sample(valid_candidates, k)

        # プロンプト作成 & タスク追加
        fewshot_text = build_fewshot_prompt(selected_shots)
        tasks.append(solve_item(target, fewshot_text, semaphore))

    # 3. 推論実行
    logger.info(f"Running inference on {len(tasks)} items")
    raw_results = await asyncio.gather(*tasks)
    logger.info("Inference finished")

    # 4. 採点 & 集計
    logger.info("Evaluating results")
    
    final_results = []
    total_correct = 0
    total_count = 0
    type_stats: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})

    for i, res in enumerate(raw_results):
        item_id = res["id"]
        original_item = data_map[item_id]
        
        gold = original_item["answer"]
        pred_raw = res["prediction"]
        item_type = original_item.get("type", "Unknown")

        # 回答抽出 & 正誤判定
        pred_clean = extract_final_answer(pred_raw)
        is_ok = equivalent(gold, pred_clean)

        if is_ok:
            total_correct += 1
            type_stats[item_type]["correct"] += 1
        
        type_stats[item_type]["total"] += 1
        total_count += 1

        final_results.append({
            "id": item_id,
            "type": item_type,
            "is_correct": is_ok,
            "gold_answer": gold,
            "pred_answer": pred_clean,
            "raw_output": pred_raw,
        })

    # 5. ファイル保存
    # タイムスタンプ付きディレクトリを作成
    run_dir = create_timestamped_dir(OUTPUT_DIR, prefix="validation_")
    logger.info(f"Saving results to: {run_dir}")
    
    # プロンプトファイルをバックアップ
    backup_prompts_to_output(run_dir)
    
    # タイプ別にファイル分割して保存
    save_results_by_type(final_results, run_dir)
    
    # 全結果も保存
    all_results_path = run_dir / "all_results.jsonl"
    final_results.sort(key=lambda x: x["id"])
    save_jsonl(final_results, all_results_path)
    logger.info(f"Saved all {len(final_results)} results to all_results.jsonl")

    # 6. 評価サマリー保存
    summary_stats = {
        "total_correct": total_correct,
        "total_count": total_count,
        "type_stats": dict(type_stats),
    }
    save_evaluation_summary(summary_stats, run_dir)

    # 7. 結果表示
    logger.info("")
    logger.info("=" * 50)
    logger.info(f" VALIDATION RESULTS (N={total_count})")
    logger.info("=" * 50)
    
    acc = total_correct / total_count if total_count > 0 else 0
    logger.info(f"Total Accuracy: {acc:.2%} ({total_correct}/{total_count})")
    logger.info("")
    
    logger.info("--- By Type ---")
    for t_type, stats in sorted(type_stats.items()):
        c, t = stats["correct"], stats["total"]
        type_acc = c / t if t > 0 else 0
        logger.info(f"  {t_type.ljust(20)}: {type_acc:.2%} ({c}/{t})")
    
    logger.info("=" * 50)
    logger.info(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    asyncio.run(run_validation())
