"""採点スクリプトモジュール。

予測ファイルと正解ファイルを比較して精度を計算する。
"""

import argparse
import logging
from collections import defaultdict

from utils import (
    load_jsonl,
    extract_final_answer,
    equivalent,
    save_evaluation_summary,
    OUTPUT_DIR,
    DATA_DIR,
    logger,
)


def score_predictions(gold_path, pred_path, verbose=True, output_dir=None):
    """予測の採点を行う。

    Args:
        gold_path: 正解データのファイルパス。
        pred_path: 予測データのファイルパス。
        verbose: 詳細な結果を表示するかどうか。
        output_dir: サマリー保存先ディレクトリ（Noneの場合は予測ファイルと同じディレクトリ）。

    Returns:
        採点結果を含む辞書。accuracy、correct、total、type_stats、wrongを含む。
    """
    gold_rows = load_jsonl(gold_path)
    pred_rows = load_jsonl(pred_path)

    gold_by_id = {r["id"]: r for r in gold_rows}
    pred_by_id = {r["id"]: r for r in pred_rows}

    total = 0
    correct = 0
    wrong = []
    type_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for gid, g in gold_by_id.items():
        total += 1
        gold_ans = g.get("answer")
        item_type = g.get("type", "Unknown")
        
        pred_raw = pred_by_id.get(gid, {}).get("prediction")
        pred_ans = extract_final_answer(pred_raw)

        ok = equivalent(gold_ans, pred_ans)
        type_stats[item_type]["total"] += 1
        
        if ok:
            correct += 1
            type_stats[item_type]["correct"] += 1
        else:
            wrong.append({
                "id": gid,
                "type": item_type,
                "gold": gold_ans,
                "pred": pred_ans,
            })

    acc = correct / total if total else 0.0
    
    if verbose:
        logger.info("=" * 50)
        logger.info(" Scoring Results")
        logger.info("=" * 50)
        logger.info(f"Accuracy: {correct}/{total} = {acc:.2%}")
        logger.info("")
        
        logger.info("--- By Type ---")
        for t_type, stats in sorted(type_stats.items()):
            c, t = stats["correct"], stats["total"]
            type_acc = c / t if t > 0 else 0
            logger.info(f"  {t_type.ljust(20)}: {type_acc:.2%} ({c}/{t})")
        
        if wrong:
            logger.info(f"")
            logger.info(f"Wrong answers: {len(wrong)}")
            logger.info("Examples (up to 5):")
            for w in wrong[:5]:
                logger.info(f"  ID {w['id']} ({w['type']}): gold={w['gold']}, pred={w['pred']}")
        
        logger.info("=" * 50)
    
    # サマリー保存（デフォルトで実行）
    from pathlib import Path
    if output_dir is None:
        # 予測ファイルと同じディレクトリに保存
        output_dir = Path(pred_path).parent
    
    summary_stats = {
        "total_correct": correct,
        "total_count": total,
        "type_stats": dict(type_stats),
    }
    save_evaluation_summary(summary_stats, Path(output_dir))

    return {
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "type_stats": dict(type_stats),
        "wrong": wrong,
    }


def main():
    """採点スクリプトのエントリーポイント。

    コマンドライン引数から正解ファイルと予測ファイルを受け取り、
    採点を実行する。
    """
    parser = argparse.ArgumentParser(description="Score predictions against gold answers")
    parser.add_argument(
        "--gold", "-g",
        type=str,
        default=str(DATA_DIR / "test_teacher.jsonl"),
        help="Path to gold (teacher) file"
    )
    parser.add_argument(
        "--pred", "-p",
        type=str,
        default=str(OUTPUT_DIR / "predictions.jsonl"),
        help="Path to prediction file"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Directory to save summary (default: same as prediction file)"
    )
    args = parser.parse_args()

    score_predictions(
        args.gold, 
        args.pred,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
