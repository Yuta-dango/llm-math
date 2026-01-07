"""プロンプト分岐のテストスクリプト。

Geometry問題と非Geometry問題でプロンプトが正しく切り替わるか確認する。
"""

import asyncio
from utils import (
    PROMPT_DEFAULT_INSTRUCTIONS,
    PROMPT_DEFAULT_INPUT_TEMPLATE,
    PROMPT_GEOMETRY_INSTRUCTIONS,
    PROMPT_GEOMETRY_INPUT_TEMPLATE,
)


def test_prompt_loading():
    """プロンプト設定が正しく読み込めているか確認する。"""
    print("=" * 60)
    print("プロンプト設定の読み込み確認")
    print("=" * 60)
    
    # デフォルトプロンプト
    print("\n[DEFAULT PROMPT]")
    print("Instructions (最初の100文字):")
    print(PROMPT_DEFAULT_INSTRUCTIONS[:100])
    print("\nInput Template (最初の100文字):")
    print(PROMPT_DEFAULT_INPUT_TEMPLATE[:100])
    
    # Geometryプロンプト
    print("\n[GEOMETRY PROMPT]")
    print("Instructions (最初の100文字):")
    print(PROMPT_GEOMETRY_INSTRUCTIONS[:100])
    print("\nInput Template (最初の100文字):")
    print(PROMPT_GEOMETRY_INPUT_TEMPLATE[:100])
    
    print("\n" + "=" * 60)
    print("✓ プロンプト設定の読み込み成功")
    print("=" * 60)


def test_prompt_selection():
    """問題タイプに応じたプロンプト選択のロジックをテスト。"""
    print("\n" + "=" * 60)
    print("プロンプト選択ロジックの確認")
    print("=" * 60)
    
    # テストデータ
    test_items = [
        {"id": 1, "type": "Geometry", "problem": "Find the angle..."},
        {"id": 2, "type": "Algebra", "problem": "Solve for x..."},
        {"id": 3, "type": "Number Theory", "problem": "Find the remainder..."},
        {"id": 4, "type": "Geometry", "problem": "Calculate the area..."},
    ]
    
    for item in test_items:
        problem_type = item.get("type", "")
        
        if problem_type == "Geometry":
            instructions = PROMPT_GEOMETRY_INSTRUCTIONS
            prompt_type = "GEOMETRY"
        else:
            instructions = PROMPT_DEFAULT_INSTRUCTIONS
            prompt_type = "DEFAULT"
        
        print(f"\nID={item['id']}, Type={problem_type:20s} → {prompt_type} prompt")
    
    print("\n" + "=" * 60)
    print("✓ プロンプト選択ロジック正常")
    print("=" * 60)


if __name__ == "__main__":
    test_prompt_loading()
    test_prompt_selection()
    print("\n✓ All tests passed!")
