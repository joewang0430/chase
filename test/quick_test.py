#!/usr/bin/env python3
"""
快速测试脚本 - 查看所有测试棋盘的基本信息
"""

import sys
import os

# 添加botzone目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
botzone_dir = os.path.join(parent_dir, 'botzone')
sys.path.insert(0, botzone_dir)

from new_main import Board, TEST_BOARDS, create_board_from_positions

def quick_analysis():
    """快速分析所有棋盘"""
    board = Board()
    
    print("=" * 80)
    print("所有测试棋盘快速分析")
    print("=" * 80)
    print(f"{'棋盘名称':<15} {'黑子':<4} {'白子':<4} {'空位':<4} {'合法移动':<8} {'最佳移动':<10} {'分数':<6}")
    print("-" * 80)
    
    for board_name, board_data in TEST_BOARDS.items():
        player, opponent = create_board_from_positions(board_data["black"], board_data["white"])
        
        black_count = board._popcount(player)
        white_count = board._popcount(opponent)
        empty_count = 64 - black_count - white_count
        
        # 获取合法移动
        legal_moves = board.get_legal_moves_sorted(player, opponent)
        legal_count = len(legal_moves)
        
        if legal_moves:
            # 简单搜索深度4来快速得到结果
            best_move = board.find_best_move(player, opponent, depth=4)
            if best_move is not None:
                row, col = best_move // 8, best_move % 8
                best_move_str = f"{chr(ord('a')+col)}{row+1}"
                
                # 获取分数
                next_p, next_o = board.make_move(player, opponent, best_move)
                score = board.pvs_search(next_o, next_p, 3, -999999999, 999999999)
                score = -score  # 从当前玩家角度
            else:
                best_move_str = "无"
                score = 0
        else:
            best_move_str = "无"
            score = 0
        
        print(f"{board_name:<15} {black_count:<4} {white_count:<4} {empty_count:<4} {legal_count:<8} {best_move_str:<10} {score:<6}")
    
    print("-" * 80)

def detailed_analysis(board_name):
    """详细分析指定棋盘"""
    if board_name not in TEST_BOARDS:
        print(f"错误: 棋盘 '{board_name}' 不存在")
        print(f"可用棋盘: {list(TEST_BOARDS.keys())}")
        return
    
    board = Board()
    board_data = TEST_BOARDS[board_name]
    
    print(f"\n{'='*60}")
    print(f"=== 详细分析: {board_name} ===")
    
    player, opponent = create_board_from_positions(board_data["black"], board_data["white"])
    
    print(f"黑子数量: {board._popcount(player)}")
    print(f"白子数量: {board._popcount(opponent)}")
    print(f"空位数量: {64 - board._popcount(player) - board._popcount(opponent)}")
    
    # 获取合法移动
    legal_moves = board.get_legal_moves_sorted(player, opponent)
    print(f"合法移动数: {len(legal_moves)}")
    
    if legal_moves:
        print("所有合法移动:")
        for i, move in enumerate(legal_moves, 1):
            row, col = move // 8, move % 8
            print(f"  {i}. {chr(ord('a')+col)}{row+1} (位置 {move}, 价值: {board.MOVE_VALUES[move]})")
        
        # 深度搜索
        import time
        start_time = time.time()
        best_move = board.find_best_move(player, opponent, depth=6)
        elapsed_time = time.time() - start_time
        
        if best_move is not None:
            row, col = best_move // 8, best_move % 8
            print(f"\n🎯 推荐移动: {chr(ord('a')+col)}{row+1} (位置 {best_move})")
            print(f"搜索深度: 6, 耗时: {elapsed_time:.3f}秒")
        else:
            print("未找到最佳移动")
    else:
        print("无合法移动")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        board_name = sys.argv[1]
        detailed_analysis(board_name)
    else:
        quick_analysis()
        print("\n使用方法:")
        print("  python quick_test.py           # 快速查看所有棋盘")
        print("  python quick_test.py 棋盘名称  # 详细分析指定棋盘")
        print(f"  可用棋盘: {['ajexsp_48', 'ajexsp_49', 'ajexsp_50', 'ajexsp_52', 'ajexsp_53', 'ajexsp_54', 'ajexsp_56', 'ajexsp_57', 'ajexsp_58']}")
