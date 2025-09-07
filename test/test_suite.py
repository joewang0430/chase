#!/usr/bin/env python3
"""
Othello测试套件 - 管理所有测试数据和测试功能
"""

import numpy as np
import time
import sys
import os

# 导入核心Board类
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
botzone_dir = os.path.join(parent_dir, 'botzone')
sys.path.insert(0, botzone_dir)

from new_main import Board

# 测试棋盘数据集
TEST_BOARDS = {
    "ajexsp_48": {
        "black": [8, 9, 10, 12, 13, 17, 19, 20, 21, 23, 26, 27, 28, 30, 31, 35, 37, 39, 46, 47, 55],
        "white": [0, 1, 2, 3, 4, 5, 6, 7, 11, 15, 16, 18, 22, 24, 25, 29, 32, 33, 34, 36, 38, 41, 42, 43, 44, 45, 49, 50, 51, 56, 58],
        "empty": [14, 40, 48, 52, 53, 54, 57, 59, 60, 61, 62, 63]
    },
    "ajexsp_49": {
        "black": [8, 9, 10, 12, 13, 14, 17, 19, 20, 21, 22, 23, 26, 27, 28, 30, 31, 35, 37, 39, 46, 47, 55],
        "white": [0, 1, 2, 3, 4, 5, 6, 7, 11, 15, 16, 18, 24, 25, 29, 32, 33, 34, 36, 38, 41, 42, 43, 44, 45, 49, 50, 51, 56, 58],
        "empty": [40, 48, 52, 53, 54, 57, 59, 60, 61, 62, 63]
    },
    "ajexsp_50": {
        "black": [8, 9, 10, 12, 13, 14, 17, 19, 20, 21, 22, 26, 27, 28, 30, 35, 37, 46],
        "white": [0, 1, 2, 3, 4, 5, 6, 7, 11, 15, 16, 18, 23, 24, 25, 29, 31, 32, 33, 34, 36, 38, 39, 41, 42, 43, 44, 45, 47, 49, 50, 51, 55, 56, 58, 63],
        "empty": [40, 48, 52, 53, 54, 57, 59, 60, 61, 62]
    },
    "ajexsp_52": {
        "black": [8, 9, 10, 12, 13, 14, 17, 19, 20, 21, 22, 26, 27, 28, 30, 35, 36, 37, 44, 46],
        "white": [0, 1, 2, 3, 4, 5, 6, 7, 11, 15, 16, 18, 23, 24, 25, 29, 31, 32, 33, 34, 38, 39, 41, 42, 43, 45, 47, 49, 50, 51, 52, 55, 56, 58, 59, 63],
        "empty": [40, 48, 53, 54, 57, 60, 61, 62]
    },
    "ajexsp_53": {
        "black": [8, 9, 10, 12, 13, 14, 17, 19, 20, 21, 22, 26, 27, 28, 30, 35, 36, 37, 44, 46, 52, 60],
        "white": [0, 1, 2, 3, 4, 5, 6, 7, 11, 15, 16, 18, 23, 24, 25, 29, 31, 32, 33, 34, 38, 39, 41, 42, 43, 45, 47, 49, 50, 51, 55, 56, 58, 59, 63],
        "empty": [40, 48, 53, 54, 57, 61, 62]
    },
    "ajexsp_54": {
        "black": [8, 9, 10, 12, 13, 14, 17, 19, 20, 21, 22, 26, 27, 28, 30, 35, 36, 37, 44, 60],
        "white": [0, 1, 2, 3, 4, 5, 6, 7, 11, 15, 16, 18, 23, 24, 25, 29, 31, 32, 33, 34, 38, 39, 41, 42, 43, 45, 46, 47, 49, 50, 51, 52, 53, 55, 56, 58, 59, 63],
        "empty": [40, 48, 54, 57, 61, 62]
    },
    "ajexsp_56": {
        "black": [8, 9, 10, 12, 13, 14, 17, 19, 20, 21, 22, 26, 27, 28, 30, 35, 36, 37, 44],
        "white": [0, 1, 2, 3, 4, 5, 6, 7, 11, 15, 16, 18, 23, 24, 25, 29, 31, 32, 33, 34, 38, 39, 41, 42, 43, 45, 46, 47, 49, 50, 51, 52, 53, 55, 56, 58, 59, 60, 61, 62, 63],
        "empty": [40, 48, 54, 57]
    },
    "ajexsp_57": {
        "black": [8, 9, 10, 12, 13, 14, 17, 19, 20, 21, 22, 26, 27, 28, 30, 34, 35, 36, 37, 41, 44, 48],
        "white": [0, 1, 2, 3, 4, 5, 6, 7, 11, 15, 16, 18, 23, 24, 25, 29, 31, 32, 33, 38, 39, 42, 43, 45, 46, 47, 49, 50, 51, 52, 53, 55, 56, 58, 59, 60, 61, 62, 63],
        "empty": [40, 54, 57]
    },
    "ajexsp_58": {
        "black": [8, 9, 10, 12, 13, 14, 17, 19, 20, 21, 22, 26, 27, 28, 30, 34, 35, 36, 37, 44],
        "white": [0, 1, 2, 3, 4, 5, 6, 7, 11, 15, 16, 18, 23, 24, 25, 29, 31, 32, 33, 38, 39, 40, 41, 42, 43, 45, 46, 47, 48, 49, 50, 51, 52, 53, 55, 56, 58, 59, 60, 61, 62, 63],
        "empty": [54, 57]
    },
}

def create_board_from_positions(black_positions, white_positions):
    """从位置列表创建位板"""
    player = np.uint64(0)
    opponent = np.uint64(0)
    
    for pos in black_positions:
        player |= np.uint64(1) << pos
    
    for pos in white_positions:
        opponent |= np.uint64(1) << pos
    
    return player, opponent

def test_board(board_obj, board_name, board_data, depth=6, silent=False):
    """测试单个棋盘"""
    if not silent:
        print(f"\n{'='*60}")
        print(f"=== 测试棋盘: {board_name} ===")
    
    player, opponent = create_board_from_positions(board_data["black"], board_data["white"])
    
    black_count = board_obj._popcount(player)
    white_count = board_obj._popcount(opponent)
    empty_count = 64 - black_count - white_count
    
    if not silent:
        print(f"黑子数量: {black_count}")
        print(f"白子数量: {white_count}")
        print(f"空位数量: {empty_count}")
    
    # 获取合法移动
    legal_moves = board_obj.get_legal_moves_sorted(player, opponent)
    legal_count = len(legal_moves)
    
    if not silent:
        print(f"合法移动数: {legal_count}")
        
        if legal_moves:
            print("合法移动:")
            for i, move in enumerate(legal_moves, 1):
                row, col = move // 8, move % 8
                print(f"  {i}. {chr(ord('a')+col)}{row+1} (位置 {move}, 价值: {board_obj.MOVE_VALUES[move]})")
    
    best_move = None
    search_time = 0
    best_score = 0
    
    if legal_count > 0:
        start_time = time.time()
        best_move = board_obj.find_best_move(player, opponent, depth=depth)
        search_time = time.time() - start_time
        
        if best_move is not None:
            # 获取最佳移动的分数
            next_p, next_o = board_obj.make_move(player, opponent, best_move)
            best_score = -board_obj.pvs_search(next_o, next_p, depth-1, -999999999, 999999999)
            
            if not silent:
                row, col = best_move // 8, best_move % 8
                print(f"\n🎯 最佳移动: {chr(ord('a')+col)}{row+1} (位置 {best_move})")
                print(f"搜索深度: {depth}, 耗时: {search_time:.3f}秒, 分数: {best_score}")
        elif not silent:
            print("未找到最佳移动")
    elif not silent:
        print("无合法移动")
    
    return {
        'board_name': board_name,
        'black_count': black_count,
        'white_count': white_count,
        'empty_count': empty_count,
        'legal_count': legal_count,
        'best_move': best_move,
        'best_score': best_score,
        'search_time': search_time
    }

def run_all_tests(depth=6, silent=False):
    """运行所有测试棋盘"""
    board = Board()
    
    if not silent:
        print("=" * 60)
        print("开始批量测试所有棋盘")
        print(f"搜索深度: {depth}")
        print("=" * 60)
    
    results = []
    
    for board_name, board_data in TEST_BOARDS.items():
        result = test_board(board, board_name, board_data, depth, silent)
        results.append(result)
    
    if not silent:
        print(f"\n{'='*60}")
        print("所有测试完成!")
    
    return results

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
            # 使用主程序的find_best_move方法，深度4快速搜索
            # 临时修改打印行为以避免输出过多信息
            import io
            import sys
            from contextlib import redirect_stdout
            
            with redirect_stdout(io.StringIO()):
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

def detailed_analysis(board_name, depth=6):
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
        start_time = time.time()
        best_move = board.find_best_move(player, opponent, depth=depth)
        elapsed_time = time.time() - start_time
        
        if best_move is not None:
            row, col = best_move // 8, best_move % 8
            print(f"\n🎯 推荐移动: {chr(ord('a')+col)}{row+1} (位置 {best_move})")
            print(f"搜索深度: {depth}, 耗时: {elapsed_time:.3f}秒")
        else:
            print("未找到最佳移动")
    else:
        print("无合法移动")

def performance_test(depth_range=[4, 6, 8], board_subset=None):
    """性能测试"""
    board = Board()
    
    if board_subset is None:
        board_subset = ["ajexsp_48", "ajexsp_52", "ajexsp_56", "ajexsp_58"]  # 选择几个代表性棋盘
    
    print("=" * 80)
    print("性能测试")
    print("=" * 80)
    print(f"{'棋盘':<15} {'深度':<4} {'合法移动':<8} {'耗时(秒)':<10} {'最佳移动':<10} {'分数':<6}")
    print("-" * 80)
    
    for board_name in board_subset:
        if board_name not in TEST_BOARDS:
            continue
            
        board_data = TEST_BOARDS[board_name]
        player, opponent = create_board_from_positions(board_data["black"], board_data["white"])
        legal_count = len(board.get_legal_moves_sorted(player, opponent))
        
        for depth in depth_range:
            start_time = time.time()
            best_move = board.find_best_move(player, opponent, depth=depth)
            elapsed_time = time.time() - start_time
            
            if best_move is not None:
                row, col = best_move // 8, best_move % 8
                best_move_str = f"{chr(ord('a')+col)}{row+1}"
                
                # 获取分数
                next_p, next_o = board.make_move(player, opponent, best_move)
                score = -board.pvs_search(next_o, next_p, depth-1, -999999999, 999999999)
            else:
                best_move_str = "无"
                score = 0
            
            print(f"{board_name:<15} {depth:<4} {legal_count:<8} {elapsed_time:<10.3f} {best_move_str:<10} {score:<6}")
    
    print("-" * 80)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "quick":
            quick_analysis()
        elif command == "all":
            depth = int(sys.argv[2]) if len(sys.argv) > 2 else 6
            run_all_tests(depth=depth)
        elif command == "perf":
            performance_test()
        elif command in TEST_BOARDS:
            # 检查是否有深度参数
            depth = int(sys.argv[2]) if len(sys.argv) > 2 else 6
            detailed_analysis(command, depth)
        else:
            print(f"未知命令: {command}")
            print("使用方法:")
            print("  python test_suite.py quick          # 快速查看所有棋盘")
            print("  python test_suite.py all [深度]     # 详细测试所有棋盘")
            print("  python test_suite.py perf           # 性能测试")
            print("  python test_suite.py 棋盘名称       # 详细分析指定棋盘")
            print(f"  可用棋盘: {list(TEST_BOARDS.keys())}")
    else:
        print("Othello 测试套件")
        print("=" * 50)
        print("使用方法:")
        print("  python test_suite.py quick          # 快速查看所有棋盘")
        print("  python test_suite.py all [深度]     # 详细测试所有棋盘 (默认深度6)")
        print("  python test_suite.py perf           # 性能测试")
        print("  python test_suite.py 棋盘名称       # 详细分析指定棋盘")
        print(f"\n可用棋盘: {list(TEST_BOARDS.keys())}")
        print("\n示例:")
        print("  python test_suite.py quick")
        print("  python test_suite.py ajexsp_52")
        print("  python test_suite.py all 8")
