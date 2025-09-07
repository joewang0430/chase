#!/usr/bin/env python3
"""
权威测试脚本：验证play.py与test_main.py的函数正确性
对比bitboard实现与grid实现的一致性
"""

import sys
import os
import json
import numpy as np
from typing import List, Tuple, Dict

# 添加botzone目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'botzone'))

from play import OthelloAI
from test_main import BotzoneCtypesBot

class CorrectnessValidator:
    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
        self.verbose = True
        
    def log(self, message, level="INFO"):
        if self.verbose:
            prefix = "✅" if level == "PASS" else "❌" if level == "FAIL" else "🔍"
            print(f"{prefix} {message}")
    
    def assert_equal(self, actual, expected, test_name):
        """断言两个值相等"""
        if actual == expected:
            self.tests_passed += 1
            self.log(f"{test_name}: PASSED", "PASS")
            return True
        else:
            self.tests_failed += 1
            self.log(f"{test_name}: FAILED - Expected {expected}, got {actual}", "FAIL")
            return False
    
    def assert_board_equal(self, play_ai: OthelloAI, test_bot: BotzoneCtypesBot, test_name):
        """对比两个AI的棋盘状态是否一致"""
        play_board = self.bitboard_to_grid(play_ai)
        test_board = test_bot.grid
        
        if self.grids_equal(play_board, test_board):
            self.tests_passed += 1
            self.log(f"{test_name}: 棋盘状态一致", "PASS")
            return True
        else:
            self.tests_failed += 1
            self.log(f"{test_name}: 棋盘状态不一致", "FAIL")
            self.log("Play.py棋盘:")
            self.print_grid(play_board)
            self.log("Test_main.py棋盘:")
            self.print_grid(test_board)
            return False
    
    def bitboard_to_grid(self, ai: OthelloAI) -> List[List[int]]:
        """将bitboard转换为8x8网格"""
        grid = [[0 for _ in range(8)] for _ in range(8)]
        
        for pos in range(64):
            x, y = ai.bit_to_xy(pos)
            if (ai.my_pieces >> pos) & 1:
                grid[y][x] = ai.my_color
            elif (ai.opp_pieces >> pos) & 1:
                grid[y][x] = -ai.my_color
        
        return grid
    
    def grids_equal(self, grid1: List[List[int]], grid2: List[List[int]]) -> bool:
        """比较两个8x8网格是否相等"""
        for i in range(8):
            for j in range(8):
                if grid1[i][j] != grid2[i][j]:
                    return False
        return True
    
    def print_grid(self, grid: List[List[int]]):
        """打印网格状态"""
        for i in range(8):
            row_str = f"  {i}: "
            for j in range(8):
                if grid[i][j] == 1:
                    row_str += "B "
                elif grid[i][j] == -1:
                    row_str += "W "
                else:
                    row_str += ". "
            self.log(row_str)
    
    def create_test_input(self, requests: List[Dict], responses: List[Dict]) -> str:
        """创建botzone格式的测试输入"""
        return json.dumps({
            "requests": requests,
            "responses": responses,
            "data": "",
            "globaldata": ""
        })
    
    def test_initial_board_setup(self):
        """测试1: 初始棋盘设置"""
        self.log("=== 测试1: 初始棋盘设置 ===")
        
        # 测试黑方开局
        play_ai = OthelloAI()
        play_ai.init_standard_board(1)  # 黑方
        
        test_bot = BotzoneCtypesBot()
        test_bot.my_color = 1  # 黑方
        
        self.assert_board_equal(play_ai, test_bot, "黑方初始棋盘")
        
        # 测试白方开局
        play_ai2 = OthelloAI()
        play_ai2.init_standard_board(-1)  # 白方
        
        test_bot2 = BotzoneCtypesBot()
        test_bot2.my_color = -1  # 白方
        
        # 手动设置test_bot2的初始状态（从白方视角）
        test_bot2.grid = [[0 for _ in range(8)] for _ in range(8)]
        test_bot2.grid[3][3] = test_bot2.grid[4][4] = -1  # 白子
        test_bot2.grid[3][4] = test_bot2.grid[4][3] = 1   # 黑子
        
        self.assert_board_equal(play_ai2, test_bot2, "白方初始棋盘")
    
    def test_coordinate_conversion(self):
        """测试2: 坐标转换函数"""
        self.log("=== 测试2: 坐标转换函数 ===")
        
        play_ai = OthelloAI()
        
        # 测试关键位置的坐标转换
        test_cases = [
            (0, 0, 0),    # 左上角
            (7, 7, 63),   # 右下角
            (3, 3, 27),   # 初始白子位置
            (3, 4, 28),   # 初始黑子位置
            (4, 3, 35),   # 初始黑子位置
            (4, 4, 36),   # 初始白子位置
        ]
        
        for x, y, expected_bit in test_cases:
            actual_bit = play_ai.xy_to_bit(x, y)
            self.assert_equal(actual_bit, expected_bit, f"xy_to_bit({x},{y})")
            
            actual_x, actual_y = play_ai.bit_to_xy(expected_bit)
            self.assert_equal((actual_x, actual_y), (x, y), f"bit_to_xy({expected_bit})")
    
    def test_single_move(self):
        """测试3: 单步移动的正确性"""
        self.log("=== 测试3: 单步移动测试 ===")
        
        # 模拟标准开局后黑方第一步下(2,3)
        test_input = self.create_test_input(
            requests=[{"x": -1, "y": -1}],  # 黑方开局
            responses=[]
        )
        
        # 使用play.py处理
        play_ai = OthelloAI()
        
        # 模拟输入解析
        requests = [{"x": -1, "y": -1}]
        responses = []
        play_ai.get_current_board(requests, responses)
        
        # 使用test_main.py处理
        test_bot = BotzoneCtypesBot()
        test_bot.my_color = 1  # 黑方
        
        # 初始状态应该一致
        self.assert_board_equal(play_ai, test_bot, "初始状态对比")
        
        # 测试黑方下棋到(2,3)
        pos_23 = play_ai.xy_to_bit(2, 3)
        new_my, new_opp = play_ai.make_move(play_ai.my_pieces, play_ai.opp_pieces, pos_23)
        
        # 更新play_ai状态
        play_ai.my_pieces = new_my
        play_ai.opp_pieces = new_opp
        
        # test_bot也执行相同移动
        test_bot.place_stone(2, 3, 1)  # 黑方下(2,3)
        
        self.assert_board_equal(play_ai, test_bot, "黑方下(2,3)后状态")
    
    def test_game_sequence(self):
        """测试4: 完整对局序列"""
        self.log("=== 测试4: 完整对局序列 ===")
        
        # 定义一个测试对局序列
        game_sequence = [
            # (requests, responses) - 模拟真实对局
            ([{"x": -1, "y": -1}], []),                    # 黑方开局
            ([{"x": -1, "y": -1}], [{"x": 2, "y": 3}]),   # 黑方下(2,3)
            ([{"x": -1, "y": -1}, {"x": 2, "y": 4}], [{"x": 2, "y": 3}]), # 白方下(2,4)
        ]
        
        for i, (requests, responses) in enumerate(game_sequence):
            self.log(f"  测试对局步骤 {i+1}")
            
            # play.py处理
            play_ai = OthelloAI()
            play_ai.get_current_board(requests, responses)
            
            # test_main.py处理（手动模拟相同输入）
            test_bot = BotzoneCtypesBot()
            
            # 模拟JSON输入处理
            test_input = self.create_test_input(requests, responses)
            
            # 手动重建test_bot状态
            if requests[0]["x"] < 0:
                test_bot.my_color = 1  # 黑方
            else:
                test_bot.my_color = -1  # 白方
            
            # 重放历史
            turn_count = len(responses)
            for j in range(turn_count):
                # 对手移动
                if j < len(requests) and requests[j]["x"] >= 0:
                    test_bot.place_stone(requests[j]["x"], requests[j]["y"], -test_bot.my_color)
                
                # 我方移动
                if responses[j]["x"] >= 0:
                    test_bot.place_stone(responses[j]["x"], responses[j]["y"], test_bot.my_color)
            
            # 当前回合对手移动
            if turn_count < len(requests) and requests[turn_count]["x"] >= 0:
                test_bot.place_stone(requests[turn_count]["x"], requests[turn_count]["y"], -test_bot.my_color)
            
            self.assert_board_equal(play_ai, test_bot, f"步骤{i+1}状态对比")
    
    def test_edge_cases(self):
        """测试5: 边界情况"""
        self.log("=== 测试5: 边界情况测试 ===")
        
        # 测试PASS移动
        play_ai = OthelloAI()
        play_ai.init_standard_board(1)
        
        # 记录初始状态
        initial_my = play_ai.my_pieces
        initial_opp = play_ai.opp_pieces
        
        # 执行PASS(-1)
        new_my, new_opp = play_ai.make_move(play_ai.my_pieces, play_ai.opp_pieces, -1)
        
        # PASS应该交换双方
        self.assert_equal(new_my, initial_opp, "PASS后我方棋子应该是原对方棋子")
        self.assert_equal(new_opp, initial_my, "PASS后对方棋子应该是原我方棋子")
        
        # 测试坐标边界
        corner_cases = [(0, 0), (0, 7), (7, 0), (7, 7)]
        for x, y in corner_cases:
            bit_pos = play_ai.xy_to_bit(x, y)
            back_x, back_y = play_ai.bit_to_xy(bit_pos)
            self.assert_equal((back_x, back_y), (x, y), f"角落坐标({x},{y})转换")
    
    def test_bitboard_operations(self):
        """测试6: Bitboard操作的正确性"""
        self.log("=== 测试6: Bitboard操作测试 ===")
        
        play_ai = OthelloAI()
        
        # 测试set_bit函数
        empty_board = np.uint64(0)
        
        # 设置标准开局位置
        board_with_33 = play_ai.set_bit(empty_board, 3, 3)
        board_with_34 = play_ai.set_bit(board_with_33, 3, 4)
        board_with_43 = play_ai.set_bit(board_with_34, 4, 3)
        board_with_44 = play_ai.set_bit(board_with_43, 4, 4)
        
        # 验证位数
        expected_bits = 4
        actual_bits = bin(board_with_44).count('1')
        self.assert_equal(actual_bits, expected_bits, "标准开局4个位置设置")
        
        # 验证具体位置
        for x, y in [(3, 3), (3, 4), (4, 3), (4, 4)]:
            bit_pos = play_ai.xy_to_bit(x, y)
            is_set = bool((board_with_44 >> bit_pos) & 1)
            self.assert_equal(is_set, True, f"位置({x},{y})已设置")
    
    def run_all_tests(self):
        """运行所有测试"""
        self.log("🚀 开始权威性测试...")
        
        try:
            self.test_initial_board_setup()
            self.test_coordinate_conversion()
            self.test_single_move()
            self.test_game_sequence()
            self.test_edge_cases()
            self.test_bitboard_operations()
            
        except Exception as e:
            self.log(f"测试过程中出现异常: {e}", "FAIL")
            self.tests_failed += 1
        
        # 输出测试结果
        total_tests = self.tests_passed + self.tests_failed
        pass_rate = (self.tests_passed / total_tests * 100) if total_tests > 0 else 0
        
        self.log(f"\n📊 测试完成!")
        self.log(f"总测试数: {total_tests}")
        self.log(f"通过: {self.tests_passed}")
        self.log(f"失败: {self.tests_failed}")
        self.log(f"通过率: {pass_rate:.1f}%")
        
        if self.tests_failed == 0:
            self.log("🎉 所有测试通过! play.py实现正确!", "PASS")
            return True
        else:
            self.log("⚠️  存在失败的测试，需要检查实现", "FAIL")
            return False

def main():
    """主函数"""
    print("=" * 60)
    print("🧪 Play.py 权威性正确性测试")
    print("=" * 60)
    
    validator = CorrectnessValidator()
    success = validator.run_all_tests()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
