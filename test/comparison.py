import numpy as np
import time

class ComparisonBoard:
    def __init__(self):
        self.direction_data = self._precompute_direction_data()
        
        # 向量化版本的预计算数据 - 修复类型问题
        self.shifts = np.array([1, 7, 8, 9, 1, 7, 8, 9], dtype=np.uint64)  # 改为uint64
        self.masks = np.array([
            ~np.uint64(0x8080808080808080), ~np.uint64(0x01010101010101FF),
            ~np.uint64(0x00000000000000FF), ~np.uint64(0x80808080808080FF),
            ~np.uint64(0x0101010101010101), ~np.uint64(0xFF80808080808080),
            ~np.uint64(0xFF00000000000000), ~np.uint64(0xFF01010101010101)
        ], dtype=np.uint64)
        self.is_left_shift = np.array([True, True, True, True, False, False, False, False])
    
    def _precompute_direction_data(self):
        """预计算8个方向的掩码、位移和操作"""
        directions = []
        
        # 8个方向：右、左下、下、右下、左、右上、上、左上
        direction_configs = [
            (1, 'right'),     # 右
            (7, 'left_down'), # 左下  
            (8, 'down'),      # 下
            (9, 'right_down'),# 右下
            (-1, 'left'),     # 左（用负数表示左移）
            (-7, 'right_up'), # 右上
            (-8, 'up'),       # 上  
            (-9, 'left_up')   # 左上
        ]
        
        for shift, name in direction_configs:
            if shift > 0:
                # 正方向（左移位）
                if shift == 1:  # 右
                    mask = ~np.uint64(0x8080808080808080)  # 不能从H列向右
                elif shift == 7:  # 左下
                    mask = ~np.uint64(0x01010101010101FF)  # 不能从A列或底行
                elif shift == 8:  # 下
                    mask = ~np.uint64(0x00000000000000FF)  # 不能从底行向下
                elif shift == 9:  # 右下
                    mask = ~np.uint64(0x80808080808080FF)  # 不能从H列或底行
                    
                directions.append((mask, shift, np.uint64.__lshift__))
            else:
                # 负方向（右移位）
                shift_abs = abs(shift)
                if shift_abs == 1:  # 左
                    mask = ~np.uint64(0x0101010101010101)  # 不能从A列向左
                elif shift_abs == 7:  # 右上
                    mask = ~np.uint64(0xFF80808080808080)  # 不能从H列或顶行
                elif shift_abs == 8:  # 上
                    mask = ~np.uint64(0xFF00000000000000)  # 不能从顶行向上
                elif shift_abs == 9:  # 左上
                    mask = ~np.uint64(0xFF01010101010101)  # 不能从A列或顶行
                    
                directions.append((mask, shift_abs, np.uint64.__rshift__))
        
        return directions
    
    def get_legal_moves_for_loop(self, player_board, opponent_board):
        """8次for循环版本"""
        legal_moves = np.uint64(0)
        empty_squares = ~(player_board | opponent_board)
        
        for mask, shift, operation in self.direction_data:
            propagating = player_board
            while propagating:
                propagating = operation((propagating & mask), shift) & opponent_board
                potential_moves = operation((propagating & mask), shift) & empty_squares
                legal_moves |= potential_moves
        
        return legal_moves
    
    def get_legal_moves_vectorized(self, player_board, opponent_board):
        """numpy向量化版本 - 修复类型问题"""
        legal_moves = np.uint64(0)
        empty_squares = ~(player_board | opponent_board)
        
        # 创建8个方向的并行数组
        player_vec = np.full(8, player_board, dtype=np.uint64)
        opponent_vec = np.full(8, opponent_board, dtype=np.uint64)
        
        # 并行传播（但要固定循环次数）
        propagating_vec = player_vec.copy()
        
        for step in range(6):  # 最多6步传播
            # 向量化的位移操作 - 使用np.left_shift和np.right_shift
            new_propagating = np.zeros(8, dtype=np.uint64)
            
            # 左移 - 使用numpy函数而不是操作符
            left_indices = self.is_left_shift
            if np.any(left_indices):
                new_propagating[left_indices] = np.left_shift(
                    (propagating_vec[left_indices] & self.masks[left_indices]), 
                    self.shifts[left_indices]
                )
            
            # 右移  
            right_indices = ~self.is_left_shift
            if np.any(right_indices):
                new_propagating[right_indices] = np.right_shift(
                    (propagating_vec[right_indices] & self.masks[right_indices]), 
                    self.shifts[right_indices]
                )
            
            propagating_vec = new_propagating & opponent_vec
            
            # 检查合法移动
            potential_moves_vec = np.zeros(8, dtype=np.uint64)
            if np.any(left_indices):
                potential_moves_vec[left_indices] = np.left_shift(
                    (propagating_vec[left_indices] & self.masks[left_indices]), 
                    self.shifts[left_indices]
                ) & empty_squares
            
            if np.any(right_indices):
                potential_moves_vec[right_indices] = np.right_shift(
                    (propagating_vec[right_indices] & self.masks[right_indices]), 
                    self.shifts[right_indices]
                ) & empty_squares
            
            legal_moves |= np.bitwise_or.reduce(potential_moves_vec)
            
            # 检查是否还有传播
            if np.all(propagating_vec == 0):
                break
        
        return legal_moves

# 性能测试
def benchmark():
    board = ComparisonBoard()
    player = np.uint64(0x0000000810000000)
    opponent = np.uint64(0x0000001008000000)
    
    print("开始性能测试...")
    
    # 先验证结果一致性
    result1 = board.get_legal_moves_for_loop(player, opponent)
    result2 = board.get_legal_moves_vectorized(player, opponent)
    print(f"结果一致性检查: {result1 == result2}")
    print(f"For循环结果: {result1:016x}")
    print(f"向量化结果: {result2:016x}")
    
    # 测试for循环版本
    start = time.time()
    for _ in range(10000):
        result1 = board.get_legal_moves_for_loop(player, opponent)
    time1 = time.time() - start
    
    # 测试向量化版本
    start = time.time()
    for _ in range(10000):
        result2 = board.get_legal_moves_vectorized(player, opponent)
    time2 = time.time() - start
    
    print(f"For循环版本: {time1:.4f}秒")
    print(f"向量化版本: {time2:.4f}秒") 
    print(f"性能比: {time2/time1:.2f}x")

if __name__ == "__main__":
    benchmark()