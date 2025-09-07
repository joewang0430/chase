"""
参考验证器 - 用简单但准确的方法验证合法移动
基于ajex的思路但更简化，专注于准确性而非性能
"""

class ReferenceValidator:
    def __init__(self):
        self.directions = [
            (-1, -1), (-1, 0), (-1, 1),  # 上左, 上, 上右
            (0, -1),           (0, 1),   # 左,       右
            (1, -1),  (1, 0),  (1, 1)    # 下左, 下, 下右
        ]
    
    def board_from_visual(self, visual_board):
        """从可视化棋盘转换为内部表示"""
        # 移除标题行和列标
        lines = visual_board.strip().split('\n')[1:]  # 跳过列标题
        board = []
        
        for line in lines:
            if line.strip().startswith(('1', '2', '3', '4', '5', '6', '7', '8')):
                # 提取棋盘内容，跳过行号
                row_content = line.split()[1:]  # 跳过行号
                row = []
                for cell in row_content:
                    if cell == '●':
                        row.append('B')  # Black
                    elif cell == '○':
                        row.append('W')  # White
                    elif cell == '·':
                        row.append('U')  # Unoccupied
                    else:
                        row.append('U')  # 默认空位
                board.append(row)
        
        return board
    
    def is_valid_position(self, row, col, n=8):
        """检查位置是否在棋盘范围内"""
        return 0 <= row < n and 0 <= col < n
    
    def check_direction(self, board, row, col, dr, dc, color):
        """检查某个方向是否可以包夹对手棋子"""
        opponent_color = 'W' if color == 'B' else 'B'
        n = len(board)
        
        # 向该方向移动一步
        r, c = row + dr, col + dc
        found_opponent = False
        
        while self.is_valid_position(r, c, n):
            if board[r][c] == opponent_color:
                found_opponent = True
                r, c = r + dr, c + dc
            elif board[r][c] == color and found_opponent:
                # 找到己方棋子且中间有对手棋子
                return True
            else:
                # 遇到空位或己方棋子但没有对手棋子
                break
        
        return False
    
    def is_legal_move(self, board, row, col, color):
        """检查某个位置是否是合法移动"""
        n = len(board)
        
        # 检查位置是否在棋盘内
        if not self.is_valid_position(row, col, n):
            return False
        
        # 检查位置是否为空
        if board[row][col] != 'U':
            return False
        
        # 检查8个方向是否至少有一个可以包夹
        for dr, dc in self.directions:
            if self.check_direction(board, row, col, dr, dc, color):
                return True
        
        return False
    
    def get_all_legal_moves(self, board, color):
        """获取所有合法移动"""
        n = len(board)
        legal_moves = []
        
        for row in range(n):
            for col in range(n):
                if self.is_legal_move(board, row, col, color):
                    legal_moves.append((row, col))
        
        return legal_moves
    
    def print_board_analysis(self, board, color):
        """打印棋盘分析"""
        n = len(board)
        
        print(f"=== 棋盘分析 (当前玩家: {'黑子' if color == 'B' else '白子'}) ===")
        
        # 统计棋子数
        black_count = sum(row.count('B') for row in board)
        white_count = sum(row.count('W') for row in board)
        empty_count = sum(row.count('U') for row in board)
        
        print(f"黑子: {black_count}, 白子: {white_count}, 空位: {empty_count}")
        
        # 获取合法移动
        legal_moves = self.get_all_legal_moves(board, color)
        
        print(f"\n合法移动数量: {len(legal_moves)}")
        print("合法移动位置:")
        
        for i, (row, col) in enumerate(legal_moves, 1):
            pos_name = f"{chr(ord('a') + col)}{row + 1}"
            print(f"  {i}. {pos_name} (位置 {row},{col})")
            
            # 详细分析这个位置为什么合法
            print(f"     原因: ", end="")
            reasons = []
            for dr, dc in self.directions:
                if self.check_direction(board, row, col, dr, dc, color):
                    dir_name = self.get_direction_name(dr, dc)
                    reasons.append(dir_name)
            print(", ".join(reasons))
        
        if not legal_moves:
            print("  无合法移动")
        
        return legal_moves
    
    def get_direction_name(self, dr, dc):
        """获取方向名称"""
        direction_names = {
            (-1, -1): "↖", (-1, 0): "↑", (-1, 1): "↗",
            (0, -1): "←",              (0, 1): "→",
            (1, -1): "↙",  (1, 0): "↓",  (1, 1): "↘"
        }
        return direction_names.get((dr, dc), "?")


def test_step52_board():
    """测试Step 52的棋盘"""
    validator = ReferenceValidator()
    
    # Step 52的棋盘状态
    visual_board = """   a b c d e f g h
 1 ○ ○ ○ ○ ○ ○ ○ ○
 2 ● ● ● ○ ● ● ● ○
 3 ○ ● ○ ● ● ● ● ○
 4 ○ ○ ● ● ● ○ ● ○
 5 ○ ○ ○ ● ● ● ○ ○
 6 · ○ ○ ○ ● ○ ● ○
 7 · ○ ○ ○ ○ · · ○
 8 ○ · ○ ○ · · · ○"""
    
    board = validator.board_from_visual(visual_board)
    
    print("原始棋盘:")
    for i, row in enumerate(board):
        print(f" {i+1} {' '.join(row)}")
    
    # 测试黑子的合法移动
    print("\n" + "="*50)
    black_moves = validator.print_board_analysis(board, 'B')
    
    print("\n" + "="*50)
    print(f"结论: 黑子有 {len(black_moves)} 个合法移动")
    
    return black_moves


if __name__ == "__main__":
    legal_moves = test_step52_board()
