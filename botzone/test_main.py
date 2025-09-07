
import sys
import time
import ctypes
import os
import json
from ctypes import c_int, c_char, POINTER, byref

class BotzoneCtypesBot:
    def __init__(self):
        self.my_color = 1  # 1为黑方，-1为白方
        self.grid = [[0 for _ in range(8)] for _ in range(8)]
        
        # 初始化标准开局
        self.grid[3][3] = self.grid[4][4] = -1  # 白子
        self.grid[3][4] = self.grid[4][3] = 1   # 黑子
        
        self.c_lib = None
        self.load_c_library()
        
    def load_c_library(self):
        """加载C动态库"""
        try:
            # 动态计算路径，相对于当前脚本文件位置
            script_dir = os.path.dirname(os.path.abspath(__file__))
            so_path = os.path.join(script_dir, 'data', 'hukaiqi1_7.so')
            
            if os.path.exists(so_path):
                self.c_lib = ctypes.CDLL(so_path)
                
                # 设置makeMove函数的参数类型
                # int makeMove(const char board[][26], int n, char turn, int *row, int *col)
                self.c_lib.makeMove.argtypes = [
                    POINTER(c_char * 26 * 26),  # board[][26]
                    c_int,                       # n
                    c_char,                      # turn  
                    POINTER(c_int),              # row
                    POINTER(c_int)               # col
                ]
                self.c_lib.makeMove.restype = c_int
                
                print(f"C库加载成功: {so_path}", file=sys.stderr)
                
            else:
                print(f"❌ C库文件不存在: {so_path}", file=sys.stderr)
                
        except Exception as e:
            print(f"❌ C库加载失败: {e}", file=sys.stderr)
    
    def grid_to_c_board(self):
        """将Python grid转换为C需要的26x26 char数组"""
        # 创建26x26的char数组
        c_board = (c_char * 26 * 26)()
        
        for i in range(8):
            for j in range(8):
                if self.grid[i][j] == 1:       # 黑子
                    c_board[i][j] = b'B'
                elif self.grid[i][j] == -1:    # 白子  
                    c_board[i][j] = b'W'
                else:                          # 空位
                    c_board[i][j] = b'U'
        
        # 填充剩余位置为'U'
        for i in range(8):
            for j in range(8, 26):
                c_board[i][j] = b'U'
        for i in range(8, 26):
            for j in range(26):
                c_board[i][j] = b'U'
                
        return c_board
    
    def place_stone(self, x, y, color):
        """在网格上放置棋子并翻转
        参数: x=列号, y=行号, color=棋子颜色
        """
        if x < 0 or y < 0:
            return False
            
        # 注意：grid[行][列] = grid[y][x]
        if self.grid[y][x] != 0:
            return False
            
        # 8个方向
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        flipped = []
        
        for dx, dy in directions:
            temp_flipped = []
            # 注意：这里nx,ny是行列坐标，不是x,y坐标
            nx, ny = y + dy, x + dx  # nx=新行号, ny=新列号
            
            # 寻找可翻转的棋子
            while 0 <= nx < 8 and 0 <= ny < 8 and self.grid[nx][ny] == -color:
                temp_flipped.append((nx, ny))
                nx += dy
                ny += dx
            
            # 如果找到同色棋子，则这个方向的棋子可以翻转
            if (0 <= nx < 8 and 0 <= ny < 8 and 
                self.grid[nx][ny] == color and len(temp_flipped) > 0):
                flipped.extend(temp_flipped)
        
        if flipped:
            # 放置新棋子
            self.grid[y][x] = color
            # 翻转棋子
            for fx, fy in flipped:
                self.grid[fx][fy] = color
            return True
        
        return False
    
    def find_best_move_c(self):
        """使用C库寻找最佳移动"""
        if not self.c_lib:
            return self.find_best_move_python()
        
        try:
            # 转换棋盘格式
            c_board = self.grid_to_c_board()
            
            # 确定当前玩家
            c_turn = b'B' if self.my_color == 1 else b'W'
            
            # 准备输出参数
            row = c_int()
            col = c_int()
            
            # 调用C函数
            start_time = time.time()
            result = self.c_lib.makeMove(byref(c_board), 8, c_turn, byref(row), byref(col))
            elapsed_time = time.time() - start_time
            
            # 获取结果 - 注意C库返回的是(row, col)，需要转换为(x, y)
            best_row = row.value
            best_col = col.value
            
            # 转换为botzone期望的(x, y)格式：x=col, y=row
            return best_col, best_row, elapsed_time, result
            
        except Exception as e:
            # 不要打印到stdout，使用stderr
            import sys
            print(f"C库调用失败: {e}", file=sys.stderr)
            return self.find_best_move_python()
    
    def find_best_move_python(self):
        """Python备用算法（简单实现）"""
        # 简单的合法移动检查
        legal_moves = []
        
        # 注意：这里i是行，j是列，需要转换为(x,y)格式
        for i in range(8):  # i = 行号 = y
            for j in range(8):  # j = 列号 = x
                if self.grid[i][j] == 0:
                    # 检查是否为合法移动，传入(x=j, y=i)
                    if self.is_legal_move(j, i, self.my_color):
                        legal_moves.append((j, i))  # 返回(x, y)格式
        
        if legal_moves:
            # 简单策略：选择第一个合法移动
            return legal_moves[0][0], legal_moves[0][1], 0.001, 0
        else:
            return -1, -1, 0.001, 0
    
    def is_legal_move(self, x, y, color):
        """检查是否为合法移动（简化版）
        参数: x=列号, y=行号, color=棋子颜色
        """
        # 注意：grid[行][列] = grid[y][x]
        if self.grid[y][x] != 0:
            return False
        
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        for dx, dy in directions:
            # 注意：这里nx,ny是行列坐标
            nx, ny = y + dy, x + dx  # nx=新行号, ny=新列号
            found_opponent = False
            
            while 0 <= nx < 8 and 0 <= ny < 8:
                if self.grid[nx][ny] == -color:
                    found_opponent = True
                elif self.grid[nx][ny] == color and found_opponent:
                    return True
                else:
                    break
                nx += dy
                ny += dx
        
        return False
    
    def debug_board(self):
        """调试：打印当前棋盘状态到stderr"""
        import sys
        print("=== 当前棋盘状态 ===", file=sys.stderr)
        for i in range(8):
            row_str = ""
            for j in range(8):
                if self.grid[i][j] == 1:
                    row_str += "B "
                elif self.grid[i][j] == -1:
                    row_str += "W "
                else:
                    row_str += ". "
            print(f"行{i}: {row_str}", file=sys.stderr)
        print("================", file=sys.stderr)
    
    def run(self):
        """主运行函数 - JSON交互模式"""
        try:
            # 1. 读取并解析单行JSON输入
            full_input_str = input()
            full_input = json.loads(full_input_str)

            # 2. 从JSON中提取历史记录
            requests = full_input["requests"]
            responses = full_input["responses"]
            turnID = len(responses)

            # 3. 根据历史记录重建棋盘状态
            # 判断颜色：第一回合若收到x=-1，则为黑方
            if requests[0]["x"] < 0:
                self.my_color = 1  # 黑方
            else:
                self.my_color = -1  # 白方

            # 遍历过往回合，恢复局面
            for i in range(turnID):
                # 模拟对手落子
                opponent_move = requests[i]
                if opponent_move["x"] >= 0:
                    self.place_stone(opponent_move["x"], opponent_move["y"], -self.my_color)
                
                # 模拟己方落子
                my_move = responses[i]
                if my_move["x"] >= 0:
                    self.place_stone(my_move["x"], my_move["y"], self.my_color)

            # 4. 处理当前回合的输入
            current_request = requests[turnID]
            if current_request["x"] >= 0:
                self.place_stone(current_request["x"], current_request["y"], -self.my_color)

            # 调试：在计算前打印棋盘状态
            self.debug_board()
            
            # 5. 使用C库计算最佳移动
            start_time = time.time()
            x, y, c_elapsed, c_result = self.find_best_move_c()
            total_elapsed = time.time() - start_time
            
            # ★★★ 重要：更新棋盘状态 ★★★
            if x >= 0 and y >= 0:
                self.place_stone(x, y, self.my_color)
            
            # 6. 构建并输出JSON响应
            my_count = sum(row.count(self.my_color) for row in self.grid)
            opp_count = sum(row.count(-self.my_color) for row in self.grid)
            empty_count = sum(row.count(0) for row in self.grid)
            
            algorithm_used = "C_LIB" if self.c_lib else "PYTHON"
            retreat_flag = "RETREAT" if c_result == 1 else "NORMAL"
            
            debug_info = f"{algorithm_used}:{retreat_flag} Time:{total_elapsed:.3f}s C:{c_elapsed:.3f}s Me:{my_count} Opp:{opp_count} Empty:{empty_count}"

            # 构建输出字典
            output_dict = {
                "response": {
                    "x": x,
                    "y": y
                },
                "debug": debug_info
            }

            # 将字典转换为紧凑的单行JSON并打印
            print(json.dumps(output_dict, separators=(',', ':')))
            
        except Exception as e:
            # 错误处理也应输出JSON格式
            error_output = {
                "response": {"x": -1, "y": -1},
                "debug": f"Error: {str(e)}"
            }
            print(json.dumps(error_output, separators=(',', ':')))

if __name__ == "__main__":
    bot = BotzoneCtypesBot()
    bot.run()