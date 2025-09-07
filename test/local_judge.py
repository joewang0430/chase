import subprocess
import numpy as np
import os
import sys

# =============================================================================
#  Othello Game Logic (Bitboard Implementation) - 保持不变，这部分是正确的
# =============================================================================

class Board:
    """
    A highly optimized Othello board class using bitboards (64-bit integers).
    This class will be used by the judge to maintain the true state of the game.
    """
    def __init__(self, black_board=None, white_board=None):
        if black_board is None or white_board is None:
            # Standard initial setup
            self.black_board = np.uint64(0x0000000810000000)
            self.white_board = np.uint64(0x0000001008000000)
        else:
            self.black_board = np.uint64(black_board)
            self.white_board = np.uint64(white_board)

    def get_legal_moves(self, player_board, opponent_board):
        """Calculates all legal moves for the current player."""
        legal_moves = np.uint64(0)
        empty_cells = ~(player_board | opponent_board)
        
        # Directions: N, S, W, E, NW, SE, NE, SW
        directions = [8, -8, 1, -1, 9, -7, 7, -9]
        # Masks to prevent wrap-around
        masks = [
            np.uint64(0xFFFFFFFFFFFFFFFF), np.uint64(0xFFFFFFFFFFFFFFFF), # N, S
            np.uint64(0xFEFEFEFEFEFEFEFE), np.uint64(0x7F7F7F7F7F7F7F7F), # W, E
            np.uint64(0xFEFEFEFEFEFEFEFE), np.uint64(0x7F7F7F7F7F7F7F7F), # NW, SE
            np.uint64(0xFEFEFEFEFEFEFEFE), np.uint64(0x7F7F7F7F7F7F7F7F)  # NE, SW
        ]

        for i in range(len(directions)):
            direction = directions[i]
            mask = masks[i]
            
            if direction > 0:
                captured = ((player_board << direction) & opponent_board & mask)
                captured |= ((captured << direction) & opponent_board & mask)
                captured |= ((captured << direction) & opponent_board & mask)
                captured |= ((captured << direction) & opponent_board & mask)
                captured |= ((captured << direction) & opponent_board & mask)
                captured |= ((captured << direction) & opponent_board & mask)
                legal_moves |= (captured << direction) & empty_cells & mask
            else:
                abs_dir = -direction
                captured = ((player_board >> abs_dir) & opponent_board & mask)
                captured |= ((captured >> abs_dir) & opponent_board & mask)
                captured |= ((captured >> abs_dir) & opponent_board & mask)
                captured |= ((captured >> abs_dir) & opponent_board & mask)
                captured |= ((captured >> abs_dir) & opponent_board & mask)
                captured |= ((captured >> abs_dir) & opponent_board & mask)
                legal_moves |= (captured >> abs_dir) & empty_cells & mask
        
        return legal_moves

    def make_move(self, player_board, opponent_board, move_pos):
        """Applies a move and returns the new board state."""
        move_bit = np.uint64(1) << move_pos
        
        new_player_board = player_board | move_bit
        new_opponent_board = opponent_board
        
        directions = [8, -8, 1, -1, 9, -7, 7, -9]
        masks = [
            np.uint64(0xFFFFFFFFFFFFFFFF), np.uint64(0xFFFFFFFFFFFFFFFF),
            np.uint64(0xFEFEFEFEFEFEFEFE), np.uint64(0x7F7F7F7F7F7F7F7F),
            np.uint64(0xFEFEFEFEFEFEFEFE), np.uint64(0x7F7F7F7F7F7F7F7F),
            np.uint64(0xFEFEFEFEFEFEFEFE), np.uint64(0x7F7F7F7F7F7F7F7F)
        ]

        flipped_pieces = np.uint64(0)

        for i in range(len(directions)):
            direction = directions[i]
            mask = masks[i]
            line_flipped = np.uint64(0)
            
            if direction > 0:
                temp = move_bit
                for _ in range(7):
                    temp = (temp << direction) & mask
                    if temp & opponent_board:
                        line_flipped |= temp
                    elif temp & player_board:
                        flipped_pieces |= line_flipped
                        break
                    else:
                        break
            else:
                abs_dir = -direction
                temp = move_bit
                for _ in range(7):
                    temp = (temp >> abs_dir) & mask
                    if temp & opponent_board:
                        line_flipped |= temp
                    elif temp & player_board:
                        flipped_pieces |= line_flipped
                        break
                    else:
                        break

        new_player_board |= flipped_pieces
        new_opponent_board &= ~flipped_pieces
        
        return new_player_board, new_opponent_board

    def __str__(self):
        """Returns a string representation of the board for printing."""
        board_str = "  a b c d e f g h\n"
        for r in range(8):
            board_str += str(r + 1) + " "
            for c in range(8):
                pos = r * 8 + c
                mask = np.uint64(1) << pos
                if self.black_board & mask:
                    board_str += "● " # Black
                elif self.white_board & mask:
                    board_str += "○ " # White
                else:
                    board_str += "· "
            board_str += "\n"
        return board_str

# =============================================================================
#  修复的Local Judge Logic
# =============================================================================

def call_bot_simple_interface(bot_script_path, turn_num, history):
    """
    使用简化交互格式调用bot
    """
    # 确保使用正确的路径
    judge_dir = os.path.dirname(os.path.abspath(__file__))
    bot_path = os.path.join(judge_dir, '..', 'botzone', 'test_main.py')
    bot_path = os.path.normpath(bot_path)

    if not os.path.exists(bot_path):
        print(f"Error: Bot script not found at {bot_path}")
        return None

    bot_cwd = os.path.dirname(bot_path)

    process = subprocess.Popen(
        ['python3', 'test_main.py'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=bot_cwd
    )
    
    # 构建简化交互格式的输入
    input_lines = [str(turn_num)]
    input_lines.extend(history)
    input_lines.append("")  # data
    input_lines.append("")  # globaldata
    
    input_text = '\n'.join(input_lines)
    
    stdout, stderr = process.communicate(input_text)
    
    if process.returncode != 0:
        print("--- Bot STDERR ---")
        print(stderr)
        return None
        
    if stderr:
        print("--- Bot Debug Info ---")
        print(stderr)
    
    # 解析bot输出
    output_lines = stdout.strip().split('\n')
    if len(output_lines) >= 1:
        try:
            x, y = map(int, output_lines[0].split())
            debug_info = output_lines[1] if len(output_lines) > 1 else ""
            return x, y, debug_info
        except:
            print(f"Error parsing bot output: {output_lines[0]}")
            return None
    
    return None

def main():
    """
    修复的主游戏循环
    """
    board = Board()
    game_history = []  # 存储所有的移动历史，按时间顺序
    
    print("🎮 本地黑白棋对局开始")
    print("Bot vs Bot (自对弈)")
    print("=" * 50)
    
    for turn_num in range(1, 61):  # 最多60回合
        print(f"\n📍 第 {turn_num} 回合")
        
        # 确定当前玩家
        is_black_turn = (turn_num % 2 == 1)
        
        if is_black_turn:
            print("当前玩家: 黑方 (●)")
            player_board, opponent_board = board.black_board, board.white_board
        else:
            print("当前玩家: 白方 (○)")
            player_board, opponent_board = board.white_board, board.black_board

        print("当前棋盘:")
        print(board)

        # 检查合法移动
        legal_moves = board.get_legal_moves(player_board, opponent_board)
        
        if legal_moves == 0:
            print("⚠️  当前玩家无合法移动，必须PASS")
            
            # 检查对手是否也无法移动（游戏结束）
            other_legal_moves = board.get_legal_moves(opponent_board, player_board)
            if other_legal_moves == 0:
                print("🏁 双方都无法移动，游戏结束!")
                break

            # 添加PASS移动到历史
            game_history.append("-1 -1")
            continue

        # 显示合法移动
        legal_count = bin(legal_moves).count('1')
        print(f"合法移动数量: {legal_count}")

        # 构建bot的历史记录（按botzone格式）
        bot_history = []
        if turn_num == 1:
            # 第一回合：黑方收到-1 -1表示自己先走
            bot_history.append("-1 -1")
        else:
            # 从第二回合开始，按正确顺序构建历史
            for i in range(len(game_history)):
                bot_history.append(game_history[i])

        # 调用bot获取移动
        result = call_bot_simple_interface("test_main.py", turn_num, bot_history)
        
        if result is None:
            print("❌ Bot调用失败，游戏结束")
            break
            
        x, y, debug_info = result
        print(f"🤖 Bot选择: ({x}, {y})")
        if debug_info:
            print(f"📊 Debug: {debug_info}")
        
        # 验证移动
        if x == -1 and y == -1:
            print("❌ Bot选择PASS，但有合法移动，无效!")
            break
        
        if not (0 <= x < 8 and 0 <= y < 8):
            print(f"❌ 移动超出边界: ({x}, {y})")
            break
            
        move_pos = y * 8 + x
        move_mask = np.uint64(1) << move_pos
        
        if not (legal_moves & move_mask):
            print(f"❌ 非法移动: ({x}, {y})")
            break
        
        # 应用移动
        new_p, new_o = board.make_move(player_board, opponent_board, move_pos)
        if is_black_turn:
            board.black_board, board.white_board = new_p, new_o
        else:
            board.white_board, board.black_board = new_p, new_o

        # 更新历史
        game_history.append(f"{x} {y}")
        
        # 显示移动后状态
        black_count = bin(board.black_board).count('1')
        white_count = bin(board.white_board).count('1')
        print(f"📈 当前比分: 黑方 {black_count} - {white_count} 白方")

    # 游戏结束，计算最终比分
    print("\n" + "=" * 50)
    print("🏁 游戏结束!")
    print("最终棋盘:")
    print(board)
    
    black_score = bin(board.black_board).count('1')
    white_score = bin(board.white_board).count('1')
    
    print(f"🏆 最终比分: 黑方 {black_score} - {white_score} 白方")
    
    if black_score > white_score:
        print("🎉 黑方获胜!")
    elif white_score > black_score:
        print("🎉 白方获胜!")
    else:
        print("🤝 平局!")

if __name__ == "__main__":
    main()