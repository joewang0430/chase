"""
棋盘编码验证器 - 对比视觉棋盘和位板编码
"""

"""
棋盘编码验证器 - 对比视觉棋盘和位板编码
"""

def analyze_visual_board(visual_board, board_name=""):
    """分析视觉棋盘并生成正确的位置编码"""
    
    lines = visual_board.strip().split('\n')[1:]  # 跳过列标题
    
    black_positions = []
    white_positions = []
    empty_positions = []
    
    print(f"=== {board_name} 棋盘位置分析 ===")
    print("位置编码: row*8 + col")
    print()
    
    for i, line in enumerate(lines):
        if line.strip().startswith(('1', '2', '3', '4', '5', '6', '7', '8')):
            row = i
            row_content = line.split()[1:]  # 跳过行号
            
            print(f"第{row+1}行: ", end="")
            for col, cell in enumerate(row_content):
                pos = row * 8 + col
                pos_name = f"{chr(ord('a') + col)}{row + 1}"
                
                if cell == '●':
                    black_positions.append(pos)
                    print(f"{pos_name}(B{pos}) ", end="")
                elif cell == '○':
                    white_positions.append(pos)
                    print(f"{pos_name}(W{pos}) ", end="")
                elif cell == '·':
                    empty_positions.append(pos)
                    print(f"{pos_name}(·{pos}) ", end="")
            print()
    
    print(f"\n黑子位置: {sorted(black_positions)}")
    print(f"白子位置: {sorted(white_positions)}")
    print(f"空位位置: {sorted(empty_positions)}")
    
    print(f"\n黑子数量: {len(black_positions)}")
    print(f"白子数量: {len(white_positions)}")
    print(f"空位数量: {len(empty_positions)}")
    
    return black_positions, white_positions, empty_positions

def process_all_test_boards():
    """处理所有测试棋盘"""
    
    test_boards = {
        "ajexsp_48": """   a b c d e f g h
 1 ○ ○ ○ ○ ○ ○ ○ ○
 2 ● ● ● ○ ● ● · ○
 3 ○ ● ○ ● ● ● ○ ●
 4 ○ ○ ● ● ● ○ ● ●
 5 ○ ○ ○ ● ○ ● ○ ●
 6 · ○ ○ ○ ○ ○ ● ●
 7 · ○ ○ ○ · · · ●
 8 ○ · ○ · · · · ·""",

        "ajexsp_49": """   a b c d e f g h
 1 ○ ○ ○ ○ ○ ○ ○ ○
 2 ● ● ● ○ ● ● ● ○
 3 ○ ● ○ ● ● ● ● ●
 4 ○ ○ ● ● ● ○ ● ●
 5 ○ ○ ○ ● ○ ● ○ ●
 6 · ○ ○ ○ ○ ○ ● ●
 7 · ○ ○ ○ · · · ●
 8 ○ · ○ · · · · ·""",

        "ajexsp_50": """   a b c d e f g h
 1 ○ ○ ○ ○ ○ ○ ○ ○
 2 ● ● ● ○ ● ● ● ○
 3 ○ ● ○ ● ● ● ● ○
 4 ○ ○ ● ● ● ○ ● ○
 5 ○ ○ ○ ● ○ ● ○ ○
 6 · ○ ○ ○ ○ ○ ● ○
 7 · ○ ○ ○ · · · ○
 8 ○ · ○ · · · · ○""",

        "ajexsp_52": """   a b c d e f g h
 1 ○ ○ ○ ○ ○ ○ ○ ○
 2 ● ● ● ○ ● ● ● ○
 3 ○ ● ○ ● ● ● ● ○
 4 ○ ○ ● ● ● ○ ● ○
 5 ○ ○ ○ ● ● ● ○ ○
 6 · ○ ○ ○ ● ○ ● ○
 7 · ○ ○ ○ ○ · · ○
 8 ○ · ○ ○ · · · ○""",

        "ajexsp_53": """   a b c d e f g h
 1 ○ ○ ○ ○ ○ ○ ○ ○
 2 ● ● ● ○ ● ● ● ○
 3 ○ ● ○ ● ● ● ● ○
 4 ○ ○ ● ● ● ○ ● ○
 5 ○ ○ ○ ● ● ● ○ ○
 6 · ○ ○ ○ ● ○ ● ○
 7 · ○ ○ ○ ● · · ○
 8 ○ · ○ ○ ● · · ○""",

        "ajexsp_54": """   a b c d e f g h
 1 ○ ○ ○ ○ ○ ○ ○ ○
 2 ● ● ● ○ ● ● ● ○
 3 ○ ● ○ ● ● ● ● ○
 4 ○ ○ ● ● ● ○ ● ○
 5 ○ ○ ○ ● ● ● ○ ○
 6 · ○ ○ ○ ● ○ ○ ○
 7 · ○ ○ ○ ○ ○ · ○
 8 ○ · ○ ○ ● · · ○""",

        "ajexsp_56": """   a b c d e f g h
 1 ○ ○ ○ ○ ○ ○ ○ ○
 2 ● ● ● ○ ● ● ● ○
 3 ○ ● ○ ● ● ● ● ○
 4 ○ ○ ● ● ● ○ ● ○
 5 ○ ○ ○ ● ● ● ○ ○
 6 · ○ ○ ○ ● ○ ○ ○
 7 · ○ ○ ○ ○ ○ · ○
 8 ○ · ○ ○ ○ ○ ○ ○""",

        "ajexsp_57": """   a b c d e f g h
 1 ○ ○ ○ ○ ○ ○ ○ ○
 2 ● ● ● ○ ● ● ● ○
 3 ○ ● ○ ● ● ● ● ○
 4 ○ ○ ● ● ● ○ ● ○
 5 ○ ○ ● ● ● ● ○ ○
 6 · ● ○ ○ ● ○ ○ ○
 7 ● ○ ○ ○ ○ ○ · ○
 8 ○ · ○ ○ ○ ○ ○ ○""",

        "ajexsp_58": """   a b c d e f g h
 1 ○ ○ ○ ○ ○ ○ ○ ○
 2 ● ● ● ○ ● ● ● ○
 3 ○ ● ○ ● ● ● ● ○
 4 ○ ○ ● ● ● ○ ● ○
 5 ○ ○ ● ● ● ● ○ ○
 6 ○ ○ ○ ○ ● ○ ○ ○
 7 ○ ○ ○ ○ ○ ○ · ○
 8 ○ · ○ ○ ○ ○ ○ ○"""
    }
    
    all_board_data = {}
    
    for board_name, visual_board in test_boards.items():
        print("\n" + "="*60)
        black_pos, white_pos, empty_pos = analyze_visual_board(visual_board, board_name)
        all_board_data[board_name] = {
            'black': black_pos,
            'white': white_pos,
            'empty': empty_pos
        }
        print("\n")
    
    # 生成代码格式的输出
    print("\n" + "="*60)
    print("=== 生成的Python代码 ===")
    print("\nTEST_BOARDS = {")
    
    for board_name, data in all_board_data.items():
        print(f'    "{board_name}": {{')
        print(f'        "black": {data["black"]},')
        print(f'        "white": {data["white"]},')
        print(f'        "empty": {data["empty"]}')
        print(f'    }},')
    
    print("}")
    
    return all_board_data

if __name__ == "__main__":
    board_data = process_all_test_boards()
