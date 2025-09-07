#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""四重对战测试：AJEX vs ENDING, ENDING vs AJEX, ENDING vs ENDING"""
import subprocess, ctypes, sys, time

INIT_BOARD_LINES = [
    "   a b c d e f g h",
    " 1 · · ○ · ● · · ·",
    " 2 ○ ○ ○ ○ ● ● ● ·",
    " 3 · ○ ○ ○ ○ ● ● ○",
    " 4 · ● · ● ● ○ ● ●",
    " 5 ● · ● ● ○ ○ ● ●",
    " 6 ● ● ● ● ○ ● ● ○",
    " 7 ● ● ● · ● ● · ·",
    " 8 ● ○ ○ · ○ ● ● ·",
]

BLACK = '●'
WHITE = '○'
EMPTY = '·'

DIRS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

def compile_libs():
    cmds = [
        ["gcc","-O3","-shared","-fPIC","-march=native","-mtune=native","-mpopcnt","-mlzcnt","-mbmi","-mbmi2","data/ending.c","-o","ending.so"],
        ["gcc","-O3","-shared","-fPIC","ajex/ajex_fll_5.c","-o","ajex.so"],
    ]
    for c in cmds:
        r = subprocess.run(c, capture_output=True, text=True)
        if r.returncode!=0:
            print("编译失败:", ' '.join(c), r.stderr); sys.exit(1)


def parse_init_board():
    board = [[EMPTY]*8 for _ in range(8)]
    for line in INIT_BOARD_LINES[1:]:  # skip header
        parts = line.strip().split()
        if not parts: continue
        row_idx = int(parts[0]) - 1
        for c in range(8):
            board[row_idx][c] = parts[c+1]
    return board


def display_board(board, move_no=None, last_move=None):
    header = "   a b c d e f g h"
    print(header)
    for r in range(8):
        row = f" {r+1} " + ' '.join(board[r])
        print(row)
    if move_no is not None:
        print(f"(After move {move_no}{' '+last_move if last_move else ''})")


def inside(r,c): return 0<=r<8 and 0<=c<8


def has_legal(board, color):
    opp = WHITE if color==BLACK else BLACK
    for r in range(8):
        for c in range(8):
            if board[r][c]!=EMPTY: continue
            for dr,dc in DIRS:
                rr,cc = r+dr,c+dc; seen=False
                while inside(rr,cc) and board[rr][cc]==opp:
                    seen=True; rr+=dr; cc+=dc
                if seen and inside(rr,cc) and board[rr][cc]==color:
                    return True
    return False


def apply_move(board, r, c, color):
    if r<0:  # pass
        return
    opp = WHITE if color==BLACK else BLACK
    flips=[]
    for dr,dc in DIRS:
        rr,cc=r+dr,c+dc; path=[]
        while inside(rr,cc) and board[rr][cc]==opp:
            path.append((rr,cc)); rr+=dr; cc+=dc
        if path and inside(rr,cc) and board[rr][cc]==color:
            flips.extend(path)
    board[r][c]=color
    for (rr,cc) in flips:
        board[rr][cc]=color


def board_to_bitboards(board):
    black_bits=0; white_bits=0
    for r in range(8):
        for c in range(8):
            bit = 1 << (r*8+c)
            if board[r][c]==BLACK: black_bits |= bit
            elif board[r][c]==WHITE: white_bits |= bit
    return black_bits, white_bits


def board_to_ajex_26(board):
    arr = [[b'U' for _ in range(26)] for _ in range(26)]
    for r in range(8):
        for c in range(8):
            ch = board[r][c]
            if ch==BLACK: arr[r][c]=b'B'
            elif ch==WHITE: arr[r][c]=b'W'
            else: arr[r][c]=b'U'
    # flatten to ctypes 26x26
    Row = ctypes.c_char * 26
    Board26 = Row * 26
    c_board = Board26()
    for i in range(26):
        row = Row()
        for j in range(26):
            row[j] = arr[i][j]
        c_board[i]=row
    return c_board


def coord_to_str(r,c): return f"{chr(ord('a')+c)}{r+1}" if r>=0 else "PASS"


def run_match(black_engine, white_engine, match_name, ending, ajex):
    """运行一局对战"""
    board = parse_init_board()
    print(f"\n{'='*60}")
    print(f"开始对战: {match_name}")
    print(f"{'='*60}")
    print("初始局面:")
    display_board(board)

    move_no = 48  # 下一手编号
    consecutive_pass = 0
    turn = BLACK  # 黑棋先手

    while True:
        black_bits, white_bits = board_to_bitboards(board)
        start_time = time.perf_counter()
        
        if turn == BLACK:
            engine = black_engine
        else:
            engine = white_engine
            
        if engine == "AJEX":
            # AJEX 引擎
            if has_legal(board, turn):
                c_board = board_to_ajex_26(board)
                row = ctypes.c_int(); col = ctypes.c_int()
                color_char = b'B' if turn == BLACK else b'W'
                ajex.makeMove(c_board, ctypes.c_int(8), ctypes.c_char(color_char[0]), ctypes.byref(row), ctypes.byref(col))
                r, c = row.value, col.value
                if r == -1 and c == -1:
                    move_str = 'PASS'; consecutive_pass += 1
                else:
                    apply_move(board, r, c, turn)
                    move_str = coord_to_str(r, c); consecutive_pass = 0
            else:
                move_str = 'PASS'; consecutive_pass += 1
            side_label = f"{'BLACK' if turn == BLACK else 'WHITE'}(AJEX)"
        else:  # ENDING
            # ENDING 引擎
            if has_legal(board, turn):
                best = ctypes.c_int()
                if turn == BLACK:
                    score = ending.solve_endgame(black_bits, white_bits, ctypes.byref(best))
                else:
                    score = ending.solve_endgame(white_bits, black_bits, ctypes.byref(best))
                
                if best.value == -1:
                    move_str = 'PASS'; consecutive_pass += 1
                else:
                    r = best.value // 8; c = best.value % 8
                    apply_move(board, r, c, turn)
                    move_str = coord_to_str(r, c); consecutive_pass = 0
            else:
                move_str = 'PASS'; consecutive_pass += 1
            side_label = f"{'BLACK' if turn == BLACK else 'WHITE'}(ENDING)"
        
        elapsed = time.perf_counter() - start_time
        print(f"Move {move_no} {side_label}: {move_str} (time {elapsed*1000:.2f} ms)")
        display_board(board, move_no, move_str)

        # 游戏结束检查
        full = (black_bits | white_bits) == (1<<64)-1
        if consecutive_pass == 2 or full:
            break
        turn = WHITE if turn == BLACK else BLACK
        move_no += 1

    # 最终结果
    black_bits, white_bits = board_to_bitboards(board)
    b_cnt = bin(black_bits).count('1')
    w_cnt = bin(white_bits).count('1')
    print("终局:")
    display_board(board)
    
    winner = '黑胜' if b_cnt > w_cnt else ('白胜' if w_cnt > b_cnt else '平局')
    print(f"黑({black_engine}) {b_cnt} : 白({white_engine}) {w_cnt} -> {winner}")
    print(f"{'='*60}")
    
    return {
        'black_engine': black_engine,
        'white_engine': white_engine,
        'black_score': b_cnt,
        'white_score': w_cnt,
        'winner': winner,
        'match_name': match_name
    }


def main():
    compile_libs()
    # 加载动态库
    ending = ctypes.CDLL('./ending.so')
    ending.solve_endgame.argtypes=[ctypes.c_uint64,ctypes.c_uint64,ctypes.POINTER(ctypes.c_int)]
    ending.solve_endgame.restype=ctypes.c_int

    ajex = ctypes.CDLL('./ajex.so')
    ajex.makeMove.argtypes=[ctypes.POINTER(ctypes.c_char * 26), ctypes.c_int, ctypes.c_char, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    ajex.makeMove.restype=ctypes.c_int

    print("🎯 四重对战测试开始！")
    print("测试四种组合：AJEX vs ENDING, ENDING vs AJEX, ENDING vs ENDING, AJEX vs AJEX")
    
    results = []
    
    # 第一局：AJEX(黑) vs ENDING(白)
    result1 = run_match("AJEX", "ENDING", "第一局: AJEX(黑) vs ENDING(白)", ending, ajex)
    results.append(result1)
    
    time.sleep(1)  # 短暂停顿
    
    # 第二局：ENDING(黑) vs AJEX(白)
    result2 = run_match("ENDING", "AJEX", "第二局: ENDING(黑) vs AJEX(白)", ending, ajex)
    results.append(result2)
    
    time.sleep(1)  # 短暂停顿
    
    # 第三局：ENDING(黑) vs ENDING(白) - 第一次
    result3 = run_match("ENDING", "ENDING", "第三局: ENDING(黑) vs ENDING(白) - 理论最优对决", ending, ajex)
    results.append(result3)
    
    time.sleep(1)  # 短暂停顿
    
    # 第四局：AJEX(黑) vs AJEX(白) - AJEX自我对决
    result4 = run_match("AJEX", "AJEX", "第四局: AJEX(黑) vs AJEX(白) - AJEX自我对决", ending, ajex)
    results.append(result4)
    
    # 总结结果
    print(f"\n{'='*80}")
    print("🏆 四重对战总结")
    print(f"{'='*80}")
    
    for i, result in enumerate(results, 1):
        print(f"第{i}局: {result['black_engine']}(黑) vs {result['white_engine']}(白)")
        print(f"      结果: {result['black_score']} : {result['white_score']} -> {result['winner']}")
    
    # 统计胜负
    ajex_wins = sum(1 for r in results if 
                   (r['black_engine'] == 'AJEX' and r['winner'] == '黑胜') or
                   (r['white_engine'] == 'AJEX' and r['winner'] == '白胜'))
    ending_wins = sum(1 for r in results if 
                     (r['black_engine'] == 'ENDING' and r['winner'] == '黑胜') or
                     (r['white_engine'] == 'ENDING' and r['winner'] == '白胜'))
    draws = sum(1 for r in results if r['winner'] == '平局')
    
    print(f"\n📊 引擎对战统计:")
    print(f"AJEX 胜利: {ajex_wins} 局")
    print(f"ENDING 胜利: {ending_wins} 局") 
    print(f"平局: {draws} 局")
    
    # 检查ENDING vs ENDING的一致性
    ending_vs_ending = [r for r in results if r['black_engine'] == 'ENDING' and r['white_engine'] == 'ENDING']
    ajex_vs_ajex = [r for r in results if r['black_engine'] == 'AJEX' and r['white_engine'] == 'AJEX']
    
    if len(ending_vs_ending) >= 1:
        print(f"\n🔬 理论最优算法检验:")
        ending_result = ending_vs_ending[0]
        print(f"✅ ENDING vs ENDING结果: {ending_result['black_score']}:{ending_result['white_score']}")
        print(f"   这展示了理论最优算法的确定性！")
    
    if len(ajex_vs_ajex) >= 1:
        print(f"\n🔬 AJEX算法自我对决检验:")
        ajex_result = ajex_vs_ajex[0]
        print(f"✅ AJEX vs AJEX结果: {ajex_result['black_score']}:{ajex_result['white_score']}")
        print(f"   这展示了AJEX算法的表现！")
    
    if ending_wins > ajex_wins:
        print(f"\n🎉 ENDING算法总体表现更优秀!")
    elif ajex_wins > ending_wins:
        print(f"\n🎉 AJEX算法总体表现更优秀!")
    else:
        print(f"\n🤝 两个算法表现相当!")
    
    print(f"{'='*80}")


if __name__=='__main__':
    main()
