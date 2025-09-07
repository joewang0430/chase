import numpy as np
import time

# 复制qszy的关键代码进行对比
move_dirs = [
    (~np.uint64(0x0101010101010101), np.uint64(1), np.uint64.__rshift__),
    (~np.uint64(0x80808080808080FF), np.uint64(7), np.uint64.__rshift__),
    (~np.uint64(0x00000000000000FF), np.uint64(8), np.uint64.__rshift__),
    (~np.uint64(0x01010101010101FF), np.uint64(9), np.uint64.__rshift__),
    (~np.uint64(0x8080808080808080), np.uint64(1), np.uint64.__lshift__),
    (~np.uint64(0xFF01010101010101), np.uint64(7), np.uint64.__lshift__),
    (~np.uint64(0xFF00000000000000), np.uint64(8), np.uint64.__lshift__),
    (~np.uint64(0xFF80808080808080), np.uint64(9), np.uint64.__lshift__),
]

def qszy_valid_move_mask(b, w):
    avail = np.uint64(0)
    for mask, move, op in move_dirs:
        alive = b
        while alive:
            alive = op((alive & mask), move) & w
            avail |= op((alive & mask), move) & ~w & ~b
    return avail

# 导入你的Board类
import sys
sys.path.append('/Users/juanjuan1/Desktop/chase/botzone')
from new_main import Board

def benchmark_vs_qszy():
    board = Board()
    
    # 测试10个有代表性的棋盘状态
    test_cases = [
        # 1. 标准开局
        (np.uint64(0x0000000810000000), np.uint64(0x0000001008000000)),
        
        # 2. 开局后几步 - 稀疏棋盘
        (np.uint64(0x0000001014000000), np.uint64(0x0000000808000000)),
        
        # 3. 早期中局 - 边角开始争夺
        (np.uint64(0x0000081834000000), np.uint64(0x0000104008000000)),
        
        # 4. 中局 - 复杂传播路径
        (np.uint64(0x0102041820408000), np.uint64(0x0081020408102000)),
        
        # 5. 中局密集 - 大量翻转可能
        (np.uint64(0x7e7e7e0000000000), np.uint64(0x00000000ff818181)),
        
        # 6. 边缘控制局面
        (np.uint64(0xff00000000000000), np.uint64(0x00ffffff00000000)),
        
        # 7. 中后期 - 空位稀少
        (np.uint64(0x3f3f3f3f3f3f3f3f), np.uint64(0xc0c0c0c0c0c0c0c0)),
        
        # 8. 角落争夺激烈
        (np.uint64(0x8142241818244281), np.uint64(0x7ebd5be7e75bbd7e)),
        
        # 9. 后期 - 高密度棋盘
        (np.uint64(0x1f1f1f1f1f1f1f1f), np.uint64(0xe0e0e0e0e0e0e0e0)),
        
        # 10. 接近终局 - 极少空位
        (np.uint64(0xffffff0000ffffff), np.uint64(0x000000ffffff0000))
    ]
    
    case_names = [
        "标准开局", "开局几步后", "早期中局", "中局复杂", "中局密集",
        "边缘控制", "中后期", "角落争夺", "后期高密度", "接近终局"
    ]
    
    total_time_ours = 0
    total_time_qszy = 0
    our_wins = 0
    qszy_wins = 0
    
    for i, (player, opponent) in enumerate(test_cases):
        print(f"\n=== 测试案例 {i+1}: {case_names[i]} ===")
        
        # 显示棋盘信息
        total_pieces = bin(player | opponent).count('1')
        legal_moves_count = bin(board.get_legal_moves(player, opponent)).count('1')
        print(f"棋子总数: {total_pieces}, 合法移动数: {legal_moves_count}")
        
        # 验证结果一致性
        result_ours = board.get_legal_moves(player, opponent)
        result_qszy = qszy_valid_move_mask(player, opponent)
        consistency = result_ours == result_qszy
        print(f"结果一致: {consistency}")
        
        if not consistency:
            print(f"  我们的结果: {result_ours:016x}")
            print(f"  qszy结果:   {result_qszy:016x}")
        
        # 性能测试
        iterations = 10000
        
        start = time.time()
        for _ in range(iterations):
            board.get_legal_moves(player, opponent)
        time_ours = time.time() - start
        
        start = time.time() 
        for _ in range(iterations):
            qszy_valid_move_mask(player, opponent)
        time_qszy = time.time() - start
        
        ratio = time_ours / time_qszy
        
        print(f"我们的版本: {time_ours:.4f}秒")
        print(f"qszy版本:   {time_qszy:.4f}秒")
        print(f"性能比:     {ratio:.3f}x", end="")
        
        if ratio < 1.0:
            print(f" (我们快 {(1/ratio-1)*100:.1f}%)")
            our_wins += 1
        else:
            print(f" (qszy快 {(ratio-1)*100:.1f}%)")
            qszy_wins += 1
            
        total_time_ours += time_ours
        total_time_qszy += time_qszy
    
    # 总结
    print(f"\n{'='*50}")
    print(f"总体性能分析:")
    print(f"我们获胜案例: {our_wins}/10")
    print(f"qszy获胜案例: {qszy_wins}/10")
    print(f"总体时间比: {total_time_ours/total_time_qszy:.3f}x")
    
    if total_time_ours < total_time_qszy:
        print(f"🎉 总体而言，我们的算法快了 {(total_time_qszy/total_time_ours-1)*100:.1f}%")
    else:
        print(f"😔 总体而言，qszy算法快了 {(total_time_ours/total_time_qszy-1)*100:.1f}%")
    
    print(f"{'='*50}")

if __name__ == "__main__":
    benchmark_vs_qszy()