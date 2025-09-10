// 终极编译器优化指令
#pragma GCC optimize("O3,unroll-loops,inline-functions")
#pragma GCC target("popcnt,lzcnt,bmi,bmi2")

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <limits.h>
#include <stdlib.h>

// 使用安全且稳定的哨兵值 
#define PLAY_TO_END_SENTINEL (INT_MIN/2)

typedef uint64_t u64;

typedef struct { u64 board[2]; } Board; // board[0]=当前方, board[1]=对方

typedef struct {
    int pos;
    int score;
} ScoredMove;

#define MY_PIECES(b)    ((b).board[0])
#define OPP_PIECES(b)   ((b).board[1])
#define ALL_PIECES(b)   (MY_PIECES(b) | OPP_PIECES(b))
#define EMPTY_SQUARES(b) (~ALL_PIECES(b))
#define COUNT_BITS(x)   __builtin_popcountll(x)
#define MY_COUNT(b)     COUNT_BITS(MY_PIECES(b))
#define OPP_COUNT(b)    COUNT_BITS(OPP_PIECES(b))
#define SWAP_BOARD(b) do { u64 t=(b).board[0]; (b).board[0]=(b).board[1]; (b).board[1]=t; } while(0)

// 基础函数
u64 generate_moves(Board board);
Board make_move(Board board, int pos);
int solve_endgame(uint64_t my_pieces, uint64_t opp_pieces, int* best_move);
static inline int final_score(Board b);

// 排序移动相关函数
static int evaluate_move(Board board, int pos);
static int compare_moves(const void *a, const void *b);
static int get_sorted_moves(Board board, u64 moves, ScoredMove *sorted_moves);

// pvs优化后
static int perfect_search_pvs(Board board, int empties, int alpha, int beta, bool is_pv);

// 结束后棋盘打分
static inline int final_score(Board b){ 
    return (int)MY_COUNT(b) - (int)OPP_COUNT(b); 
} 

// 完全枚举搜索 (Negamax + AlphaBeta)，同时pvs优化
static int perfect_search_pvs(Board board, int empties, int alpha, int beta, bool is_pv){
    u64 moves = generate_moves(board);
    if (!moves){
        Board opp = board; SWAP_BOARD(opp);
        u64 opp_moves = generate_moves(opp);
        if (!opp_moves){ // 双方都无棋 -> 终局
            return final_score(board);
        }
        // 单方 PASS，不减少剩余空位(棋盘未变化)，换边取负
        int val = -perfect_search_pvs(opp, empties, -beta, -alpha, is_pv);
        return val;
    }
    if (empties == 0){ // 理论上不应发生(有合法着说明仍有空格) 保险返回
        return final_score(board);
    }
    
    int best = INT_MIN;
    ScoredMove sorted_moves[32];
    int move_count = get_sorted_moves(board, moves, sorted_moves);
    
    bool first_move = true;
    for (int i = 0; i < move_count; i++) {
        int pos = sorted_moves[i].pos;
        Board nb = make_move(board, pos);
        SWAP_BOARD(nb);
        
        int val;
        if (first_move) {
            // 第一个移动：用全窗口搜索（这很可能是最佳移动）
            val = -perfect_search_pvs(nb, empties - 1, -beta, -alpha, true);
            first_move = false;
        } else {
            // 后续移动：先用null window试探
            val = -perfect_search_pvs(nb, empties - 1, -alpha - 1, -alpha, false);
            
            // 如果试探发现这个移动很好，重新用全窗口搜索
            if (val > alpha && val < beta && is_pv) {
                val = -perfect_search_pvs(nb, empties - 1, -beta, -alpha, true);
            }
        }
        if (val > best) best = val;
        if (val > alpha) alpha = val;
        if (alpha >= beta) break; // 剪枝
    }
    return best;
}

int solve_endgame(uint64_t my_pieces, uint64_t opp_pieces, int* best_move){
    Board board = {{my_pieces, opp_pieces}};
    int empties = 64 - COUNT_BITS(my_pieces | opp_pieces);

    u64 moves = generate_moves(board);
    if (!moves){
        Board opp = board; SWAP_BOARD(opp);
        u64 opp_moves = generate_moves(opp);
        if (!opp_moves){ *best_move = -1; return final_score(board); }
        int score = -perfect_search_pvs(opp, empties, -INT_MAX, INT_MAX, true);
        *best_move = -1; return score;
    }

    int alpha = -INT_MAX, beta = INT_MAX;
    int best_score = INT_MIN; int best_pos = -1;
    u64 tmp = moves;
    // 使用排序后的移动
    ScoredMove sorted_moves[32];
    int move_count = get_sorted_moves(board, moves, sorted_moves);

    for (int i = 0; i < move_count; i++) {
        int pos = sorted_moves[i].pos;
        Board nb = make_move(board, pos);
        SWAP_BOARD(nb);
        int score = -perfect_search_pvs(nb, empties - 1, -beta, -alpha, true); // 使用PVS
        if (score > best_score){ best_score = score; best_pos = pos; }
        if (score > alpha) alpha = score;
        if (alpha >= beta) break;
    }
    *best_move = best_pos;
    return best_score;
}

u64 generate_moves(Board board) {
    u64 my = MY_PIECES(board);
    u64 opp = OPP_PIECES(board);
    u64 opp_inner = opp & 0x7E7E7E7E7E7E7E7EULL; // 排除边界，用于水平/对角移动
    u64 moves = 0;
    u64 b_flip, b_opp_adj;
    
    // 水平向右 (>>1)
    b_flip = (my >> 1) & opp_inner;
    b_flip |= (b_flip >> 1) & opp_inner;
    b_opp_adj = opp_inner & (opp_inner >> 1);
    b_flip |= (b_flip >> 2) & b_opp_adj;
    b_flip |= (b_flip >> 2) & b_opp_adj;
    moves |= (b_flip >> 1) & ~(my | opp);
    
    // 水平向左 (<<1)
    b_flip = (my << 1) & opp_inner;
    b_flip |= (b_flip << 1) & opp_inner;
    b_opp_adj = opp_inner & (opp_inner << 1);
    b_flip |= (b_flip << 2) & b_opp_adj;
    b_flip |= (b_flip << 2) & b_opp_adj;
    moves |= (b_flip << 1) & ~(my | opp);
    
    // 垂直向下 (>>8)
    b_flip = (my >> 8) & opp;
    b_flip |= (b_flip >> 8) & opp;
    b_opp_adj = opp & (opp >> 8);
    b_flip |= (b_flip >> 16) & b_opp_adj;
    b_flip |= (b_flip >> 16) & b_opp_adj;
    moves |= (b_flip >> 8) & ~(my | opp);
    
    // 垂直向上 (<<8)
    b_flip = (my << 8) & opp;
    b_flip |= (b_flip << 8) & opp;
    b_opp_adj = opp & (opp << 8);
    b_flip |= (b_flip << 16) & b_opp_adj;
    b_flip |= (b_flip << 16) & b_opp_adj;
    moves |= (b_flip << 8) & ~(my | opp);
    
    // 对角线右下 (>>7)
    b_flip = (my >> 7) & opp_inner;
    b_flip |= (b_flip >> 7) & opp_inner;
    b_opp_adj = opp_inner & (opp_inner >> 7);
    b_flip |= (b_flip >> 14) & b_opp_adj;
    b_flip |= (b_flip >> 14) & b_opp_adj;
    moves |= (b_flip >> 7) & ~(my | opp);
    
    // 对角线左上 (<<7)
    b_flip = (my << 7) & opp_inner;
    b_flip |= (b_flip << 7) & opp_inner;
    b_opp_adj = opp_inner & (opp_inner << 7);
    b_flip |= (b_flip << 14) & b_opp_adj;
    b_flip |= (b_flip << 14) & b_opp_adj;
    moves |= (b_flip << 7) & ~(my | opp);
    
    // 对角线右上 (>>9)
    b_flip = (my >> 9) & opp_inner;
    b_flip |= (b_flip >> 9) & opp_inner;
    b_opp_adj = opp_inner & (opp_inner >> 9);
    b_flip |= (b_flip >> 18) & b_opp_adj;
    b_flip |= (b_flip >> 18) & b_opp_adj;
    moves |= (b_flip >> 9) & ~(my | opp);
    
    // 对角线左下 (<<9)
    b_flip = (my << 9) & opp_inner;
    b_flip |= (b_flip << 9) & opp_inner;
    b_opp_adj = opp_inner & (opp_inner << 9);
    b_flip |= (b_flip << 18) & b_opp_adj;
    b_flip |= (b_flip << 18) & b_opp_adj;
    moves |= (b_flip << 9) & ~(my | opp);
    
    return moves;
}

Board make_move(Board board, int pos) {
    u64 my = MY_PIECES(board);
    u64 opp = OPP_PIECES(board);
    u64 opp_inner = opp & 0x7E7E7E7E7E7E7E7EULL;
    u64 pos_bit = 1ULL << pos;
    u64 flipped = 0;
    u64 b_flip, b_opp_adj;
    
    // 水平向右 (>>1)
    b_flip = (pos_bit >> 1) & opp_inner;
    b_flip |= (b_flip >> 1) & opp_inner;
    b_opp_adj = opp_inner & (opp_inner >> 1);
    b_flip |= (b_flip >> 2) & b_opp_adj;
    b_flip |= (b_flip >> 2) & b_opp_adj;
    if ((b_flip >> 1) & my) flipped |= b_flip;
    
    // 水平向左 (<<1)
    b_flip = (pos_bit << 1) & opp_inner;
    b_flip |= (b_flip << 1) & opp_inner;
    b_opp_adj = opp_inner & (opp_inner << 1);
    b_flip |= (b_flip << 2) & b_opp_adj;
    b_flip |= (b_flip << 2) & b_opp_adj;
    if ((b_flip << 1) & my) flipped |= b_flip;
    
    // 垂直向下 (>>8)
    b_flip = (pos_bit >> 8) & opp;
    b_flip |= (b_flip >> 8) & opp;
    b_opp_adj = opp & (opp >> 8);
    b_flip |= (b_flip >> 16) & b_opp_adj;
    b_flip |= (b_flip >> 16) & b_opp_adj;
    if ((b_flip >> 8) & my) flipped |= b_flip;
    
    // 垂直向上 (<<8)
    b_flip = (pos_bit << 8) & opp;
    b_flip |= (b_flip << 8) & opp;
    b_opp_adj = opp & (opp << 8);
    b_flip |= (b_flip << 16) & b_opp_adj;
    b_flip |= (b_flip << 16) & b_opp_adj;
    if ((b_flip << 8) & my) flipped |= b_flip;
    
    // 对角线右下 (>>7)
    b_flip = (pos_bit >> 7) & opp_inner;
    b_flip |= (b_flip >> 7) & opp_inner;
    b_opp_adj = opp_inner & (opp_inner >> 7);
    b_flip |= (b_flip >> 14) & b_opp_adj;
    b_flip |= (b_flip >> 14) & b_opp_adj;
    if ((b_flip >> 7) & my) flipped |= b_flip;
    
    // 对角线左上 (<<7)
    b_flip = (pos_bit << 7) & opp_inner;
    b_flip |= (b_flip << 7) & opp_inner;
    b_opp_adj = opp_inner & (opp_inner << 7);
    b_flip |= (b_flip << 14) & b_opp_adj;
    b_flip |= (b_flip << 14) & b_opp_adj;
    if ((b_flip << 7) & my) flipped |= b_flip;
    
    // 对角线右上 (>>9)
    b_flip = (pos_bit >> 9) & opp_inner;
    b_flip |= (b_flip >> 9) & opp_inner;
    b_opp_adj = opp_inner & (opp_inner >> 9);
    b_flip |= (b_flip >> 18) & b_opp_adj;
    b_flip |= (b_flip >> 18) & b_opp_adj;
    if ((b_flip >> 9) & my) flipped |= b_flip;
    
    // 对角线左下 (<<9)
    b_flip = (pos_bit << 9) & opp_inner;
    b_flip |= (b_flip << 9) & opp_inner;
    b_opp_adj = opp_inner & (opp_inner << 9);
    b_flip |= (b_flip << 18) & b_opp_adj;
    b_flip |= (b_flip << 18) & b_opp_adj;
    if ((b_flip << 9) & my) flipped |= b_flip;
    
    Board result;
    result.board[0] = my | pos_bit | flipped;
    result.board[1] = opp & ~flipped;
    return result;
}

// 移动评估函数
static int evaluate_move(Board board, int pos) {
    // 1. 角落权重最高
    if (pos == 0 || pos == 7 || pos == 56 || pos == 63) {
        return 1000;
    }
    // 2. 边缘次优
    if (pos < 8 || pos >= 56 || (pos % 8) == 0 || (pos % 8) == 7) {
        return 100;
    }
    // 3. 翻转子数量
    u64 my = MY_PIECES(board);
    u64 opp = OPP_PIECES(board);
    Board test_board = make_move(board, pos);
    int flips = COUNT_BITS(MY_PIECES(test_board)) - COUNT_BITS(my) - 1;
    return flips * 10;
}

// 比较函数：分数高的排在前面
static int compare_moves(const void *a, const void *b) {
    const ScoredMove *ma = (const ScoredMove*)a;
    const ScoredMove *mb = (const ScoredMove*)b;
    return mb->score - ma->score;  // 降序排列
}

static int get_sorted_moves(Board board, u64 moves, ScoredMove *sorted_moves) {
    int count = 0;
    // 遍历所有合法移动并评分
    u64 temp_moves = moves;
    while (temp_moves) {
        int pos = __builtin_ctzll(temp_moves);
        temp_moves &= (temp_moves - 1);
        
        sorted_moves[count].pos = pos;
        sorted_moves[count].score = evaluate_move(board, pos);
        count++;
    }
    // 按分数排序
    qsort(sorted_moves, count, sizeof(ScoredMove), compare_moves);
    return count;
}

// uint64_t gen_moves(uint64_t my_pieces, uint64_t opp_pieces){
//     Board b = {{my_pieces, opp_pieces}};
//     return generate_moves(b);
// }

// 新语义：play_to_end(my, opp, &final_my, &final_opp)
// - 输入 my/opp：my 始终为“当前准备走子一方”的棋盘，opp 为对方
// - 输出 final_my/final_opp：按相同语义返回终局棋盘（初始 my 一方的最终棋盘，初始 opp 一方的最终棋盘）
// - 返回值：>0 表示 my 获胜，<0 表示 opp 获胜，0 表示平局；当不处理（如空位>8）时返回 PLAY_TO_END_SENTINEL
int play_to_end(uint64_t my,
                uint64_t opp,
                uint64_t *final_my,
                uint64_t *final_opp)
{
    int empties = 64 - COUNT_BITS(my | opp);
    if (empties > 8) {
        if (final_my)  *final_my  = my;
        if (final_opp) *final_opp = opp;
        return PLAY_TO_END_SENTINEL;
    }
    // 构造以“当前行动方”视角的 Board；初始轮到 my
    Board board = {{my, opp}};
    bool board0_is_my = true;  // board.board[0] 当前是否表示初始 my 方

    while (1) {
        u64 moves = generate_moves(board);
        if (!moves) {
            // 检查对手是否也无棋 -> 终局
            Board opp = board; SWAP_BOARD(opp);
            u64 opp_moves = generate_moves(opp);
            if (!opp_moves) break;          // 双无可走 -> 结束
            board = opp;                    // PASS：交换行动方
            board0_is_my = !board0_is_my;
            continue;
        }
        // 使用宏 EMPTY_SQUARES 计算剩余空格（只用于防御/调试）
        int cur_empties = COUNT_BITS(EMPTY_SQUARES(board));
        if (cur_empties <= 0) break; // 保险退出

        // 选最佳着法（根层也采用 PVS/alpha-beta 策略）
        ScoredMove sorted_moves[32];
        int move_count = get_sorted_moves(board, moves, sorted_moves);

        int best_pos   = -1;
        int best_score = -INT_MAX;
        int alpha = -INT_MAX;
        int beta  = INT_MAX;
        bool first = true;

        for (int i = 0; i < move_count; i++) {
            int pos = sorted_moves[i].pos;
            Board nb = make_move(board, pos);
            SWAP_BOARD(nb); // 下完后轮到对手
            int val;
            if (first) {
                // 第一个全窗口
                val = -perfect_search_pvs(nb, cur_empties - 1, -beta, -alpha, true);
                first = false;
            } else {
                // Null-window 试探
                val = -perfect_search_pvs(nb, cur_empties - 1, -alpha - 1, -alpha, false);
                if (val > alpha && val < beta) {
                    // 需要重搜
                    val = -perfect_search_pvs(nb, cur_empties - 1, -beta, -alpha, true);
                }
            }
            if (val > best_score) {
                best_score = val;
                best_pos = pos;
            }
            if (val > alpha) alpha = val;
            if (alpha >= beta) break; // 剪枝
        }

        // 应用找到的最佳走法，推进棋局状态
        if (best_pos >= 0) {
            board = make_move(board, best_pos); // 当前方落子
            SWAP_BOARD(board);                  // 轮到对手（保持 generate_moves 视角一致）
            board0_is_my = !board0_is_my;       // 当前视角标识翻转
        } else {
            // 理论上不应发生：有 moves 却未选出 best_pos
            break;
        }
    }
    // 终局统计（按初始 my/opp 标签返回）
    uint64_t fmy, fopp;
    if (board0_is_my) {
        fmy  = board.board[0];
        fopp = board.board[1];
    } else {
        fmy  = board.board[1];
        fopp = board.board[0];
    }
    if (final_my)  *final_my  = fmy;
    if (final_opp) *final_opp = fopp;

    int my_cnt  = COUNT_BITS(fmy);
    int opp_cnt = COUNT_BITS(fopp);
    if (my_cnt > opp_cnt) return 1;
    if (my_cnt < opp_cnt) return -1;
    return 0;
}